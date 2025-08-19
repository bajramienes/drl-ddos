import os
import json
import time
import psutil
import docker
import GPUtil
import gymnasium as gym
import torch
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from stable_baselines3 import PPO, SAC, DQN, A2C, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from gymnasium import spaces

# =====================
# USER CONFIG
# =====================
ENV_ID = "CartPole-v1"  # TODO: replace with your real env id
CONTAINERS = ["ddos_attacker", "ddos_target", "ddos_monitor"]

TIMESTEPS_PER_TRAIN_EPISODE = 1000

PHASES = [
    (120,  80,  "Early Phase"),
    (320,  80,  "Mid Phase"),
    (520,  80,  "Extended Phase"),
    (700, 100,  "Pre-Final Phase"),
    (900, 100,  "Final Phase"),
]

WORK_SCALE = 1.0
N_ENVS     = 4

RESULTS_DIR     = "./results"
TENSORBOARD_DIR = "./tensorboard_logs"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] PyTorch device = {DEVICE} (torch.cuda.is_available()={torch.cuda.is_available()})")

docker_client = docker.from_env()

# =====================
# METRICS HELPERS
# =====================
def get_host_metrics() -> Dict[str, Any]:
    cpu = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory().percent
    temperature_c = None
    try:
        temps = psutil.sensors_temperatures(fahrenheit=False)
        if temps:
            grp = temps.get("coretemp") or next(iter(temps.values()))
            vals = [t.current for t in grp if hasattr(t, "current")]
            if vals:
                temperature_c = round(sum(vals) / len(vals), 2)
    except Exception:
        temperature_c = None
    return {"cpu_percent": round(cpu, 2), "ram_percent": round(ram, 2), "temperature_c": temperature_c}

def get_gpu_metrics() -> List[Dict[str, Any]]:
    out = []
    try:
        for g in GPUtil.getGPUs():
            out.append({
                "id": g.id,
                "name": g.name,
                "load_percent": round(g.load * 100.0, 2),
                "mem_used_mb": round(g.memoryUsed, 2),
                "mem_total_mb": round(g.memoryTotal, 2),
                "mem_util_percent": round((g.memoryUsed / max(g.memoryTotal, 1e-6)) * 100.0, 2),
                "temperature_c": getattr(g, "temperature", None),
            })
    except Exception:
        pass
    return out

def get_docker_stats() -> Dict[str, Any]:
    stats = {}
    for name in CONTAINERS:
        try:
            c = docker_client.containers.get(name)
            s = c.stats(stream=False)
            sys_cpu = max(s.get("cpu_stats", {}).get("system_cpu_usage", 1), 1)
            total_cpu = s.get("cpu_stats", {}).get("cpu_usage", {}).get("total_usage", 0)
            cpu_percent = (total_cpu / sys_cpu) * 100.0
            mem_usage = s.get("memory_stats", {}).get("usage", 0) / 1e6
            net = s.get("networks", {})
            if net:
                first = next(iter(net.values()))
                rx_mb = first.get("rx_bytes", 0) / 1e6
                tx_mb = first.get("tx_bytes", 0) / 1e6
            else:
                rx_mb = tx_mb = 0.0
            stats[name] = {
                "cpu_percent": round(cpu_percent, 2),
                "mem_mb": round(mem_usage, 2),
                "net_rx_mb": round(rx_mb, 2),
                "net_tx_mb": round(tx_mb, 2),
            }
        except Exception as e:
            stats[name] = {"error": str(e)}
    return stats

# =====================
# DOCKER CONTROL
# =====================
def start_docker_containers():
    for name in CONTAINERS:
        try:
            c = docker_client.containers.get(name)
            if c.status != "running":
                c.start()
                print(f"Started container: {name}")
            else:
                print(f"Already running: {name}")
        except docker.errors.NotFound:
            print(f"Container not found: {name}")

def stop_docker_containers():
    for name in CONTAINERS:
        try:
            docker_client.containers.get(name).stop()
            print(f"Stopped container: {name}")
        except docker.errors.NotFound:
            pass

# =====================
# ACTION WRAPPERS (force SAC/TD3 on discrete envs)
# =====================
class DiscreteToBoxArgmaxWrapper(gym.ActionWrapper):
    """
    If original action space is Discrete(n), expose Box([-1,1], shape=(n,))
    and map continuous vector -> argmax -> discrete action index.
    """
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, spaces.Discrete), "Wrapper expects Discrete action space"
        n = env.action_space.n
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n,), dtype=np.float32)

    def action(self, act):
        a = np.asarray(act, dtype=np.float32).reshape(-1)
        if a.size != self.action_space.shape[0]:
            # fall back: pick first valid action
            return 0
        a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=-1.0)
        return int(np.argmax(a))

class IdentityActionWrapper(gym.ActionWrapper):
    """Pass-through wrapper for algorithms that already match the env's action space."""
    def __init__(self, env):
        super().__init__(env)
        # keep original action_space
        self.action_space = env.action_space

    def action(self, action):
        # return the action unchanged
        return action

# =====================
# ENV FACTORIES
# =====================
def base_env():
    return gym.make(ENV_ID)

def make_train_env_for_algo(algo_name: str):
    e = base_env()
    if isinstance(e.action_space, spaces.Discrete):
        if algo_name in {"SAC", "TD3"}:
            e = DiscreteToBoxArgmaxWrapper(e)
        else:
            e = IdentityActionWrapper(e)
    # For Box envs, no change needed.
    return e

def make_eval_env_for_algo(algo_name: str):
    e = make_train_env_for_algo(algo_name)
    e = gym.wrappers.RecordEpisodeStatistics(e)
    e = Monitor(e)
    return e

def make_vec_env_for_algo(algo_name: str, n_envs: int) -> VecMonitor:
    env_fns = [lambda: make_train_env_for_algo(algo_name) for _ in range(n_envs)]
    vec = DummyVecEnv(env_fns)
    vec = VecMonitor(vec)
    return vec

# =====================
# EVALUATION
# =====================
def evaluate_deterministic(policy_like, env, episodes: int, is_sb3_model: bool) -> Dict[str, Any]:
    rewards = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_r = 0.0
        while not done:
            if is_sb3_model:
                action, _ = policy_like.predict(obs, deterministic=True)
            else:
                action, _ = policy_like.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_r += float(reward)
        rewards.append(ep_r)
    return {
        "episodes": episodes,
        "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
        "reward_std": float(np.std(rewards)) if rewards else 0.0,
        "reward_min": float(np.min(rewards)) if rewards else 0.0,
        "reward_max": float(np.max(rewards)) if rewards else 0.0,
    }

def calc_iterations(model, total_timesteps: int, n_envs: int) -> Optional[int]:
    steps_per_iter = None
    if hasattr(model, "n_steps"):  # PPO/A2C
        try:
            steps_per_iter = int(model.n_steps) * int(n_envs)
        except Exception:
            pass
    if steps_per_iter:
        return max(1, total_timesteps // steps_per_iter)
    return None

def print_phase_summary(phase_name: str, algo_name: str, t_start: float, total_timesteps: int, iterations: Optional[int]):
    elapsed = time.time() - t_start
    fps = total_timesteps / max(elapsed, 1e-9)
    print("\n------------------------------")
    print(f"| phase               | {phase_name}")
    print(f"| algo                | {algo_name}")
    print(f"| time/               |")
    print(f"|    fps              | {int(fps)}")
    print(f"|    iterations       | {iterations if iterations is not None else '-'}")
    print(f"|    time_elapsed(s)  | {int(elapsed)}")
    print(f"|    total_timesteps  | {total_timesteps}")
    print("------------------------------")

# =====================
# MAIN
# =====================
def main():
    # scale phases
    scaled_phases = []
    for train_eps, test_eps, label in PHASES:
        scaled_train_eps = max(1, int(train_eps * WORK_SCALE))
        scaled_test_eps  = max(1, int(test_eps  * WORK_SCALE))
        scaled_phases.append((scaled_train_eps, scaled_test_eps, label))

    start_docker_containers()
    time.sleep(3)

    master_log: Dict[str, Any] = {
        "device": DEVICE,
        "env_id": ENV_ID,
        "work_scale": WORK_SCALE,
        "n_envs": N_ENVS,
        "algorithms": {},
        "baseline": {},
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    algos: Dict[str, Any] = {"PPO": PPO, "A2C": A2C, "DQN": DQN, "SAC": SAC, "TD3": TD3}

    print(f"\n[INFO] Test session START: {master_log['started_at']}")
    for algo_name, AlgoClass in algos.items():
        master_log["algorithms"][algo_name] = {}
        for train_eps, test_eps, phase_name in scaled_phases:
            print(f"\n=== {phase_name} START ({algo_name}) ===")
            print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            vec_env = make_vec_env_for_algo(algo_name, N_ENVS)

            log_dir = os.path.join(TENSORBOARD_DIR, f"{algo_name}_{phase_name.replace(' ', '_')}")
            model = AlgoClass("MlpPolicy", vec_env, verbose=0, tensorboard_log=log_dir, device=DEVICE)
            model.set_logger(configure(log_dir, ["tensorboard"]))

            total_timesteps = int(train_eps * TIMESTEPS_PER_TRAIN_EPISODE)
            t0 = time.time()
            model.learn(total_timesteps=total_timesteps)
            train_secs = time.time() - t0

            model_path = os.path.join(RESULTS_DIR, f"{algo_name.lower()}_{phase_name.replace(' ', '_')}.zip")
            model.save(model_path)

            eval_env = make_eval_env_for_algo(algo_name)
            eval_stats = evaluate_deterministic(model, eval_env, test_eps, is_sb3_model=True)
            eval_env.close()

            iterations = calc_iterations(model, total_timesteps, N_ENVS)
            print_phase_summary(phase_name, algo_name, t0, total_timesteps, iterations)

            master_log["algorithms"][algo_name][phase_name] = {
                "phase": phase_name,
                "train_episodes": int(train_eps),
                "test_episodes": int(test_eps),
                "timesteps_per_train_episode": TIMESTEPS_PER_TRAIN_EPISODE,
                "total_timesteps": total_timesteps,
                "training_time_sec": round(train_secs, 2),
                "fps": round(total_timesteps / max(train_secs, 1e-9), 2),
                "iterations_est": iterations,
                "eval_reward_mean": eval_stats["reward_mean"],
                "eval_reward_std":  eval_stats["reward_std"],
                "eval_reward_min":  eval_stats["reward_min"],
                "eval_reward_max":  eval_stats["reward_max"],
                "host_metrics_end": get_host_metrics(),
                "gpu_metrics_end":  get_gpu_metrics(),
                "docker_stats_end": get_docker_stats(),
                "model_path": model_path,
            }

            print(f"=== {phase_name} END ({algo_name}) ===")
            vec_env.close()

    # === Rule-based baseline
    print("\n=== BASELINE (rule-based) START ===")
    baseline_all: Dict[str, Any] = {}
    for _, test_eps, phase_name in scaled_phases:
        env = gym.make(ENV_ID)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = Monitor(env)

        class RuleBasedPolicy:
            def __init__(self, action_space):
                self.aspace = action_space
            def predict(self, obs, deterministic=True):
                if isinstance(self.aspace, spaces.Discrete):
                    if self.aspace.n == 2:
                        act = 1 if float(np.nan_to_num(np.asarray(obs)).mean()) > 0 else 0
                    else:
                        score = float(np.nan_to_num(np.asarray(obs)).mean())
                        scores = [score * (i + 1) for i in range(self.aspace.n)]
                        act = int(np.argmax(scores))
                    return act, None
                elif isinstance(self.aspace, spaces.Box):
                    return np.zeros(self.aspace.shape, dtype=np.float32), None
                try:
                    return self.aspace.sample(), None
                except Exception:
                    return 0, None

        baseline = RuleBasedPolicy(env.action_space)
        t0 = time.time()
        stats = evaluate_deterministic(baseline, env, test_eps, is_sb3_model=False)
        secs = time.time() - t0
        env.close()

        print_phase_summary(phase_name, "BASELINE", t0, total_timesteps=test_eps, iterations=None)

        baseline_all[phase_name] = {
            "phase": phase_name,
            "test_episodes": int(test_eps),
            "eval_reward_mean": stats["reward_mean"],
            "eval_reward_std":  stats["reward_std"],
            "eval_reward_min":  stats["reward_min"],
            "eval_reward_max":  stats["reward_max"],
            "testing_time_sec": round(secs, 2),
            "host_metrics_end": get_host_metrics(),
            "gpu_metrics_end":  get_gpu_metrics(),
            "docker_stats_end": get_docker_stats(),
        }

    master_log["baseline"] = baseline_all
    master_log["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    master_path = os.path.join(RESULTS_DIR, "full_experiment_log.json")
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(master_log, f, indent=4)

    print("\nAll testing completed.")
    print(f"   Master log: {master_path}")

    stop_docker_containers()

if __name__ == "__main__":
    main()
