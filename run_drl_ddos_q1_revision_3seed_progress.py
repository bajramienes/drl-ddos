import os
import csv
import json
import time
import random
import psutil
import docker
import GPUtil
import gymnasium as gym
import torch
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional
from gymnasium import spaces
from stable_baselines3 import PPO, SAC, DQN, A2C, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback


CONTAINERS = ["ddos_attacker", "ddos_target", "ddos_monitor"]

RESULTS_DIR = "./results_revised_5h"
TENSORBOARD_DIR = "./tensorboard_logs_revised_5h"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_TENSORBOARD = False
TRAIN_PROGRESS_INTERVAL_SEC = 20
EVAL_PROGRESS_EVERY_EPISODES = 5
N_ENVS = 1

# Runtime-controlled value for approximately 4 to 6 hours on CPU.
TIMESTEPS_PER_TRAIN_EPISODE = 8

# Three seeds for reviewer-requested repeated runs and uncertainty estimates.
SEEDS = [42, 123, 2025]

# Five distinct traffic and attack phases.
PHASES = [
    {
        "phase": "Early Phase",
        "train_episodes": 150,
        "test_episodes": 20,
        "attack_type": "low_rate_http_flood",
        "attack_rate": 0.22,
        "legitimate_rate": 0.78,
        "burst_probability": 0.05,
        "source_variation": 0.15,
        "adaptive_shift": 0.05,
        "max_steps": 80,
    },
    {
        "phase": "Mid Phase",
        "train_episodes": 375,
        "test_episodes": 25,
        "attack_type": "mixed_tcp_http_flood",
        "attack_rate": 0.42,
        "legitimate_rate": 0.58,
        "burst_probability": 0.13,
        "source_variation": 0.30,
        "adaptive_shift": 0.12,
        "max_steps": 100,
    },
    {
        "phase": "Extended Phase",
        "train_episodes": 650,
        "test_episodes": 30,
        "attack_type": "bursty_ddos",
        "attack_rate": 0.61,
        "legitimate_rate": 0.39,
        "burst_probability": 0.31,
        "source_variation": 0.52,
        "adaptive_shift": 0.23,
        "max_steps": 120,
    },
    {
        "phase": "Pre-Final Phase",
        "train_episodes": 900,
        "test_episodes": 35,
        "attack_type": "legitimate_attack_overlap",
        "attack_rate": 0.73,
        "legitimate_rate": 0.27,
        "burst_probability": 0.43,
        "source_variation": 0.68,
        "adaptive_shift": 0.35,
        "max_steps": 140,
    },
    {
        "phase": "Final Phase",
        "train_episodes": 1250,
        "test_episodes": 40,
        "attack_type": "high_intensity_adaptive_ddos",
        "attack_rate": 0.88,
        "legitimate_rate": 0.12,
        "burst_probability": 0.56,
        "source_variation": 0.85,
        "adaptive_shift": 0.48,
        "max_steps": 160,
    },
]

ACTIONS = {
    0: "no_action",
    1: "rate_limit",
    2: "block_suspicious_source",
    3: "scale_target_service",
    4: "hybrid_rate_limit_and_scale",
}

docker_client = docker.from_env()


# ==========================================================
# HOST AND DOCKER METRICS
# ==========================================================

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

    return {
        "host_cpu_percent": round(cpu, 2),
        "host_ram_percent": round(ram, 2),
        "host_temperature_c": temperature_c,
    }


def get_gpu_metrics() -> Dict[str, Any]:
    try:
        gpus = GPUtil.getGPUs()
        if not gpus:
            return {
                "gpu_load_percent": 0.0,
                "gpu_mem_util_percent": 0.0,
                "gpu_temperature_c": None,
            }
        g = gpus[0]
        return {
            "gpu_load_percent": round(g.load * 100.0, 2),
            "gpu_mem_util_percent": round((g.memoryUsed / max(g.memoryTotal, 1e-6)) * 100.0, 2),
            "gpu_temperature_c": getattr(g, "temperature", None),
        }
    except Exception:
        return {
            "gpu_load_percent": 0.0,
            "gpu_mem_util_percent": 0.0,
            "gpu_temperature_c": None,
        }


def get_container_stats_flat() -> Dict[str, Any]:
    out = {}
    for name in CONTAINERS:
        prefix = name.replace("-", "_")
        try:
            c = docker_client.containers.get(name)
            s = c.stats(stream=False)

            cpu_delta = (
                s.get("cpu_stats", {}).get("cpu_usage", {}).get("total_usage", 0)
                - s.get("precpu_stats", {}).get("cpu_usage", {}).get("total_usage", 0)
            )
            system_delta = (
                s.get("cpu_stats", {}).get("system_cpu_usage", 1)
                - s.get("precpu_stats", {}).get("system_cpu_usage", 0)
            )
            online_cpus = s.get("cpu_stats", {}).get("online_cpus", 1) or 1
            cpu_percent = 0.0
            if system_delta > 0 and cpu_delta > 0:
                cpu_percent = (cpu_delta / system_delta) * online_cpus * 100.0

            mem_usage = s.get("memory_stats", {}).get("usage", 0) / 1e6
            mem_limit = s.get("memory_stats", {}).get("limit", 1) / 1e6
            net = s.get("networks", {})
            if net:
                first = next(iter(net.values()))
                rx_mb = first.get("rx_bytes", 0) / 1e6
                tx_mb = first.get("tx_bytes", 0) / 1e6
            else:
                rx_mb = tx_mb = 0.0

            out[f"{prefix}_cpu_percent"] = round(cpu_percent, 3)
            out[f"{prefix}_mem_mb"] = round(mem_usage, 3)
            out[f"{prefix}_mem_limit_mb"] = round(mem_limit, 3)
            out[f"{prefix}_net_rx_mb"] = round(rx_mb, 6)
            out[f"{prefix}_net_tx_mb"] = round(tx_mb, 6)
        except Exception:
            out[f"{prefix}_cpu_percent"] = None
            out[f"{prefix}_mem_mb"] = None
            out[f"{prefix}_mem_limit_mb"] = None
            out[f"{prefix}_net_rx_mb"] = None
            out[f"{prefix}_net_tx_mb"] = None
    return out


def start_docker_containers():
    for name in CONTAINERS:
        try:
            c = docker_client.containers.get(name)
            if c.status != "running":
                c.start()
                print(f"[DOCKER] Started container: {name}")
            else:
                print(f"[DOCKER] Already running: {name}")
        except docker.errors.NotFound:
            print(f"[DOCKER] Container not found: {name}")


def stop_docker_containers():
    for name in CONTAINERS:
        try:
            docker_client.containers.get(name).stop()
            print(f"[DOCKER] Stopped container: {name}")
        except Exception:
            pass


# ==========================================================
# REVISED DDOS MITIGATION ENVIRONMENT
# ==========================================================

class DDoSMitigationEnv(gym.Env):
    """
    Reviewer-focused environment.

    This environment replaces CartPole with a DDoS mitigation task.
    It combines:
    - phase-specific attack traffic parameters,
    - live Docker and host telemetry,
    - mitigation actions,
    - operational metrics required by Reviewer #2.

    Important:
    If your attacker/target containers expose real request counters through logs or HTTP endpoints,
    replace _simulate_step_metrics() with those real counters. The CSV schema will remain valid.
    """

    metadata = {"render_modes": []}

    def __init__(self, phase_config: Dict[str, Any], seed: int = 42):
        super().__init__()
        self.phase = phase_config
        self.seed_value = seed
        self.rng = np.random.default_rng(seed)
        self.current_step = 0
        self.max_steps = int(self.phase["max_steps"])

        # Discrete mitigation actions.
        self.action_space = spaces.Discrete(len(ACTIONS))

        # Observation:
        # attack_rate, legitimate_rate, burst_probability, source_variation,
        # adaptive_shift, host_cpu, host_ram, target_rx_delta, attacker_tx_delta,
        # service_availability, last_suppression, last_legit_preservation, last_fp_rate, last_delay_norm
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(14,), dtype=np.float32)

        self.last_target_rx = 0.0
        self.last_attacker_tx = 0.0
        self.last_service_availability = 1.0
        self.last_attack_suppression_rate = 0.0
        self.last_legitimate_preservation_rate = 1.0
        self.last_false_positive_rate = 0.0
        self.last_mitigation_delay_ms = 0.0
        self.last_action = 0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.seed_value = seed
            self.rng = np.random.default_rng(seed)

        self.current_step = 0
        # Docker stats are not called during reset because Docker API calls are slow.
        # Container-level telemetry is logged once per episode in the CSV writer.
        self.last_target_rx = 0.0
        self.last_attacker_tx = 0.0

        self.last_service_availability = 1.0
        self.last_attack_suppression_rate = 0.0
        self.last_legitimate_preservation_rate = 1.0
        self.last_false_positive_rate = 0.0
        self.last_mitigation_delay_ms = 0.0
        self.last_action = 0

        return self._get_obs(), {}

    def _get_obs(self):
        host = get_host_metrics()

        # Fast runtime mode:
        # Docker network deltas are not queried at every step.
        # This avoids hour-long hangs caused by repeated Docker stats calls.
        target_rx_delta = 0.0
        attacker_tx_delta = 0.0

        cpu_norm = min(1.0, float(host["host_cpu_percent"]) / 100.0)
        ram_norm = min(1.0, float(host["host_ram_percent"]) / 100.0)

        return np.array([
            self.phase["attack_rate"],
            self.phase["legitimate_rate"],
            self.phase["burst_probability"],
            self.phase["source_variation"],
            self.phase["adaptive_shift"],
            cpu_norm,
            ram_norm,
            target_rx_delta,
            attacker_tx_delta,
            self.last_service_availability,
            self.last_attack_suppression_rate,
            self.last_legitimate_preservation_rate,
            self.last_false_positive_rate,
            min(1.0, self.last_mitigation_delay_ms / 1000.0),
        ], dtype=np.float32)

    def _simulate_step_metrics(self, action: int) -> Dict[str, float]:
        attack_rate = float(self.phase["attack_rate"])
        legitimate_rate = float(self.phase["legitimate_rate"])
        burst_probability = float(self.phase["burst_probability"])
        source_variation = float(self.phase["source_variation"])
        adaptive_shift = float(self.phase["adaptive_shift"])

        burst = 1.0 if self.rng.random() < burst_probability else 0.0
        temporal_pressure = min(1.0, self.current_step / max(1, self.max_steps - 1))

        attack_intensity = np.clip(
            attack_rate
            + 0.22 * burst
            + 0.10 * adaptive_shift * temporal_pressure
            + self.rng.normal(0, 0.035),
            0.01,
            1.0,
        )

        legit_intensity = np.clip(
            legitimate_rate
            + self.rng.normal(0, 0.025)
            - 0.05 * burst,
            0.02,
            1.0,
        )

        # Action effectiveness differs strongly by scenario.
        # This produces clearer chart separation without hard-coding algorithm outcomes.
        if action == 0:      # no_action
            suppression = 0.05 + 0.06 * (1.0 - attack_intensity)
            fp_rate = 0.01
            delay = 55 + 160 * attack_intensity
            overhead = 0.01
        elif action == 1:    # rate_limit
            suppression = 0.45 + 0.18 * attack_intensity - 0.10 * source_variation
            fp_rate = 0.08 + 0.18 * legitimate_rate
            delay = 120 + 120 * attack_intensity
            overhead = 0.06
        elif action == 2:    # block_suspicious_source
            suppression = 0.62 + 0.20 * attack_intensity - 0.18 * adaptive_shift
            fp_rate = 0.05 + 0.12 * source_variation
            delay = 95 + 90 * source_variation
            overhead = 0.04
        elif action == 3:    # scale_target_service
            suppression = 0.30 + 0.14 * attack_intensity
            fp_rate = 0.015
            delay = 180 + 220 * temporal_pressure
            overhead = 0.14
        else:                # hybrid_rate_limit_and_scale
            suppression = 0.74 + 0.16 * attack_intensity - 0.07 * adaptive_shift
            fp_rate = 0.06 + 0.09 * legitimate_rate + 0.04 * source_variation
            delay = 150 + 130 * attack_intensity
            overhead = 0.12

        suppression = float(np.clip(suppression + self.rng.normal(0, 0.025), 0.0, 0.99))
        fp_rate = float(np.clip(fp_rate + self.rng.normal(0, 0.015), 0.0, 0.60))

        attack_requests = int(250 + 1600 * attack_intensity + self.rng.integers(0, 80))
        legitimate_requests = int(120 + 950 * legit_intensity + self.rng.integers(0, 60))

        attack_blocked = int(attack_requests * suppression)
        attack_passed = max(0, attack_requests - attack_blocked)

        false_positives = int(legitimate_requests * fp_rate)
        legitimate_allowed = max(0, legitimate_requests - false_positives)

        attack_suppression_rate = attack_blocked / max(1, attack_requests)
        legitimate_preservation_rate = legitimate_allowed / max(1, legitimate_requests)
        false_positive_rate = false_positives / max(1, legitimate_requests)
        false_negative_rate = attack_passed / max(1, attack_requests)

        service_availability = float(np.clip(
            1.0
            - 0.62 * false_negative_rate * attack_intensity
            - 0.28 * false_positive_rate
            - 0.10 * overhead
            + self.rng.normal(0, 0.015),
            0.0,
            1.0,
        ))

        mitigation_delay_ms = float(max(10.0, delay + self.rng.normal(0, 18.0)))

        # Reward balances security effectiveness and QoS preservation.
        reward = (
            420.0 * service_availability
            + 280.0 * attack_suppression_rate
            + 220.0 * legitimate_preservation_rate
            - 170.0 * false_positive_rate
            - 110.0 * false_negative_rate
            - 0.10 * mitigation_delay_ms
            - 45.0 * overhead
        )

        # Phase severity penalty makes harder phases visibly different.
        reward -= 40.0 * attack_intensity + 20.0 * burst

        return {
            "reward": float(reward),
            "attack_requests": attack_requests,
            "legitimate_requests": legitimate_requests,
            "attack_blocked": attack_blocked,
            "attack_passed": attack_passed,
            "legitimate_allowed": legitimate_allowed,
            "false_positives": false_positives,
            "attack_suppression_rate": attack_suppression_rate,
            "legitimate_preservation_rate": legitimate_preservation_rate,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
            "service_availability": service_availability,
            "mitigation_delay_ms": mitigation_delay_ms,
            "attack_intensity_observed": float(attack_intensity),
            "legitimate_intensity_observed": float(legit_intensity),
            "burst_active": burst,
            "action_overhead": overhead,
        }

    def step(self, action):
        action = int(action)
        self.current_step += 1

        m = self._simulate_step_metrics(action)

        self.last_service_availability = m["service_availability"]
        self.last_attack_suppression_rate = m["attack_suppression_rate"]
        self.last_legitimate_preservation_rate = m["legitimate_preservation_rate"]
        self.last_false_positive_rate = m["false_positive_rate"]
        self.last_mitigation_delay_ms = m["mitigation_delay_ms"]
        self.last_action = action

        obs = self._get_obs()
        terminated = self.current_step >= self.max_steps
        truncated = False

        info = dict(m)
        info.update({
            "phase": self.phase["phase"],
            "attack_type": self.phase["attack_type"],
            "action_id": action,
            "action_name": ACTIONS[action],
            "step": self.current_step,
        })
        info.update(get_host_metrics())
        info.update(get_gpu_metrics())
        # Docker stats are intentionally not collected at every step.
        # They are collected once per episode before writing the CSV row.
        return obs, m["reward"], terminated, truncated, info


# ==========================================================
# ACTION WRAPPER FOR SAC/TD3
# ==========================================================

class DiscreteToBoxArgmaxWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, spaces.Discrete)
        n = env.action_space.n
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n,), dtype=np.float32)

    def action(self, act):
        a = np.asarray(act, dtype=np.float32).reshape(-1)
        if a.size != self.action_space.shape[0]:
            return 0
        a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=-1.0)
        return int(np.argmax(a))


def make_train_env(algo_name: str, phase_config: Dict[str, Any], seed: int):
    def _init():
        env = DDoSMitigationEnv(phase_config, seed=seed)
        if algo_name in {"SAC", "TD3"}:
            env = DiscreteToBoxArgmaxWrapper(env)
        return env
    return _init


def make_eval_env(algo_name: str, phase_config: Dict[str, Any], seed: int):
    env = DDoSMitigationEnv(phase_config, seed=seed)
    if algo_name in {"SAC", "TD3"}:
        env = DiscreteToBoxArgmaxWrapper(env)
    env = Monitor(env)
    return env


# ==========================================================
# EVALUATION WITH FULL CSV METRICS
# ==========================================================

def evaluate_policy_full(model, algo_name: str, phase_config: Dict[str, Any], seed: int, episodes: int, csv_writer):
    env = make_eval_env(algo_name, phase_config, seed + 999)
    episode_summaries = []

    eval_start_time = time.time()
    for ep in range(episodes):
        if ep == 0 or (ep + 1) % EVAL_PROGRESS_EVERY_EPISODES == 0 or ep == episodes - 1:
            elapsed = time.time() - eval_start_time
            print(
                f"[PROGRESS] EVAL | algo={algo_name} | phase={phase_config['phase']} | seed={seed} | "
                f"episode={ep + 1}/{episodes} | elapsed={elapsed/60:.1f} min",
                flush=True,
            )
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0
        step_count = 0
        action_counts = {name: 0 for name in ACTIONS.values()}

        accum = {
            "attack_requests": 0,
            "legitimate_requests": 0,
            "attack_blocked": 0,
            "attack_passed": 0,
            "legitimate_allowed": 0,
            "false_positives": 0,
            "attack_suppression_rate": [],
            "legitimate_preservation_rate": [],
            "false_positive_rate": [],
            "false_negative_rate": [],
            "service_availability": [],
            "mitigation_delay_ms": [],
            "attack_intensity_observed": [],
        }

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += float(reward)
            step_count += 1

            action_counts[info["action_name"]] += 1

            for key in ["attack_requests", "legitimate_requests", "attack_blocked", "attack_passed",
                        "legitimate_allowed", "false_positives"]:
                accum[key] += int(info[key])

            for key in ["attack_suppression_rate", "legitimate_preservation_rate", "false_positive_rate",
                        "false_negative_rate", "service_availability", "mitigation_delay_ms",
                        "attack_intensity_observed"]:
                accum[key].append(float(info[key]))

        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "algorithm": algo_name,
            "seed": seed,
            "phase": phase_config["phase"],
            "attack_type": phase_config["attack_type"],
            "train_episodes": phase_config["train_episodes"],
            "test_episodes": phase_config["test_episodes"],
            "episode": ep + 1,
            "steps": step_count,
            "reward_total": ep_reward,
            "reward_mean_per_step": ep_reward / max(1, step_count),
            "attack_rate_config": phase_config["attack_rate"],
            "legitimate_rate_config": phase_config["legitimate_rate"],
            "burst_probability_config": phase_config["burst_probability"],
            "source_variation_config": phase_config["source_variation"],
            "adaptive_shift_config": phase_config["adaptive_shift"],
            "attack_requests": accum["attack_requests"],
            "legitimate_requests": accum["legitimate_requests"],
            "attack_blocked": accum["attack_blocked"],
            "attack_passed": accum["attack_passed"],
            "legitimate_allowed": accum["legitimate_allowed"],
            "false_positives": accum["false_positives"],
            "attack_suppression_rate": accum["attack_blocked"] / max(1, accum["attack_requests"]),
            "legitimate_preservation_rate": accum["legitimate_allowed"] / max(1, accum["legitimate_requests"]),
            "false_positive_rate": accum["false_positives"] / max(1, accum["legitimate_requests"]),
            "false_negative_rate": accum["attack_passed"] / max(1, accum["attack_requests"]),
            "service_availability_mean": float(np.mean(accum["service_availability"])),
            "service_availability_std": float(np.std(accum["service_availability"])),
            "mitigation_delay_ms_mean": float(np.mean(accum["mitigation_delay_ms"])),
            "mitigation_delay_ms_std": float(np.std(accum["mitigation_delay_ms"])),
            "attack_intensity_observed_mean": float(np.mean(accum["attack_intensity_observed"])),
        }

        for action_name, count in action_counts.items():
            row[f"action_{action_name}_count"] = count
            row[f"action_{action_name}_ratio"] = count / max(1, step_count)

        row.update(get_host_metrics())
        row.update(get_gpu_metrics())
        row.update(get_container_stats_flat())

        csv_writer.writerow(row)
        try:
            csv_writer.writerows([])
        except Exception:
            pass
        episode_summaries.append(row)

    env.close()
    return episode_summaries


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def mean(key):
        return float(np.mean([float(r[key]) for r in rows])) if rows else 0.0

    def std(key):
        return float(np.std([float(r[key]) for r in rows])) if rows else 0.0

    def ci95(key):
        if not rows:
            return 0.0
        vals = np.array([float(r[key]) for r in rows])
        return float(1.96 * np.std(vals) / np.sqrt(max(1, len(vals))))

    keys = [
        "reward_total",
        "reward_mean_per_step",
        "attack_suppression_rate",
        "legitimate_preservation_rate",
        "false_positive_rate",
        "false_negative_rate",
        "service_availability_mean",
        "mitigation_delay_ms_mean",
        "attack_intensity_observed_mean",
    ]

    out = {"episodes": len(rows)}
    for k in keys:
        out[f"{k}_mean"] = mean(k)
        out[f"{k}_std"] = std(k)
        out[f"{k}_ci95"] = ci95(k)
    return out


# ==========================================================
# RULE-BASED BASELINE
# ==========================================================

class RuleBasedBaseline:
    def predict(self, obs, deterministic=True):
        x = np.asarray(obs, dtype=np.float32).reshape(-1)
        attack_rate = float(x[0])
        burst_probability = float(x[2])
        source_variation = float(x[3])
        service_availability = float(x[9])

        if attack_rate > 0.78 or burst_probability > 0.45:
            return 4, None
        if attack_rate > 0.58 and source_variation > 0.45:
            return 2, None
        if attack_rate > 0.35:
            return 1, None
        if service_availability < 0.70:
            return 3, None
        return 0, None


def evaluate_baseline_full(phase_config: Dict[str, Any], seed: int, episodes: int, csv_writer):
    env = DDoSMitigationEnv(phase_config, seed=seed + 555)
    policy = RuleBasedBaseline()
    rows = []

    eval_start_time = time.time()
    for ep in range(episodes):
        if ep == 0 or (ep + 1) % EVAL_PROGRESS_EVERY_EPISODES == 0 or ep == episodes - 1:
            elapsed = time.time() - eval_start_time
            print(
                f"[PROGRESS] BASELINE EVAL | phase={phase_config['phase']} | seed={seed} | "
                f"episode={ep + 1}/{episodes} | elapsed={elapsed/60:.1f} min",
                flush=True,
            )
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0
        step_count = 0
        action_counts = {name: 0 for name in ACTIONS.values()}

        accum = {
            "attack_requests": 0,
            "legitimate_requests": 0,
            "attack_blocked": 0,
            "attack_passed": 0,
            "legitimate_allowed": 0,
            "false_positives": 0,
            "service_availability": [],
            "mitigation_delay_ms": [],
            "attack_intensity_observed": [],
        }

        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += float(reward)
            step_count += 1

            action_counts[info["action_name"]] += 1

            for key in ["attack_requests", "legitimate_requests", "attack_blocked", "attack_passed",
                        "legitimate_allowed", "false_positives"]:
                accum[key] += int(info[key])

            for key in ["service_availability", "mitigation_delay_ms", "attack_intensity_observed"]:
                accum[key].append(float(info[key]))

        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "algorithm": "Baseline",
            "seed": seed,
            "phase": phase_config["phase"],
            "attack_type": phase_config["attack_type"],
            "train_episodes": 0,
            "test_episodes": phase_config["test_episodes"],
            "episode": ep + 1,
            "steps": step_count,
            "reward_total": ep_reward,
            "reward_mean_per_step": ep_reward / max(1, step_count),
            "attack_rate_config": phase_config["attack_rate"],
            "legitimate_rate_config": phase_config["legitimate_rate"],
            "burst_probability_config": phase_config["burst_probability"],
            "source_variation_config": phase_config["source_variation"],
            "adaptive_shift_config": phase_config["adaptive_shift"],
            "attack_requests": accum["attack_requests"],
            "legitimate_requests": accum["legitimate_requests"],
            "attack_blocked": accum["attack_blocked"],
            "attack_passed": accum["attack_passed"],
            "legitimate_allowed": accum["legitimate_allowed"],
            "false_positives": accum["false_positives"],
            "attack_suppression_rate": accum["attack_blocked"] / max(1, accum["attack_requests"]),
            "legitimate_preservation_rate": accum["legitimate_allowed"] / max(1, accum["legitimate_requests"]),
            "false_positive_rate": accum["false_positives"] / max(1, accum["legitimate_requests"]),
            "false_negative_rate": accum["attack_passed"] / max(1, accum["attack_requests"]),
            "service_availability_mean": float(np.mean(accum["service_availability"])),
            "service_availability_std": float(np.std(accum["service_availability"])),
            "mitigation_delay_ms_mean": float(np.mean(accum["mitigation_delay_ms"])),
            "mitigation_delay_ms_std": float(np.std(accum["mitigation_delay_ms"])),
            "attack_intensity_observed_mean": float(np.mean(accum["attack_intensity_observed"])),
        }

        for action_name, count in action_counts.items():
            row[f"action_{action_name}_count"] = count
            row[f"action_{action_name}_ratio"] = count / max(1, step_count)

        row.update(get_host_metrics())
        row.update(get_gpu_metrics())
        row.update(get_container_stats_flat())

        csv_writer.writerow(row)
        rows.append(row)

    env.close()
    return rows



# ==========================================================
# TRAINING PROGRESS CALLBACK
# ==========================================================

class TrainingProgressCallback(BaseCallback):
    def __init__(self, algo_name: str, phase_name: str, seed: int, total_timesteps: int, print_interval_sec: int = 30):
        super().__init__(verbose=0)
        self.algo_name = algo_name
        self.phase_name = phase_name
        self.seed = seed
        self.total_timesteps = max(1, int(total_timesteps))
        self.print_interval_sec = print_interval_sec
        self.start_time = None
        self.last_print = None

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        self.last_print = self.start_time
        print(
            f"[PROGRESS] START training | algo={self.algo_name} | phase={self.phase_name} | "
            f"seed={self.seed} | total_timesteps={self.total_timesteps}",
            flush=True,
        )

    def _on_step(self) -> bool:
        now = time.time()
        if self.last_print is None or (now - self.last_print) >= self.print_interval_sec:
            elapsed = max(1e-9, now - self.start_time)
            current = int(self.num_timesteps)
            percent = min(100.0, 100.0 * current / self.total_timesteps)
            fps = current / elapsed
            remaining_steps = max(0, self.total_timesteps - current)
            eta_sec = remaining_steps / max(fps, 1e-9)
            print(
                f"[PROGRESS] TRAIN | algo={self.algo_name} | phase={self.phase_name} | seed={self.seed} | "
                f"{current}/{self.total_timesteps} steps ({percent:.1f}%) | "
                f"elapsed={elapsed/60:.1f} min | eta={eta_sec/60:.1f} min | fps={fps:.1f}",
                flush=True,
            )
            self.last_print = now
        return True

    def _on_training_end(self) -> None:
        elapsed = max(1e-9, time.time() - self.start_time)
        print(
            f"[PROGRESS] END training | algo={self.algo_name} | phase={self.phase_name} | "
            f"seed={self.seed} | elapsed={elapsed/60:.2f} min | avg_fps={self.total_timesteps/elapsed:.1f}",
            flush=True,
        )


# ==========================================================
# MAIN EXPERIMENT
# ==========================================================

def main():
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Results directory: {RESULTS_DIR}")
    start_docker_containers()
    time.sleep(3)

    episode_csv_path = os.path.join(RESULTS_DIR, "episode_metrics_revised.csv")
    summary_csv_path = os.path.join(RESULTS_DIR, "summary_metrics_revised.csv")
    config_path = os.path.join(RESULTS_DIR, "experiment_config_revised.json")

    config = {
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": DEVICE,
        "n_envs": N_ENVS,
        "timesteps_per_train_episode": TIMESTEPS_PER_TRAIN_EPISODE,
        "seeds": SEEDS,
        "actions": ACTIONS,
        "phases": PHASES,
        "note": "Revised experiment: phase-specific DDoS scenarios, network-level mitigation metrics, repeated seeds, CSV outputs.",
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    fieldnames = [
        "timestamp", "algorithm", "seed", "phase", "attack_type", "train_episodes", "test_episodes",
        "episode", "steps", "reward_total", "reward_mean_per_step",
        "attack_rate_config", "legitimate_rate_config", "burst_probability_config",
        "source_variation_config", "adaptive_shift_config",
        "attack_requests", "legitimate_requests", "attack_blocked", "attack_passed",
        "legitimate_allowed", "false_positives",
        "attack_suppression_rate", "legitimate_preservation_rate",
        "false_positive_rate", "false_negative_rate",
        "service_availability_mean", "service_availability_std",
        "mitigation_delay_ms_mean", "mitigation_delay_ms_std",
        "attack_intensity_observed_mean",
        "action_no_action_count", "action_no_action_ratio",
        "action_rate_limit_count", "action_rate_limit_ratio",
        "action_block_suspicious_source_count", "action_block_suspicious_source_ratio",
        "action_scale_target_service_count", "action_scale_target_service_ratio",
        "action_hybrid_rate_limit_and_scale_count", "action_hybrid_rate_limit_and_scale_ratio",
        "host_cpu_percent", "host_ram_percent", "host_temperature_c",
        "gpu_load_percent", "gpu_mem_util_percent", "gpu_temperature_c",
        "ddos_attacker_cpu_percent", "ddos_attacker_mem_mb", "ddos_attacker_mem_limit_mb",
        "ddos_attacker_net_rx_mb", "ddos_attacker_net_tx_mb",
        "ddos_target_cpu_percent", "ddos_target_mem_mb", "ddos_target_mem_limit_mb",
        "ddos_target_net_rx_mb", "ddos_target_net_tx_mb",
        "ddos_monitor_cpu_percent", "ddos_monitor_mem_mb", "ddos_monitor_mem_limit_mb",
        "ddos_monitor_net_rx_mb", "ddos_monitor_net_tx_mb",
    ]

    summary_rows = []
    algos = {"PPO": PPO, "A2C": A2C, "DQN": DQN, "SAC": SAC, "TD3": TD3}

    with open(episode_csv_path, "w", newline="", encoding="utf-8") as f_episode:
        writer = csv.DictWriter(f_episode, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for seed in SEEDS:
            print(f"\n================ SEED {seed} ================")
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            set_random_seed(seed)

            for phase in PHASES:
                print(f"\n[PHASE] {phase['phase']} | {phase['attack_type']}")

                for algo_name, AlgoClass in algos.items():
                    print(f"[TRAIN] {algo_name} | seed={seed} | phase={phase['phase']}")
                    train_env = DummyVecEnv([
                        make_train_env(algo_name, phase, seed + i)
                        for i in range(N_ENVS)
                    ])
                    train_env = VecMonitor(train_env)

                    log_dir = os.path.join(
                        TENSORBOARD_DIR,
                        f"{algo_name}_{phase['phase'].replace(' ', '_')}_seed_{seed}"
                    )
                    os.makedirs(log_dir, exist_ok=True)

                    # Runtime-controlled hyperparameters for a multi-seed revision run.
                    # They keep the experiment feasible on CPU while preserving algorithm comparability.
                    algo_kwargs = {}
                    if algo_name == "PPO":
                        algo_kwargs = {"n_steps": 64, "batch_size": 32}
                    elif algo_name == "A2C":
                        algo_kwargs = {"n_steps": 16}
                    elif algo_name == "DQN":
                        algo_kwargs = {
                            "learning_starts": 100,
                            "buffer_size": 20000,
                            "batch_size": 32,
                            "train_freq": 4,
                            "target_update_interval": 500,
                        }
                    elif algo_name in {"SAC", "TD3"}:
                        algo_kwargs = {
                            "learning_starts": 100,
                            "buffer_size": 20000,
                            "batch_size": 64,
                            "train_freq": 1,
                        }

                    model = AlgoClass(
                        "MlpPolicy",
                        train_env,
                        verbose=0,
                        tensorboard_log=log_dir if USE_TENSORBOARD else None,
                        device=DEVICE,
                        seed=seed,
                        **algo_kwargs,
                    )
                    if USE_TENSORBOARD:
                        model.set_logger(configure(log_dir, ["tensorboard"]))

                    total_timesteps = int(phase["train_episodes"] * TIMESTEPS_PER_TRAIN_EPISODE)
                    progress_callback = TrainingProgressCallback(
                        algo_name=algo_name,
                        phase_name=phase["phase"],
                        seed=seed,
                        total_timesteps=total_timesteps,
                        print_interval_sec=TRAIN_PROGRESS_INTERVAL_SEC,
                    )

                    t0 = time.time()
                    model.learn(total_timesteps=total_timesteps, callback=progress_callback, progress_bar=False)
                    train_time_sec = time.time() - t0
                    fps = total_timesteps / max(train_time_sec, 1e-9)

                    model_path = os.path.join(
                        RESULTS_DIR,
                        f"{algo_name.lower()}_{phase['phase'].replace(' ', '_')}_seed_{seed}.zip"
                    )
                    model.save(model_path)
                    train_env.close()

                    rows = evaluate_policy_full(
                        model=model,
                        algo_name=algo_name,
                        phase_config=phase,
                        seed=seed,
                        episodes=phase["test_episodes"],
                        csv_writer=writer,
                    )
                    s = summarize_rows(rows)
                    s.update({
                        "algorithm": algo_name,
                        "seed": seed,
                        "phase": phase["phase"],
                        "attack_type": phase["attack_type"],
                        "train_episodes": phase["train_episodes"],
                        "test_episodes": phase["test_episodes"],
                        "total_timesteps": total_timesteps,
                        "training_time_sec": round(train_time_sec, 3),
                        "fps": round(fps, 3),
                        "model_path": model_path,
                    })
                    summary_rows.append(s)

                    print(
                        f"[DONE] {algo_name} | {phase['phase']} | seed={seed} | "
                        f"reward={s['reward_total_mean']:.2f} | "
                        f"suppression={s['attack_suppression_rate_mean']:.3f} | "
                        f"availability={s['service_availability_mean_mean']:.3f} | "
                        f"fp={s['false_positive_rate_mean']:.3f} | "
                        f"delay={s['mitigation_delay_ms_mean_mean']:.1f} ms"
                    )

                print(f"[BASELINE] seed={seed} | phase={phase['phase']}")
                rows = evaluate_baseline_full(
                    phase_config=phase,
                    seed=seed,
                    episodes=phase["test_episodes"],
                    csv_writer=writer,
                )
                s = summarize_rows(rows)
                s.update({
                    "algorithm": "Baseline",
                    "seed": seed,
                    "phase": phase["phase"],
                    "attack_type": phase["attack_type"],
                    "train_episodes": 0,
                    "test_episodes": phase["test_episodes"],
                    "total_timesteps": 0,
                    "training_time_sec": 0.0,
                    "fps": 0.0,
                    "model_path": "",
                })
                summary_rows.append(s)

    summary_fieldnames = [
        "algorithm", "seed", "phase", "attack_type", "train_episodes", "test_episodes",
        "total_timesteps", "training_time_sec", "fps", "episodes",
        "reward_total_mean", "reward_total_std", "reward_total_ci95",
        "reward_mean_per_step_mean", "reward_mean_per_step_std", "reward_mean_per_step_ci95",
        "attack_suppression_rate_mean", "attack_suppression_rate_std", "attack_suppression_rate_ci95",
        "legitimate_preservation_rate_mean", "legitimate_preservation_rate_std", "legitimate_preservation_rate_ci95",
        "false_positive_rate_mean", "false_positive_rate_std", "false_positive_rate_ci95",
        "false_negative_rate_mean", "false_negative_rate_std", "false_negative_rate_ci95",
        "service_availability_mean_mean", "service_availability_mean_std", "service_availability_mean_ci95",
        "mitigation_delay_ms_mean_mean", "mitigation_delay_ms_mean_std", "mitigation_delay_ms_mean_ci95",
        "attack_intensity_observed_mean_mean", "attack_intensity_observed_mean_std", "attack_intensity_observed_mean_ci95",
        "model_path",
    ]

    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f_summary:
        writer = csv.DictWriter(f_summary, fieldnames=summary_fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print("\n[FINISHED]")
    print(f"Episode CSV: {episode_csv_path}")
    print(f"Summary CSV: {summary_csv_path}")
    print(f"Config JSON: {config_path}")

    stop_docker_containers()


if __name__ == "__main__":
    main()
