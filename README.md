# Deep Reinforcement Learning for DDoS Mitigation

This repository contains the code, configuration, and experimental results for the paper:

**"Deep Reinforcement Learning Driven Mitigation of Distributed Denial of Service Attacks in Cloud Server Environments"**

## 📂 Repository Structure

- `run_drl_ddos_realtime.py`  
  Main script for training and testing DRL agents (PPO, A2C, DQN, SAC, TD3) in a live Docker-based cloud environment.

- `results/`  
  Contains experiment logs, JSON master logs, and generated plots.


## 🧪 Testing Procedure

1. **Start Docker containers**  
   ```bash
   docker compose up -d
   ```

2. **Run DRL experiment**  
   Example for PPO:  
   ```bash
   python run_drl_ddos_realtime.py --algo PPO --episodes 900000
   ```

3. **View results**  
   - Logs: `results/full_experiment_log.json`  
   - Charts: `results/plots/`

## 📊 Algorithms Tested

- Proximal Policy Optimization (PPO)  
- Advantage Actor-Critic (A2C)  
- Deep Q-Network (DQN)  
- Soft Actor-Critic (SAC)  
- Twin Delayed Deep Deterministic Policy Gradient (TD3)  
- Rule-based Baseline  

## ⚙️ Requirements

- Python 3.10+  
- PyTorch 2.x  
- Stable-Baselines3  
- Docker & Docker Compose  

## 📬 Contact

**Enes Bajrami**  
PhD Candidate in Software Engineering and Artificial Intelligence  
Faculty of Computer Science and Engineering, Ss. Cyril and Methodius University
Skopje, North Macedonia

Email: enes.bajrami@students.finki.ukim.mk
