# Adaptive DDoS Mitigation using Deep Reinforcement Learning

[![Journal](https://img.shields.io/badge/Journal-Computer%20Networks-blue)](https://doi.org/10.1016/j.comnet.2026.112530)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.comnet.2026.112530-green)](https://doi.org/10.1016/j.comnet.2026.112530)
[![Python](https://img.shields.io/badge/Python-3.11+-yellow)]()
[![Docker](https://img.shields.io/badge/Docker-Required-blue)]()

Official implementation accompanying the paper:

> **Adaptive DDoS Mitigation using Deep Reinforcement Learning: A Comparative Study in a Live Docker Testbed**

Published in **Computer Networks (Elsevier), Volume 287, 2026**.

---

## Overview

This repository contains the experimental implementation used in the paper, where multiple Deep Reinforcement Learning (DRL) algorithms are evaluated for adaptive Distributed Denial-of-Service (DDoS) mitigation inside a **live Docker-based cloud environment**.

Unlike simulation-only studies, the proposed framework performs mitigation under dynamically generated attack traffic while continuously monitoring system-level and network-level metrics.

The repository includes:

- PPO implementation
- SAC implementation
- DQN implementation
- A2C implementation
- TD3 implementation
- Rule-based baseline
- Docker attack environment
- Evaluation scripts
- Experimental results
- Figures used in the publication

---

## Paper

Enes Bajrami and Florim Idrizi.

**Adaptive DDoS Mitigation using Deep Reinforcement Learning: A Comparative Study in a Live Docker Testbed**

Computer Networks, Volume 287, 2026.

DOI:

https://doi.org/10.1016/j.comnet.2026.112530

---

## Repository Structure

```text
.
├── charts_output/                         Publication-quality figures generated for the paper
├── results/                               Experimental results and evaluation outputs
├── README.md                              Repository documentation
├── docker_testbed.png                     Live Docker-based DDoS experimental environment
├── run_drl_ddos_q1_revision_3seed_progress.py
│                                          Main training and evaluation script
├── start_ddos_env.ps1                     Starts the Docker-based DDoS testbed
└── stop_clean_ddos_env.ps1                Stops containers and cleans the environment
```

---

## Evaluated Algorithms

The following DRL algorithms are implemented and compared:

- Proximal Policy Optimization (PPO)
- Soft Actor-Critic (SAC)
- Deep Q-Network (DQN)
- Advantage Actor-Critic (A2C)
- Twin Delayed DDPG (TD3)

Additionally, a non-learning rule-based mitigation strategy is included as a baseline.

---

## Experimental Environment

- Python
- Stable-Baselines3
- Gymnasium
- PyTorch
- Docker
- NumPy
- Pandas
- Matplotlib

The experiments are executed in a live Docker-based cloud testbed consisting of attacker, victim, and monitoring containers.

---

## Evaluation Metrics

The paper evaluates mitigation performance using:

- Attack Suppression Rate (ASR)
- Legitimate Traffic Preservation (LTP)
- False Positive Rate (FPR)
- False Negative Rate (FNR)
- Service Availability (SA)
- Mitigation Delay (MD)
- Episode Reward

All experiments are repeated using three independent random seeds.

---

## Citation

If you use this repository in your research, please cite:

```bibtex
@article{BAJRAMI2026112530,
  title   = {Adaptive DDoS mitigation using Deep reinforcement learning: A comparative study in a live docker testbed},
  journal = {Computer Networks},
  volume  = {287},
  pages   = {112530},
  year    = {2026},
  issn    = {1389-1286},
  doi     = {10.1016/j.comnet.2026.112530},
  url     = {https://www.sciencedirect.com/science/article/pii/S1389128626005426},
  author  = {Enes Bajrami and Florim Idrizi}
}
```

---

## License

This repository is released for academic and research purposes.

Please cite the associated publication when using this code or any derived results.

---

## Contact

**Enes Bajrami**

Faculty of Computer Science and Engineering (FINKI)

Ss. Cyril and Methodius University in Skopje

Email:
enes.bajrami@students.finki.ukim.mk
