# Cloud RL Optimizer

A deep reinforcement learning agent that autonomously manages task routing across a simulated 3-server cloud data center, optimising for energy efficiency while preventing thermal and resource overloads.

Inspired by [DeepMind's 40% cooling reduction at Google](https://deepmind.google/discover/blog/deepmind-ai-reduces-google-data-centre-cooling-bill-by-40/): energy cost scales as cpu² × temperature, so keeping servers cool **and** balanced is strictly more efficient than overloading any single one.

---

## Results

![Evaluation results](results.png)

The PPO agent is benchmarked against two classical baselines over 30 episodes:
- **Round-Robin** — cycles through servers in fixed order
- **Least-Loaded** — always assigns to the server with the lowest current CPU

---

## Repository Layout

```
cloud_env.py   — Custom Gymnasium environment (3-server physics simulation)
train.py       — PPO training with 4 parallel environments + EvalCallback
evaluate.py    — 30-episode evaluation: RL vs Round-Robin vs Least-Loaded
app.py         — Streamlit live dashboard (real-time server cards + energy chart)
```

---

## Environment

| Property | Value |
|---|---|
| Observation space | 11-dim continuous [0, 1] |
| Action space | Discrete(3) — choose target server |
| Max episode length | 200 steps |
| Traffic model | Poisson arrivals (λ = 1.2), variable task sizes |

**Observation vector:** `[cpu₀, cpu₁, cpu₂, ram₀, ram₁, ram₂, temp₀, temp₁, temp₂, queue_norm, next_task_size_norm]`

**Reward:** survival bonus + balance bonus − energy cost (cpu² × temp) − latency cost − thermal risk − crash penalty (−150)

---

## Agent

PPO (Stable-Baselines3) with a `[256, 256]` MLP policy, 4 parallel `SubprocVecEnv` workers, and an `EvalCallback` that retains the best checkpoint throughout training (300 000 timesteps).

---

## Quick Start

```bash
pip install -r requirements.txt

# Train
python train.py

# Evaluate — generates results.png
python evaluate.py

# Live dashboard
streamlit run app.py
```
