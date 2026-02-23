import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from stable_baselines3 import PPO
from cloud_env import CloudDataCenterEnv


N_EPISODES  = 30
MAX_STEPS   = 200
MODEL_PATH  = "ppo_cloud_optimizer"
RESULTS_IMG = "results.png"


try:
    model = PPO.load(MODEL_PATH)
    print(f"✔  Loaded model from {MODEL_PATH}.zip")
except FileNotFoundError:
    try:
        model = PPO.load("best_model/best_model")
        print("✔  Loaded best_model checkpoint")
    except FileNotFoundError:
        raise FileNotFoundError(
            "No trained model found. Run  python train.py  first."
        )

env = CloudDataCenterEnv(max_steps=MAX_STEPS)



def round_robin_action(step: int, **_) -> int:
    return step % 3


def least_loaded_action(obs: np.ndarray, **_) -> int:
    cpu_loads = obs[:3]
    return int(np.argmin(cpu_loads))



def run_episode(strategy: str, seed: int):
    obs, _ = env.reset(seed=seed)
    total_reward  = 0.0
    energy_log    = []
    survived      = True

    for step in range(MAX_STEPS):
        if strategy == "RL":
            action, _ = model.predict(obs, deterministic=True)
        elif strategy == "Round-Robin":
            action = round_robin_action(step=step)
        else:  # Least-Loaded
            action = least_loaded_action(obs=obs)

        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
        energy_log.append(info["energy_cost"])

        if terminated:
            survived = False
            break
        if truncated:
            break

    return {
        "survived":      survived,
        "total_reward":  total_reward,
        "energy_log":    energy_log,
        "final_queue":   info["queue"],
        "total_energy":  sum(energy_log),
    }



strategies = ["RL", "Round-Robin", "Least-Loaded"]
results    = {s: [] for s in strategies}

print(f"\nRunning {N_EPISODES} evaluation episodes per strategy …\n")
for seed in range(N_EPISODES):
    for s in strategies:
        results[s].append(run_episode(s, seed=seed))


def stats(strategy):
    eps = results[strategy]
    return {
        "survival_rate":  np.mean([e["survived"]     for e in eps]) * 100,
        "mean_reward":    np.mean([e["total_reward"]  for e in eps]),
        "mean_energy":    np.mean([e["total_energy"]  for e in eps]),
        "mean_queue":     np.mean([e["final_queue"]   for e in eps]),
    }

summary = {s: stats(s) for s in strategies}

col_w = 16
header = f"{'Strategy':<{col_w}}{'Survival %':>12}{'Mean Reward':>14}{'Mean Energy':>14}{'Mean Queue':>12}"
divider = "─" * len(header)
print(divider)
print(header)
print(divider)
for s in strategies:
    st = summary[s]
    print(
        f"{s:<{col_w}}"
        f"{st['survival_rate']:>11.1f}%"
        f"{st['mean_reward']:>14.1f}"
        f"{st['mean_energy']:>14.2f}"
        f"{st['mean_queue']:>12.1f}"
    )
print(divider)

rl, rr = summary["RL"], summary["Round-Robin"]
print(f"\nRL vs Round-Robin:")
print(f"  Energy reduction : {(rr['mean_energy'] - rl['mean_energy']) / rr['mean_energy'] * 100:+.1f}%")
print(f"  Reward gain      : {(rl['mean_reward']  - rr['mean_reward'])  / abs(rr['mean_reward']) * 100:+.1f}%")
print(f"  Survival delta   : {rl['survival_rate'] - rr['survival_rate']:+.1f} pp")


def mean_trace(strategy):
    """Average per-step energy across all episodes (pad shorter epis with last value)."""
    logs = [e["energy_log"] for e in results[strategy]]
    max_len = max(len(l) for l in logs)
    padded  = [l + [l[-1]] * (max_len - len(l)) for l in logs]
    return np.mean(padded, axis=0)

traces = {s: mean_trace(s) for s in strategies}


COLORS = {"RL": "#2196F3", "Round-Robin": "#F44336", "Least-Loaded": "#FF9800"}

fig = plt.figure(figsize=(16, 10), facecolor="#0f1117")
fig.suptitle(
    "Cloud Data Center RL Optimiser — Evaluation Report",
    fontsize=16, fontweight="bold", color="white", y=0.98,
)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]

def style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor("#1a1d27")
    ax.set_title(title,  color="white", fontsize=12, pad=8)
    ax.set_xlabel(xlabel, color="#aaaaaa", fontsize=9)
    ax.set_ylabel(ylabel, color="#aaaaaa", fontsize=9)
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_color("#444444")
    ax.grid(True, color="#333333", linewidth=0.5, linestyle="--")

ax = axes[0]
for s in strategies:
    ax.plot(traces[s], label=s, color=COLORS[s], linewidth=2,
            linestyle="-" if s == "RL" else "--")
style_ax(ax, "⚡ Energy Cost Over Time (avg across episodes)",
         "Time Step", "Energy Cost (lower = better)")
ax.legend(facecolor="#252830", labelcolor="white", fontsize=9)

ax = axes[1]
rates  = [summary[s]["survival_rate"] for s in strategies]
colors = [COLORS[s] for s in strategies]
bars   = ax.bar(strategies, rates, color=colors, edgecolor="#555555", width=0.5)
for bar, rate in zip(bars, rates):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{rate:.0f}%", ha="center", va="bottom", color="white", fontsize=11)
style_ax(ax, "🛡️ Survival Rate (%)", "Strategy", "% Episodes Survived")
ax.set_ylim(0, 115)

ax = axes[2]
rewards = [summary[s]["mean_reward"] for s in strategies]
bars    = ax.bar(strategies, rewards, color=colors, edgecolor="#555555", width=0.5)
for bar, val in zip(bars, rewards):
    ypos = bar.get_height() + (5 if val >= 0 else -15)
    ax.text(bar.get_x() + bar.get_width() / 2, ypos,
            f"{val:.0f}", ha="center", va="bottom", color="white", fontsize=11)
style_ax(ax, "🏆 Mean Cumulative Reward", "Strategy", "Reward (higher = better)")

ax = axes[3]
all_energies = [[e["total_energy"] for e in results[s]] for s in strategies]
bp = ax.boxplot(
    all_energies,
    labels=strategies,
    patch_artist=True,
    medianprops=dict(color="white", linewidth=2),
    boxprops=dict(linewidth=1.2),
    whiskerprops=dict(color="#888888"),
    capprops=dict(color="#888888"),
    flierprops=dict(marker="o", markersize=4, color="#888888", alpha=0.5),
)
for patch, color in zip(bp["boxes"], [COLORS[s] for s in strategies]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
style_ax(ax, "📦 Total Energy Distribution", "Strategy", "Total Energy Cost")

plt.savefig(RESULTS_IMG, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\n✔  Evaluation plot saved → {RESULTS_IMG}")
plt.show()