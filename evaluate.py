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


def run_episode(seed: int):
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    energy_log   = []
    survived     = True

    for step in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
        energy_log.append(info["energy_cost"])

        if terminated:
            survived = False
            break
        if truncated:
            break

    return {
        "survived":     survived,
        "total_reward": total_reward,
        "energy_log":   energy_log,
        "final_queue":  info["queue"],
        "total_energy": sum(energy_log),
    }


print(f"\nRunning {N_EPISODES} evaluation episodes …\n")
episodes = [run_episode(seed=s) for s in range(N_EPISODES)]

survival_rate = np.mean([e["survived"]     for e in episodes]) * 100
mean_reward   = np.mean([e["total_reward"] for e in episodes])
mean_energy   = np.mean([e["total_energy"] for e in episodes])
mean_queue    = np.mean([e["final_queue"]  for e in episodes])

col_w = 18
header  = f"{'Metric':<{col_w}}{'Value':>14}"
divider = "─" * (col_w + 14)
print(divider)
print(header)
print(divider)
print(f"{'Survival Rate':<{col_w}}{survival_rate:>13.1f}%")
print(f"{'Mean Reward':<{col_w}}{mean_reward:>14.1f}")
print(f"{'Mean Total Energy':<{col_w}}{mean_energy:>14.2f}")
print(f"{'Mean Final Queue':<{col_w}}{mean_queue:>14.1f}")
print(divider)


def mean_trace_with_std():
    """Per-step mean ± std energy, padded to the same length."""
    logs    = [e["energy_log"] for e in episodes]
    max_len = max(len(l) for l in logs)
    padded  = np.array([l + [l[-1]] * (max_len - len(l)) for l in logs])
    return padded.mean(axis=0), padded.std(axis=0)


mean_e, std_e   = mean_trace_with_std()
rewards_per_ep  = [e["total_reward"] for e in episodes]
energies_per_ep = [e["total_energy"] for e in episodes]
survived_count  = int(sum(e["survived"] for e in episodes))
crashed_count   = N_EPISODES - survived_count

RL_COLOR = "#2196F3"
RL_FILL  = "#2196F344"

fig = plt.figure(figsize=(16, 10), facecolor="#0f1117")
fig.suptitle(
    "Cloud Data Center RL Optimiser — PPO Agent Evaluation  (30 episodes)",
    fontsize=16, fontweight="bold", color="white", y=0.98,
)
gs   = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]


def style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor("#1a1d27")
    ax.set_title(title,  color="white",   fontsize=12, pad=8)
    ax.set_xlabel(xlabel, color="#aaaaaa", fontsize=9)
    ax.set_ylabel(ylabel, color="#aaaaaa", fontsize=9)
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_color("#444444")
    ax.grid(True, color="#333333", linewidth=0.5, linestyle="--")


# ── Panel 1: Energy cost over time ──────────────────────────────────────────
ax = axes[0]
x  = np.arange(len(mean_e))
ax.plot(x, mean_e, color=RL_COLOR, linewidth=2, label="PPO Agent")
ax.fill_between(x, mean_e - std_e, mean_e + std_e, color=RL_FILL)
style_ax(ax, "⚡ Energy Cost Over Time (mean ± std)",
         "Time Step", "Energy Cost (lower = better)")
ax.legend(facecolor="#252830", labelcolor="white", fontsize=9)

# ── Panel 2: Survival pie / bar ──────────────────────────────────────────────
ax = axes[1]
wedge_colors = ["#4caf50", "#f44336"]
wedges, texts, autotexts = ax.pie(
    [survived_count, crashed_count],
    labels=["Survived", "Crashed"],
    colors=wedge_colors,
    autopct="%1.0f%%",
    startangle=90,
    textprops={"color": "white", "fontsize": 11},
    wedgeprops={"edgecolor": "#0f1117", "linewidth": 2},
)
for at in autotexts:
    at.set_fontsize(13)
    at.set_fontweight("bold")
ax.set_facecolor("#1a1d27")
ax.set_title("🛡️ Survival Rate  (30 episodes)", color="white", fontsize=12, pad=8)

# ── Panel 3: Reward distribution ─────────────────────────────────────────────
ax = axes[2]
ax.hist(rewards_per_ep, bins=12, color=RL_COLOR, edgecolor="#0f1117",
        alpha=0.85, linewidth=0.8)
ax.axvline(mean_reward, color="#ffeb3b", linewidth=1.8,
           linestyle="--", label=f"Mean = {mean_reward:.0f}")
style_ax(ax, "🏆 Cumulative Reward Distribution",
         "Total Reward per Episode", "# Episodes")
ax.legend(facecolor="#252830", labelcolor="white", fontsize=9)

# ── Panel 4: Total energy distribution ───────────────────────────────────────
ax = axes[3]
bp = ax.boxplot(
    energies_per_ep,
    vert=True,
    patch_artist=True,
    widths=0.4,
    medianprops=dict(color="white",    linewidth=2.5),
    boxprops=dict(linewidth=1.2),
    whiskerprops=dict(color="#888888", linewidth=1.2),
    capprops=dict(color="#888888",     linewidth=1.2),
    flierprops=dict(marker="o", markersize=5, color=RL_COLOR, alpha=0.6),
)
bp["boxes"][0].set_facecolor(RL_COLOR)
bp["boxes"][0].set_alpha(0.75)
ax.set_xticks([1])
ax.set_xticklabels(["PPO Agent"], color="#aaaaaa", fontsize=10)
style_ax(ax, "📦 Total Energy Cost Distribution",
         "Strategy", "Total Energy Cost per Episode")

plt.savefig(RESULTS_IMG, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print(f"\n✔  Evaluation plot saved → {RESULTS_IMG}")
plt.show()
