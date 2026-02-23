import streamlit as st
import numpy as np
import time
import plotly.graph_objects as go
from stable_baselines3 import PPO
from cloud_env import CloudDataCenterEnv


st.set_page_config(
    page_title="AI Cloud Optimizer",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  /* Streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 0.6rem 1.4rem 0 1.4rem !important; max-width: 100% !important; }
  div[data-testid="stVerticalBlock"] { gap: 0.3rem !important; }
  div[data-testid="column"] { padding: 0 6px !important; }

  /* Hero bar */
  .hero { display:flex; align-items:center; gap:12px; padding:4px 0 6px 0; }
  .hero-title { font-size:1.35rem; font-weight:700; color:#fff; white-space:nowrap; }
  .hero-sub   { font-size:0.75rem; color:#888; }

  /* Info badges */
  .badge { display:inline-block; background:#1e2130; border:1px solid #333;
           border-radius:6px; padding:3px 10px; font-size:0.72rem;
           color:#ccc; white-space:nowrap; }
  .badge b { color:#fff; font-size:0.82rem; }

  /* Panel header */
  .ph-rl { color:#2196F3; font-size:0.95rem; font-weight:700;
           border-bottom:2px solid #2196F3; padding-bottom:3px; margin:0 0 4px 0; }

  /* Server card */
  .sc { background:#13151f; border:1px solid #2a2d3a; border-radius:8px;
        padding:7px 10px; margin:3px 0; }
  .sc-target { border-color:#2196F3 !important; box-shadow:0 0 8px #2196F344; }
  .sc-crash  { opacity:0.45; border-color:#555 !important; }
  .sc-hdr { display:flex; justify-content:space-between; margin-bottom:3px; }
  .sc-name { color:#fff; font-weight:600; font-size:0.78rem; }
  .sc-lbl  { color:#777; font-size:0.68rem; margin-top:3px; }
  .bar-bg  { background:#232638; border-radius:3px; height:6px;
             overflow:hidden; margin:2px 0 0 0; }
  .bar-fg  { height:100%; border-radius:3px; }

  /* Crash banner */
  .crash  { background:#b71c1c; border-radius:6px; padding:5px 8px;
            text-align:center; color:#fff; font-size:0.75rem; font-weight:700;
            margin:3px 0; }

  /* Stat tile */
  .tile { background:#13151f; border:1px solid #2a2d3a; border-radius:8px;
          padding:7px 10px; text-align:center; }
  .tile-val { font-size:1.1rem; font-weight:700; color:#fff; line-height:1.1; }
  .tile-lbl { font-size:0.65rem; color:#777; margin-top:1px; }
  .tile-sub { font-size:0.65rem; color:#555; }

  /* Log entries */
  .le { font-size:0.7rem; color:#bbb; padding:2px 0;
        border-bottom:1px solid #1e2130; font-family:monospace; }

  /* Result banners */
  .win  { background:#1b5e20; border:1px solid #4caf50; border-radius:8px;
          padding:8px 14px; color:#a5d6a7; font-size:0.85rem;
          font-weight:700; text-align:center; margin-top:6px; }
  .done { background:#1a1d27; border:1px solid #333; border-radius:8px;
          padding:8px 14px; color:#aaa; font-size:0.82rem;
          text-align:center; margin-top:6px; }
</style>
""", unsafe_allow_html=True)


def _cpu_color(v):
    if v < 0.50: return "#4caf50"
    if v < 0.75: return "#ff9800"
    if v < 0.90: return "#f44336"
    return "#b71c1c"

def _temp_color(v):
    if v < 0.40: return "#2196f3"
    if v < 0.70: return "#ff9800"
    return "#f44336"

def _dot(v):
    if v < 0.50: return "🟢"
    if v < 0.75: return "🟡"
    return "🔴"

def bar(value, color, height=6):
    pct = int(np.clip(value, 0, 1) * 100)
    return (f'<div class="bar-bg"><div class="bar-fg" '
            f'style="width:{pct}%;background:{color};height:{height}px;"></div></div>')

def server_card(sid, cpu, ram, temp, is_target=False, is_dead=False):
    extra = ' sc-target' if is_target else (' sc-crash' if is_dead else '')
    tag   = ("⚡ " if is_target else ("💀 " if is_dead else ""))
    return f"""
<div class="sc{extra}">
  <div class="sc-hdr">
    <span class="sc-name">{tag}Server {sid}</span>
    <span style="font-size:0.75rem">{_dot(cpu)}</span>
  </div>
  <div class="sc-lbl">CPU {cpu*100:.1f}%</div>{bar(cpu, _cpu_color(cpu))}
  <div class="sc-lbl">RAM {ram*100:.1f}%</div>{bar(ram, "#7c4dff")}
  <div class="sc-lbl">Temp {temp*100:.1f}°</div>{bar(temp, _temp_color(temp), 4)}
</div>"""

def tile(label, value, sub="", color="#fff"):
    return (f'<div class="tile"><div class="tile-val" style="color:{color}">{value}</div>'
            f'<div class="tile-lbl">{label}</div>'
            + (f'<div class="tile-sub">{sub}</div>' if sub else '')
            + '</div>')

def badge(label, val, color="#ccc"):
    return f'<span class="badge"><b style="color:{color}">{val}</b> {label}</span>'



@st.cache_resource
def load_resources():
    env = CloudDataCenterEnv()
    for path in ("ppo_cloud_optimizer", "best_model/best_model"):
        try:
            agent = PPO.load(path)
            return env, agent, path
        except Exception:
            continue
    raise FileNotFoundError("No trained model found. Run  python train.py  first.")

env_rl, model, model_path = load_resources()


hdr_col, ctrl_col = st.columns([3, 2])

with hdr_col:
    st.markdown(
        '<div class="hero">'
        '<div><div class="hero-title">☁️ Autonomous Cloud Resource Optimizer</div>'
        '<div class="hero-sub">Deep Reinforcement Learning (PPO) &nbsp;·&nbsp; '
        'Inspired by DeepMind\'s 40% cooling reduction at Google</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        badge("PPO Agent", "PPO", "#2196F3") + "  " +
        badge("State space", "11-dim", "#7c4dff") + "  " +
        badge("Servers", "3", "#4caf50") + "  " +
        badge("Reward", "Energy·Latency·Temp", "#ff9800") + "  " +
        badge("Model", model_path.split("/")[-1], "#888"),
        unsafe_allow_html=True,
    )

with ctrl_col:
    cc1, cc2, cc3 = st.columns([1.2, 1, 1.4])
    speed = cc1.selectbox("Speed", ["Fast", "Normal", "Slow"], index=0, label_visibility="collapsed")
    seed  = cc2.number_input("Seed", value=42, min_value=0, max_value=999, step=1, label_visibility="collapsed")
    run_btn = cc3.button("▶️ Run Simulation", type="primary", use_container_width=True)

speed_map = {"Fast": 0.07, "Normal": 0.18, "Slow": 0.38}

st.markdown('<hr style="margin:4px 0; border-color:#222;">', unsafe_allow_html=True)

if run_btn:
    rl_obs, _ = env_rl.reset(seed=int(seed))

    rl_cpu  = env_rl.cpu_loads.copy()
    rl_ram  = env_rl.ram_loads.copy()
    rl_temp = env_rl.temps.copy()
    rl_info: dict = {"energy_cost": 0.0, "queue": env_rl.queue}
    rl_action    = 0
    rl_alive     = True
    rl_term      = False
    rl_showed_crash = False
    rl_energy_hist: list = []
    task_log: list = []
    total_steps = 200

    col_srv, col_stat = st.columns([1, 1.6])

    col_srv.markdown('<div class="ph-rl">🤖 PPO RL Agent — Live Server Status</div>',
                     unsafe_allow_html=True)
    col_stat.markdown(
        '<span style="color:#aaa;font-size:0.85rem;font-weight:600;">📊 Metrics &amp; Live Chart</span>',
        unsafe_allow_html=True,
    )

    rl_s = [col_srv.empty() for _ in range(3)]
    rl_banner = col_srv.empty()

    t_row1  = col_stat.columns(2)
    t_energy = t_row1[0].empty()
    t_reward = t_row1[1].empty()
    t_row2  = col_stat.columns(2)
    t_queue = t_row2[0].empty()
    t_step  = t_row2[1].empty()

    col_stat.markdown(
        '<div style="color:#555;font-size:0.65rem;margin:4px 0 1px 0;">ENERGY COST OVER TIME</div>',
        unsafe_allow_html=True,
    )
    chart_slot = col_stat.empty()

    col_stat.markdown(
        '<div style="color:#555;font-size:0.65rem;margin:4px 0 1px 0;">TASK LOG</div>',
        unsafe_allow_html=True,
    )
    log_slot = col_stat.empty()

    summary_slot = st.empty()

    cumulative_reward = 0.0

    for step in range(total_steps):

        if rl_alive:
            try:
                raw_action, _ = model.predict(rl_obs, deterministic=True)
                rl_action = int(raw_action)
                rl_obs, step_reward, rl_term, rl_trunc, rl_info = env_rl.step(rl_action)
                rl_cpu   = rl_info["cpu_loads"].copy()
                rl_ram   = rl_info["ram_loads"].copy()
                rl_temp  = rl_info["temps"].copy()
                rl_alive = not (rl_term or rl_trunc)
                cumulative_reward += float(step_reward)
            except Exception as exc:
                rl_term  = True
                rl_alive = False
                rl_info["crash_reason"] = f"predict error: {exc}"

        rl_energy_hist.append(float(rl_info.get("energy_cost", 0.0)))


        raw_size = float(rl_obs[10])
        size_tag = "LRG" if raw_size > 0.60 else ("MED" if raw_size > 0.30 else "SML")
        task_log.insert(0, {"step": step + 1, "size": size_tag, "server": rl_action + 1})
        task_log = task_log[:8]

        for i, slot in enumerate(rl_s):
            slot.markdown(
                server_card(i + 1, rl_cpu[i], rl_ram[i], rl_temp[i],
                            is_target=(rl_alive and rl_action == i),
                            is_dead=not rl_alive),
                unsafe_allow_html=True,
            )
        if rl_term and not rl_showed_crash:
            reason = rl_info.get("crash_reason", "overload")
            rl_banner.markdown(
                f'<div class="crash">💥 RL CRASHED — {reason}</div>',
                unsafe_allow_html=True,
            )
            rl_showed_crash = True

        rl_e = rl_energy_hist[-1]
        q_val = int(rl_info.get("queue", 0))

        t_energy.markdown(tile("Energy Cost", f"{rl_e:.2f}", "lower = better", "#2196F3"),
                          unsafe_allow_html=True)
        t_reward.markdown(tile("Cumulative Reward", f"{cumulative_reward:.0f}",
                               "higher = better",
                               "#4caf50" if cumulative_reward >= 0 else "#f44336"),
                          unsafe_allow_html=True)
        t_queue.markdown(tile("Queue Backlog", f"{q_val} tasks", color="#ff9800"),
                         unsafe_allow_html=True)
        t_step.markdown(tile("Step", f"{step+1}/{total_steps}", color="#888"),
                        unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=rl_energy_hist, mode="lines", name="RL Agent — Energy Cost",
            line=dict(color="#2196F3", width=2),
            fill="tozeroy", fillcolor="rgba(33,150,243,0.08)",
        ))
        fig.update_layout(
            margin=dict(l=30, r=10, t=8, b=25),
            height=150,
            plot_bgcolor="#0d1117",
            paper_bgcolor="#0d1117",
            font=dict(color="#888", size=9),
            showlegend=False,
            xaxis=dict(gridcolor="#1e2130", showgrid=True, title="Step"),
            yaxis=dict(gridcolor="#1e2130", showgrid=True, title="Energy"),
        )
        chart_slot.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        size_colors = {"SML": "#4caf50", "MED": "#ff9800", "LRG": "#f44336"}
        rows = "".join(
            f'<div class="le">s{r["step"]:>3} '
            f'<span style="color:{size_colors[r["size"]]}">[{r["size"]}]</span>'
            f' → Server {r["server"]}</div>'
            for r in task_log
        )
        log_slot.markdown(rows, unsafe_allow_html=True)

        time.sleep(speed_map[speed])

        if not rl_alive:
            break

    total_energy = sum(rl_energy_hist)
    steps_run    = len(rl_energy_hist)
    avg_energy   = total_energy / max(steps_run, 1)
    rl_status    = "✅ SURVIVED" if (rl_alive or not rl_term) else "❌ CRASHED"
    color        = "#1b5e20" if (rl_alive or not rl_term) else "#b71c1c"
    border       = "#4caf50" if (rl_alive or not rl_term) else "#f44336"
    txt_col      = "#a5d6a7" if (rl_alive or not rl_term) else "#ef9a9a"

    summary_slot.markdown(
        f'<div style="background:{color};border:1px solid {border};border-radius:8px;'
        f'padding:8px 14px;color:{txt_col};font-size:0.85rem;font-weight:700;'
        f'text-align:center;margin-top:6px;">'
        f'{rl_status} &nbsp;·&nbsp; Steps: {steps_run}/{total_steps}'
        f' &nbsp;·&nbsp; Total Energy: {total_energy:.1f}'
        f' &nbsp;·&nbsp; Avg Energy/Step: {avg_energy:.2f}'
        f' &nbsp;·&nbsp; Cumulative Reward: {cumulative_reward:.0f}'
        f'</div>',
        unsafe_allow_html=True,
    )