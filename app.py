"""
Streamlit dashboard for the Max Pressure SUMO traffic signal controller.

Run:
    streamlit run dashboard.py

The dashboard launches SUMO (headless by default) in a background thread
and reads shared state every second to display live metrics.

Requirements:
    pip install streamlit plotly pandas
    SUMO_HOME must be set in your environment.
"""

import os
import sys
import time
import threading
import statistics
import collections
import csv
from datetime import datetime
from copy import deepcopy

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ── SUMO / TraCI setup ────────────────────────────────────────────────────────
if "SUMO_HOME" not in os.environ:
    st.error(
        "**SUMO_HOME is not set.**\n\n"
        "Set it before launching Streamlit:\n"
        "```\nexport SUMO_HOME=/usr/share/sumo   # Linux\n"
        "set SUMO_HOME=C:\\Program Files\\Eclipse\\Sumo  # Windows\n```"
    )
    st.stop()

sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
import traci

# ── Intersection topology ─────────────────────────────────────────────────────
TL_ID = "J9"

APPROACHES = {
    "North_SB": {
        "lanes": ["-E5.164_0", "-E5.164_1", "-E5.164_2"],
        "link_indices": [6, 7, 8, 9],
        "color": "#3b82f6",
    },
    "South_NB": {
        "lanes": ["E6.160_0", "E6.160_1"],
        "link_indices": [0, 1, 2],
        "color": "#f59e0b",
    },
    "East_WB": {
        "lanes": ["E9.162_0", "E9.162_1"],
        "link_indices": [3, 4, 5],
        "color": "#10b981",
    },
    "West_EB": {
        "lanes": ["E7.160_0", "E7.160_1"],
        "link_indices": [10, 11, 12],
        "color": "#ef4444",
    },
}

APPROACH_NAMES = list(APPROACHES.keys())
APPROACH_COLORS = [APPROACHES[a]["color"] for a in APPROACH_NAMES]
YELLOW_DURATION = 3

# ── Shared simulation state (written by sim thread, read by Streamlit) ────────
_lock = threading.Lock()

_sim_state = {
    "running": False,
    "stopped": False,
    "error": None,
    "sim_time": 0,
    "total_vehicles": 0,
    "current_phase": "—",
    "phase_timer": 0,
    "phase_duration": 0,
    "approaches": {
        name: {"q": 0, "d": 0.0, "g": 0.0, "P": 0.0, "T": 0.0, "green": False}
        for name in APPROACH_NAMES
    },
    # Rolling history for charts — list of dicts, one per dt tick
    "history": [],
    "phases_detected": [],
    "log": [],          # recent console lines
}


def _log(msg):
    with _lock:
        _sim_state["log"].append(f"[{datetime.now():%H:%M:%S}] {msg}")
        if len(_sim_state["log"]) > 200:
            _sim_state["log"] = _sim_state["log"][-200:]


# ── Controller helpers ────────────────────────────────────────────────────────
def safe_median(seq, fallback=1.0):
    values = [v for v in seq if v > 0]
    return statistics.median(values) if values else fallback


def compute_pressures(state, alpha, beta):
    pressures = {}
    for name, data in state.items():
        q_ref = safe_median(data["q_hist"])
        d_ref = safe_median(data["d_hist"])
        g_ref = safe_median(data["g_hist"])
        P = (data["q"] / q_ref) * (1.0 + alpha * data["d"] / d_ref) + beta * abs(data["g"]) / g_ref
        pressures[name] = max(P, 0.0)
    return pressures


def allocate_green_times(pressures, C, ds, N):
    total_P = sum(pressures.values()) or 1.0
    flexible = max(0, C - N * ds)
    return {name: max(ds, (flexible * P / total_P) + ds) for name, P in pressures.items()}


def detect_phases(tl_id):
    logic = traci.trafficlight.getAllProgramLogics(tl_id)[0]
    link_to_approach = {}
    for aname, meta in APPROACHES.items():
        for idx in meta["link_indices"]:
            link_to_approach[idx] = aname

    phases = []
    for i, phase in enumerate(logic.phases):
        state_str = phase.state
        green_set = set()
        for link_idx, char in enumerate(state_str):
            if char in ("G", "g"):
                approach = link_to_approach.get(link_idx)
                if approach:
                    green_set.add(approach)
        is_yellow = not green_set
        name = (
            "_".join(sorted(green_set)) + "_green" if green_set
            else f"yellow_{i}"
        )
        phases.append({
            "sumo_index": i,
            "name": name,
            "green": sorted(green_set),
            "is_yellow": is_yellow,
            "state_str": state_str,
        })
    return phases


# ── Simulation thread ─────────────────────────────────────────────────────────
def simulation_thread(params):
    alpha    = params["alpha"]
    beta     = params["beta"]
    C        = params["cycle"]
    ds       = params["ds"]
    dt       = params["dt"]
    hist_len = params["history_len"]
    cfg      = params["sumocfg"]
    output   = params["output_csv"]
    use_gui  = params["gui"]
    demand   = params["demand"]   # dict {flow_id: veh_per_hour}

    try:
        binary = "sumo-gui" if use_gui else "sumo"

        # Build per-flow demand overrides as SUMO options
        sumo_cmd = [binary, "-c", cfg, "--no-warnings"]
        traci.start(sumo_cmd)

        # Apply demand overrides via TraCI flow scaling
        # (done by adjusting route file at startup — demand dict is written
        #  to a temp route file before launch; handled in launcher below)

        PHASES = detect_phases(TL_ID)
        with _lock:
            _sim_state["phases_detected"] = PHASES

        N_phases = sum(1 for p in PHASES if not p["is_yellow"])
        if N_phases == 0:
            raise RuntimeError("No green phases detected. Check link_indices in APPROACHES.")

        _log(f"Detected {len(PHASES)} phases ({N_phases} green).")
        for p in PHASES:
            _log(f"  Phase {p['sumo_index']}: {p['name']}  state={p['state_str']}")

        per_approach = {
            name: {
                "q": 0.0, "d": 0.0, "g": 0.0, "q_prev": 0.0,
                "q_hist": collections.deque(maxlen=hist_len),
                "d_hist": collections.deque(maxlen=hist_len),
                "g_hist": collections.deque(maxlen=hist_len),
                "T": float(ds),
            }
            for name in APPROACH_NAMES
        }

        current_phase_idx = 0
        phase_timer       = 0
        pressures         = {n: 1.0 for n in APPROACH_NAMES}
        green_times       = allocate_green_times(pressures, C, ds, N_phases)
        last_dt_tick      = 0

        traci.trafficlight.setPhase(TL_ID, PHASES[current_phase_idx]["sumo_index"])

        csv_file = open(output, "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(["time", "phase", "approach", "Q", "D", "G", "P", "T_allocated"])

        _log("Simulation running.")

        while traci.simulation.getMinExpectedNumber() > 0:
            # Check for stop signal
            with _lock:
                if not _sim_state["running"]:
                    break

            step = int(traci.simulation.getTime())

            # Sample lane metrics
            for name, meta in APPROACHES.items():
                q_sum = sum(traci.lane.getLastStepVehicleNumber(ln) for ln in meta["lanes"])
                d_sum = sum(traci.lane.getWaitingTime(ln) for ln in meta["lanes"])
                n_lanes = len(meta["lanes"])
                per_approach[name]["q"] = float(q_sum)
                per_approach[name]["d"] = d_sum / n_lanes if n_lanes else 0.0

            # Every dt seconds: update growth, refs, pressures
            if step - last_dt_tick >= dt:
                last_dt_tick = step
                for name, data in per_approach.items():
                    data["g"] = (data["q"] - data["q_prev"]) / dt
                    data["q_prev"] = data["q"]
                    data["q_hist"].append(data["q"])
                    data["d_hist"].append(data["d"])
                    data["g_hist"].append(abs(data["g"]))

                pressures   = compute_pressures(per_approach, alpha, beta)
                green_times = allocate_green_times(pressures, C, ds, N_phases)
                for name in per_approach:
                    per_approach[name]["T"] = green_times.get(name, ds)

                # Write CSV
                phase_name = PHASES[current_phase_idx]["name"]
                for name, data in per_approach.items():
                    writer.writerow([
                        step, phase_name, name,
                        round(data["q"], 2), round(data["d"], 2),
                        round(data["g"], 4),
                        round(pressures.get(name, 0), 4),
                        round(data["T"], 1),
                    ])

                # Push to shared state
                history_row = {"time": step}
                for name in APPROACH_NAMES:
                    history_row[f"{name}_q"] = per_approach[name]["q"]
                    history_row[f"{name}_d"] = per_approach[name]["d"]
                    history_row[f"{name}_P"] = pressures.get(name, 0)
                    history_row[f"{name}_T"] = per_approach[name]["T"]

                with _lock:
                    _sim_state["history"].append(history_row)
                    if len(_sim_state["history"]) > 500:
                        _sim_state["history"] = _sim_state["history"][-500:]

            # Phase switching
            phase_timer += 1
            phase_def = PHASES[current_phase_idx]
            if phase_def["is_yellow"]:
                target = YELLOW_DURATION
            else:
                ga = phase_def["green"]
                target = max(ds, round(sum(green_times.get(a, ds) for a in ga) / len(ga))) if ga else ds

            if phase_timer >= target:
                phase_timer = 0
                current_phase_idx = (current_phase_idx + 1) % len(PHASES)
                traci.trafficlight.setPhase(TL_ID, PHASES[current_phase_idx]["sumo_index"])

            # Update shared display state
            total_veh = int(traci.simulation.getMinExpectedNumber())
            green_set = set(PHASES[current_phase_idx]["green"])
            with _lock:
                _sim_state["sim_time"]       = step
                _sim_state["total_vehicles"] = total_veh
                _sim_state["current_phase"]  = PHASES[current_phase_idx]["name"]
                _sim_state["phase_timer"]    = phase_timer
                _sim_state["phase_duration"] = target
                for name in APPROACH_NAMES:
                    _sim_state["approaches"][name] = {
                        "q":     per_approach[name]["q"],
                        "d":     per_approach[name]["d"],
                        "g":     per_approach[name]["g"],
                        "P":     pressures.get(name, 0),
                        "T":     per_approach[name]["T"],
                        "green": name in green_set,
                    }

            traci.simulationStep()

        csv_file.close()
        traci.close()
        _log("Simulation complete.")
        with _lock:
            _sim_state["running"] = False
            _sim_state["stopped"] = True

    except Exception as exc:
        _log(f"ERROR: {exc}")
        with _lock:
            _sim_state["running"] = False
            _sim_state["error"]   = str(exc)
        try:
            traci.close()
        except Exception:
            pass


# ── Route file writer (applies demand overrides) ──────────────────────────────
FLOW_DEFS = {
    "f_0":  ("E7",  "-E6.38", "E7.160 -E6"),
    "f_1":  ("E7",  "E5.38",  "E8"),
    "f_2":  ("E7",  "-E9.41", "E7.160 -E9"),
    "f_3":  ("E6",  "-E7.39", "E2"),
    "f_4":  ("E6",  "E5.38",  "E6.160 E5"),
    "f_5":  ("-E5", "-E9.41", "E10"),
    "f_6":  ("-E5", "-E6.38", "-E5.164 -E6"),
    "f_7":  ("-E5", "-E7.39", "-E5.164 -E7"),
    "f_8":  ("E6",  "-E9.41", "E6.160 -E9"),
    "f_9":  ("E9",  "-E6.38", "E1"),
    "f_10": ("E9",  "-E7.39", "E9.162 -E7"),
    "f_11": ("E9",  "E5.38",  "E9.162 E5"),
}

FLOW_BEGINS = {
    "f_0": 10.0, "f_1": 10.0, "f_2": 10.0,
    "f_3": 0.0,  "f_4": 0.0,  "f_5": 2.5,
    "f_6": 2.5,  "f_7": 2.5,  "f_8": 0.0,
    "f_9": 5.0,  "f_10": 5.0, "f_11": 5.0,
}


def write_route_file(demand_per_flow: dict, path: str):
    lines = ['<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
             'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">',
             "    <!-- Generated by dashboard.py -->"]
    for fid, (frm, to, via) in sorted(FLOW_DEFS.items()):
        vph = demand_per_flow.get(fid, 100.0)
        begin = FLOW_BEGINS.get(fid, 0.0)
        lines.append(
            f'    <flow id="{fid}" begin="{begin:.2f}" end="3600.00" '
            f'perHour="{vph:.2f}" from="{frm}" to="{to}" via="{via}"/>'
        )
    lines.append("</routes>")
    with open(path, "w") as f:
        f.write("\n".join(lines))


# ── Streamlit page ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Max Pressure — SUMO Dashboard",
    page_icon="🚦",
    layout="wide",
)

st.title("🚦 Max Pressure Signal Controller")
st.caption("SUMO simulation dashboard — J9 intersection")

# ── Session state initialisation ──────────────────────────────────────────────
if "thread" not in st.session_state:
    st.session_state.thread = None
if "demand" not in st.session_state:
    st.session_state.demand = {fid: 100.0 for fid in FLOW_DEFS}

# ── Sidebar — settings ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Algorithm parameters")
    alpha    = st.slider("α — wait weight",   0.0, 2.0, 0.5, 0.05)
    beta     = st.slider("β — growth weight", 0.0, 2.0, 0.3, 0.05)
    cycle    = st.slider("Cycle C (s)",       30,  180, 90,  5)
    ds       = st.slider("Min green ds (s)",   4,   30,  8,  1)
    dt       = st.slider("Growth window dt (s)", 5, 60, 15,  5)
    use_gui  = st.checkbox("Launch SUMO GUI", value=False)
    cfg_path = st.text_input("Config file", value="intersection.sumocfg")
    out_csv  = st.text_input("CSV output",  value="results.csv")
    rou_path = st.text_input("Route file (will be overwritten)", value="intersection.rou.xml")

    st.divider()
    st.header("Traffic demand (veh/hr)")

    # Group flows by origin approach for clarity
    approach_flows = {
        "North (−E5)": ["f_5", "f_6", "f_7"],
        "South (E6)":  ["f_3", "f_4", "f_8"],
        "East (E9)":   ["f_9", "f_10", "f_11"],
        "West (E7)":   ["f_0", "f_1", "f_2"],
    }
    dest_labels = {
        "f_0": "→ North (-E6.38)",  "f_1": "→ South (E5.38)",   "f_2": "→ East (-E9.41)",
        "f_3": "→ West (-E7.39)",   "f_4": "→ South (E5.38)",   "f_5": "→ East (-E9.41)",
        "f_6": "→ North (-E6.38)", "f_7": "→ West (-E7.39)",   "f_8": "→ East (-E9.41)",
        "f_9": "→ North (-E6.38)", "f_10": "→ West (-E7.39)",  "f_11": "→ South (E5.38)",
    }

    for group, fids in approach_flows.items():
        with st.expander(group, expanded=False):
            for fid in fids:
                st.session_state.demand[fid] = st.number_input(
                    dest_labels.get(fid, fid),
                    min_value=0, max_value=1800,
                    value=int(st.session_state.demand[fid]),
                    step=10, key=f"demand_{fid}",
                )

    st.divider()

    col_start, col_stop = st.columns(2)
    with col_start:
        start_btn = st.button("▶ Start", type="primary", use_container_width=True,
                              disabled=_sim_state["running"])
    with col_stop:
        stop_btn = st.button("⏹ Stop", use_container_width=True,
                             disabled=not _sim_state["running"])

    if start_btn and not _sim_state["running"]:
        # Reset history
        with _lock:
            _sim_state.update({
                "running": True, "stopped": False, "error": None,
                "sim_time": 0, "total_vehicles": 0,
                "current_phase": "—", "phase_timer": 0, "phase_duration": 0,
                "history": [], "log": [], "phases_detected": [],
                "approaches": {
                    name: {"q": 0, "d": 0.0, "g": 0.0, "P": 0.0, "T": 0.0, "green": False}
                    for name in APPROACH_NAMES
                },
            })

        write_route_file(st.session_state.demand, rou_path)

        params = {
            "alpha": alpha, "beta": beta, "cycle": cycle,
            "ds": ds, "dt": dt, "history_len": 20,
            "gui": use_gui, "sumocfg": cfg_path,
            "output_csv": out_csv, "demand": dict(st.session_state.demand),
        }
        t = threading.Thread(target=simulation_thread, args=(params,), daemon=True)
        t.start()
        st.session_state.thread = t

    if stop_btn and _sim_state["running"]:
        with _lock:
            _sim_state["running"] = False

# ── Main area ─────────────────────────────────────────────────────────────────
with _lock:
    snap = deepcopy(_sim_state)

# Status banner
status_col, time_col, veh_col, phase_col = st.columns(4)
with status_col:
    if snap["error"]:
        st.error(f"Error: {snap['error']}")
    elif snap["running"]:
        st.success("🟢 Running")
    elif snap["stopped"]:
        st.info("✅ Complete")
    else:
        st.warning("⏸ Stopped")

with time_col:
    st.metric("Sim time", f"{snap['sim_time']} s")

with veh_col:
    st.metric("Vehicles in network", snap["total_vehicles"])

with phase_col:
    pname = snap["current_phase"]
    ptimer = snap["phase_timer"]
    pdur   = snap["phase_duration"]
    st.metric("Current phase", pname, delta=f"{ptimer}/{pdur} s" if pdur else None)

st.divider()

# ── Intersection diagram ──────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 2])

with left_col:
    st.subheader("Intersection view")

    approach_data = snap["approaches"]

    # Build SVG intersection diagram
    def make_intersection_svg(approaches):
        cx, cy = 160, 160
        W, H = 320, 320

        def road(x, y, w, h, color="#1a1a1a"):
            return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{color}"/>'

        def bubble(px, py, q, is_green, name):
            r = max(14, min(28, 14 + q * 0.8))
            color = "#22c55e" if is_green else ("#ef4444" if q > 8 else "#f59e0b")
            label = str(int(q))
            return (
                f'<circle cx="{px}" cy="{py}" r="{r}" fill="{color}" fill-opacity="0.88"/>'
                f'<text x="{px}" y="{py+1}" text-anchor="middle" dominant-baseline="middle" '
                f'fill="white" font-size="11" font-weight="bold">{label}</text>'
                f'<text x="{px}" y="{py+r+9}" text-anchor="middle" fill="#555" font-size="8">{name.replace("_"," ")}</text>'
            )

        # Traffic light dots
        tl_phase = snap["current_phase"]
        tl_green = not ("yellow" in tl_phase or tl_phase == "—")
        tl_color_r = "#ef4444" if not tl_green else "#555"
        tl_color_g = "#22c55e" if tl_green else "#555"

        svg = f"""<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg">
  <rect width="{W}" height="{H}" fill="#f8f8f8"/>
  {road(cx-20, 0, 40, H)}
  {road(0, cy-20, W, 40)}
  <rect x="{cx-32}" y="{cy-32}" width="64" height="64" fill="#7f1d1d"/>
  <circle cx="{cx-16}" cy="{cy-16}" r="14" fill="#7f1d1d"/>
  <circle cx="{cx+16}" cy="{cy-16}" r="14" fill="#7f1d1d"/>
  <circle cx="{cx-16}" cy="{cy+16}" r="14" fill="#7f1d1d"/>
  <circle cx="{cx+16}" cy="{cy+16}" r="14" fill="#7f1d1d"/>
  <rect x="{cx-20}" y="{cy-20}" width="40" height="40" fill="#1a1a1a"/>
  <circle cx="{cx+4}" cy="{cy-8}" r="4" fill="{tl_color_r}"/>
  <circle cx="{cx+4}" cy="{cy}" r="4" fill="#f59e0b"/>
  <circle cx="{cx+4}" cy="{cy+8}" r="4" fill="{tl_color_g}"/>
  <text x="{cx}" y="12" text-anchor="middle" fill="#555" font-size="10">N</text>
  <text x="{cx}" y="{H-4}" text-anchor="middle" fill="#555" font-size="10">S</text>
  <text x="10" y="{cy+4}" text-anchor="middle" fill="#555" font-size="10">W</text>
  <text x="{W-10}" y="{cy+4}" text-anchor="middle" fill="#555" font-size="10">E</text>
  {bubble(cx, cy-75, approaches['North_SB']['q'], approaches['North_SB']['green'], 'North SB')}
  {bubble(cx, cy+75, approaches['South_NB']['q'], approaches['South_NB']['green'], 'South NB')}
  {bubble(cx+75, cy, approaches['East_WB']['q'],  approaches['East_WB']['green'],  'East WB')}
  {bubble(cx-75, cy, approaches['West_EB']['q'],  approaches['West_EB']['green'],  'West EB')}
</svg>"""
        return svg

    st.markdown(
        make_intersection_svg(approach_data),
        unsafe_allow_html=True,
    )

# ── Live metrics table ────────────────────────────────────────────────────────
with right_col:
    st.subheader("Lane metrics")

    rows = []
    for name in APPROACH_NAMES:
        a = approach_data[name]
        rows.append({
            "Approach":    name,
            "Signal":      "🟢 Green" if a["green"] else "🔴 Red",
            "Vehicles Qᵢ": int(a["q"]),
            "Avg wait Dᵢ (s)": round(a["d"], 1),
            "Growth Gᵢ":   round(a["g"], 3),
            "Pressure Pᵢ": round(a["P"], 3),
            "Green time Tᵢ (s)": round(a["T"], 1),
        })

    df_metrics = pd.DataFrame(rows).set_index("Approach")
    st.dataframe(
        df_metrics,
        use_container_width=True,
        height=210,
    )

    # Pressure bar chart
    if any(approach_data[n]["P"] > 0 for n in APPROACH_NAMES):
        fig_p = go.Figure(go.Bar(
            x=APPROACH_NAMES,
            y=[approach_data[n]["P"] for n in APPROACH_NAMES],
            marker_color=APPROACH_COLORS,
            text=[f"{approach_data[n]['P']:.2f}" for n in APPROACH_NAMES],
            textposition="outside",
        ))
        fig_p.update_layout(
            title="Traffic pressure Pᵢ",
            yaxis_title="Pᵢ",
            height=220,
            margin=dict(t=36, b=20, l=10, r=10),
            showlegend=False,
        )
        st.plotly_chart(fig_p, use_container_width=True)

st.divider()

# ── Time-series charts ────────────────────────────────────────────────────────
history = snap["history"]

if len(history) >= 2:
    df_hist = pd.DataFrame(history)

    chart_tab1, chart_tab2, chart_tab3 = st.tabs(
        ["Vehicle counts (Qᵢ)", "Avg wait time (Dᵢ)", "Green time allocation (Tᵢ)"]
    )

    with chart_tab1:
        fig_q = go.Figure()
        for name, color in zip(APPROACH_NAMES, APPROACH_COLORS):
            fig_q.add_trace(go.Scatter(
                x=df_hist["time"], y=df_hist[f"{name}_q"],
                name=name, line=dict(color=color, width=2), mode="lines",
            ))
        fig_q.update_layout(
            yaxis_title="Vehicles in approach",
            xaxis_title="Sim time (s)",
            height=300, margin=dict(t=10, b=30, l=40, r=10),
            legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig_q, use_container_width=True)

    with chart_tab2:
        fig_d = go.Figure()
        for name, color in zip(APPROACH_NAMES, APPROACH_COLORS):
            fig_d.add_trace(go.Scatter(
                x=df_hist["time"], y=df_hist[f"{name}_d"],
                name=name, line=dict(color=color, width=2), mode="lines",
            ))
        fig_d.update_layout(
            yaxis_title="Avg waiting time (s)",
            xaxis_title="Sim time (s)",
            height=300, margin=dict(t=10, b=30, l=40, r=10),
            legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig_d, use_container_width=True)

    with chart_tab3:
        fig_t = go.Figure()
        for name, color in zip(APPROACH_NAMES, APPROACH_COLORS):
            fig_t.add_trace(go.Bar(
                x=df_hist["time"], y=df_hist[f"{name}_T"],
                name=name, marker_color=color,
            ))
        fig_t.update_layout(
            barmode="group",
            yaxis_title="Allocated green time (s)",
            xaxis_title="Sim time (s)",
            height=300, margin=dict(t=10, b=30, l=40, r=10),
            legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig_t, use_container_width=True)
else:
    st.info("Charts will appear once the simulation starts and produces data.")

st.divider()

# ── Phase detection table ─────────────────────────────────────────────────────
if snap["phases_detected"]:
    st.subheader("Detected signal phases")
    phase_rows = []
    for p in snap["phases_detected"]:
        phase_rows.append({
            "SUMO index": p["sumo_index"],
            "Name": p["name"],
            "Green approaches": ", ".join(p["green"]) if p["green"] else "(yellow/red)",
            "State string": p["state_str"],
        })
    st.dataframe(pd.DataFrame(phase_rows), use_container_width=True, hide_index=True)

# ── Log ───────────────────────────────────────────────────────────────────────
with st.expander("Simulation log", expanded=False):
    log_lines = snap["log"]
    st.code("\n".join(log_lines[-50:]) if log_lines else "No log yet.", language=None)

# ── Auto-refresh ──────────────────────────────────────────────────────────────
if snap["running"]:
    time.sleep(1)
    st.rerun()
