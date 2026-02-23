import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.mixture import GaussianMixture
from pathlib import Path
import networkx as nx

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PASSIVE_LOG_PATH = RAW_DATA_DIR / "passive_log_100k.csv"
ACTIVE_LOG_PATH = RAW_DATA_DIR / "active_radr_log_100k.csv"
SENSITIVITY_PATH = BASE_DIR / "data" / "output" / "sensitivity_analysis.csv"

FIG_DIR = BASE_DIR / "figures"
OUTPUT_DIR = BASE_DIR / "data" / "output"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ACTIVITY = "Underwriting"
SLA_THRESHOLD_MINUTES = 60

# Only require operational columns; ignore extra attribute columns if present.
REQUIRED_COLS = ["CaseID", "Activity", "Arrival_Time", "Start_Time", "End_Time", "Resource", "Regime_Mode"]

# Set Typography: Use Serif to match standard manuscript styles (e.g., Times New Roman)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "stix",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.autolayout": True
})
sns.set_context("paper")
sns.set_style("ticks")  # cleaner journal style


def save_ijds_fig(path_base):
    """
    Saves figure in PDF + high-res PNG formats.
    path_base: Path object or string without extension.
    """
    plt.savefig(f"{path_base}.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(f"{path_base}.png", dpi=300, bbox_inches="tight")
    print(f"Exported PDF and PNG for: {Path(path_base).name}")


def load_and_validate_data(filepath: Path):
    if not filepath.exists():
        return None

    df = pd.read_csv(filepath)

    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required operational columns: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    # numeric cleanup for time columns; ignore extras
    for c in ["Arrival_Time", "Start_Time", "End_Time"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# --- LEGACY FIGURES (PRESERVED / UPDATED FOR ROBUSTNESS) ---
def plot_fidelity_comparison(data):
    """Fig 3: Two panels (3a Full, 3b Tail)"""
    print("Generating Figure 3 (Two-Panel Fidelity Comparison)...")

    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data) & (data > 0)]
    if data.size == 0:
        print("No valid underwriting durations found; skipping Fig 3.")
        return

    x_full = np.linspace(0, 600, 2000)

    # Baseline (Normal)
    mu, std = stats.norm.fit(data)
    norm_pdf = stats.norm.pdf(x_full, mu, std)

    # Method A (Log-Normal)
    shape, loc, scale = stats.lognorm.fit(data, floc=0)
    lognorm_pdf = stats.lognorm.pdf(x_full, shape, loc, scale)

    # Method B (Mixture) -- keep legacy 2 components for visualization only
    gmm = GaussianMixture(n_components=2, random_state=42).fit(np.log(data).reshape(-1, 1))
    weights, means, stds = gmm.weights_, gmm.means_.flatten(), np.sqrt(gmm.covariances_.flatten())

    mix_pdf = np.zeros_like(x_full)
    for i in range(len(weights)):
        mix_pdf += weights[i] * stats.lognorm.pdf(x_full, s=stds[i], scale=np.exp(means[i]))

    # PANEL A
    plt.figure(figsize=(8, 6))
    sns.kdeplot(data, color="grey", fill=True, alpha=0.3, label="Ground Truth", linewidth=0)
    plt.plot(x_full, norm_pdf, "r--", linewidth=2, label="Normal (Baseline)")
    plt.plot(x_full, lognorm_pdf, "b:", linewidth=2, label="Log-Normal (Method A)")
    plt.plot(x_full, mix_pdf, "g-", linewidth=3, label="Mixture (Method B)")
    plt.annotate("Standard Mode\n(Routine Peak)", xy=(45, 0.012), xytext=(100, 0.015),
                 arrowprops=dict(facecolor="black", shrink=0.05), fontsize=10)
    plt.xlim(0, 600)
    plt.legend()
    plt.xlabel("Cycle Time (Minutes)")
    plt.ylabel("Probability Density")
    plt.title("Overall Distribution Fit (Macro View)")
    save_ijds_fig(FIG_DIR / "fig3a_full_distribution")
    plt.close()

    # PANEL B
    plt.figure(figsize=(8, 6))
    sns.kdeplot(data, color="grey", fill=True, alpha=0.3, label="Ground Truth", linewidth=0)
    plt.plot(x_full, norm_pdf, "r--", linewidth=2, label="Normal")
    plt.plot(x_full, lognorm_pdf, "b:", linewidth=2, label="Log-Normal")
    plt.plot(x_full, mix_pdf, "g-", linewidth=3, label="Mixture (Method B)")
    plt.xlim(150, 600)
    plt.ylim(0, 0.004)
    plt.annotate("Exception Tail\n(Method B captures this)", xy=(300, 0.0015), xytext=(350, 0.003),
                 arrowprops=dict(facecolor="black", arrowstyle="->"), fontsize=10)
    plt.legend(loc="upper right")
    plt.xlabel("Cycle Time (Minutes)")
    plt.ylabel("Probability Density")
    plt.title("Tail Behavior Detail (Micro View)")
    save_ijds_fig(FIG_DIR / "fig3b_tail_detail")
    plt.close()


def plot_sensitivity(sens_df):
    """Fig 4: Sensitivity Analysis (supports old and new analyzer outputs)"""
    print("Generating Figure 4 (Sensitivity Analysis)...")

    # Support either schema:
    # old: Perturbation_% and KS_Statistic
    # new: scale_shift and ks_stat
    if "scale_shift" in sens_df.columns:
        x_col = "scale_shift"
    elif "Perturbation_%" in sens_df.columns:
        x_col = "Perturbation_%"
    else:
        raise ValueError(f"Unknown sensitivity schema. Columns: {list(sens_df.columns)}")

    if "ks_stat" in sens_df.columns:
        y_col = "ks_stat"
    elif "KS_Statistic" in sens_df.columns:
        y_col = "KS_Statistic"
    else:
        raise ValueError(f"Unknown sensitivity schema. Columns: {list(sens_df.columns)}")

    # Ensure numeric
    dfp = sens_df.copy()
    dfp[x_col] = pd.to_numeric(dfp[x_col], errors="coerce")
    dfp[y_col] = pd.to_numeric(dfp[y_col], errors="coerce")
    dfp = dfp.dropna(subset=[x_col, y_col]).sort_values(by=x_col)

    plt.figure(figsize=(8, 6))

    # Plot line
    sns.lineplot(
        data=dfp,
        x=x_col,
        y=y_col,
        marker="o",
        color="purple",
        linewidth=2,
        label="Sensitivity curve"
    )

    # Reference line (keep if you want it; it is not a p-value threshold)
    plt.axhline(y=0.05, color="r", linestyle="--", linewidth=1.2, label="Reference (0.05)")

    # Mark MLE / 0 shift if present
    if np.any(np.isclose(dfp[x_col].to_numpy(dtype=float), 0.0)):
        mle_val = float(dfp.loc[np.isclose(dfp[x_col], 0.0), y_col].iloc[0])
        plt.plot(0.0, mle_val, "ko", markersize=7, label="MLE (0 shift)")

        # IMPORTANT: place annotation using axes-fraction coords to avoid huge whitespace
        plt.annotate(
            f"MLE (0 shift)\nKS={mle_val:.3f}",
            xy=(0.0, mle_val),
            xycoords="data",
            xytext=(0.67, 0.85),
            textcoords="axes fraction",
            arrowprops=dict(facecolor="black", arrowstyle="->", shrinkA=0, shrinkB=5),
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", alpha=0.9),
            ha="left",
            va="top"
        )

    # Labels / limits
    plt.title("Sensitivity Analysis (KS vs. Scale Shift)")
    plt.xlabel("Scale shift (%)")
    plt.ylabel("KS statistic (error)")

    # Show x in percent (your data is -0.05..0.05)
    xticks = sorted(dfp[x_col].unique())
    plt.xticks(xticks, [f"{int(round(x*100))}" for x in xticks])

    # Tight x-limits around available points
    xmin, xmax = float(dfp[x_col].min()), float(dfp[x_col].max())
    pad = max(0.01, 0.1 * (xmax - xmin) if xmax > xmin else 0.01)
    plt.xlim(xmin - pad, xmax + pad)

    y_max = float(np.nanmax(dfp[y_col])) if len(dfp) else 0.2
    plt.ylim(0, max(0.22, y_max * 1.15))

    # Legend strictly inside axes to prevent canvas expansion
    plt.legend(loc="upper left", frameon=True, fancybox=True, framealpha=0.9)

    plt.grid(True, alpha=0.3)

    save_ijds_fig(FIG_DIR / "fig4_sensitivity_analysis")
    plt.close()


def plot_bimodal_evidence(data):
    """Fig 1: Two-Panel Bimodal Evidence"""
    print("Generating Figure 1 (Two-Panel Bimodal Evidence)...")

    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data) & (data > 0)]
    if data.size == 0:
        print("No valid underwriting durations found; skipping Fig 1.")
        return

    # PANEL A: Macro
    plt.figure(figsize=(8, 6))
    sns.histplot(data, bins=100, kde=False, color="#34495E", stat="density", alpha=0.5, label="Raw Data")
    sns.kdeplot(data, color="black", linewidth=2, label="Density Trend (KDE)")
    plt.annotate("Standard Mode\n(Routine Peak)", xy=(45, 0.012), xytext=(100, 0.015),
                 arrowprops=dict(facecolor="black", shrink=0.05), fontsize=10)
    plt.xlim(0, 600)
    plt.ylim(0, 0.025)
    plt.xlabel("Processing Time (Minutes)")
    plt.ylabel("Density Probability")
    plt.title("Empirical Data Distribution (Macro View)")
    plt.legend(loc="upper right")
    save_ijds_fig(FIG_DIR / "fig1a_bimodal_macro")
    plt.close()

    # PANEL B: Micro
    plt.figure(figsize=(8, 6))
    sns.histplot(data, bins=40, kde=False, color="#34495E", stat="density", alpha=0.6, label="Raw Data (Aggregated)")
    sns.kdeplot(data, color="black", linewidth=2, label="Density Trend")
    plt.xlim(150, 600)
    plt.ylim(0, 0.005)
    plt.annotate("Exception Mode\n(Distinct Cluster)", xy=(220, 0.002), xytext=(250, 0.004),
                 arrowprops=dict(facecolor="black", shrink=0.05), fontsize=10)
    plt.xlabel("Processing Time (Minutes)")
    plt.ylabel("Density Probability")
    plt.title("Tail Behavior Evidence (Micro View)")
    plt.legend(loc="upper right")
    save_ijds_fig(FIG_DIR / "fig1b_bimodal_micro")
    plt.close()


def plot_queue_dynamics(df):
    """
    Figure 5: Queue Dynamics (Passive) -- preserved from previous script.
    Uses 5-minute sampling over 48 hours.
    """
    print("Generating Figure 5 (Queue Dynamics)...")

    uw = df[df["Activity"] == "Underwriting"].copy()
    if uw.empty:
        print("No underwriting events found; skipping Fig 5.")
        return

    timeline = np.arange(0, 2880, 5)  # 48 hours, 5 min steps
    queue_counts = []
    active_counts = []

    for t in timeline:
        q = ((uw["Arrival_Time"] <= t) & (uw["Start_Time"] > t)).sum()
        a = ((uw["Start_Time"] <= t) & (uw["End_Time"] > t)).sum()
        queue_counts.append(int(q))
        active_counts.append(int(a))

    plt.figure(figsize=(10, 6))
    plt.plot(timeline / 60, active_counts, label="Active Service (Staff)", color="#2ECC71", linewidth=2)
    plt.plot(timeline / 60, queue_counts, label="Queue Backlog", color="#E74C3C", linestyle="--", linewidth=1.5)
    plt.fill_between(timeline / 60, queue_counts, color="#E74C3C", alpha=0.2)

    plt.xlabel("Simulation Time (Hours)")
    plt.ylabel("Number of Cases")
    plt.title("Resource Contention: Queue Formation during Peak Load")
    plt.xlim(0, 48)
    plt.legend(loc="upper right", frameon=True, fancybox=True, framealpha=0.9)
    plt.grid(True, alpha=0.3)

    save_ijds_fig(FIG_DIR / "fig5_queue_dynamics")
    plt.close()


def plot_process_map(df):
    """Fig 2: Process Map (Discovered Flow)"""
    print("Generating Figure 2 (Process Map)...")

    df_sorted = df.sort_values(by=["CaseID", "Start_Time"]).copy()
    df_sorted["Next_Activity"] = df_sorted.groupby("CaseID")["Activity"].shift(-1)
    transitions = df_sorted.dropna(subset=["Next_Activity"])
    edges = transitions.groupby(["Activity", "Next_Activity"]).size().reset_index(name="weight")

    G = nx.DiGraph()
    for _, row in edges.iterrows():
        G.add_edge(row["Activity"], row["Next_Activity"], weight=row["weight"])

    in_degrees = dict(G.in_degree())
    try:
        start_node = [n for n, d in in_degrees.items() if d == 0][0]
    except IndexError:
        start_node = min(in_degrees, key=in_degrees.get) if in_degrees else None

    if start_node is None:
        print("Process map: could not infer start node; skipping.")
        return

    layers = {start_node: 0}
    queue = [start_node]
    visited = {start_node}
    while queue:
        current = queue.pop(0)
        for neighbor in G.successors(current):
            if neighbor not in visited:
                layers[neighbor] = layers[current] + 1
                visited.add(neighbor)
                queue.append(neighbor)

    layer_groups = {}
    for node, layer in layers.items():
        layer_groups.setdefault(layer, []).append(node)

    pos = {}
    max_layer = max(layers.values()) if layers else 1
    for layer, nodes in layer_groups.items():
        x = 0.1 + (layer / max_layer) * 0.8
        nodes.sort()
        ys = np.linspace(0.4, 0.6, len(nodes) + 2)[1:-1]
        for i, node in enumerate(nodes):
            pos[node] = (x, ys[i])

    plt.figure(figsize=(12, 5))
    nx.draw_networkx_nodes(G, pos, node_size=6000, node_color="#ECF0F1", edgecolors="#2C3E50", linewidths=2)
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(weights) if weights else 1
    widths = [(w / max_w) * 6 + 1 for w in weights]
    nx.draw_networkx_edges(G, pos, width=widths, edge_color="#7F8C8D",
                           arrowstyle="->", arrowsize=30, connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_labels(G, pos, font_size=11, font_weight="bold", font_color="#2C3E50")

    edge_labels = {(u, v): f"{d['weight']}" for u, v, d in G.edges(data=True) if d["weight"] > max_w * 0.05}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10,
                                 bbox=dict(alpha=1.0, edgecolor="white", facecolor="white"))

    plt.title("Discovered Process Flow: The 'Happy Path' and Exceptions", fontsize=14, y=0.85)
    plt.ylim(0, 1)
    plt.axis("off")
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.05, right=0.95)

    save_ijds_fig(FIG_DIR / "fig2_process_map")
    plt.close()


# --- Event-sweep engine for Figure 6a/6b + ops metrics ---
def calc_queue_sweep(df, mode="backlog"):
    """
    O(N log N) event sweep defining queue level for underwriting.
    mode="backlog": +1 arrival, -1 start (waiting only)
    mode="wip":     +1 arrival, -1 end   (waiting + in service)
    Returns: t_series (minutes), q_series, q_max, mean_q (time-weighted)
    """
    uw = df[df["Activity"] == ACTIVITY].copy()
    if uw.empty:
        return np.array([]), np.array([]), 0, 0.0

    arrivals = pd.DataFrame({"time": uw["Arrival_Time"], "change": 1})
    if mode == "backlog":
        departures = pd.DataFrame({"time": uw["Start_Time"], "change": -1})
    else:
        departures = pd.DataFrame({"time": uw["End_Time"], "change": -1})

    events = pd.concat([arrivals, departures], ignore_index=True)
    events = events.dropna(subset=["time"])

    # Stable tie-break: departures (-1) before arrivals (+1) at same time
    events = events.sort_values(["time", "change"], ascending=[True, True]).reset_index(drop=True)

    t_series = events["time"].to_numpy(dtype=float)
    q_series = events["change"].to_numpy(dtype=int).cumsum()

    q_max = int(q_series.max()) if q_series.size else 0

    if t_series.size < 2 or t_series[-1] <= t_series[0]:
        mean_q = 0.0
    else:
        dt = np.diff(t_series)
        mean_q = float(np.sum(q_series[:-1] * dt) / (t_series[-1] - t_series[0]))

    return t_series, q_series, q_max, mean_q


def calc_ops_metrics(df, policy_name):
    uw = df[df["Activity"] == ACTIVITY].copy()
    if uw.empty:
        return {}

    wait = (uw["Start_Time"] - uw["Arrival_Time"]).to_numpy(dtype=float)
    wait = wait[np.isfinite(wait) & (wait >= 0)]

    if wait.size:
        mean_wait = float(np.mean(wait))
        p95_wait = float(np.percentile(wait, 95))
        p99_wait = float(np.percentile(wait, 99))
        sla_breach_rate = float(np.mean(wait > SLA_THRESHOLD_MINUTES))
    else:
        mean_wait = p95_wait = p99_wait = sla_breach_rate = 0.0

    rerouted = int((uw["Resource"] == "Reserve_UW").sum()) if "Resource" in uw.columns else 0
    t0 = float(df["Arrival_Time"].min())
    t1 = float(df["End_Time"].max())
    span_minutes = max(0.0, t1 - t0)
    span_hours = span_minutes / 60.0 if span_minutes > 0 else 0.0
    reroutes_per_hour = float(rerouted / span_hours) if span_hours > 0 else 0.0

    _, _, q_max_backlog, mean_backlog = calc_queue_sweep(df, mode="backlog")
    _, _, q_max_wip, mean_wip = calc_queue_sweep(df, mode="wip")

    return {
        "policy": policy_name,
        "q_max_backlog": q_max_backlog,
        "mean_backlog": mean_backlog,
        "q_max_wip": q_max_wip,
        "mean_wip": mean_wip,
        "mean_wait": mean_wait,
        "p95_wait": p95_wait,
        "p99_wait": p99_wait,
        "sla_breach_rate": sla_breach_rate,
        "reroutes_per_hour": reroutes_per_hour,
    }


def generate_figure_6a_6b(df_passive, df_active):
    """
    Produces:
      - Figure 6a: backlog (waiting only)
      - Figure 6b: WIP (waiting + in service)
    """
    print("Generating Figure 6a/6b (Passive vs RADR)...")

    t_p_b, q_p_b, _, _ = calc_queue_sweep(df_passive, "backlog")
    t_a_b, q_a_b, _, _ = calc_queue_sweep(df_active, "backlog")

    t_p_w, q_p_w, _, _ = calc_queue_sweep(df_passive, "wip")
    t_a_w, q_a_w, _, _ = calc_queue_sweep(df_active, "wip")

    VIEW_HOURS = 48
    VIEW_MINS = VIEW_HOURS * 60

    # 6a: Backlog (waiting only)
    plt.figure(figsize=(10, 5))
    if t_p_b.size:
        mask_p = t_p_b <= VIEW_MINS
        plt.fill_between(t_p_b[mask_p] / 60.0, q_p_b[mask_p],
                         alpha=0.2, step="post", label="Passive Twin", rasterized=True)
        plt.step(t_p_b[mask_p] / 60.0, q_p_b[mask_p], where="post", linewidth=1.2, rasterized=True)
    if t_a_b.size:
        mask_a = t_a_b <= VIEW_MINS
        plt.step(t_a_b[mask_a] / 60.0, q_a_b[mask_a],
                 where="post", linewidth=1.5, label="Active Twin (RADR)", rasterized=True)

    plt.title("Queue Dynamics: Underwriting Backlog (waiting only)", fontweight="bold")
    plt.xlabel("Simulation Time (Hours)")
    plt.ylabel("Number of Cases Waiting")
    plt.legend(loc="upper right", frameon=True)
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.xlim(0, VIEW_HOURS)
    save_ijds_fig(FIG_DIR / "fig6a_underwriting_backlog_passive_vs_radr")
    plt.close()

    # 6b: WIP (queue + in service)
    plt.figure(figsize=(10, 5))
    if t_p_w.size:
        mask_p = t_p_w <= VIEW_MINS
        plt.fill_between(t_p_w[mask_p] / 60.0, q_p_w[mask_p],
                         alpha=0.2, step="post", label="Passive Twin", rasterized=True)
        plt.step(t_p_w[mask_p] / 60.0, q_p_w[mask_p], where="post", linewidth=1.2, rasterized=True)
    if t_a_w.size:
        mask_a = t_a_w <= VIEW_MINS
        plt.step(t_a_w[mask_a] / 60.0, q_a_w[mask_a],
                 where="post", linewidth=1.5, label="Active Twin (RADR)", rasterized=True)

    plt.title("Queue Dynamics: Underwriting WIP (waiting + in service)", fontweight="bold")
    plt.xlabel("Simulation Time (Hours)")
    plt.ylabel("Number of Cases In System")
    plt.legend(loc="upper right", frameon=True)
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.xlim(0, VIEW_HOURS)
    save_ijds_fig(FIG_DIR / "fig6b_underwriting_wip_passive_vs_radr")
    plt.close()


def main():
    if not PASSIVE_LOG_PATH.exists():
        print(f"ERROR: Data file not found at {PASSIVE_LOG_PATH}")
        return

    print(f"Loading data from {PASSIVE_LOG_PATH}...")
    df_passive = load_and_validate_data(PASSIVE_LOG_PATH)
    if df_passive is None:
        print("Passive log missing; aborting.")
        return

    # Underwriting durations for fidelity/bimodality figures
    uw_passive = df_passive[df_passive["Activity"] == ACTIVITY].copy()
    durations = (uw_passive["End_Time"] - uw_passive["Start_Time"]).to_numpy(dtype=float)
    durations = durations[np.isfinite(durations) & (durations > 0)]

    plot_fidelity_comparison(durations)

    sens_df = None
    if SENSITIVITY_PATH.exists():
        sens_df = pd.read_csv(SENSITIVITY_PATH)
        plot_sensitivity(sens_df)

    plot_bimodal_evidence(durations)

    # Figure 5 
    plot_queue_dynamics(df_passive)

    # Process map
    plot_process_map(df_passive)

    # Figure 6a/6b + ops metrics
    df_active = load_and_validate_data(ACTIVE_LOG_PATH)
    metrics_list = []
    metrics_list.append(calc_ops_metrics(df_passive, "Passive"))

    if df_active is not None:
        metrics_list.append(calc_ops_metrics(df_active, "RADR"))
        generate_figure_6a_6b(df_passive, df_active)
    else:
        print("Active RADR log not found; skipping Fig 6 and RADR ops metrics.")

    ops_df = pd.DataFrame([m for m in metrics_list if m])
    if not ops_df.empty:
        ops_path = OUTPUT_DIR / "ops_metrics.csv"
        ops_df.to_csv(ops_path, index=False)
        print("\n--- OPS METRICS SUMMARY ---")
        print(ops_df[[
            "policy", "mean_backlog", "q_max_backlog", "mean_wip", "q_max_wip",
            "mean_wait", "p95_wait", "p99_wait", "sla_breach_rate", "reroutes_per_hour"
        ]])
        print(f"\nSaved ops metrics to: {ops_path}")

    print("\nVisualizer complete.")
    print(f"Figures: {FIG_DIR}")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()