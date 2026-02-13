import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.mixture import GaussianMixture
from pathlib import Path
import networkx as nx 

# --- CONFIGURATION ---
# Use current directory for Sandbox/Local portability
BASE_DIR = Path('.').resolve()
DATA_PATH = BASE_DIR / 'data' / 'raw' / 'financial_log_100k.csv'
SENSITIVITY_PATH = BASE_DIR / 'data' / 'output' / 'sensitivity_analysis.csv'
FIG_DIR = BASE_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Set Typography: Use Serif to match tgtermes/Times New Roman (IJDS Standard)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "stix", # Matches LaTeX math style
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.autolayout": True
})

sns.set_context("paper")
sns.set_style("ticks") # Cleaner style for top-tier journals

def save_ijds_fig(path_base):
    """
    Saves figure in all required IJDS formats.
    path_base: Path object or string without extension.
    """
    # 1. PDF for final publication (Vector - highest priority)
    plt.savefig(f"{path_base}.pdf", format='pdf', bbox_inches='tight')
    # 2. EPS for legacy typesetting (Vector)
    plt.savefig(f"{path_base}.eps", format='eps', bbox_inches='tight')
    # 3. High-res PNG for quick review
    plt.savefig(f"{path_base}.png", dpi=300, bbox_inches='tight')
    
    print(f"Exported PDF, EPS, and PNG for: {Path(path_base).name}")

def plot_fidelity_comparison(data):
    """Fig 1: Generates TWO panels (1a Full, 1b Tail)"""
    print("Generating Figure 1 (Two-Panel Fidelity Comparison)...")
    
    x_full = np.linspace(0, 600, 2000)
    
    # Baseline (Normal)
    mu, std = stats.norm.fit(data)
    norm_pdf = stats.norm.pdf(x_full, mu, std)
    
    # Method A (Log-Normal)
    shape, loc, scale = stats.lognorm.fit(data, floc=0)
    lognorm_pdf = stats.lognorm.pdf(x_full, shape, loc, scale)
    
    # Method B (Mixture)
    gmm = GaussianMixture(n_components=2, random_state=42).fit(np.log(data + 1e-9).reshape(-1,1))
    weights, means, stds = gmm.weights_, gmm.means_.flatten(), np.sqrt(gmm.covariances_.flatten())
    
    mix_pdf = np.zeros_like(x_full)
    for i in range(len(weights)):
        mix_pdf += weights[i] * stats.lognorm.pdf(x_full, s=stds[i], scale=np.exp(means[i]))

    # --- PANEL A: Full Distribution ---
    plt.figure(figsize=(8, 6))
    sns.kdeplot(data, color='grey', fill=True, alpha=0.3, label='Ground Truth', linewidth=0)
    plt.plot(x_full, norm_pdf, 'r--', linewidth=2, label='Normal (Baseline)')
    plt.plot(x_full, lognorm_pdf, 'b:', linewidth=2, label='Log-Normal (Method A)')
    plt.plot(x_full, mix_pdf, 'g-', linewidth=3, label='Mixture (Method B)')
    
    plt.annotate('Standard Mode\n(Routine Peak)', xy=(45, 0.012), xytext=(100, 0.015),
                 arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)
    
    plt.xlim(0, 600)
    plt.legend()
    plt.xlabel('Cycle Time (Minutes)')
    plt.ylabel('Probability Density')
    plt.title('Fig 1a: Overall Distribution Fit (Macro View)')
    
    save_ijds_fig(FIG_DIR / 'fig1a_full_distribution')
    plt.close()

    # --- PANEL B: Tail Detail ---
    plt.figure(figsize=(8, 6))
    sns.kdeplot(data, color='grey', fill=True, alpha=0.3, label='Ground Truth', linewidth=0)
    plt.plot(x_full, norm_pdf, 'r--', linewidth=2, label='Normal')
    plt.plot(x_full, lognorm_pdf, 'b:', linewidth=2, label='Log-Normal')
    plt.plot(x_full, mix_pdf, 'g-', linewidth=3, label='Mixture (Method B)')
    
    plt.xlim(150, 600)
    plt.ylim(0, 0.004)
    
    plt.annotate('Exception Tail\n(Method B captures this)', xy=(300, 0.0015), xytext=(350, 0.003),
                 arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=10)
    
    plt.legend(loc='upper right')
    plt.xlabel('Cycle Time (Minutes)')
    plt.ylabel('Probability Density')
    plt.title('Fig 1b: Tail Behavior Detail (Micro View)')
    
    save_ijds_fig(FIG_DIR / 'fig1b_tail_detail')
    plt.close()

def plot_sensitivity(sens_df):
    """Fig 2: Sensitivity Analysis"""
    print("Generating Figure 2 (Sensitivity Analysis)...")
    plt.figure(figsize=(8, 6))
    
    sns.lineplot(data=sens_df, x='Perturbation_%', y='KS_Statistic', marker='o', color='purple', linewidth=2, label='Sensitivity Curve')
    plt.axhline(y=0.05, color='r', linestyle='--', label='Acceptance Threshold (p>0.05)')
    
    mle_val = sens_df[sens_df['Perturbation_%'] == 0]['KS_Statistic'].values[0]
    plt.plot(0, mle_val, 'ko', markersize=8)
    
    plt.annotate(f'Structural Bias:\nMLE ({mle_val:.2f}) $\\neq$ Min Error', 
                 xy=(0, mle_val), xytext=(-4.8, 0.17),
                 arrowprops=dict(facecolor='black', arrowstyle='->', connectionstyle="arc3,rad=0.2"),
                 fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.9))

    better_val = sens_df[sens_df['Perturbation_%'] == -5]['KS_Statistic'].values[0]
    if better_val < mle_val:
         plt.annotate('Lowest Error\n(-5%)', xy=(-5, better_val), xytext=(-3.5, better_val - 0.005),
                     arrowprops=dict(facecolor='black', arrowstyle='->'),
                     fontsize=9, color='purple', fontweight='bold')

    plt.title('Sensitivity Analysis: Structural Instability of Model B')
    plt.xlabel('Parameter Perturbation (%)')
    plt.ylabel('KS Statistic (Error)')
    plt.ylim(0, 0.22) 
    plt.legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    
    save_ijds_fig(FIG_DIR / 'fig2_sensitivity_analysis')
    plt.close()

def plot_bimodal_evidence(data):
    """Fig 3: Empirical Evidence"""
    print("Generating Figure 3 (Two-Panel Bimodal Evidence)...")
    
    # --- PANEL A: Macro View ---
    plt.figure(figsize=(8, 6))
    sns.histplot(data, bins=100, kde=False, color='#34495E', stat='density', alpha=0.5, label='Raw Data')
    sns.kdeplot(data, color='black', linewidth=2, label='Density Trend (KDE)')
    
    plt.annotate('Standard Mode\n(Routine Peak)', xy=(45, 0.012), xytext=(100, 0.015),
                 arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)
    
    plt.xlim(0, 600)
    plt.ylim(0, 0.025)
    plt.xlabel('Processing Time (Minutes)')
    plt.ylabel('Density Probability')
    plt.title('Fig 3a: Empirical Data Distribution (Macro View)')
    plt.legend(loc='upper right')
    
    save_ijds_fig(FIG_DIR / 'fig3a_bimodal_macro')
    plt.close()

    # --- PANEL B: Micro View ---
    plt.figure(figsize=(8, 6))
    sns.histplot(data, bins=40, kde=False, color='#34495E', stat='density', alpha=0.6, label='Raw Data (Aggregated)')
    sns.kdeplot(data, color='black', linewidth=2, label='Density Trend')
    
    plt.xlim(150, 600)
    plt.ylim(0, 0.005)
    
    plt.annotate('Exception Mode\n(Distinct Cluster)', xy=(220, 0.002), xytext=(250, 0.004),
                 arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)
    
    plt.xlabel('Processing Time (Minutes)')
    plt.ylabel('Density Probability')
    plt.title('Fig 3b: Tail Behavior Evidence (Micro View)')
    plt.legend(loc='upper right')
    
    save_ijds_fig(FIG_DIR / 'fig3b_bimodal_micro')
    plt.close()

def plot_queue_dynamics(df):
    """Fig 4: Queue Dynamics"""
    print("Generating Figure 4 (Queue Dynamics)...")
    
    uw = df[df['Activity'] == 'Underwriting'].copy()
    timeline = np.arange(0, 2880, 5) 
    queue_counts = []
    active_counts = []
    
    for t in timeline:
        q = ((uw['Arrival_Time'] <= t) & (uw['Start_Time'] > t)).sum()
        a = ((uw['Start_Time'] <= t) & (uw['End_Time'] > t)).sum()
        queue_counts.append(q)
        active_counts.append(a)
        
    plt.figure(figsize=(10, 6))
    plt.plot(timeline/60, active_counts, label='Active Service (Staff)', color='#2ECC71', linewidth=2)
    plt.plot(timeline/60, queue_counts, label='Queue Backlog', color='#E74C3C', linestyle='--', linewidth=1.5)
    plt.fill_between(timeline/60, queue_counts, color='#E74C3C', alpha=0.2)
    
    plt.xlabel('Simulation Time (Hours)')
    plt.ylabel('Number of Cases')
    plt.title('Resource Contention: Queue Formation during Peak Load')
    plt.xlim(0, 48)
    plt.legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    
    save_ijds_fig(FIG_DIR / 'fig4_queue_dynamics')
    plt.close()

def plot_process_map(df):
    """Fig 5: Process Map (Discovered Flow)"""
    print("Generating Figure 5 (Process Map)...")
    df = df.sort_values(by=['CaseID', 'Start_Time'])
    df['Next_Activity'] = df.groupby('CaseID')['Activity'].shift(-1)
    transitions = df.dropna(subset=['Next_Activity'])
    edges = transitions.groupby(['Activity', 'Next_Activity']).size().reset_index(name='weight')
    
    G = nx.DiGraph()
    for _, row in edges.iterrows():
        G.add_edge(row['Activity'], row['Next_Activity'], weight=row['weight'])
        
    # Standard Linear Layout
    in_degrees = dict(G.in_degree())
    try:
        start_node = [n for n, d in in_degrees.items() if d == 0][0]
    except IndexError:
        start_node = min(in_degrees, key=in_degrees.get)
    
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
        if layer not in layer_groups: layer_groups[layer] = []
        layer_groups[layer].append(node)
        
    pos = {}
    max_layer = max(layers.values())
    for layer, nodes in layer_groups.items():
        # Center the nodes horizontally
        x = 0.1 + (layer / max_layer) * 0.8 
        nodes.sort() 
        for i, node in enumerate(nodes):
            # Center vertically (0.5 is middle)
            pos[node] = (x, np.linspace(0.4, 0.6, len(nodes)+2)[1:-1][i])

    # 1. Figure Dimensions: Make it wide and not too tall
    plt.figure(figsize=(12, 5)) 
    
    nx.draw_networkx_nodes(G, pos, node_size=6000, node_color='#ECF0F1', edgecolors='#2C3E50', linewidths=2)
    
    weights = [G[u][v]['weight'] for u,v in G.edges()]
    widths = [(w / max(weights)) * 6 + 1 for w in weights]
    nx.draw_networkx_edges(G, pos, width=widths, edge_color='#7F8C8D', arrowstyle='->', arrowsize=30, connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold', font_color='#2C3E50')
    
    edge_labels = { (u,v): f"{d['weight']}" for u,v,d in G.edges(data=True) if d['weight'] > max(weights)*0.05 }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, bbox=dict(alpha=1.0, edgecolor='white', facecolor='white'))
    
    # 2. FIXED TITLE: Use y=0.85 to place it safely inside the plot area but above nodes
    plt.title("Discovered Process Flow: The 'Happy Path' and Exceptions", fontsize=14, y=0.85)
    
    # 3. FIXED MARGINS: Expand limits to (0, 1) so nodes (at 0.5) don't get cut off
    plt.ylim(0, 1) 
    plt.axis('off')
    
    # 4. Final layout tweak to ensure no clipping
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.05, right=0.95)
    
    save_ijds_fig(FIG_DIR / 'fig5_process_map')
    plt.close()

def main():
    if not DATA_PATH.exists():
        print(f"ERROR: Data file not found at {DATA_PATH}")
        return

    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    durations = df[df['Activity'] == 'Underwriting']['End_Time'] - df[df['Activity'] == 'Underwriting']['Start_Time']
    
    plot_fidelity_comparison(durations.values)
    
    if SENSITIVITY_PATH.exists():
        sens_df = pd.read_csv(SENSITIVITY_PATH)
        plot_sensitivity(sens_df)
    
    plot_bimodal_evidence(durations.values)
    plot_queue_dynamics(df)
    plot_process_map(df)

if __name__ == "__main__":
    main()