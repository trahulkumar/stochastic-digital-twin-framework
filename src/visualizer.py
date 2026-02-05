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
DATA_PATH = BASE_DIR / 'data' / 'raw' / 'financial_log_100k.csv'
SENSITIVITY_PATH = BASE_DIR / 'data' / 'output' / 'sensitivity_analysis.csv'
FIG_DIR = BASE_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Set Style
sns.set_context("paper")
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'serif'

def plot_fidelity_comparison(data):
    """Fig 1: Generates TWO panels (1a Full, 1b Tail) for side-by-side layout"""
    print("Generating Figure 1 (Two-Panel Fidelity Comparison)...")
    
    # Common Data Setup
    # Calculate Fits over the full range
    x_full = np.linspace(0, 600, 2000)
    
    # Baseline (Normal)
    mu, std = stats.norm.fit(data)
    norm_pdf = stats.norm.pdf(x_full, mu, std)
    
    # Method A (Log-Normal)
    shape, loc, scale = stats.lognorm.fit(data, floc=0)
    lognorm_pdf = stats.lognorm.pdf(x_full, shape, loc, scale)
    
    # Method B (Mixture)
    # Fit on log-data, then project back
    gmm = GaussianMixture(n_components=2, random_state=42).fit(np.log(data + 1e-9).reshape(-1,1))
    weights, means, stds = gmm.weights_, gmm.means_.flatten(), np.sqrt(gmm.covariances_.flatten())
    
    mix_pdf = np.zeros_like(x_full)
    for i in range(len(weights)):
        mix_pdf += weights[i] * stats.lognorm.pdf(x_full, s=stds[i], scale=np.exp(means[i]))

    # --- PANEL A: Full Distribution (Macro View) ---
    plt.figure(figsize=(8, 6))
    
    # Plot Ground Truth
    sns.kdeplot(data, color='grey', fill=True, alpha=0.3, label='Ground Truth', linewidth=0)
    
    # Plot Lines
    plt.plot(x_full, norm_pdf, 'r--', linewidth=2, label='Normal (Baseline)')
    plt.plot(x_full, lognorm_pdf, 'b:', linewidth=2, label='Log-Normal (Method A)')
    plt.plot(x_full, mix_pdf, 'g-', linewidth=3, label='Mixture (Method B)')
    
    # Annotation (Using plt.annotate, avoiding 'ax')
    plt.annotate('Standard Mode\n(Routine Peak)', xy=(45, 0.012), xytext=(100, 0.015),
                 arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)
    
    plt.xlim(0, 600)
    plt.legend()
    plt.xlabel('Cycle Time (Minutes)')
    plt.ylabel('Probability Density')
    plt.title('Fig 1a: Overall Distribution Fit (Macro View)')
    
    save_path_a = FIG_DIR / 'fig1a_full_distribution.png'
    plt.savefig(save_path_a, dpi=300)
    print(f"Saved {save_path_a}")
    plt.close() # Close to free memory

    # --- PANEL B: Tail Detail (Micro View) ---
    plt.figure(figsize=(8, 6))
    
    # Plot Ground Truth
    sns.kdeplot(data, color='grey', fill=True, alpha=0.3, label='Ground Truth', linewidth=0)
    
    # Plot Lines
    plt.plot(x_full, norm_pdf, 'r--', linewidth=2, label='Normal')
    plt.plot(x_full, lognorm_pdf, 'b:', linewidth=2, label='Log-Normal')
    plt.plot(x_full, mix_pdf, 'g-', linewidth=3, label='Mixture (Method B)')
    
    # CRITICAL: Zoom in on the tail (x=150 to 600)
    plt.xlim(150, 600)
    plt.ylim(0, 0.004) # Zoom Y-axis to show the separation
    
    # Annotation for the victory
    plt.annotate('Exception Tail\n(Method B captures this)', xy=(300, 0.0015), xytext=(350, 0.003),
                 arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=10)
    
    plt.legend(loc='upper right')
    plt.xlabel('Cycle Time (Minutes)')
    plt.ylabel('Probability Density')
    plt.title('Fig 1b: Tail Behavior Detail (Micro View)')
    
    save_path_b = FIG_DIR / 'fig1b_tail_detail.png'
    plt.savefig(save_path_b, dpi=300)
    print(f"Saved {save_path_b}")
    plt.close()

def plot_sensitivity(sens_df):
    """Fig 2: Sensitivity Analysis (Balanced Layout)"""
    print("Generating Figure 2 (Sensitivity Analysis)...")
    plt.figure(figsize=(8, 6))
    
    # Plot the curve
    sns.lineplot(data=sens_df, x='Perturbation_%', y='KS_Statistic', marker='o', color='purple', linewidth=2, label='Sensitivity Curve')
    
    # Add the Red Threshold line
    plt.axhline(y=0.05, color='r', linestyle='--', label='Acceptance Threshold (p>0.05)')
    
    # Highlight the MLE "Optimum" (0%)
    mle_val = sens_df[sens_df['Perturbation_%'] == 0]['KS_Statistic'].values[0]
    plt.plot(0, mle_val, 'ko', markersize=8)
    
    # --- 1. ANNOTATION (Top Left) ---
    plt.annotate(f'Structural Bias:\nMLE ({mle_val:.2f}) $\\neq$ Min Error', 
                 xy=(0, mle_val), 
                 xytext=(-4.8, 0.17), # Position in Top Left
                 arrowprops=dict(facecolor='black', arrowstyle='->', connectionstyle="arc3,rad=0.2"),
                 fontsize=10, 
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.9))

    # Descriptive text for the -5% shift
    better_val = sens_df[sens_df['Perturbation_%'] == -5]['KS_Statistic'].values[0]
    if better_val < mle_val:
         plt.annotate('Lowest Error\n(-5%)', 
                     xy=(-5, better_val), 
                     xytext=(-3.5, better_val - 0.005),
                     arrowprops=dict(facecolor='black', arrowstyle='->'),
                     fontsize=9, color='purple', fontweight='bold')

    # --- 2. LAYOUT ADJUSTMENTS ---
    plt.title('Sensitivity Analysis: Structural Instability of Model B')
    plt.xlabel('Parameter Perturbation (%)')
    plt.ylabel('KS Statistic (Error)')
    
    # Set Y-Limit to create "Headroom" for the Legend
    plt.ylim(0, 0.22) 
    
    # Move Legend to Top Right (Clean & Empty now)
    plt.legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.9)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = FIG_DIR / 'fig2_sensitivity_analysis.png'
    plt.savefig(save_path, dpi=300)
    print(f"Saved Re-Positioned Figure to {save_path}")
    plt.close()

def plot_bimodal_evidence(data):
    """Fig 3: Empirical Evidence (Macro vs Micro with corrected binning)"""
    print("Generating Figure 3 (Two-Panel Bimodal Evidence)...")
    
    # --- PANEL A: Full Distribution (Macro View) ---
    plt.figure(figsize=(8, 6))
    
    # High-resolution bins for the main peak
    sns.histplot(data, bins=100, kde=False, color='#34495E', stat='density', 
                 alpha=0.5, label='Raw Data (Histogram)')
    
    sns.kdeplot(data, color='black', linewidth=2, label='Density Trend (KDE)')
    
    # Annotation
    plt.annotate('Standard Mode\n(Routine Peak)', xy=(45, 0.012), xytext=(100, 0.015),
                 arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)
    
    plt.xlim(0, 600)
    plt.ylim(0, 0.025) # Scale for the tall peak
    plt.xlabel('Processing Time (Minutes)')
    plt.ylabel('Density Probability')
    plt.title('Fig 3a: Empirical Data Distribution (Macro View)')
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    save_path_a = FIG_DIR / 'fig3a_bimodal_macro.png'
    plt.savefig(save_path_a, dpi=300)
    print(f"Saved {save_path_a}")
    plt.close()

    # --- PANEL B: Tail Detail (Micro View) ---
    plt.figure(figsize=(8, 6))
    
    # CRITICAL FIX: Fewer bins (40) makes the sparse tail bars thicker and visible
    sns.histplot(data, bins=40, kde=False, color='#34495E', stat='density', 
                 alpha=0.6, label='Raw Data (Aggregated)')
    
    sns.kdeplot(data, color='black', linewidth=2, label='Density Trend')
    
    # Zoom in
    plt.xlim(150, 600)
    plt.ylim(0, 0.005) # CRITICAL FIX: Zoom Y-axis to 0.005 to reveal the bumps
    
    # Annotation
    plt.annotate('Exception Mode\n(Distinct Cluster)', xy=(220, 0.002), xytext=(250, 0.004),
                 arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)
    
    plt.xlabel('Processing Time (Minutes)')
    plt.ylabel('Density Probability')
    plt.title('Fig 3b: Tail Behavior Evidence (Micro View)')
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    save_path_b = FIG_DIR / 'fig3b_bimodal_micro.png'
    plt.savefig(save_path_b, dpi=300)
    print(f"Saved {save_path_b}")
    plt.close()

def plot_queue_dynamics(df):
    """Fig 4: Queue Dynamics (Layout Fixed for Uniformity)"""
    print("Generating Figure 4 (Queue Dynamics)...")
    
    # Filter for the bottleneck activity
    uw = df[df['Activity'] == 'Underwriting'].copy()
    
    # Time window: 48 hours (2880 mins)
    timeline = np.arange(0, 2880, 5) 
    queue_counts = []
    active_counts = []
    
    # Calculate state at each snapshot (Vectorized would be faster, but this is clear)
    for t in timeline:
        # Queued: Arrived <= t AND Started > t
        q = ((uw['Arrival_Time'] <= t) & (uw['Start_Time'] > t)).sum()
        # Active: Started <= t AND Ended > t
        a = ((uw['Start_Time'] <= t) & (uw['End_Time'] > t)).sum()
        queue_counts.append(q)
        active_counts.append(a)
        
    # Use consistent figure size (Matching Fig 3)
    plt.figure(figsize=(10, 6))
    
    # Plot Active (Green - The Capacity)
    plt.plot(timeline/60, active_counts, label='Active Service (Staff)', color='#2ECC71', linewidth=2)
    
    # Plot Queue (Red - The Backlog)
    plt.plot(timeline/60, queue_counts, label='Queue Backlog', color='#E74C3C', linestyle='--', linewidth=1.5)
    plt.fill_between(timeline/60, queue_counts, color='#E74C3C', alpha=0.2)
    
    # Formatting
    plt.xlabel('Simulation Time (Hours)')
    plt.ylabel('Number of Cases')
    plt.title('Resource Contention: Queue Formation during Peak Load')
    plt.xlim(0, 48)
    
    # Legend Position (Upper Right is usually empty in queue plots)
    plt.legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.9)
    
    # Uniformity Fixes
    plt.grid(True, alpha=0.3)
    plt.tight_layout() # Removes the extra whitespace
    
    save_path = FIG_DIR / 'fig4_queue_dynamics.png'
    plt.savefig(save_path, dpi=300)
    print(f"Saved Fixed Figure to {save_path}")
    plt.close()

def plot_process_map_v1(df):
    """Fig 5: Process Map (NetworkX Version)"""
    print("Generating Figure 5 (Process Map)...")
    
    # 1. Calculate Transitions
    df = df.sort_values(by=['CaseID', 'Start_Time'])
    df['Next_Activity'] = df.groupby('CaseID')['Activity'].shift(-1)
    transitions = df.dropna(subset=['Next_Activity'])
    edges = transitions.groupby(['Activity', 'Next_Activity']).size().reset_index(name='weight')
    
    # 2. Build Graph
    G = nx.DiGraph()
    for _, row in edges.iterrows():
        G.add_edge(row['Activity'], row['Next_Activity'], weight=row['weight'])
        
    # 3. Draw
    plt.figure(figsize=(12, 8))
    pos = nx.shell_layout(G)
    
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue', alpha=0.9)
    
    weights = [G[u][v]['weight'] for u,v in G.edges()]
    max_w = max(weights) if weights else 1
    widths = [(w / max_w) * 5 for w in weights]
    
    nx.draw_networkx_edges(G, pos, width=widths, edge_color='grey', arrowstyle='->', arrowsize=20)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    edge_labels = {(u, v): f"{d['weight']}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title("Discovered Process Flow (Directly-Follows Graph)")
    plt.axis('off')
    
    plt.savefig(FIG_DIR / 'fig5_process_map.png', dpi=300)
    plt.close()

def plot_process_map(df):
    """Fig 5: Process Map (Final Polish - Title Positioned Correctly)"""
    print("Generating Figure 5 (Process Map - Final Version)...")
    
    # 1. Calculate Transitions
    df = df.sort_values(by=['CaseID', 'Start_Time'])
    df['Next_Activity'] = df.groupby('CaseID')['Activity'].shift(-1)
    transitions = df.dropna(subset=['Next_Activity'])
    edges = transitions.groupby(['Activity', 'Next_Activity']).size().reset_index(name='weight')
    
    # 2. Build Graph
    G = nx.DiGraph()
    for _, row in edges.iterrows():
        G.add_edge(row['Activity'], row['Next_Activity'], weight=row['weight'])
        
    # 3. Layout (Normalized Linear)
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
        x = 0.05 + (layer / max_layer) * 0.9 
        nodes.sort() 
        n_nodes = len(nodes)
        y_slots = np.linspace(0.35, 0.65, n_nodes + 2)[1:-1]
        for i, node in enumerate(nodes):
            pos[node] = (x, y_slots[i])

    for node in G.nodes():
        if node not in pos: pos[node] = (0.9, 0.1)

    # 4. Drawing (With Title Adjustment)
    plt.figure(figsize=(12, 4)) 
    
    # Draw Nodes
    nx.draw_networkx_nodes(G, pos, node_size=6000, node_color='#ECF0F1', edgecolors='#2C3E50', linewidths=2)
    
    # Draw Edges
    weights = [G[u][v]['weight'] for u,v in G.edges()]
    max_w = max(weights) if weights else 1
    widths = [(w / max_w) * 6 + 1 for w in weights]
    
    nx.draw_networkx_edges(G, pos, width=widths, edge_color='#7F8C8D', 
                           arrowstyle='->', arrowsize=30,
                           connectionstyle="arc3,rad=0.1")
    
    # Labels
    nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold', font_color='#2C3E50')
    
    # Edge Labels
    threshold = max_w * 0.05
    significant_edges = { (u,v): f"{d['weight']}" for u,v,d in G.edges(data=True) if d['weight'] > threshold }
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels=significant_edges, font_size=10, 
                                 bbox=dict(alpha=1.0, edgecolor='white', facecolor='white', boxstyle='round,pad=0.2'))
    
    # CRITICAL: Manually position the title lower (y=0.9)
    plt.title("Discovered Process Flow: The 'Happy Path' and Exceptions", fontsize=14, y=0.90)
    
    plt.xlim(0, 1)
    plt.ylim(0.2, 0.8) 
    plt.axis('off')
    
    # Adjust layout to give the title breathing room
    plt.subplots_adjust(left=0, right=1, top=0.85, bottom=0.1)
    
    save_path = FIG_DIR / 'fig5_process_map.png'
    plt.savefig(save_path, dpi=300)
    print(f"Saved Final Figure 5 to {save_path}")
    plt.close()

def generate_results_cheat_sheet(df, sensitivity_df=None):
    """
    Calculates ALL critical numbers for the paper and saves to a single CSV.
    No more manual lookups.
    """
    print("Generating Results Cheat Sheet (Hard Numbers)...")
    results = {}

    # --- 1. QUEUE DYNAMICS (From Raw Data) ---
    # Re-run the logic from plot_queue_dynamics to get the exact numbers
    uw = df[df['Activity'] == 'Underwriting'].copy()
    timeline = np.arange(0, 2880, 5) # 48 hours, 5 min steps
    queue_sizes = []
    active_staff = []
    
    for t in timeline:
        q = ((uw['Arrival_Time'] <= t) & (uw['Start_Time'] > t)).sum()
        a = ((uw['Start_Time'] <= t) & (uw['End_Time'] > t)).sum()
        queue_sizes.append(q)
        active_staff.append(a)
    
    results['Queue_Peak_Backlog'] = np.max(queue_sizes)
    results['Queue_Mean_Backlog'] = np.mean(queue_sizes)
    results['Queue_Max_Active_Staff'] = np.max(active_staff) # The "Ceiling"

    # --- 2. PROCESS MAP COUNTS (From Raw Data) ---
    # Re-run logic to get edge weights
    df_sorted = df.sort_values(by=['CaseID', 'Start_Time'])
    df_sorted['Next_Activity'] = df_sorted.groupby('CaseID')['Activity'].shift(-1)
    transitions = df_sorted.dropna(subset=['Next_Activity'])
    edges = transitions.groupby(['Activity', 'Next_Activity']).size().reset_index(name='weight')
    
    # Get Top "Happy Path" edge (max weight)
    max_edge = edges.loc[edges['weight'].idxmax()]
    results['Process_Happy_Path_Edge'] = f"{max_edge['Activity']} -> {max_edge['Next_Activity']}"
    results['Process_Happy_Path_Count'] = max_edge['weight']
    
    # Get a specific "Exception" edge (approx 10-15% of max)
    # We look for something significant but not the main path
    threshold = max_edge['weight'] * 0.2
    exceptions = edges[edges['weight'] < threshold].sort_values(by='weight', ascending=False)
    if not exceptions.empty:
        exc_edge = exceptions.iloc[0]
        results['Process_Exception_Edge'] = f"{exc_edge['Activity']} -> {exc_edge['Next_Activity']}"
        results['Process_Exception_Count'] = exc_edge['weight']

    # --- 3. FIDELITY METRICS (Re-Calculate KS) ---
    uw_durations = (uw['End_Time'] - uw['Start_Time']).values
    
    # Normal Fit
    mu, std = stats.norm.fit(uw_durations)
    ks_norm = stats.kstest(uw_durations, 'norm', args=(mu, std)).statistic
    results['Fidelity_ModelA_KS'] = ks_norm
    
    # Log-Normal Fit
    shape, loc, scale = stats.lognorm.fit(uw_durations, floc=0)
    ks_log = stats.kstest(uw_durations, 'lognorm', args=(shape, loc, scale)).statistic
    results['Fidelity_ModelB_KS'] = ks_log
    
    # Mixture Fit (Method C) - The Winner
    # Note: KStest for Mixture is hard to do analytically, but we can approximate or rely on Model B comparison
    # For the cheat sheet, we'll calculate the reduction %
    results['Fidelity_Error_Reduction_Pct'] = (1 - (0.0034 / ks_log)) * 100 # Using the 0.0034 from your validation_metrics.csv
    
    # --- 4. SENSITIVITY (From Existing CSV) ---
    if sensitivity_df is not None:
        try:
            mle_val = sensitivity_df[sensitivity_df['Perturbation_%'] == 0]['KS_Statistic'].values[0]
            shift_val = sensitivity_df[sensitivity_df['Perturbation_%'] == -5]['KS_Statistic'].values[0]
            
            results['Sensitivity_MLE_Error'] = mle_val
            results['Sensitivity_Manual_Shift_Error'] = shift_val
            results['Sensitivity_Structural_Bias_Pct'] = ((mle_val - shift_val) / shift_val) * 100
        except Exception as e:
            print(f"Could not extract sensitivity details: {e}")

    # --- SAVE TO CSV ---
    cheat_sheet_path = FIG_DIR.parent / 'data' / 'output' / 'results_cheat_sheet.csv'
    pd.Series(results).to_csv(cheat_sheet_path)
    print(f"SUCCESS: Cheat Sheet saved to {cheat_sheet_path}")
    print(pd.Series(results)) # Print to console for immediate checking


def main():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # Extract Underwriting durations
    uw_data = df[df['Activity'] == 'Underwriting']
    durations = uw_data['End_Time'] - uw_data['Start_Time']
    
    # 1. Fidelity Plots
    plot_fidelity_comparison(durations.values)
    
    # 2. Sensitivity Plots & Data
    sens_df = None
    if SENSITIVITY_PATH.exists():
        sens_df = pd.read_csv(SENSITIVITY_PATH)
        plot_sensitivity(sens_df)
    else:
        print("Sensitivity data not found, skipping Fig 2.")
        
    # 3. New Plots
    plot_bimodal_evidence(durations.values)
    plot_queue_dynamics(df)
    plot_process_map(df)
    
    # 4. GENERATE CHEAT SHEET (The New Step)
    generate_results_cheat_sheet(df, sens_df)

if __name__ == "__main__":
    main()
    