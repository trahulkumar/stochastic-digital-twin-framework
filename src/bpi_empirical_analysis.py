import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from sklearn.mixture import GaussianMixture

# --- CONFIGURATION ---
SEED = 42
TEST_FRACTION = 0.30
ACTIVITY_NAME = "W_Validate application"
MIN_DURATION = 0.0  # Filter out <= 0

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "raw" / "bpi_2017_cleaned.csv"
METRICS_OUT = BASE_DIR / "data" / "output" / "bpi_empirical_metrics.csv"
FIGURE_MACRO = BASE_DIR / "figures" / "fig_empirical_bpi_macro"
FIGURE_TAIL = BASE_DIR / "figures" / "fig_empirical_bpi_tail"

# Styling
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]


def extract_and_preprocess():
    print(f"Reading BPI dataset: {INPUT_FILE.name}...")
    df = pd.read_csv(INPUT_FILE)

    # 1. Filter for specific activity
    print(f"Filtering for activity: {ACTIVITY_NAME}")
    df_act = df[df["concept:name"] == ACTIVITY_NAME].copy()

    # 2. Extract Duration
    # Group by Case ID to find min/max for the specific activity
    # Note: Some activities might occur multiple times per case; 
    # we take the span of that activity within the case.
    print("Calculating service times...")
    df_act["time:timestamp"] = pd.to_datetime(df_act["time:timestamp"], format='ISO8601', utc=True)
    
    grouped = df_act.groupby("case:concept:name").agg(
        Start_Time=("time:timestamp", "min"),
        End_Time=("time:timestamp", "max")
    ).reset_index()

    # Calculate duration in minutes
    grouped["duration"] = (grouped["End_Time"] - grouped["Start_Time"]).dt.total_seconds() / 60.0
    
    # Filter valid durations
    valid = grouped[grouped["duration"] > MIN_DURATION]["duration"].to_numpy()
    print(f"Extracted {len(valid)} valid service time samples.")
    return valid


def fit_models(data):
    # Train/Test Split
    print(f"Performing {int((1-TEST_FRACTION)*100)}/{int(TEST_FRACTION*100)} split...")
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(data))
    rng.shuffle(idx)
    split_point = int(len(data) * (1.0 - TEST_FRACTION))
    train_data = data[idx[:split_point]]
    test_data = data[idx[split_point:]]

    # --- Fit LogNormal (Baseline) ---
    print("Fitting LogNormal model...")
    ln_shape, ln_loc, ln_scale = stats.lognorm.fit(train_data, floc=0)
    
    # --- Fit GMM (2 components on log-scale) ---
    print("Fitting 2-component GMM on log-scale...")
    train_log = np.log(train_data).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=SEED, n_init=5)
    gmm.fit(train_log)

    # --- Evaluation ---
    # LogNormal KS
    ks_ln, p_ln = stats.kstest(test_data, "lognorm", args=(ln_shape, ln_loc, ln_scale))
    
    # GMM KS (Two-sample on log-scale)
    test_log = np.log(test_data)
    # Generate synthetic samples from GMM
    sim_log, _ = gmm.sample(len(test_data))
    ks_gmm, p_gmm = stats.ks_2samp(test_log, sim_log.flatten())

    reduction = 100.0 * (ks_ln - ks_gmm) / ks_ln
    
    print(f"\nResults on test set (N={len(test_data)}):")
    print(f"  LogNormal KS: {ks_ln:.4f}")
    print(f"  GMM(K=2) KS:  {ks_gmm:.4f}")
    print(f"  Error Reduction: {reduction:.2f}%")

    metrics = {
        "dataset": "BPI_2017",
        "activity": ACTIVITY_NAME,
        "train_n": len(train_data),
        "test_n": len(test_data),
        "ks_lognormal": ks_ln,
        "ks_gmm": ks_gmm,
        "error_reduction_pct": reduction,
        "ln_params": {"shape": ln_shape, "loc": ln_loc, "scale": ln_scale},
        "gmm_params": {
            "weights": gmm.weights_.tolist(),
            "means": gmm.means_.flatten().tolist(),
            "covars": gmm.covariances_.flatten().tolist()
        }
    }
    
    # Save Metrics
    pd.DataFrame([metrics]).to_csv(METRICS_OUT, index=False)
    print(f"Metrics saved to {METRICS_OUT}")

    return train_data, test_data, metrics, (ln_shape, ln_loc, ln_scale), gmm


def visualize(train_data, metrics, ln_params, gmm):
    print("Generating visualizations...")
    sns.set_theme(style="whitegrid", font="serif")
    
    # Pre-calculate PDFs for plotting
    x_raw = np.linspace(0, np.percentile(train_data, 99.5), 500)
    pdf_ln = stats.lognorm.pdf(x_raw, *ln_params)
    
    # GMM PDF requires combining components on log scale then transforming
    # Or more simply, sample heavily and KDE, but let's do exact mix if possible
    # For plotting raw scale, we can sample 100k points
    sim_log, _ = gmm.sample(100000)
    sim_raw = np.exp(sim_log.flatten())
    
    # Figure A: Macro View
    plt.figure(figsize=(8, 6))
    sns.histplot(train_data, stat="density", color="gray", alpha=0.3, label="BPI Data")
    plt.plot(x_raw, pdf_ln, 'r--', lw=2, label="LogNormal (Unimodal)")
    sns.kdeplot(sim_raw, color="blue", lw=2, label="GMM (Bimodal)")
    
    plt.title(f"Macro View: {ACTIVITY_NAME} Durations", fontsize=14)
    plt.xlabel("Duration (Minutes)")
    plt.ylabel("Density")
    plt.xlim(0, np.percentile(train_data, 98))
    plt.legend()
    
    plt.annotate("Routine Mode", xy=(np.median(train_data), 0.01), xytext=(np.median(train_data)*2, 0.02),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

    for ext in ['pdf', 'png']:
        plt.savefig(f"{FIGURE_MACRO}.{ext}", dpi=300, bbox_inches='tight')
    plt.close()

    # Figure B: Tail View
    # Figure B: Tail View
    plt.figure(figsize=(8, 6))
    tail_thresh = np.percentile(train_data, 90)
    
    # Filter the data for the tail to get a dynamic y-limit
    tail_data = train_data[train_data > tail_thresh]
    
    sns.histplot(train_data, stat="density", color="gray", alpha=0.3)
    plt.plot(x_raw, pdf_ln, 'r--', lw=2, label="LogNormal")
    sns.kdeplot(sim_raw, color="blue", lw=2, label="GMM")
    
    plt.title("Micro/Tail View: Exception Regime", fontsize=14)
    plt.xlabel("Duration (Minutes)")
    plt.ylabel("Density")
    plt.xlim(tail_thresh, np.percentile(train_data, 99.9))
    
    # --- THE FIX ---
    # Remove the hardcoded plt.ylim(0, 0.005). 
    # Instead, dynamically zoom the y-axis based on the maximum density in the tail region
    plt.ylim(0, np.max(pdf_ln[x_raw > tail_thresh]) * 2.5) 
    
    plt.legend()
    
    # Adjust annotation position to fit the new dynamic scale
    y_max = plt.gca().get_ylim()[1]
    plt.annotate("Exception Tail", 
                 xy=(np.percentile(train_data, 95), y_max * 0.2), 
                 xytext=(np.percentile(train_data, 96), y_max * 0.4),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

    for ext in ['pdf', 'png']:
        plt.savefig(f"{FIGURE_TAIL}.{ext}", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Figures saved as {FIGURE_MACRO}.* and {FIGURE_TAIL}.*")


def main():
    try:
        data = extract_and_preprocess()
        train, test, metrics, ln_params, gmm = fit_models(data)
        visualize(train, metrics, ln_params, gmm)
        print("\nEmpirical demonstration complete.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
