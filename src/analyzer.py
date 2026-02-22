import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.mixture import GaussianMixture
from pathlib import Path

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / 'data' / 'raw' / 'passive_log_100k.csv'
OUTPUT_METRICS = BASE_DIR / 'data' / 'output' / 'validation_metrics.csv'
OUTPUT_SENSITIVITY = BASE_DIR / 'data' / 'output' / 'sensitivity_analysis.csv'

# Ensure output dir
OUTPUT_METRICS.parent.mkdir(parents=True, exist_ok=True)

def hierarchical_validation(data):
    print("\n--- Performing Hierarchical Model Fit ---")
    results = []
    
    # 1. Model A: Normal (Baseline)
    mu, std = stats.norm.fit(data)
    ks_stat, p_val = stats.kstest(data, 'norm', args=(mu, std))
    results.append({
        'Model': 'Model A (Normal)',
        'KS_Statistic': ks_stat,
        'p_value': p_val,
        'Parameters': f"mu={mu:.2f}, std={std:.2f}"
    })
    
    # 2. Model B: Log-Normal (Parametric)
    shape, loc, scale = stats.lognorm.fit(data, floc=0)
    ks_stat, p_val = stats.kstest(data, 'lognorm', args=(shape, loc, scale))
    results.append({
        'Model': 'Model B (Log-Normal)',
        'KS_Statistic': ks_stat,
        'p_value': p_val,
        'Parameters': f"s={shape:.2f}, scale={scale:.2f}"
    })
    
    # 3. Model C: Mixture Model (Proposed)
    # Fit GMM on Log-transformed data (since durations are > 0 and skewed)
    # Note: Logic is to fit on log data then project back, or standard GMM on raw/log.
    # Prompt says "Gaussian Mixture Model on log-data".
    log_data = np.log(data + 1e-9).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
    gmm.fit(log_data)
    
    # Generate Synthetic Samples to test KS
    n_samples = len(data)
    weights = gmm.weights_
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    
    # Sample component indices
    comps = np.random.choice(len(weights), size=n_samples, p=weights)
    # Sample from Normal(mean, std) for each chosen component
    sim_log = np.array([np.random.normal(means[c], stds[c]) for c in comps])
    sim_data = np.exp(sim_log) # Transform back
    
    ks_stat, p_val = stats.ks_2samp(data, sim_data)
    results.append({
        'Model': 'Model C (Mixture)',
        'KS_Statistic': ks_stat,
        'p_value': p_val,
        'Parameters': f"w={weights.round(2)}, mu={means.round(2)}"
    })
    
    return pd.DataFrame(results), (shape, loc, scale)

def sensitivity_test(data, lognorm_params):
    print("\n--- Performing Sensitivity Analysis (Model B) ---")
    shape, loc, scale = lognorm_params
    perturbations = [-0.05, 0.0, 0.05] # +/- 5%
    results = []
    
    for p in perturbations:
        # Perturb Scale (which relates to the median/mean)
        new_scale = scale * (1 + p)
        ks_stat, p_val = stats.kstest(data, 'lognorm', args=(shape, loc, new_scale))
        results.append({
            'Perturbation_%': p * 100,
            'KS_Statistic': ks_stat
        })
    
    return pd.DataFrame(results)

def main():
    print(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    # Filter for Underwriting
    data = df[df['Activity'] == 'Underwriting']['End_Time'] - df[df['Activity'] == 'Underwriting']['Start_Time']
    # Or simpler if we have 'Duration' column? Generator doesn't save Duration explicitly, but Start/End.
    # Let's compute it.
    
    data = data.values
    print(f"Analyzing {len(data)} Underwriting cases...")
    
    # Run Validation
    metrics_df, lognorm_params = hierarchical_validation(data)
    print("\nValidation Metrics:")
    print(metrics_df)
    metrics_df.to_csv(OUTPUT_METRICS, index=False)
    print(f"Saved to {OUTPUT_METRICS}")
    
    # Run Sensitivity
    sens_df = sensitivity_test(data, lognorm_params)
    print("\nSensitivity Analysis:")
    print(sens_df)
    sens_df.to_csv(OUTPUT_SENSITIVITY, index=False)
    print(f"Saved to {OUTPUT_SENSITIVITY}")

if __name__ == "__main__":
    main()
