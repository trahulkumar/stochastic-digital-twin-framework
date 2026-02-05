import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / 'outputs'
# Ensure output directory exists (though it should already)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = OUTPUT_DIR / 'financial_event_log_100k.csv'
FIG_PATH = OUTPUT_DIR / 'fig_fidelity_hierarchy.png'

# Load Data
print(f"Loading data from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)
data = df[df['Activity'] == 'Underwriting']['Duration'].values
log_data = np.log(data + 1e-9).reshape(-1, 1)

print("\n--- HIERARCHICAL VALIDATION RESULTS ---")

# 1. Baseline: Normal
mu_n, std_n = stats.norm.fit(data)
ks_norm = stats.kstest(data, 'norm', args=(mu_n, std_n)).statistic
# EMD approx for Normal
sim_norm = stats.norm.rvs(mu_n, std_n, size=len(data))
emd_norm = stats.wasserstein_distance(data, sim_norm)

print(f"[Baseline] Normal     -> KS: {ks_norm:.3f} | EMD: {emd_norm:.2f}")

# 2. Method A: Log-Normal
shape, loc, scale = stats.lognorm.fit(data, floc=0)
ks_log = stats.kstest(data, 'lognorm', args=(shape, loc, scale)).statistic
sim_log = stats.lognorm.rvs(shape, loc=loc, scale=scale, size=len(data))
emd_log = stats.wasserstein_distance(data, sim_log)

print(f"[Method A] Log-Normal -> KS: {ks_log:.3f} | EMD: {emd_log:.2f}")

# 3. Method B: Mixture Model
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
gmm.fit(log_data)
weights = gmm.weights_
means = gmm.means_.flatten()
stds = np.sqrt(gmm.covariances_.flatten())

# Sample from Mixture
comps = np.random.choice([0, 1], size=len(data), p=weights)
sim_mix_log = np.array([np.random.normal(means[c], stds[c]) for c in comps])
sim_mix = np.exp(sim_mix_log)

ks_mix, p_mix = stats.ks_2samp(data, sim_mix)
emd_mix = stats.wasserstein_distance(data, sim_mix)

print(f"[Method B] Mixture    -> KS: {ks_mix:.3f} | p-value: {p_mix:.3f} | EMD: {emd_mix:.2f}")

# Plotting Comparison
plt.figure(figsize=(10, 6))
sns.kdeplot(data, color='black', fill=True, alpha=0.1, linewidth=2, label='Ground Truth')
sns.kdeplot(sim_norm, color='red', linestyle='--', label='Baseline (Normal)')
sns.kdeplot(sim_log, color='blue', linestyle=':', label='Method A (Log-Normal)')
sns.kdeplot(sim_mix, color='green', linestyle='-', linewidth=2, label='Method B (Mixture)')
plt.xlim(0, 150)
plt.title("Evolution of Fidelity: From Baseline to Mixture Model")
plt.legend()
plt.savefig(FIG_PATH, dpi=300)
print(f"Figure saved to {FIG_PATH}")
