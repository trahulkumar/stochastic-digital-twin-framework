import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.mixture import GaussianMixture

# --- CONFIGURATION ---
SEED = 42
TEST_FRACTION = 0.30
K_CANDIDATES = [1, 2, 3, 4]
ACTIVITY = "Underwriting"

BASE_DIR = Path(__file__).resolve().parent.parent
PASSIVE_LOG = BASE_DIR / "data" / "raw" / "passive_log_100k.csv"
OUTPUT_DIR = BASE_DIR / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Only require operational columns; ignore attribute columns if present.
REQUIRED_COLS = [
    "CaseID",
    "Activity",
    "Arrival_Time",
    "Start_Time",
    "End_Time",
    "Resource",
    "Regime_Mode",
]


def _ensure_numeric(df: pd.DataFrame, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_and_validate_service_times(filepath: Path) -> np.ndarray:
    print(f"Loading log: {filepath}")
    if not filepath.exists():
        raise FileNotFoundError(f"Log file not found: {filepath}")

    df = pd.read_csv(filepath)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required operational columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    # Keep only underwriting rows
    act_df = df[df["Activity"] == ACTIVITY].copy()

    # Ensure numeric times
    act_df = _ensure_numeric(act_df, ["Arrival_Time", "Start_Time", "End_Time"])

    # Service time = End - Start
    act_df["service_time"] = act_df["End_Time"] - act_df["Start_Time"]

    # Drop invalid service times
    before = len(act_df)
    act_df = act_df[(act_df["service_time"].notna()) & (act_df["service_time"] > 0)]
    dropped = before - len(act_df)
    if dropped > 0:
        print(f"Warning: Dropped {dropped} invalid underwriting rows (service_time <= 0 or NaN).")

    data = act_df["service_time"].to_numpy(dtype=np.float64)
    if data.size == 0:
        raise ValueError("No valid underwriting service_time values after filtering.")

    return data


def train_test_split(data: np.ndarray, test_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(data))
    rng.shuffle(idx)
    split = int(len(data) * (1.0 - test_fraction))
    train = data[idx[:split]]
    test = data[idx[split:]]
    return train, test


def held_out_validation():
    np.random.seed(SEED)

    data = load_and_validate_service_times(PASSIVE_LOG)

    train, test = train_test_split(data, TEST_FRACTION, SEED)
    train_n, test_n = len(train), len(test)
    print(f"Underwriting samples: total={len(data)} train={train_n} test={test_n}")

    metrics = []

    # ---- Normal baseline (fit train, test KS on raw scale)
    mu, std = stats.norm.fit(train)
    ks_stat, p_val = stats.kstest(test, "norm", args=(mu, std))
    metrics.append(
        {
            "model_id": "M1",
            "model_name": "Normal",
            "family": "Unimodal",
            "ks_stat": float(ks_stat),
            "p_value": float(p_val),
            "test_type": "one-sample",
            "scale": "raw",
            "train_n": train_n,
            "test_n": test_n,
            "params_json": json.dumps({"mu": float(mu), "std": float(std)}),
            "notes": "Baseline",
        }
    )

    # ---- LogNormal baseline (fit train, test KS on raw scale)
    ln_shape, ln_loc, ln_scale = stats.lognorm.fit(train, floc=0)
    ks_stat, p_val = stats.kstest(test, "lognorm", args=(ln_shape, ln_loc, ln_scale))
    metrics.append(
        {
            "model_id": "M2",
            "model_name": "LogNormal",
            "family": "Unimodal",
            "ks_stat": float(ks_stat),
            "p_value": float(p_val),
            "test_type": "one-sample",
            "scale": "raw",
            "train_n": train_n,
            "test_n": test_n,
            "params_json": json.dumps(
                {"shape": float(ln_shape), "loc": float(ln_loc), "scale": float(ln_scale)}
            ),
            "notes": "Common service-time baseline",
        }
    )

    # ---- Gamma baseline (fit train, test KS on raw scale)
    g_shape, g_loc, g_scale = stats.gamma.fit(train, floc=0)
    ks_stat, p_val = stats.kstest(test, "gamma", args=(g_shape, g_loc, g_scale))
    metrics.append(
        {
            "model_id": "M3",
            "model_name": "Gamma",
            "family": "Unimodal",
            "ks_stat": float(ks_stat),
            "p_value": float(p_val),
            "test_type": "one-sample",
            "scale": "raw",
            "train_n": train_n,
            "test_n": test_n,
            "params_json": json.dumps(
                {"shape": float(g_shape), "loc": float(g_loc), "scale": float(g_scale)}
            ),
            "notes": "Alternative service-time baseline",
        }
    )

    # ---- Empirical resampling baseline (two-sample KS on raw scale)
    np.random.seed(SEED)
    sim_emp = np.random.choice(train, size=test_n, replace=True)
    ks_stat, p_val = stats.ks_2samp(test, sim_emp)
    metrics.append(
        {
            "model_id": "M4",
            "model_name": "EmpiricalResample",
            "family": "NonParametric",
            "ks_stat": float(ks_stat),
            "p_value": float(p_val),
            "test_type": "two-sample",
            "scale": "raw",
            "train_n": train_n,
            "test_n": test_n,
            "params_json": "{}",
            "notes": "Bootstrap resampling from train",
        }
    )

    # ---- GMM on log(service_time): select K via BIC on train; evaluate via two-sample KS on log scale
    X_train = np.log(train).reshape(-1, 1)
    X_test = np.log(test)  # 1D array

    bic_rows = []
    best_bic = np.inf
    best_k = None
    best_gmm = None

    for k in K_CANDIDATES:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=SEED,
            n_init=10,
        )
        gmm.fit(X_train)
        bic = float(gmm.bic(X_train))

        bic_rows.append(
            {"K": k, "bic": bic, "converged": bool(gmm.converged_), "n_iter": int(gmm.n_iter_)}
        )

        if bic < best_bic:
            best_bic = bic
            best_k = k
            best_gmm = gmm

    pd.DataFrame(bic_rows).to_csv(OUTPUT_DIR / "gmm_model_selection.csv", index=False)

    np.random.seed(SEED)
    x_sim, _ = best_gmm.sample(n_samples=test_n)
    x_sim = x_sim.flatten()

    ks_stat, p_val = stats.ks_2samp(X_test, x_sim)

    gmm_params = {
        "K": int(best_k),
        "weights": best_gmm.weights_.tolist(),
        "means": best_gmm.means_.flatten().tolist(),
        "covariances": best_gmm.covariances_.flatten().tolist(),
    }

    metrics.append(
        {
            "model_id": "M5",
            "model_name": f"GMM(K={best_k})",
            "family": "Mixture",
            "ks_stat": float(ks_stat),
            "p_value": float(p_val),
            "test_type": "two-sample",
            "scale": "log",
            "train_n": train_n,
            "test_n": test_n,
            "params_json": json.dumps(gmm_params),
            "notes": "BIC-selected mixture on log(service_time)",
        }
    )

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(OUTPUT_DIR / "validation_metrics.csv", index=False)

    # ---- Leakage-free sensitivity analysis (LogNormal): perturb fitted scale, evaluate on TEST
    print("\n--- Leakage-Free Sensitivity Analysis (LogNormal scale perturbation) ---")
    sens_rows = []
    for shift in [-0.05, 0.0, 0.05]:
        new_scale = ln_scale * (1.0 + shift)
        ks_s, p_s = stats.kstest(test, "lognorm", args=(ln_shape, ln_loc, new_scale))
        sens_rows.append(
            {
                "scale_shift": shift,
                "ks_stat": float(ks_s),
                "p_value": float(p_s),
                "train_n": train_n,
                "test_n": test_n,
            }
        )
    pd.DataFrame(sens_rows).to_csv(OUTPUT_DIR / "sensitivity_analysis.csv", index=False)

    # ---- Results summary / structural bias computed from held-out KS
    unimodal = metrics_df[metrics_df["family"] == "Unimodal"]["ks_stat"]
    best_unimodal_ks = float(unimodal.min())
    best_overall_ks = float(metrics_df["ks_stat"].min())

    # Structural bias (relative gap) and percent error reduction vs best unimodal
    structural_bias = (best_unimodal_ks - best_overall_ks) / best_unimodal_ks
    error_reduction_percent = 100.0 * structural_bias

    summary_df = pd.DataFrame(
        [
            {
                "best_unimodal_ks": best_unimodal_ks,
                "best_overall_ks": best_overall_ks,
                "structural_bias": float(structural_bias),
                "error_reduction_percent": float(error_reduction_percent),
                "best_k": int(best_k),
                "train_n": train_n,
                "test_n": test_n,
                "seed": SEED,
            }
        ]
    )
    summary_df.to_csv(OUTPUT_DIR / "results_summary.csv", index=False)

    # ---- CLI printout
    print("\n--- Held-out KS results (lower is better) ---")
    print(metrics_df[["model_name", "family", "ks_stat", "p_value", "scale", "test_type"]])

    print(f"\nBest K by BIC: {best_k}")
    print(f"Best unimodal KS: {best_unimodal_ks:.6f}")
    print(f"Best overall KS: {best_overall_ks:.6f}")
    print(f"Structural bias (relative gap): {structural_bias:.4f}")
    print(f"Error reduction vs best unimodal: {error_reduction_percent:.2f}%")

    print("\nOutputs written to:")
    print(f"  {OUTPUT_DIR / 'validation_metrics.csv'}")
    print(f"  {OUTPUT_DIR / 'gmm_model_selection.csv'}")
    print(f"  {OUTPUT_DIR / 'sensitivity_analysis.csv'}")
    print(f"  {OUTPUT_DIR / 'results_summary.csv'}")


if __name__ == "__main__":
    held_out_validation()