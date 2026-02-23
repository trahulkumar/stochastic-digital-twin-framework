import simpy
import random
import pandas as pd
import numpy as np
from pathlib import Path

# --- CONFIGURATION ---
NUM_CASES = 100000
ARRIVAL_RATE = 5  # cases per hour
SEED = 42

# Calibration Constants (attribute-driven exception rate)
TARGET_EXCEPTION_RATE = 0.20
N_CAL = 20000
CAL_TOL = 1e-3
CAL_MAX_ITERS = 60

# Resource Capacities (Baseline Staffing)
N_CLERKS = 8
N_UNDERWRITERS_PRIMARY = 5  # "Routine Lane"
N_UNDERWRITERS_RESERVE = 2  # "Exception Lane" (RADR Capacity)
N_VPS = 2

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_OUTPUT_DIR = BASE_DIR / "data" / "output"
SUMMARY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_lognorm_duration(mu, sigma):
    # Interpreting mu/sigma as log-space parameters.
    return float(np.random.lognormal(mean=mu, sigma=sigma))


def sigmoid(x):
    # Guard against overflow in exp for extreme inputs.
    x = np.clip(x, -60, 60)
    return 1.0 / (1.0 + np.exp(-x))


def generate_attributes():
    """
    Generates observable case attributes (covariates).
    These are repeated on every event row for that case.
    """
    channel = np.random.choice(["Online", "Branch", "Partner"], p=[0.60, 0.25, 0.15])
    credit_band = np.random.choice(["A", "B", "C", "D"], p=[0.35, 0.30, 0.22, 0.13])

    # DTI beta distribution scaled to [0, 0.8]
    dti = 0.8 * np.random.beta(2.2, 5.0)
    dti = float(np.clip(dti, 0.0, 0.8))

    # DocsComplete probability depends on channel
    if channel == "Online":
        p_docs = 0.88
    elif channel == "Branch":
        p_docs = 0.95
    else:  # Partner
        p_docs = 0.80
    docs_complete = 1 if np.random.random() < p_docs else 0

    # Optional LoanAmountBand
    loan_amount_band = np.random.choice(["Small", "Medium", "Large"], p=[0.40, 0.40, 0.20])

    return {
        "Channel": channel,
        "CreditBand": credit_band,
        "DTI": dti,
        "DocsComplete": int(docs_complete),
        "LoanAmountBand": loan_amount_band,
    }


def linear_predictor_no_intercept(attrs):
    """
    Linear predictor for exception risk EXCLUDING intercept b0.
    Intercept is calibrated to hit TARGET_EXCEPTION_RATE.
    """
    b_credit = {"A": -1.0, "B": -0.4, "C": 0.4, "D": 1.0}
    b_channel = {"Online": 0.2, "Branch": -0.2, "Partner": 0.4}
    b_dti = 3.0
    b_docs = 1.2

    lp = (
        b_credit[attrs["CreditBand"]]
        + b_channel[attrs["Channel"]]
        + b_dti * (attrs["DTI"] - 0.25)
        + b_docs * (1 - attrs["DocsComplete"])
    )
    return float(lp)


def calibrate_intercept():
    """
    Calibrate b0 so that mean(sigmoid(b0 + lp)) ~= TARGET_EXCEPTION_RATE
    over a calibration sample of attributes.
    """
    print(f"Calibrating intercept for target exception rate = {TARGET_EXCEPTION_RATE:.0%} ...")

    cal_attrs = [generate_attributes() for _ in range(N_CAL)]
    cal_lp = np.array([linear_predictor_no_intercept(a) for a in cal_attrs], dtype=np.float64)

    low, high = -6.0, 2.0
    b0 = 0.0
    p_exc = None

    for _ in range(CAL_MAX_ITERS):
        b0 = (low + high) / 2.0
        p_exc = float(np.mean(sigmoid(b0 + cal_lp)))
        if abs(p_exc - TARGET_EXCEPTION_RATE) < CAL_TOL:
            break
        if p_exc < TARGET_EXCEPTION_RATE:
            low = b0
        else:
            high = b0

    print(f"Calibration complete: b0={b0:.4f}, achieved rate≈{p_exc:.4f}")
    return float(b0)


class FinancialProcess:
    def __init__(self, env, b0, use_radr=False):
        self.env = env
        self.use_radr = use_radr
        self.b0 = b0

        self.clerk = simpy.Resource(env, capacity=N_CLERKS)
        self.underwriter_primary = simpy.Resource(env, capacity=N_UNDERWRITERS_PRIMARY)
        self.underwriter_reserve = simpy.Resource(env, capacity=N_UNDERWRITERS_RESERVE)
        self.vp = simpy.Resource(env, capacity=N_VPS)

        self.log = []

    def log_event(self, case_id, activity, arrival, start, end, resource, mode, attrs):
        event = {
            "CaseID": case_id,
            "Activity": activity,
            "Arrival_Time": float(arrival),
            "Start_Time": float(start),
            "End_Time": float(end),
            "Resource": resource,
            "Regime_Mode": mode,
        }
        event.update(attrs)  # add attribute columns
        self.log.append(event)

    def process_case(self, case_id):
        # --- Generate observable covariates ---
        attrs = generate_attributes()

        # --- Attribute-driven exception probability ---
        lp = linear_predictor_no_intercept(attrs)
        p_exc = sigmoid(self.b0 + lp)
        is_exception = (random.random() < p_exc)

        # Optional: exception affects probability of VP Approval (structural branch)
        is_high_value = (random.random() < (0.80 if is_exception else 0.05))
        mode_label = "Exception" if is_exception else "Routine"

        # --- STEP 1: CLERK DATA ENTRY ---
        arrival = self.env.now
        with self.clerk.request() as req:
            yield req
            start = self.env.now
            yield self.env.timeout(get_lognorm_duration(2.1, 0.3))
        self.log_event(case_id, "Data_Entry", arrival, start, self.env.now, "Clerk", mode_label, attrs)

        # --- STEP 2: UNDERWRITING (with RADR routing + mild attribute effects) ---
        arrival = self.env.now

        selected_resource = self.underwriter_primary
        resource_name = "Primary_UW"
        if self.use_radr and is_exception:
            selected_resource = self.underwriter_reserve
            resource_name = "Reserve_UW"

        # Base regime parameters (log-space)
        if is_exception:
            base_mean, base_sigma = 5.5, 0.6
        else:
            base_mean, base_sigma = 3.8, 0.4

        # Mild attribute effects (log-mean offsets)
        if attrs["DocsComplete"] == 0:
            base_mean += 0.15
        if attrs["Channel"] == "Partner":
            base_mean += 0.10
        if attrs["CreditBand"] == "D":
            base_mean += 0.10

        with selected_resource.request() as req:
            yield req
            start = self.env.now
            yield self.env.timeout(get_lognorm_duration(base_mean, base_sigma))
        self.log_event(case_id, "Underwriting", arrival, start, self.env.now, resource_name, mode_label, attrs)

        # --- STEP 3: VP APPROVAL (exception branch more likely for exceptions) ---
        if is_high_value:
            arrival = self.env.now
            with self.vp.request() as req:
                yield req
                start = self.env.now
                yield self.env.timeout(get_lognorm_duration(4.1, 0.8))
            self.log_event(case_id, "VP_Approval", arrival, start, self.env.now, "VP", mode_label, attrs)


def run_simulation(b0, use_radr=False, filename="financial_log.csv"):
    env = simpy.Environment()
    process = FinancialProcess(env, b0, use_radr=use_radr)

    def arrival_generator():
        case_id = 0
        while case_id < NUM_CASES:
            env.process(process.process_case(case_id))
            interarrival = random.expovariate(ARRIVAL_RATE / 60.0)  # minutes
            yield env.timeout(interarrival)
            case_id += 1

    env.process(arrival_generator())
    env.run()

    df = pd.DataFrame(process.log)
    output_path = OUTPUT_DIR / filename
    df.to_csv(output_path, index=False)
    print(f"Simulation complete. Saved to {output_path}")
    return df


def perform_sanity_check(df):
    # Use one row per case (attributes/regime are case-level in this generator)
    case_df = df.drop_duplicates(subset=["CaseID"]).copy()

    overall_exception_rate = float((case_df["Regime_Mode"] == "Exception").mean())
    print("\n--- SANITY CHECK SUMMARY ---")
    print(f"Overall Exception Rate: {overall_exception_rate:.2%}")

    print("\nMean DTI by Regime:")
    print(case_df.groupby("Regime_Mode")["DTI"].mean())

    # Group summary
    summary_df = (
        case_df.groupby(["CreditBand", "Channel"], as_index=False)
        .agg(
            n_cases=("CaseID", "count"),
            exception_rate=("Regime_Mode", lambda x: float((x == "Exception").mean())),
            avg_dti=("DTI", "mean"),
            docs_complete_rate=("DocsComplete", "mean"),
        )
        .sort_values(["CreditBand", "Channel"])
    )

    summary_path = SUMMARY_OUTPUT_DIR / "attribute_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary metrics written to {summary_path}")
    print("----------------------------\n")


if __name__ == "__main__":
    # Ensure reproducibility
    random.seed(SEED)
    np.random.seed(SEED)

    # 1) Calibrate intercept b0 to hit TARGET_EXCEPTION_RATE
    b0_calibrated = calibrate_intercept()

    # 2) Generate Passive Log
    random.seed(SEED)
    np.random.seed(SEED)
    print("\nStarting Passive Log Generation...")
    passive_df = run_simulation(b0=b0_calibrated, use_radr=False, filename="passive_log_100k.csv")

    # 3) Generate Active RADR Log
    # Re-seed so the attribute stream is reproducible run-to-run.
    # Note: because routing changes event timing, downstream random draws (durations) can consume RNG
    # in a different order between passive and RADR runs; that's acceptable for this synthetic experiment.
    random.seed(SEED)
    np.random.seed(SEED)
    print("\nStarting Active RADR Log Generation...")
    active_df = run_simulation(b0=b0_calibrated, use_radr=True, filename="active_radr_log_100k.csv")

    # 4) Sanity check (passive is sufficient; attributes are generated the same way in both)
    perform_sanity_check(passive_df)