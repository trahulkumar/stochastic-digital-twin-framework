# Beyond the Average: Overcoming Fidelity Barriers in Stochastic Digital Twins for Financial Services

**Reproducibility Suite**

## Abstract
This repository contains the reproducibility suite for the manuscript *"Beyond the Average: Overcoming Fidelity Barriers in Stochastic Digital Twins for Financial Services"*.

Many workflow-oriented Digital Twin deployments stall at a **fidelity barrier**: the twin matches averages but fails to reproduce (and therefore govern) **multi-regime stochastic volatility** in human-centric service processes. This project provides a reproducible pipeline that integrates:

- **Synthetic event-log generation** for a representative loan workflow,
- **Process-aware stochastic calibration** using baseline families and **Gaussian Mixture Models (GMM)**,
- A bidirectional intervention mechanism: **Regime-Aware Dynamic Routing (RADR)**,
- Distributional validation and operational metrics that connect fidelity improvements to queueing outcomes.

This repository is designed to be fully reproducible without using any sensitive institutional data.

---

## Key Features
1. **Stochastic calibration (Digital Twin “brain”)**  
   Held-out evaluation of service-time models (Normal, Log-Normal, Gamma, Empirical Resampling) and a BIC-selected **GMM**.

2. **Structural discovery (process map)**  
   Process-flow visualization from event transitions to separate routine flow from exceptions.

3. **Closed-loop control (RADR)**  
   Simulation of routing to isolate likely exception work and reduce routine-capacity contamination.

4. **Reproducibility-first artifacts**  
   All numerical values used for manuscript tables are exported as CSVs in `data/output/`.

---

## Quickstart

### Requirements
- Python + `uv` installed

### Install dependencies
```bash
uv sync
```

### Run the full reproducibility suite

#### Option A (recommended): one command on Windows PowerShell

```powershell
.\run_reproducibility.ps1
```

#### Option B: run the scripts manually (any OS)

```bash
uv run src/generator.py
uv run src/analyzer.py
uv run src/visualizer.py
```

---

## Reproducibility Workflow

### 1. Data Generation (Physical System Simulation)

Generates large-scale synthetic event logs (100k cases) simulating a loan workflow with resource contention.

**New in this version:** exception probability is driven by **case covariates** (e.g., `Channel`, `CreditBand`, `DTI`, `DocsComplete`) using a logistic model with automatic intercept calibration to achieve a target exception rate.

```bash
uv run src/generator.py
```

**Outputs**

* `data/raw/passive_log_100k.csv`
* `data/raw/active_radr_log_100k.csv`
* `data/output/attribute_summary.csv` (exception rate and summary stats by segment)

---

### 2. Statistical Analysis & GMM Calibration (Digital Twin Brain)

Performs strict **held-out validation** (Train/Test split) to evaluate model fidelity for underwriting service times.

**New in this version:**

* Adds **Gamma** and **Empirical Resampling** baselines
* Adds automatic **GMM component selection** using **BIC**
* Produces leakage-free sensitivity results

```bash
uv run src/analyzer.py
```

**Outputs**

* `data/output/validation_metrics.csv` (held-out KS metrics for all models)
* `data/output/gmm_model_selection.csv` (BIC scores for K candidates)
* `data/output/sensitivity_analysis.csv` (leakage-free lognormal perturbation analysis)
* `data/output/results_summary.csv` (best unimodal vs best overall + structural bias summary)

---

### 3. Visualization & Operational Dynamics (RADR Intervention)

Generates manuscript-ready figures and computes operational metrics comparing passive routing vs RADR.

**New in this version:**

* Efficient `O(N log N)` event-sweep queue engine
* Produces two Figure 6 subfigures:

  * **Fig 6a:** Underwriting **Backlog** (waiting only; arrival → start)
  * **Fig 6b:** Underwriting **WIP** (waiting + in service; arrival → end)
* Exports an operational metrics table used for manuscript reporting

```bash
uv run src/visualizer.py
```

**Outputs**

* `data/output/ops_metrics.csv` (queue + wait metrics for Passive vs RADR)
* Figures in `figures/` (PDF + PNG)

---

### 4. Empirical Validation (BPI Challenge 2017)

Validates the GMM approach on a real-world dataset (BPI Challenge 2017) by extracting service times for specific activities (e.g., `W_Validate application`) andComparing GMM fidelity against unimodal baselines.

**Key results:**
* Demonstrates high-fidelity capture of bimodal "Routine" vs "Exception" modes.
* Significant reduction in KS distance compared to unimodal LogNormal benchmarks.

```bash
uv run src/bpi_empirical_analysis.py
```

**Outputs**

* `data/output/bpi_empirical_metrics.csv` (KS metrics and model parameters)
* `figures/fig_empirical_bpi_macro.png` (Macro-level bimodality view)
* `figures/fig_empirical_bpi_tail.png` (Micro-level exception tail zoom)

---

## Manuscript Figures & Artifacts

| Manuscript Ref | Description                                | Output Filename (saved as PDF + PNG)           |
| -------------- | ------------------------------------------ | ---------------------------------------------- |
| **Figure 1**   | Empirical Bimodality Evidence              | `fig1a_bimodal_macro`, `fig1b_bimodal_micro`   |
| **Figure 2**   | Discovered Process Flow                    | `fig2_process_map`                             |
| **Figure 3**   | Fidelity Comparison (Mixture vs Baselines) | `fig3a_full_distribution`, `fig3b_tail_detail` |
| **Figure 4**   | Sensitivity Analysis (Structural Bias)     | `fig4_sensitivity_analysis`                    |
| **Figure 5**   | Resource Contention & Queue Build-Up       | `fig5_queue_dynamics`                          |
| **Figure 6a**  | Underwriting Backlog (Passive vs RADR)     | `fig6a_underwriting_backlog_passive_vs_radr`   |
| **Figure 6b**  | Underwriting WIP (Passive vs RADR)         | `fig6b_underwriting_wip_passive_vs_radr`       |
| **Appendix Fig**| BPI Empirical Bimodality                   | `fig_empirical_bpi_macro`, `fig_empirical_bpi_tail` |

---

## Data & Software Availability

* **Synthetic data:** generated locally by `src/generator.py` and written to `data/raw/`.
* **Reproducibility artifacts:** exported to `data/output/` (CSV outputs) and `figures/` (PDF/PNG).
* This project is designed to support open research standards without exposing sensitive institutional data.

**Zenodo Archive DOI:** *[replace with your DOI]*

---

## Notes on Interpretation

This suite reports fidelity and control outcomes from synthetic experiments. The resulting metrics depend on configuration (arrival rate, staffing, RADR capacity, covariate model, and random seed). For manuscript reporting, use the exported CSV files in `data/output/` as the single source of truth for numeric results.

If you want, I can also produce a short “How to cite” section and a “Configuration knobs” section (ARRIVAL_RATE, staffing, TARGET_EXCEPTION_RATE, SLA threshold) — but I won’t add it unless you explicitly ask.
