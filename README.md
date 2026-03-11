# Beyond the Average: Overcoming Fidelity Barriers in Stochastic Digital Twins for Financial Services Workflows

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Reproducibility Suite](https://img.shields.io/badge/Reproducibility-100%25-brightgreen.svg)]()

This repository contains the official reproducibility suite and source code for the manuscript **"Beyond the Average: Overcoming Fidelity Barriers in Stochastic Digital Twins for Financial Services Workflows."**

Many workflow-oriented Digital Twin deployments stall at a **fidelity barrier**: the twin matches average processing times but fails to reproduce the **multi-regime stochastic volatility** inherent in human-centric service processes. This project provides a fully reproducible computational pipeline that overcomes this barrier by integrating process-aware structural discovery, bimodal stochastic calibration, and closed-loop routing control. 

---

## 🚀 Key Features & Contributions

1. **Stochastic Calibration (Digital Twin "Brain")** Rigorous held-out evaluation of service-time models (Normal, Log-Normal, Gamma, Empirical Resampling). Features automatic model selection using Bayesian Information Criterion (BIC) to identify the optimal **Gaussian Mixture Model (GMM)**.
2. **Empirical Grounding (BPI 2017)** Validates the GMM approach against real-world data, demonstrating a **56.55% reduction** in Kolmogorov-Smirnov (KS) distribution error compared to standard unimodal baselines.
3. **Closed-Loop Control: Regime-Aware Dynamic Routing (RADR)** Simulates a bidirectional routing policy to isolate exception-prone work, reducing routine-capacity contamination. Demonstrates a reduction in peak underwriting backlog from 42,207 to 12,526 cases.
4. **Reproducibility-First Architecture** Designed to be fully reproducible without exposing sensitive institutional PII. All numerical values, metrics, and figures used in the manuscript are automatically generated and exported to `data/output/`.

---

## ⚙️ Quickstart

### Requirements
* Python 3.10+
* `uv` (fast Python package and project manager)

### Install Dependencies
```bash
uv sync

```

### Run the Full Reproducibility Suite

**Option A (Recommended): One command on Windows PowerShell**

```powershell
.\run_reproducibility.ps1

```

**Option B: Run the scripts manually (Any OS)**

```bash
uv run src/generator.py
uv run src/analyzer.py
uv run src/visualizer.py
uv run src/bpi_empirical_analysis.py

```

---

## 🧪 Reproducibility Workflow

### 1. Data Generation (Physical System Simulation)

Generates large-scale synthetic event logs (100,000 cases) simulating a loan workflow with resource contention. Exception probability is driven by **case covariates** (e.g., `Channel`, `CreditBand`, `DTI`, `DocsComplete`) using a logistic model with automatic intercept calibration to achieve a target exception rate.

* **Command:** `uv run src/generator.py`
* **Outputs:** `passive_log_100k.csv`, `active_radr_log_100k.csv`, `attribute_summary.csv`

### 2. Statistical Analysis & GMM Calibration

Performs strict **held-out validation** (Train/Test split) to evaluate model fidelity for underwriting service times. Produces leakage-free sensitivity results and verifies the 97.5% KS error reduction achieved by the GMM on synthetic data.

* **Command:** `uv run src/analyzer.py`
* **Outputs:** `validation_metrics.csv`, `gmm_model_selection.csv`, `sensitivity_analysis.csv`

### 3. Visualization & Operational Dynamics

Runs an efficient `O(N log N)` event-sweep queue engine to compute operational metrics comparing passive routing versus the RADR intervention. Generates all queueing and backlog visualizations.

* **Command:** `uv run src/visualizer.py`
* **Outputs:** `ops_metrics.csv`, plus all simulated figures in `figures/`

### 4. Empirical Validation (BPI Challenge 2017)

Extracts service times for specific manual activities (e.g., `W_Validate application`) from the real-world Business Process Intelligence (BPI) 2017 dataset to empirically prove the existence of bimodal "Routine" vs. "Exception" regimes.

* **Command:** `uv run src/bpi_empirical_analysis.py`
* **Outputs:** `bpi_empirical_metrics.csv`, BPI macro and tail figures in `figures/`.

---

## 📊 Manuscript Figures & Artifacts

All figures referenced in the manuscript are generated locally by the suite and saved as both high-resolution PDF and PNG files in the `figures/` directory.

| Manuscript Ref | Description | Generated Filenames |
| --- | --- | --- |
| **Figure 1** | Empirical Bimodality Evidence | `fig1a_bimodal_macro`, `fig1b_bimodal_micro` |
| **Figure 2** | Discovered Process Flow | `fig2_process_map` |
| **Figure 3** | Fidelity Comparison (Mixture vs Baselines) | `fig3a_full_distribution`, `fig3b_tail_detail` |
| **Figure 4** | Sensitivity Analysis (Structural Bias) | `fig4_sensitivity_analysis` |
| **Figure 5** | Resource Contention & Queue Build-Up | `fig5_queue_dynamics` |
| **Figure 6** | Underwriting Backlog & WIP (Passive vs RADR) | `fig6a_underwriting_backlog...`, `fig6b_underwriting_wip...` |
| **Empirical** | BPI 2017 Real-World Validation | `fig_empirical_bpi_macro`, `fig_empirical_bpi_tail` |

---

## 📂 Data & Software Availability

* **Software Repository:** [https://github.com/trahulkumar/stochastic-digital-twin-framework](https://github.com/trahulkumar/stochastic-digital-twin-framework)
* **Synthetic Data:** Generated locally by `src/generator.py` and written to `data/raw/`.
* **Empirical Data:** The BPI Challenge 2017 event log utilized for validation is publicly available via the official 4TU.ResearchData repository ([DOI: 10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b](https://doi.org/10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b)).
* **Operational Metrics:** Use the exported CSV files in `data/output/` as the single source of truth for numeric results reported in the paper.

---

## 📝 How to Cite

If you use this code, the synthetic generator, or the RADR methodology in your research, please cite the accompanying paper:

> Thatikonda, R. K., & Donepudi, S. (2026). Beyond the Average: Overcoming Fidelity Barriers in Stochastic Digital Twins for Financial Services Workflows. *Digital Twin*. [Submitted / Under Review]

```
