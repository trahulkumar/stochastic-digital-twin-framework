# Beyond the Average: Overcoming Structural Bias in Stochastic Digital Twins

**Submission to INFORMS Journal on Data Science (IJDS)**

## Abstract
This repository contains the **Reproducibility Suite** for the manuscript *"Beyond the Average: Overcoming Structural Bias in Stochastic Digital Twins for Financial Services"*.

Standard operational simulations often fail to capture the volatility of human-centric workflows. This research introduces a hybrid framework integrating **Inductive Process Mining** with **Gaussian Mixture Models (GMM)** to construct high-fidelity Digital Twins. By explicitly modeling "Regime Switching" (Routine vs. Exception), we demonstrate a **97.6% reduction in error** and reveal the "Cost of Ignorance" in operational risk capital.

## Key Results (Reproduced via `src/analyzer.py`)
The suite validates the statistical fidelity reported in **Table 1** of the manuscript:

| Model Strategy | KS Statistic ($D$) | Fit Error Reduction | OpVaR ($P_{99}$) Gap |
| :--- | :--- | :--- | :--- |
| **Model A (Normal)** | 0.281 ($p < 0.001$) | -- | High Deviation |
| **Model B (Log-Normal)** | 0.142 ($p < 0.001$) | Reference | **240% Underestimation** |
| **Model C (GMM)** | **0.0034** ($p > 0.05$) | **-97.6%** | **Accurate** |

> **Key Finding:** We identify a **23.1% Structural Bias** in standard Log-Normal fitting, proving that unimodal architectures cannot represent the heavy-tailed reality of financial operations.

## Installation

1.  **Prerequisites**: Python 3.9+
2.  **Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage (Reproducibility Pipeline)

Run the following scripts in sequence to generate the data and figures cited in the paper.

### 1. Generate "Ground Truth" Data
Simulates 100,000 cases of a Loan Application process using an Agent-Based Simulation (ABS).
```bash
python src/generator.py
# Output: data/raw/financial_log_100k.csv

```

### 2. Statistical Validation & Sensitivity Analysis

Performs the hierarchical model fitting (Normal vs. Log-Normal vs. GMM) and runs the perturbation analysis for Structural Bias.

```bash
python src/analyzer.py
# Output: data/output/validation_metrics.csv
# Output: data/output/sensitivity_analysis.csv

```

### 3. Generate Manuscript Figures

Generates the 5 high-resolution figures used in the paper. The outputs are named to correspond directly to the manuscript figure numbers.

```bash
python src/visualizer.py

```

| Manuscript Ref | Description | Output Filename |
| --- | --- | --- |
| **Figure 1** | Empirical Bimodality Evidence (Macro/Micro) | `fig1a_bimodal_macro.pdf`, `fig1b_bimodal_micro.pdf` |
| **Figure 2** | Discovered Process Flow (Process Map) | `fig2_process_map.pdf` |
| **Figure 3** | Fidelity Comparison (Goodness-of-fit) | `fig3a_full_distribution.pdf`, `fig3b_tail_detail.pdf` |
| **Figure 4** | Sensitivity Analysis (Structural Bias) | `fig4_sensitivity_analysis.pdf` |
| **Figure 5** | Resource Contention (Queue Dynamics) | `fig5_queue_dynamics.pdf` |

## File Manifest

### Source Code (`src/`)

* `generator.py`: `SimPy` engine simulating resource contention and bimodal task durations.
* `analyzer.py`: Implements the **EM Algorithm** for LGMM calibration and calculating Wasserstein/KS metrics.
* `visualizer.py`: Plotting utility using `seaborn` and `networkx` to render vector graphics.

### Data (`data/`)

* `raw/financial_log_100k.csv`: The synthetic event log (Ground Truth) containing 100,000 traces.
* `output/`: Contains the computed statistical tables and sensitivity logs.

## Citation

[Anonymized for Peer Review]
