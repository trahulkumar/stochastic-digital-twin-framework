# Process-Mining Driven Construction of Stochastic Digital Twins

**Submission to INFORMS Journal on Data Science**

## Abstract
This repository contains the "Reproducibility Suite" for the paper *Process-Mining Driven Construction of Stochastic Digital Twins*. It provides the complete source code to:
1.  **Generate Ground Truth:** Run an Agent-Based Simulation (ABS) of a complex financial workflow with bimodal "Standard vs. Exception" behavior.
2.  **Validate Statistically:** Perform a Hierarchical Model Fit comparing Normal, Log-Normal, and **2-Component Mixture Models** against the ground truth.
3.  **Visualize Evidence:** Generate the 5 high-resolution figures cited in the manuscript, proving statistical indistinguishability ($p > 0.05$).

## Key Results (Reproduced via `src/analyzer.py`)
This suite validates the "Hierarchy of Fidelity" presented in **Section 6.2** of the manuscript:

| Model Strategy | KS Statistic (Error) | p-value | Verdict |
| :--- | :--- | :--- | :--- |
| **Model A (Baseline Normal)** | 0.102 | < 0.001 | **Reject** |
| **Model B (Log-Normal)** | 0.011 | < 0.001 | **Reject** (Structural Bias) |
| **Model C (Mixture Model)** | **0.004** | **> 0.100** | **Pass (Indistinguishable)** |

## Installation

This project uses modern Python packaging.

1.  **Prerequisites**: Python 3.8+
2.  **Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *or using uv (recommended)*:
    ```bash
    uv pip install -r requirements.txt
    ```

## Usage (Reproduction Steps)

To reproduce the full analysis pipeline, run the following scripts in order:

### 1. Generate "Ground Truth" Data
Simulates 100,000 cases of a financial Loan Application process. The simulation engine (`src/generator.py`) implements resource contention (queues), non-stationary arrivals, and bimodal task durations.
```bash
python src/generator.py
# Output: data/raw/financial_log_100k.csv

```

### 2. Statistical Validation

Performs the hierarchical fitting and calculates Kolmogorov-Smirnov (KS) statistics and Earth Mover's Distance (EMD) for all candidate models.

```bash
python src/analyzer.py
# Output: data/output/validation_metrics.csv, data/output/sensitivity_analysis.csv

```

### 3. Generate Figures

Generates the 5 publication-ready figures used in the manuscript.

```bash
python src/visualizer.py

```

**Generated Outputs in `figures/`:**

* `fig1_fidelity_comparison.png`: KDE overlay showing Model C's superior fit.
* `fig2_sensitivity_analysis.png`: Robustness check for MLE calibration.
* `fig3_bimodal_evidence.png`: Empirical evidence of "Exception" handling modes.
* `fig4_queue_dynamics.png`: Time-series analysis of resource bottlenecks.
* `fig5_process_map.png`: The discovered Directly-Follows Graph (DFG).

## File Manifest

### Source Code (`src/`)

* `generator.py`: SimPy-based Agent-Based Simulation engine. Logs `Arrival_Time` and `Start_Time` to enable queue analysis.
* `analyzer.py`: Validation script. Implements the 2-Component Log-Gaussian Mixture Model (LGMM) using Expectation-Maximization.
* `visualizer.py`: Plotting utility using `seaborn` and `networkx` to generate all paper figures.

### Data (`data/`)

* `raw/financial_log_100k.csv`: The synthetic event log (Ground Truth).
* `output/validation_metrics.csv`: The exact statistical values cited in Table 3 of the paper.
* `output/sensitivity_analysis.csv`: The perturbation data for Figure 4 of the manuscript (saved as fig2 here).

## Citation

[Placeholder: Insert final paper citation here]
