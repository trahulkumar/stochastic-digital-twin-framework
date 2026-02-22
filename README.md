# Beyond the Average: Overcoming Fidelity Barriers in Stochastic Digital Twins for Financial Services

**Reproducibility Suite**

## Abstract
This repository contains the **Reproducibility Suite** for the manuscript *"Beyond the Average: Overcoming Fidelity Barriers in Stochastic Digital Twins for Financial Services"*.

Standard Digital Twin deployments often suffer from a "Fidelity Barrier"—the inability to replicate the bimodal stochastic volatility of human-centric workflows. This research introduces a closed-loop architecture that integrates **Inductive Process Mining** with **Gaussian Mixture Models (GMM)** and a bidirectional control layer: **Regime-Aware Dynamic Routing (RADR)**. We demonstrate that by explicitly modeling "Regime Switching" (Routine vs. Exception), we can achieve a **97.6% reduction in error** and a **92% reduction in peak queue backlogs** through automated intervention.



## Key Features
1. **Stochastic Calibration:** EM-algorithm implementation for GMM-based service time estimation.
2. **Structural Discovery:** Inductive Mining pipeline to isolate "Happy Path" vs. "Exception" workflows.
3. **Closed-Loop Control:** Simulation of the RADR protocol for real-time resource reallocation.
4. **Risk Quantification:** Analysis of the "Average Fallacy" and its $9M annual impact on Operational Value at Risk (OpVaR).

## Reproducibility Workflow

### 1. Data Generation (Physical System Simulation)
Generates a large-scale event log (100k traces) simulating a Loan Origination System (LOS) with embedded bimodal bottlenecks.
```bash
python src/generator.py
```

### 2. Statistical Analysis & GMM Calibration (Digital Twin Brain)

Performs the hierarchical model fit (Normal vs. Log-Normal vs. GMM) and quantifies the **23.1% Structural Bias**.

```bash
python src/analyzer.py
```

### 3. Closed-Loop Simulation (RADR Intervention)

*New Analysis:* Simulates the bidirectional feedback loop where the Twin reroutes cases to prevent predicted congestion.

```bash
python src/control_sim.py
```

### 4. Visualization

Generates high-resolution figures for the manuscript, including the new **Figure 6 (Queue Reduction)**.

```bash
python src/visualizer.py
```

## Manuscript Figures & Artifacts

| Manuscript Ref | Description | Output Filename |
| --- | --- | --- |
| **Figure 1** | Empirical Bimodality Evidence | `fig1_bimodal.pdf` |
| **Figure 2** | Discovered Process Flow | `fig2_process_map.pdf` |
| **Figure 3** | Fidelity Comparison (GMM vs. Baseline) | `fig3_fidelity.pdf` |
| **Figure 4** | Sensitivity Analysis (Structural Bias) | `fig4_sensitivity.pdf` |
| **Figure 5** | OpVaR & Financial Impact | `fig5_opvar.pdf` |
| **Figure 6** | **RADR Control Loop (Queue Reduction)** | `fig6_radr_dynamics.pdf` |

## Data & Software Availability

This repository is archived on Zenodo (DOI: [Insert DOI]) and follows the Open Research standards required by the journal. The synthetic data generator is provided to ensure full reproducibility without exposing sensitive institutional PII.
