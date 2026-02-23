$ErrorActionPreference = "Stop"

Write-Host "--------------------------------------------------------"
Write-Host "Starting Reproducibility Suite: Manuscript Complement"
Write-Host "Project: Beyond the Average - Stochastic Digital Twins"
Write-Host "--------------------------------------------------------"

Write-Host "Step 0: Syncing dependencies with uv..."
uv sync

Write-Host "Step 1: Simulating Physical System & Digital Twin Control..."
# Generates both logs with case attributes:
#   data/raw/passive_log_100k.csv
#   data/raw/active_radr_log_100k.csv
# Also writes:
#   data/output/attribute_summary.csv
uv run src/generator.py

Write-Host "Step 2: Performing held-out statistical calibration (baselines + BIC-selected GMM)..."
# Writes:
#   data/output/validation_metrics.csv
#   data/output/gmm_model_selection.csv
#   data/output/sensitivity_analysis.csv
#   data/output/results_summary.csv
uv run src/analyzer.py

Write-Host "Step 3: Generating manuscript figures and operational metrics (Passive vs RADR)..."
# Writes:
#   data/output/ops_metrics.csv
# Produces figures including:
#   figures/fig6a_underwriting_backlog_passive_vs_radr.pdf
#   figures/fig6b_underwriting_wip_passive_vs_radr.pdf
uv run src/visualizer.py

Write-Host "--------------------------------------------------------"
Write-Host "Reproducibility Run Complete."
Write-Host "Key outputs:"
Write-Host "  /data/raw: passive_log_100k.csv, active_radr_log_100k.csv"
Write-Host "  /data/output: validation_metrics.csv, gmm_model_selection.csv, sensitivity_analysis.csv, results_summary.csv, ops_metrics.csv, attribute_summary.csv"
Write-Host "  /figures: all figures generated in figures folder"
Write-Host "--------------------------------------------------------"
