Write-Host "--------------------------------------------------------"
Write-Host "Starting Reproducibility Suite: Manuscript Complement"
Write-Host "Project: Beyond the Average - Stochastic Digital Twins"
Write-Host "--------------------------------------------------------"

Write-Host "Step 1: Simulating Physical System & Digital Twin Control..."
# This now generates both 'passive_log_100k.csv' and 'active_radr_log_100k.csv'
uv run src/generator.py

Write-Host "Step 2: Performing Statistical Calibration (GMM vs. Baselines)..."
# Validates the 97.6% error reduction and 23.1% structural bias
uv run src/analyzer.py

Write-Host "Step 3: Generating Manuscript Figures (including new Figure 6)..."
# Produces high-res PDFs for the paper, now including the RADR queue reduction chart
uv run src/visualizer.py

Write-Host "--------------------------------------------------------"
Write-Host "Reproducibility Run Complete."
Write-Host "Outputs available in /data/output and /figures"
Write-Host "--------------------------------------------------------"