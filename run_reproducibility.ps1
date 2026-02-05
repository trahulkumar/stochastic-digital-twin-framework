Write-Host "Starting Reproducibility Suite..."

Write-Host "Step 1: Generating Ground Truth Data..."
uv run src/generator.py

Write-Host "Step 2: Performing Statistical Validation..."
uv run src/analyzer.py

Write-Host "Step 3: Generating Figures..."
uv run src/visualizer.py

Write-Host "Reproducibility Run Complete."
