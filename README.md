# monte-carlo-neutron-ml
Monte Carlo neutron transport simulator + Gradient Boosting surrogate model | R² 0.9884 | 36,000× faster than simulation | Python · Plotly · scikit-learn
What Is This?
This project combines nuclear physics simulation with machine learning.
Each neutron is tracked through a material using real physics equations until it
scatters, gets absorbed, or causes fission. After 10,000 simulation runs, a
Gradient Boosting model learns to predict the fission rate from material properties
alone — 36,000× faster than the simulation.

Results
ModelR²RMSEMAELinear Regression0.90520.04140.0322Random Forest0.98810.01470.0115Gradient Boosting0.98840.01450.0113 ← Best

Speedup: ~36,000× faster than Monte Carlo simulation
Cross-validation std: 0.0002 (extremely stable)
Overfitting gap: 0.0021 (model genuinely learned, not memorised)


Core Physics
Distance to next neutron collision (the Monte Carlo formula):
s = -ln(ξ) / Σₜ
where ξ is a uniform random number and Σₜ = Σₛ + Σₐ + Σ_f is the total
cross-section. Outcome probabilities at each collision:
P(scatter) = Σₛ / Σₜ
P(absorb)  = Σₐ / Σₜ
P(fission) = Σ_f / Σₜ

Project Structure
monte-carlo-neutron-ml/
├── config.py               ← All physics parameters (edit here only)
├── simulation.py           ← Monte Carlo neutron simulator + Plotly visuals
├── dataset_generator.py    ← Loops configs, runs simulation, saves CSV
├── clean_dataset.py        ← Validates and cleans the dataset
├── train_model.py          ← ML pipeline: LR → RF → Gradient Boosting
├── generate_report.py      ← Builds PDF report (reportlab)
├── report.tex              ← Full LaTeX report with equations
├── .gitignore
└── README.md

How to Run
1. Install dependencies (Arch Linux)
bashsudo pacman -S python-numpy python-pandas python-scikit-learn python-plotly
pip install reportlab --break-system-packages
2. Run the simulator (physics test + 3D plots)
bashpython simulation.py
3. Generate the dataset (10,000 rows, ~3 min)
bashpython dataset_generator.py
4. Clean the dataset
bashpython clean_dataset.py
5. Train and evaluate all models
bashpython train_model.py
6. Build the PDF report
bashpython generate_report.py   # reportlab version
pdflatex report.tex         # LaTeX version (run twice for TOC)

Visualisations
All plots are interactive HTML files (Plotly) saved to results/:
FileContentsimulation_diagnostics.htmlCollision histogram, outcome pie, fission rate sweepfission_surface_3d.htmlRotatable 3D fission rate surfaceml_dashboard.htmlPredicted vs actual, feature importance, residuals, R² comparisonpredictions_3d.html3D actual vs predicted vs residual (Gradient Boosting)

Key Insight
Monte Carlo simulation is accurate but slow.
ML acts as a surrogate model — it learns the physics patterns and
predicts results instantly. This is exactly how modern nuclear engineering
research accelerates reactor design and material screening.

Author
JINX_KAL — github.com
