
# Agroforestry Risk & Cost-Benefit Analysis

This repository provides tools and workflows for building, evaluating, and analyzing agroforestry systems—focusing on shade trees, risk assessment, and cost–benefit trade-offs—using the climate-risk modeling framework **CLIMADA**.

---

## Repository Structure

```

.
├── Build\_typical\_agroforest\_plot.ipynb  # Generate typical agroforestry plots
├── agroforestry\_systems/               # Excel outputs of typical plots
├── agroforest\_risk/                    # Risk analysis workflows
│   ├── get\_point\_locations.ipynb       # Gather species occurrence points
│   ├── suitability\_typical\_agroforest\_plots.ipynb  # Run suitability models
│   ├── risk\_agroforest\_system.ipynb    # Compute risk (suitability loss + extreme weather)
│   ├── species\_thresholds.json         # Thresholds for suitability analysis
│   ├── utils\_agroforestry.py           # Agroforestry helper functions
│   ├── utils\_hazards.py                # Hazard helper functions
│   └── utils\_suitability\_modelling.py  # Suitability modelling functions
├── cost\_benefit/                       # Cost–benefit evaluation workflows
│   ├── make\_canopy\_alternatives.ipynb  # Generate canopy composition scenarios
│   ├── utils\_cb.py                     # Helpers for cost–benefit routines
│   └── CostBenefit\_Canopy.ipynb        # Main cost–benefit analysis notebook
├── climate\_data/                       # Climate datasets and preprocessing
├── experiments/                        # Exploratory / test notebooks
├── config.py                           # Central configuration (paths for data, outputs)
├── README.md                           # This documentation file

````

---

## Workflow Overview

1. **Build Typical Plots**  
   Run `Build_typical_agroforest_plot.ipynb` to generate representative agroforestry plots.  
   Results are exported as Excel files into `agroforestry_systems/`.

2. **Risk Analysis** (`agroforest_risk/`)  
   - `get_point_locations.ipynb`: collect occurrence points for coffee, cacao, and associated species.  
   - `suitability_typical_agroforest_plots.ipynb`: apply suitability models to the typical plots.  
   - `risk_agroforest_system.ipynb`: compute risk by combining impacts from suitability loss and extreme weather.  
   - Supporting files: `species_thresholds.json`, plus utility modules (`utils_agroforestry.py`, `utils_hazards.py`, `utils_suitability_modelling.py`).

3. **Cost–Benefit Analysis** (`cost_benefit/`)  
   - `CostBenefit_Canopy.ipynb`: perform cost–benefit evaluation.  
   - Utility: `utils_cb.py`.

4. **Experiments**  
   Use the `experiments/` directory for exploratory and ad-hoc analyses.

---

## Dependencies

This project builds on **CLIMADA** (Climate Adaptation), a powerful open-source Python framework designed for probabilistic climate risk assessments and adaptation analysis.

- As of March 2025, the latest stable version is CLIMADA **v6.0.1**.
- A full installation guide is available here:  
  **[CLIMADA Installation Guide — Read the Docs](https://climada-python.readthedocs.io/en/stable/guide/install.html)** :contentReference[oaicite:1]{index=1}

---

## License & Citation

* **CLIMADA** is licensed under **GPL-3.0**. Please ensure compliance when distributing or adapting code.
* If using this repository in publications or presentations, please cite both CLIMADA and your project appropriately.

