
# Agroforestry Risk & Cost-Benefit Analysis

This repository provides tools and workflows for building, evaluating, and analyzing agroforestry systems—focusing on shade trees, risk assessment, and cost–benefit trade-offs—using the climate-risk modeling framework **CLIMADA**.

---

## Repository Structure

```

.
├── agroforestry_systems/                # Typical agroforestry plot generation
│   ├── Build_typical_agroforest_plot.ipynb  # Generate typical agroforestry plots
│   └── *.xlsx                           # Excel outputs of typical plots
├── agroforest_risk/                     # Risk analysis workflows
│   ├── get_point_locations.ipynb        # Gather species occurrence points
│   ├── suitability_typical_agroforest_plots.ipynb  # Run suitability models
│   ├── risk_agroforest_system.ipynb     # Compute risk (suitability loss + extreme weather)
│   ├── species_thresholds.json          # Thresholds for suitability analysis
│   ├── utils_agroforestry.py            # Agroforestry helper functions
│   ├── utils_hazards.py                 # Hazard helper functions
│   └── utils_suitability_modelling.py   # Suitability modelling functions
├── cost_benefit/                        # Cost–benefit evaluation workflows
│   ├── make_canopy_alternatives.ipynb   # Generate canopy composition scenarios
│   ├── utils_cb.py                      # Helpers for cost–benefit routines
│   └── CostBenefit_Canopy.ipynb         # Main cost–benefit analysis notebook
├── climate_data/                        # Climate datasets and preprocessing
├── experiments/                         # Exploratory / test notebooks
├── config.py                            # Central configuration (paths for data, outputs)
├── README.md                            # This documentation file

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
   - Supporting files: `species_thresholds.json` generated in the suitability modelling, plus utility modules (`utils_agroforestry.py`, `utils_hazards.py`, `utils_suitability_modelling.py`).

3. **Cost–Benefit Analysis** (`cost_benefit/`)  
   - `CostBenefit_Canopy.ipynb`: perform cost–benefit evaluation.  
   - Utility: `utils_cb.py`.

4. **Experiments**  
   If you are brave enough, those are experiments that we tried. We do not guarantee that it works or that they are well documented.

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

