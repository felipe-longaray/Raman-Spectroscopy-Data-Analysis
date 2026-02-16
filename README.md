# Raman Spectroscopy Analysis Pipeline

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains a specialized Python pipeline for processing raw Raman spectroscopy data. It is designed to transform noisy experimental spectra into clean, normalized, and publication-ready visualizations.

Unlike basic plotting scripts, this tool implements advanced signal processing algorithms to handle common spectral artifacts automatically.

## Key Features
* **Asymmetric Least Squares (ALS) Baseline Correction:** Implements the algorithm by *Eilers & Boelens (2005)* to mathematically strip fluorescence backgrounds without manual point selection.
* **Cosmic Ray Removal:** Automatically detects and zaps single-point spikes (high-energy particle impacts on the CCD) using a robust statistical rolling median filter.
* **Structural Metrics:** Automatically calculates the $I_D/I_G$ ratio (Defect/Graphite band ratio), a critical metric for characterizing carbonaceous materials and graphene derivatives.
* **Stacked Visualization:** Generates vertically offset "waterfall" plots for easy comparison of multiple samples, compliant with academic journal standards.

## Dependencies
* `numpy`, `scipy` (Sparse matrix operations for ALS)
* `pandas` (Data handling)
* `matplotlib` (Visualization)

## Usage
1. Place raw `.txt` spectral files (xy format) in the `/data` directory.
2. Run the script:
   ```bash
   python raman_processing.py
3. The script will output a stacked TIFF figure with calculated metrics in the legend.

**Mathematical Context**
The baseline correction minimizes the following cost function:$$ S = \sum_i w_i (y_i - z_i)^2 + \lambda \sum_i (\Delta^2 z_i)^2 $$Where $y$ is the signal, $z$ is the baseline, $\lambda$ controls smoothness, and $w_i$ assigns asymmetric weights to ignore positive peaks while fitting the underlying background.

**Author**

