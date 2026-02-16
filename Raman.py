"""
Raman Spectroscopy Analysis Pipeline
====================================
Author: [Your Name]
Description: Automated processing of Raman spectra including cosmic ray removal,
             Asymmetric Least Squares (ALS) baseline correction, and D/G ratio calculation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from pathlib import Path
from typing import Tuple, Optional, List

# ==========================================
# CONFIGURATION
# ==========================================
# Use raw strings (r"...") for paths
DATA_DIR = Path(__file__).parent / "data"
OUTPUT_FILENAME = "Raman_Stacked_Spectra.tiff"

# Spectral Region of Interest (ROI)
X_MIN, X_MAX = 600, 3000 

# Baseline Correction Parameters (ALS)
LAMBDA_ALS = 100000  # Smoothness (Higher = Smoother baseline)
P_ALS = 0.001        # Asymmetry (0.001-0.01 is standard for Raman)

# Plotting
STACK_OFFSET = 0.5   # Vertical separation between spectra
DPI = 1200
FIG_SIZE = (7, 6)

class RamanAnalyzer:
    """
    Encapsulates spectral processing algorithms: spike removal, 
    baseline correction, and normalization.
    """
    
    @staticmethod
    def baseline_als(y: np.ndarray, lam: float, p: float, niter: int = 10) -> np.ndarray:
        """
        Asymmetric Least Squares Smoothing for baseline correction.
        Source: Eilers, P.H.C. and Boelens, H.F.M. (2005).
        """
        L = len(y)
        D = sparse.diags([1,-2,1], [0,-1,-2], shape=(L,L-2))
        w = np.ones(L)
        
        for i in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w*y)
            w = p * (y > z) + (1-p) * (y < z)
        return z

    @staticmethod
    def remove_cosmic_rays(y: np.ndarray, window: int = 5, threshold: int = 7) -> np.ndarray:
        """
        Removes sharp spikes (cosmic rays) using a rolling median filter.
        """
        y_clean = y.copy()
        series = pd.Series(y)
        
        # Calculate rolling median
        y_median = series.rolling(window=window, center=True).median().fillna(method='bfill').fillna(method='ffill').values
        
        residual = np.abs(y - y_median)
        mad = np.median(residual) # Median Absolute Deviation
        sigma = 1.4826 * mad
        
        if sigma > 0:
            spikes = residual > (threshold * sigma)
            y_clean[spikes] = y_median[spikes]
            if np.sum(spikes) > 0:
                print(f"   [INFO] Removed {np.sum(spikes)} cosmic ray spikes.")
                
        return y_clean

    @staticmethod
    def calculate_dg_ratio(x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculates the intensity ratio between D-band (~1350 cm-1) and G-band (~1580 cm-1).
        This is a standard metric for carbonaceous materials/graphene.
        """
        # Define integration windows
        d_mask = (x > 1330) & (x < 1370)
        g_mask = (x > 1560) & (x < 1600)
        
        if np.any(d_mask) and np.any(g_mask):
            id_max = np.max(y[d_mask])
            ig_max = np.max(y[g_mask])
            if ig_max > 0:
                return id_max / ig_max
        return 0.0

    def process_file(self, filepath: Path) -> Optional[Tuple[np.ndarray, np.ndarray, float, str]]:
        """
        Reads and processes a single Raman spectral file.
        Returns: x, y_processed, dg_ratio, label
        """
        try:
            # Flexible reading: tries tab, then whitespace
            df = pd.read_csv(filepath, sep=None, engine='python', header=None)
            
            # Ensure numeric
            df = df.apply(pd.to_numeric, errors='coerce').dropna()
            
            if df.shape[1] < 2:
                print(f"   [WARN] Skipping {filepath.name}: Invalid format.")
                return None
                
            x = df.iloc[:, 0].values
            y = df.iloc[:, 1].values
            
            # Sort by wavenumber
            sort_idx = np.argsort(x)
            x, y = x[sort_idx], y[sort_idx]
            
            # 1. Despiking
            y = self.remove_cosmic_rays(y)
            
            # 2. Baseline Correction
            baseline = self.baseline_als(y, LAMBDA_ALS, P_ALS)
            y_corr = y - baseline
            
            # 3. Min-Max Normalization
            y_norm = (y_corr - y_corr.min()) / (y_corr.max() - y_corr.min())
            
            # 4. Metrics
            dg_ratio = self.calculate_dg_ratio(x, y_norm)
            
            return x, y_norm, dg_ratio, filepath.stem
            
        except Exception as e:
            print(f"   [ERROR] Failed to process {filepath.name}: {e}")
            return None

def plot_spectra(processed_data: List[Tuple], output_path: Path):
    """
    Generates a stacked publication-quality plot.
    """
    plt.figure(figsize=FIG_SIZE)
    
    # Sort data if needed (e.g., by filename or ratio)
    # processed_data.sort(key=lambda x: x[3]) 

    for i, (x, y, ratio, name) in enumerate(processed_data):
        # Stack spectra vertically
        y_stacked = y + (i * STACK_OFFSET)
        
        label_text = f"{name} ($I_D/I_G$: {ratio:.2f})"
        
        plt.plot(x, y_stacked, label=label_text, linewidth=1.2)

    plt.xlabel(r"Raman Shift (cm$^{-1}$)")
    plt.ylabel("Normalized Intensity (a.u.)")
    plt.xlim(X_MIN, X_MAX)
    
    # Remove Y-ticks for stacked plots (standard convention)
    plt.yticks([])
    
    # Legend settings
    plt.legend(loc='upper right', frameon=False, fontsize=9)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=DPI, pil_kwargs={"compression": "tiff_lzw"})
    print(f"\n[SUCCESS] Figure saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    print("--- Raman Spectroscopy Pipeline ---")
    
    if not DATA_DIR.exists():
        print(f"[ERROR] Data directory not found: {DATA_DIR}")
        exit()
        
    analyzer = RamanAnalyzer()
    processed_spectra = []
    
    files = sorted(list(DATA_DIR.glob("*.txt")))
    print(f"Found {len(files)} files.")
    
    for f in files:
        print(f"Processing: {f.name}...")
        result = analyzer.process_file(f)
        if result:
            processed_spectra.append(result)
            
    if processed_spectra:
        plot_spectra(processed_spectra, DATA_DIR / OUTPUT_FILENAME)
    else:
        print("[WARN] No valid spectra found to plot.")
