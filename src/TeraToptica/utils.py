# src/terapy/utils.py

from __future__ import annotations
from typing import Literal, Optional, Sequence, Tuple

import numpy as np
from astropy.convolution import Box1DKernel, Gaussian1DKernel, convolve
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt

WindowType = Literal["median", "boxcar", "bartlett", "blackman", "gaussian", "hanning", "hamming"]


def build_mask_from_bounds(
    freq: np.ndarray,
    mask_bounds: Optional[Sequence[Tuple[float, float]]],
) -> np.ndarray:
    mask = np.zeros_like(freq, dtype=bool)
    if mask_bounds:
        for start, stop in mask_bounds:
            mask |= (freq >= start) & (freq <= stop)
    return mask


def compute_boxnum_from_window_size(freq: np.ndarray, window_size_ghz: float) -> int:
    """
    Convert a window size in GHz into an odd integer box length based on freq sampling.
    """
    # use median spacing to be robust against tiny irregularities
    df = np.median(np.diff(freq))
    if not np.isfinite(df) or df <= 0:
        raise ValueError("Frequency array must be strictly increasing with finite spacing.")
    boxnum = int(window_size_ghz / df)
    if boxnum < 1:
        boxnum = 1
    # enforce odd for symmetric smoothing
    if boxnum % 2 == 0:
        boxnum -= 1
        if boxnum < 1:
            boxnum = 1
    return boxnum


def convolve_1d(arr: np.ndarray, boxnum: int, *, window: WindowType = "boxcar") -> np.ndarray:
    if window not in ["median", "boxcar", "bartlett", "blackman", "gaussian", "hanning", "hamming"]:
        raise ValueError("window must be one of: median, boxcar, gaussian, hanning, hamming, bartlett, blackman")

    if window == "median":
        # Median filter is not a convolution
        return median_filter(arr, size=boxnum, mode="nearest")

    else: 
        kernel_map = {
            "boxcar": Box1DKernel(boxnum),
            "gaussian": Gaussian1DKernel(boxnum),
            # astropy.convolve can accept an ndarray kernel too
            "bartlett": np.bartlett(boxnum),
            "blackman": np.blackman(boxnum),
            "hanning": np.hanning(boxnum),
            "hamming": np.hamming(boxnum),
        }
        return convolve(arr, kernel_map[window])


def rolling_std_error(arr: np.ndarray, boxnum: int) -> np.ndarray:
    """
    Rolling std / sqrt(N) matching your original smoothing_err intent.
    Pads edges with edge values (same as your prior behavior).
    """
    n = len(arr)
    if n < boxnum or boxnum <= 1:
        print(f"Warning: Inputted window_size returned invalid kernel size (boxnum = {boxnum} larger than array size = {n} or = 1; returning NaNs for error.")
        return np.full_like(arr, np.nan)

    windows = np.lib.stride_tricks.sliding_window_view(arr, boxnum)
    stds = np.std(windows, axis=1) / np.sqrt(boxnum)
    pad = boxnum // 2
    return np.pad(stds, (pad, pad), mode="edge")


# ----------------------------
# "Getter" replacements
# ----------------------------

def get_spectra(self) -> Tuple[np.ndarray, np.ndarray]:
    """Return (freq_GHz, unsmoothed fractional power)."""
    return self.base_freq, self.normed_unsmoothed

def get_smoothed_spectra(self):
    """Return (freq_GHz, smoothed, unsmoothed, mask, err, boxnum)."""
    return (
        self.base_freq,
        self.normed,
        self.normed_unsmoothed,
        self.mask,
        self.err,
        self.boxnum,
    )

def get_open_current(self):
    return self.base_freq, self.base_amp

def get_sample_current(self):
    return self.samp_freq, self.samp_amp

def get_norm_factor(self):
    return self.C

def get_mask(self):
    return self.mask


def plot_spectra(freq, frac_power, mode=None, label=None, mask_bounds=None, xlims=(100, 1000), ylims=(0, 1.6)):
    plt.figure(figsize=(16, 9))
    plt.plot(freq, frac_power, color='#1845FB', label=label)

    if mask_bounds is not None:
        for i, (start, stop) in enumerate(mask_bounds):
                plt.fill_between((start, stop),  2,  edgecolor='black', lw=1, facecolor='none', hatch="//",  label='Masked Water Absorption Lines' if i == 0 else None)

    plt.legend(loc='upper right')
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.xlabel('Freq [GHz]')
    plt.ylabel('Fractional Transmission' if mode == 'transmission' else ('Fractional Reflection' if mode == 'reflection' else 'Fractional Power'))
    plt.show()