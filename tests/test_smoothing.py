"""
Regression test for the window_size_ghz unit mismatch fix in convolve_1d
(src/TeraToptica/utils.py). Plain-assert script (no pytest dependency) --
run with: python tests/test_smoothing.py

Bug: Gaussian1DKernel's constructor parameter is a standard deviation, not
a full width (unlike every other window type here), so passing the same
GHz-derived boxnum to both Box1DKernel(width) and Gaussian1DKernel(stddev)
produced a Gaussian kernel with ~8x wider support than a boxcar of the
"same" nominal width. The fix converts boxnum to the matching stddev via
the standard FWHM<->sigma relation before building the Gaussian kernel, so
window_size_ghz means the same practical width (FWHM) regardless of window
choice.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from TeraToptica.utils import convolve_1d, compute_boxnum_from_window_size


def half_max_width(x, y):
    """Width (in x-units) of the region where y is above half its peak-to-trough range."""
    y_hi, y_lo = y.max(), y.min()
    half = (y_hi + y_lo) / 2.0
    above = x[y >= half]
    if len(above) == 0:
        return 0.0
    return above.max() - above.min()


def main():
    # Synthetic step: a sharp transition from 1 -> 0, like an idealized filter cutoff.
    freq = np.arange(0.0, 500.0, 1.0)  # 1 GHz spacing, matching this repo's TeraFlash convention
    step = np.where(freq < 250.0, 1.0, 0.0).astype(float)

    window_size_ghz = 25.0
    boxnum = compute_boxnum_from_window_size(freq, window_size_ghz)

    box_smoothed = convolve_1d(step, boxnum, window="boxcar")
    gauss_smoothed = convolve_1d(step, boxnum, window="gaussian")

    # A boxcar of full width W has a well-defined "response width" of ~W when
    # convolved with a step (the transition zone it smears the step across).
    # After the fix, a same-window_size_ghz Gaussian's FWHM should be
    # comparable to (not wildly larger than) the boxcar's width -- prior to
    # the fix this was off by a factor of ~8 (Gaussian1DKernel's default
    # 8*stddev+1 truncation applied to a stddev that should have been ~2.35x
    # smaller).
    box_transition = np.sum((box_smoothed > 0.05) & (box_smoothed < 0.95))
    gauss_transition = np.sum((gauss_smoothed > 0.05) & (gauss_smoothed < 0.95))

    print(f"window_size_ghz={window_size_ghz}, boxnum={boxnum}")
    print(f"boxcar transition width (samples): {box_transition}")
    print(f"gaussian transition width (samples): {gauss_transition}")

    assert box_transition > 0, "boxcar produced no measurable transition"
    assert gauss_transition > 0, "gaussian produced no measurable transition"

    # Comparable widths: within a factor of 2 of each other, NOT the ~8x
    # blowout the bug produced.
    ratio = gauss_transition / box_transition
    print(f"gaussian/boxcar transition-width ratio: {ratio:.2f}")
    assert 0.5 <= ratio <= 2.0, (
        f"Gaussian transition width ({gauss_transition}) is not comparable to "
        f"boxcar's ({box_transition}); ratio={ratio:.2f} -- window_size_ghz "
        "unit mismatch may have regressed."
    )

    print("PASS: gaussian and boxcar smoothing widths are FWHM-comparable for the same window_size_ghz.")


if __name__ == "__main__":
    main()
