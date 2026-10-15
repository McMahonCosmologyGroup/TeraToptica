"""Unit tests for the array helpers in TeraToptica.utils."""
import numpy as np
import pytest

from TeraToptica.utils import (
    build_mask_from_bounds,
    compute_boxnum_from_window_size,
    convolve_1d,
    rolling_std_error,
)

WINDOWS = ["boxcar", "bartlett", "blackman", "gaussian", "hanning", "hamming", "median"]


def test_mask_from_bounds_is_inclusive_union():
    f = np.arange(0.0, 100.0, 1.0)
    m = build_mask_from_bounds(f, [(10, 20), (50, 55)])
    assert m.dtype == bool and m.shape == f.shape
    assert m[10] and m[20] and m[50] and m[55]          # bounds are inclusive
    assert not m[9] and not m[21] and not m[56]
    assert m.sum() == 11 + 6


@pytest.mark.parametrize("bounds", [None, ()])
def test_mask_from_bounds_empty(bounds):
    assert not build_mask_from_bounds(np.arange(10.0), bounds).any()


@pytest.mark.parametrize("window_ghz, expected", [(15.0, 3), (25.0, 5), (20.0, 3), (1.0, 1)])
def test_boxnum_is_odd_and_scales_with_grid(window_ghz, expected):
    f = np.arange(0.0, 1000.0, 5.0)                     # TeraFlash's 5 GHz grid
    assert compute_boxnum_from_window_size(f, window_ghz) == expected


def test_boxnum_rejects_bad_grid():
    with pytest.raises(ValueError):
        compute_boxnum_from_window_size(np.array([3.0, 2.0, 1.0]), 10.0)


@pytest.mark.parametrize("window", WINDOWS)
def test_convolve_preserves_a_constant(window):
    x = np.full(200, 0.7)
    y = convolve_1d(x, 9, window=window)
    assert y.shape == x.shape
    assert np.allclose(y[20:-20], 0.7)                  # interior untouched; edges depend on the kernel's padding


def test_convolve_rejects_unknown_window():
    with pytest.raises(ValueError):
        convolve_1d(np.ones(10), 3, window="triangle")


def test_rolling_std_error_matches_definition():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(500)
    err = rolling_std_error(x, 11)
    assert err.shape == x.shape and np.all(np.isfinite(err))
    assert np.isclose(err[100], np.std(x[95:106]) / np.sqrt(11))
    assert np.allclose(rolling_std_error(np.ones(50), 5), 0.0)


def test_rolling_std_error_invalid_window_gives_nan():
    assert np.all(np.isnan(rolling_std_error(np.ones(5), 7)))
