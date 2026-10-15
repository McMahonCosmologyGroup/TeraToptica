"""
Regression test for the orientation of the phase difference in get_phase_diff
(src/TeraToptica/utils.py). Plain-assert script (no pytest dependency) --
run with: python tests/test_phase_orientation.py

Bug: get_phase_diff(absolute=True) took abs(theta - theta_base). The exported phases carry
arbitrary 2*pi offsets, so the difference can be negative over part of the band (a small
phase plus a -2*pi offset crosses zero inside it). abs() folded that part, and thin,
low-index samples (HD30 foams) came out with n < 1. The fix orients the difference by the
measured pulse delay, so abs() is no longer needed and absolute=True is a deprecated no-op on
the pulse path.
"""
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from TeraToptica import TeraFlashAnalyzer
from TeraToptica import utils as tu

C = 299792458.0


def pulse(t, t0, sigma=0.25):
    """Single-cycle-like THz pulse (derivative of a Gaussian) centred at t0 (ps)."""
    return -(t - t0) / sigma**2 * np.exp(-0.5 * ((t - t0) / sigma) ** 2)


def fake_measurement(delay_ps, samp_phase_offset=0.0, flip=False):
    """Sample and reference pulses on a 200 ps record, spectra from the FFT path, and an extra
    constant offset on the sample phase (as the exported files carry). flip=True mimics data
    with the opposite sign convention."""
    t = np.arange(0.0, 200.0 + 1e-9, 0.05)
    t_ref = 27.0
    s_ref, s_smp = pulse(t, t_ref), pulse(t, t_ref + delay_ps)
    kw = dict(dfreq=None, rel_start=0.01, rel_end=0.01, norm="ortho")
    f, a_ref, ph_ref = TeraFlashAnalyzer._spectr_from_pulse(t, s_ref, **kw)
    _, a_smp, ph_smp = TeraFlashAnalyzer._spectr_from_pulse(t, s_smp, **kw)
    ph_smp = ph_smp + samp_phase_offset
    if flip:
        ph_ref, ph_smp = -ph_ref, -ph_smp
    mask = (f < 150) | (f > 2000)
    return SimpleNamespace(base_freq=f, samp_phase=ph_smp, base_phase=ph_ref, samp_amp=a_smp, base_amp=a_ref,
                           samp_time=t, samp_signal=s_smp, base_time=t, base_signal=s_ref, mask=mask, samp="synthetic")


def n_from(obj, w_m, **kw):
    pd_, n_ulim, info = tu.get_phase_diff(obj, return_info=True, **kw)
    f = obj.base_freq
    sel = (f >= 150) & (f <= 1000)
    n = 1 + pd_[sel] * C / (2 * np.pi * f[sel] * 1e9 * w_m)
    return n, info


def main():
    w = 3.0e-3                      # 3 mm slab
    n_true = 1.10                   # delay (n - 1) w / c = 1.0 ps
    delay = (n_true - 1) * w / C * 1e12

    # 1) a -2*pi offset on the sample phase makes the raw difference cross zero at 1 THz, inside the band
    obj = fake_measurement(delay, samp_phase_offset=-2 * np.pi)
    raw = obj.samp_phase - obj.base_phase
    f = obj.base_freq
    band = (f >= 150) & (f <= 1000)
    assert raw[band].min() < 0 < raw[(f >= 1000) & (f <= 2000)].max(), "test setup: raw difference should cross zero near 1 THz"
    n, info = n_from(obj, w)
    assert np.allclose(n, n_true, atol=2e-3), f"zero-crossing difference: n = {n.min():.4f}-{n.max():.4f}, expected {n_true}"
    assert info["orientation"] == 1

    # what the old workaround did with the same data: fold below 1 THz, a wrong, frequency-dependent n
    folded = np.abs(raw)
    k = np.round((2 * np.pi * f[band] * 1e9 * delay * 1e-12 - folded[band]).mean() / (2 * np.pi))
    n_old = 1 + (folded[band] + 2 * np.pi * k) * C / (2 * np.pi * f[band] * 1e9 * w)
    assert np.max(np.abs(n_old - n_true)) > 0.05, "test setup: abs() should have corrupted this case"

    # 2) opposite sign convention: flipped and reported, same n
    n, info = n_from(fake_measurement(delay, samp_phase_offset=4 * np.pi, flip=True), w)
    assert info["orientation"] == -1, "opposite-convention phases should be flipped"
    assert np.allclose(n, n_true, atol=2e-3), f"flipped convention: n = {n.min():.4f}-{n.max():.4f}, expected {n_true}"

    # 3) thin, low-index sample (HD30-like: n - 1 = 0.0134 over 3.1 mm, delay ~0.14 ps) stays above 1
    w2, n2 = 3.1e-3, 1.0134
    n, info = n_from(fake_measurement((n2 - 1) * w2 / C * 1e12, samp_phase_offset=-2 * np.pi), w2)
    assert np.allclose(n, n2, atol=1e-3) and info["orientation"] == 1, f"thin foam: n = {n.min():.4f}-{n.max():.4f}"

    # 4) absolute=True is a deprecated no-op on the pulse path
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        n_abs, _ = n_from(fake_measurement(delay, samp_phase_offset=-2 * np.pi), w, absolute=True)
    assert any(issubclass(c.category, FutureWarning) for c in caught), "absolute=True should warn"
    assert np.allclose(n_abs, n_true, atol=2e-3), "absolute=True should not change the result on the pulse path"

    print(f"OK: zero-crossing, flipped-convention, thin-foam and absolute=True cases all give n = {n_true} / {n2} "
          f"(old abs() result was off by up to {np.max(np.abs(n_old - n_true)):.2f})")


if __name__ == "__main__":
    main()
