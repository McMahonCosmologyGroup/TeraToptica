"""
Regression test for the FFT-mode phase sign (TeraFlashAnalyzer._spectr_from_pulse in
src/TeraToptica/tf.py). Runs under pytest, or standalone --
run with: python tests/test_fft_phase_sign.py

Bug: numpy's rfft follows the e^{-i w t} convention, so the FFT-mode phase of a pulse at time t0
was -w t0, while the TF5 .spectr export (the default mode) gives +w t0. A sample pulse delayed
behind its reference therefore had a negative phase_diff in FFT mode, and n = 1 + dL/w came out
mirrored about 1 (e.g. 0.98 instead of 1.02). The fix negates the FFT phase so both modes agree.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from TeraToptica import TeraFlashAnalyzer


def pulse(t, t0, sigma=0.25):
    """Single-cycle-like THz pulse (derivative of a Gaussian) centred at t0 (ps)."""
    return -(t - t0) / sigma**2 * np.exp(-0.5 * ((t - t0) / sigma) ** 2)


def phase_slope_ps(freq, phase, lo=200.0, hi=500.0):
    """d(phase)/d(omega) over [lo, hi] GHz, in ps: +t0 in the .spectr convention."""
    sel = (freq >= lo) & (freq <= hi)
    return np.polyfit(freq[sel], phase[sel], 1)[0] / (2 * np.pi) * 1e3


def test_fft_phase_sign_matches_spectr_convention():
    t = np.arange(0.0, 200.0 + 1e-9, 0.05)  # ps, like a 200 ps TeraFlash record
    t_ref, delay = 27.0, 0.40                # reference peak position and a sample delay (ps)

    f, _, ph_ref = TeraFlashAnalyzer._spectr_from_pulse(t, pulse(t, t_ref), dfreq=None,
                                                        rel_start=0.01, rel_end=0.01, norm="ortho")
    _, _, ph_smp = TeraFlashAnalyzer._spectr_from_pulse(t, pulse(t, t_ref + delay), dfreq=None,
                                                        rel_start=0.01, rel_end=0.01, norm="ortho")

    slope = phase_slope_ps(f, ph_ref)
    assert abs(slope - t_ref) < 0.05, f"reference phase slope {slope:.3f} ps, expected +{t_ref} ps (.spectr convention)"

    dslope = phase_slope_ps(f, ph_smp - ph_ref)
    assert abs(dslope - delay) < 0.01, f"sample - reference phase slope {dslope:.4f} ps, expected +{delay} ps"
    print(f"OK: reference phase slope {slope:+.3f} ps (expected +{t_ref}), delay {dslope:+.4f} ps (expected +{delay})")


if __name__ == "__main__":
    test_fft_phase_sign_matches_spectr_convention()
