"""Shared fixtures: paths to the bundled example data, and a writer for synthetic TF5 exports."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))   # tests run against the checkout, installed or not

from TeraToptica import TeraFlashAnalyzer  # noqa: E402

C = 299792458.0
EXAMPLE = ROOT / "example" / "data"


def thz_pulse(t, t0, sigma=0.25):
    """Single-cycle-like THz pulse centred at t0 (ps): a derivative of a Gaussian plus a smaller Gaussian.

    The Gaussian term makes the pulse asymmetric, as real TeraFlash pulses are. A pure derivative of a Gaussian
    has two lobes of equal |signal|, so the pulse-peak delay could jump between them by 2*sigma.
    """
    x = (t - t0) / sigma
    return (-x + 0.8) * np.exp(-0.5 * x**2) / sigma


def write_tf5(folder, stem, t, signal, *, phase_offset=0.0, amp_scale=None):
    """Write `<stem>.pulse.csv` and `<stem>.spectr.csv` in the TF5 'separate_files' layout.

    The spectrum is computed from the pulse with the package's own FFT path (e^{+i w t}
    convention, as the instrument exports it). `phase_offset` is added to the exported phase,
    mimicking the constant offset the instrument's running phase average can leave in the
    .spectr.csv file. `amp_scale(f)` multiplies the exported amplitude (e.g. to add absorption lines).
    """
    folder = Path(folder)
    f, amp, ph = TeraFlashAnalyzer._spectr_from_pulse(t, signal, dfreq=None, rel_start=0.01, rel_end=0.01, norm="ortho")
    if amp_scale is not None:
        amp = amp * amp_scale(f)
    np.savetxt(folder / f"{stem}.pulse.csv", np.column_stack([t, signal]), delimiter=",", fmt="%.6f",
               header="Time_abs/ps, Signal 1/nA", comments="")
    np.savetxt(folder / f"{stem}.spectr.csv", np.column_stack([f, amp, ph + phase_offset]), delimiter=",",
               fmt="%.8e", header="Frequency/GHz, Amplitude rel. 1, Phase 1", comments="")


@pytest.fixture
def slab(tmp_path):
    """Factory: write an open-beam reference and a non-dispersive slab (index n, thickness w_m) behind it.

    Returns (folder, sample_stem, base_stem). The slab delays the pulse by (n - 1) w / c and scales it by
    `transmission` (field); `noise` adds white noise to both traces.
    """
    def make(n=1.5, w_m=5e-3, transmission=0.8, phase_offset=0.0, noise=1e-4, seed=0, amp_scale=None):
        rng = np.random.default_rng(seed)
        t = np.arange(0.0, 200.0 + 1e-9, 0.05)          # 200 ps record, 0.05 ps steps: 5 GHz grid to 10 THz
        t_ref = 27.0
        delay = (n - 1) * w_m / C * 1e12
        ref = thz_pulse(t, t_ref) + noise * rng.standard_normal(t.size)
        smp = transmission * thz_pulse(t, t_ref + delay) + noise * rng.standard_normal(t.size)
        write_tf5(tmp_path, "open", t, ref, amp_scale=amp_scale)
        write_tf5(tmp_path, "slab", t, smp, phase_offset=phase_offset, amp_scale=amp_scale)
        return tmp_path, "slab", "open"
    return make


@pytest.fixture
def example_dir():
    return EXAMPLE
