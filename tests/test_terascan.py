"""TeraScanAnalyzer on the bundled example data."""
import numpy as np
import pytest

from TeraToptica import FourierCutConfig, TeraScanAnalyzer as TSA, TeraScanConfig as TSC

SAMPLE, BASE = "LPE_5o7cm^-1_90-180ghz", "open_90-180ghz"


@pytest.mark.parametrize("kw", [dict(), dict(fiber_stretcher=False), dict(fourier_cut=FourierCutConfig(enabled=True))],
                         ids=["stretcher", "hilbert", "fourier_cut"])
def test_spectra_are_sane(example_dir, kw):
    obj = TSA(example_dir / "TeraScan", SAMPLE, BASE, config=TSC(window="boxcar", window_size=2, **kw))
    f, t, ut, mask, err, boxnum = obj.get_smoothed_spectra()
    assert all(len(a) == len(f) for a in (t, ut, mask, err))
    sel = (f > 95) & (f < 175)
    assert np.all(np.isfinite(t[sel])) and np.all(np.isfinite(err[sel])) and np.all(err[sel] >= 0)
    assert 0.5 < np.median(t[sel]) < 1.1            # a low-pass filter, mostly transparent in this band
    amp_s, amp_b = obj.get_currents()
    assert amp_s.shape == amp_b.shape == f.shape


def test_unequal_frequency_domains_are_trimmed(example_dir):
    """Sample (110-180 GHz) and base (90-160 GHz) are cut to their common range and put on one grid."""
    obj = TSA(example_dir / "TeraScan", "LPE_5o7cm^-1_110-180ghz", "open_90-160ghz", config=TSC())
    f = obj.base_freq
    assert f.min() >= 109 and f.max() <= 161
    assert len(obj.samp_freq) == len(f)


def test_phase_diff_uses_linear_branch_pick(example_dir):
    obj = TSA(example_dir / "TeraScan", SAMPLE, BASE, config=TSC())
    # the default fit_range falls back to the data's own band (~90-180 GHz); (200, 500) used to raise here
    pd_, n_ulim, info = obj.get_phase_diff(return_info=True)
    assert info["offset_source"] == "linear"
    assert np.all(np.isfinite(pd_))
    pd_explicit, _ = obj.get_phase_diff(fit_range=(90.0, 180.0))
    assert np.allclose(pd_, pd_explicit)
    with pytest.raises(ValueError):                 # an explicit range outside the data still raises
        obj.get_phase_diff(fit_range=(200.0, 500.0))
    with pytest.raises(ValueError):                 # no time-domain pulses on a CW instrument
        obj.get_phase_diff(branch_method="pulse")
