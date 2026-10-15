"""TeraFlashAnalyzer: loading, spectra, pulse delay, refractive index and atmospheric-line handling."""
import warnings

import numpy as np
import pytest

from TeraToptica import TeraFlashAnalyzer as TFA, TeraFlashConfig as TFC

from conftest import C

BAND = (150.0, 1000.0)
MASK = ((0, 100), (2000, 10000))


def n_of(obj, w_m, **kw):
    """Median refractive index over BAND, plus get_phase_diff's info dict."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pd_, n_ulim, info = obj.get_phase_diff(return_info=True, **kw)
    f = obj.base_freq
    sel = (f >= BAND[0]) & (f <= BAND[1])
    return np.median(1 + pd_[sel] * C / (2 * np.pi * f[sel] * 1e9 * w_m)), info


# ---------------- bundled example data ----------------

@pytest.mark.parametrize("use_fft", [False, True])
def test_example_data_loads(example_dir, use_fft):
    obj = TFA(example_dir / "TeraFlash", "al_LF_575-775ps_1000avgs", "open_575-775ps_1000avgs",
              config=TFC(mask_bounds=MASK, use_FFT=use_fft))
    out = obj.get_smoothed_spectra()
    assert len(out) == 6
    f, t, ut, mask, err, boxnum = out
    assert all(len(a) == len(f) for a in (t, ut, mask, err))
    assert boxnum % 2 == 1 and mask[f < 100].all() and not mask[(f > 150) & (f < 1900)].any()
    sel = (f > 150) & (f < 1000)
    assert np.all(np.isfinite(t[sel])) and np.all(np.isfinite(err[sel]))
    assert np.all((t[sel] >= 0) & (t[sel] < 1.5))
    assert obj.get_pulse_delay() > 0                     # a sample in the beam delays the pulse


def test_exported_and_fft_transmission_agree(example_dir):
    """Both spectrum sources should give the same transmittance where there is signal."""
    kw = dict(mask_bounds=MASK, window_size=25)
    a = TFA(example_dir / "TeraFlash", "al_LF_575-775ps_1000avgs", "open_575-775ps_1000avgs", config=TFC(**kw))
    b = TFA(example_dir / "TeraFlash", "al_LF_575-775ps_1000avgs", "open_575-775ps_1000avgs",
            config=TFC(use_FFT=True, **kw))
    f, ta = a.get_smoothed_spectra()[:2]
    fb, tb = b.get_smoothed_spectra()[:2]
    band = (f > 200) & (f < 700)
    sel = band & (ta > 0.05 * ta[band].max())          # this filter is nearly opaque; compare where it transmits
    assert sel.sum() > 10
    assert np.median(np.abs(np.interp(f[sel], fb, tb) / ta[sel] - 1)) < 0.05


def test_open_in_open_is_unity(slab):
    folder, _, base = slab()
    f, t = TFA(folder, base, base, config=TFC(mask_bounds=MASK)).get_spectra()
    sel = (f > 150) & (f < 1500)
    assert np.allclose(t[sel], 1.0)


def test_layout_argument_errors(slab):
    folder, sample, base = slab()
    with pytest.raises(ValueError):
        TFA(folder, sample, config=TFC(layout="separate_files"))            # base missing
    with pytest.raises(ValueError):
        TFA(folder, sample, config=TFC(layout="reference_included", open_stem=base))


# ---------------- pulse delay and refractive index on a known slab ----------------

def test_pulse_delay_recovers_slab_delay(slab):
    n, w = 1.53, 6e-3
    folder, sample, base = slab(n=n, w_m=w)
    delay = TFA(folder, sample, base, config=TFC(mask_bounds=MASK)).get_pulse_delay()
    assert delay == pytest.approx((n - 1) * w / C * 1e12, abs=0.01)    # ps


@pytest.mark.parametrize("use_fft", [False, True])
@pytest.mark.parametrize("n, w", [(1.53, 6e-3), (1.0134, 3.1e-3), (3.1, 2e-3)])
@pytest.mark.parametrize("phase_offset", [0.0, -2 * np.pi], ids=["no_offset", "minus_2pi_offset"])
def test_index_of_known_slab(slab, use_fft, n, w, phase_offset):
    """HDPE-like, thin HD30-foam-like and alumina-like slabs, on both phase paths. A -2*pi offset in the
    exported phase makes the raw difference cross zero for the thin foam: the case abs() used to mirror to n < 1."""
    folder, sample, base = slab(n=n, w_m=w, phase_offset=phase_offset)
    n_meas, info = n_of(TFA(folder, sample, base, config=TFC(mask_bounds=MASK, use_FFT=use_fft)), w)
    assert n_meas == pytest.approx(n, abs=1e-3)
    assert info["offset_source"] == "pulse" and info["orientation"] == 1 and not info["dealiased"]
    assert not info["offset_ambiguous"] and not info["offset_inconsistent"]


@pytest.mark.parametrize("use_fft", [False, True])
def test_late_pulse_is_dealiased(slab, use_fft):
    """A 20 mm alumina-like slab puts the sample pulse ~167 ps into its 200 ps record: a phase step of
    ~5.2 rad per 5 GHz bin, which bin-by-bin unwrapping aliases (the bundled LF alumina example)."""
    n, w = 3.1, 20e-3
    folder, sample, base = slab(n=n, w_m=w)
    obj = TFA(folder, sample, base, config=TFC(mask_bounds=MASK, use_FFT=use_fft))
    raw = obj.samp_phase - obj.base_phase
    f = obj.base_freq
    band = (f >= BAND[0]) & (f <= BAND[1])
    expected_slope = 2 * np.pi * obj.get_pulse_delay() * 1e-3                 # rad/GHz
    assert abs(np.polyfit(f[band], raw[band], 1)[0] - expected_slope) > 0.5   # test setup: the input is aliased
    n_meas, info = n_of(obj, w)
    assert info["dealiased"]
    assert n_meas == pytest.approx(n, abs=1e-3)


def test_whole_cycle_export_offset_is_absorbed(slab):
    """A 2*pi multiple in the exported phase is a branch choice, not an error."""
    folder, sample, base = slab(n=1.53, w_m=6e-3, phase_offset=-4 * np.pi)
    n_meas, info = n_of(TFA(folder, sample, base, config=TFC(mask_bounds=MASK)), 6e-3)
    assert n_meas == pytest.approx(1.53, abs=1e-3)


def test_fractional_export_offset_is_flagged_and_fft_is_immune(slab):
    """A half-cycle constant in the exported phase (as the instrument's running phase average can
    leave) biases n and must be flagged; the FFT of the pulse doesn't see it."""
    folder, sample, base = slab(n=1.53, w_m=6e-3, phase_offset=np.pi)
    _, info = n_of(TFA(folder, sample, base, config=TFC(mask_bounds=MASK)), 6e-3)
    assert info["offset_ambiguous"] and abs(abs(info["offset_frac"]) - 0.5) < 0.05
    n_fft, info_fft = n_of(TFA(folder, sample, base, config=TFC(mask_bounds=MASK, use_FFT=True)), 6e-3)
    assert n_fft == pytest.approx(1.53, abs=1e-3) and not info_fft["offset_ambiguous"]


def test_absolute_keyword_is_deprecated_noop(slab):
    folder, sample, base = slab(n=1.0134, w_m=3.1e-3, phase_offset=-2 * np.pi)
    obj = TFA(folder, sample, base, config=TFC(mask_bounds=MASK))
    with pytest.warns(FutureWarning):
        pd_abs, _ = obj.get_phase_diff(absolute=True)
    pd_new, _ = obj.get_phase_diff()
    assert np.allclose(pd_abs, pd_new)


def test_linear_branch_method_still_available(slab):
    folder, sample, base = slab(n=1.53, w_m=6e-3)
    n_meas, info = n_of(TFA(folder, sample, base, config=TFC(mask_bounds=MASK)), 6e-3, branch_method="linear")
    assert info["offset_source"] == "linear"
    assert n_meas == pytest.approx(1.53, abs=2e-3)


# ---------------- atmospheric-line detection and filling ----------------

def lines(centers, depth=0.8, width_ghz=8.0):
    """Amplitude factor with narrow Gaussian absorption lines (field depth `depth`)."""
    def scale(f):
        s = np.ones_like(f)
        for c in centers:
            s *= 1 - depth * np.exp(-0.5 * ((f - c) / width_ghz) ** 2)
        return s
    return scale


def test_find_atm_mask_flags_narrow_dip_not_broad_fringe():
    f = np.arange(0.0, 2000.0, 5.0)
    smooth = np.exp(-f / 800.0)
    fringe = smooth * (1 + 0.3 * np.cos(2 * np.pi * f / 600.0))   # broad, real feature
    dip = smooth * lines([557.0])(f)
    assert TFA.find_atm_mask(f, dip)[np.argmin(np.abs(f - 557))]
    assert not TFA.find_atm_mask(f, fringe).any()


def test_atm_correction_fills_lines_and_can_be_restricted(slab):
    folder, sample, base = slab(n=1.53, w_m=6e-3, amp_scale=lines([557.0, 752.0]))
    near = lambda f, c: np.abs(f - c) <= 5
    on = TFA(folder, sample, base, config=TFC(mask_bounds=MASK, atm_correction=True))
    assert on.atm_mask[near(on.base_freq, 557)].any() and on.atm_mask[near(on.base_freq, 752)].any()
    restricted = TFA(folder, sample, base, config=TFC(mask_bounds=MASK + ((545, 570),), atm_correction=True,
                                                   atm_restrict_to_mask_bounds=True))
    f = restricted.base_freq
    assert restricted.atm_mask[near(f, 557)].any()
    assert not restricted.atm_mask[near(f, 752)].any()
    # the line divides out of the ratio either way: transmission stays flat across it
    t = on.get_spectra()[1]
    assert np.allclose(t[near(f, 752)], np.median(t[(f > 700) & (f < 800)]), rtol=0.05)
