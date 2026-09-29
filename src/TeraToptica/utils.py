# src/terapy/utils.py

from __future__ import annotations
from typing import Literal, Optional, Sequence, Tuple

import numpy as np
from astropy.convolution import Box1DKernel, Gaussian1DKernel, convolve
from scipy.ndimage import median_filter
from scipy.signal import find_peaks
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

    if window == "gaussian":
        # Gaussian1DKernel's parameter is the standard deviation, not the
        # full width, unlike every other window here. We therefore convert
        # boxnum to an equivalent standard deviation: FWHM = 2*sqrt(2*ln2)*stddev.
        stddev = boxnum / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        kernel = Gaussian1DKernel(stddev)
    else:
        kernel_map = {
            "boxcar": Box1DKernel(boxnum),
            # astropy.convolve can accept an ndarray kernel too
            "bartlett": np.bartlett(boxnum),
            "blackman": np.blackman(boxnum),
            "hanning": np.hanning(boxnum),
            "hamming": np.hamming(boxnum),
        }
        kernel = kernel_map[window]

    return convolve(arr, kernel)


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

def get_base_current(self):
    return self.base_freq, self.base_amp

def get_sample_current(self):
    return self.samp_freq, self.samp_amp

def get_open_current(self):
    return self.open_freq, self.open_amp

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


# ----------------------------
# Phase-difference cleaning (2*pi branch pick + mid-spectrum jump repair)
# ----------------------------

# ----------------------------
# Phase-difference cleaning (2*pi branch pick + mid-spectrum jump repair)
# ----------------------------

def _pick_branch_offset(
    freq: np.ndarray,
    phase_diff: np.ndarray,
    fit_range: Tuple[float, float],
    max_branch_cycles: int,
):
    """Zero-intercept 2*pi branch pick. See get_phase_diff for the rationale.

    Fits `phase_diff` over `fit_range` and chooses whichever multiple of 2*pi (within
    +-max_branch_cycles) makes the fit's f=0 intercept closest to zero.

    Returns (offset, intercept, at_edge). `at_edge` is True when the chosen offset is
    the largest one searched in either direction, i.e. the true branch may lie outside
    +-max_branch_cycles and the search range should be widened.
    """
    lo, hi = fit_range
    fit = (freq >= lo) & (freq <= hi)
    if fit.sum() < 2:
        raise ValueError(f"fit_range={fit_range} GHz contains fewer than 2 frequency points.")
    _, intercept = np.polyfit(freq[fit], phase_diff[fit], 1)
    offsets = 2 * np.pi * np.arange(-max_branch_cycles, max_branch_cycles + 1)
    i = int(np.argmin(np.abs(intercept + offsets)))
    return offsets[i], float(intercept), i in (0, len(offsets) - 1)

def _detect_jump_candidates(
    f_good: np.ndarray,
    phase_good: np.ndarray,
    jump_height: float,
    jump_distance: int,
    slope_window_ghz: float,
    max_smear_bins: int,
):
    """Detrended, multi-bin-aware candidate detector for _repair_phase_jumps.

    A genuine 2*pi branch slip landing between two grid points splits its magnitude
    across up to a few adjacent bins. Raw |diff(phase_diff)| cannot distinguish this from 
    the sample's own continuous dispersion slope, which has no fixed ratio to a slip's 
    magnitude and can be arbitrarily large for a thick/high-index sample. Detection 
    therefore happens AFTER removing the local slope, not on the raw difference.

    K=1 candidates are found on the RAW `|d|` (exactly `_repair_phase_jumps`'s pre-
    existing single-bin statistic), NOT the detrended `|r|` -- verified on real data that
    this distinction matters: detrending a real (nonzero-slope) sample can push an
    ordinary bin's residual ABOVE its own raw magnitude wherever the local slope estimate
    happens to have the opposite sign there, so a purely-synthetic zero-slope equivalence
    check is not sufficient proof that detrended-K=1 matches legacy (a real HDPE
    measurement produced exactly this: a new single-bin false positive at 1095 GHz that
    raw legacy never had). Using raw `|d|` for K=1 makes legacy-equivalence a STRUCTURAL
    guarantee for every existing caller by construction, not a coincidence of zero slope.
    Only K=2..max_smear_bins (opt-in via `max_smear_bins > 1`; the default leaves this
    loop empty) uses the detrended `r`, for exactly the multi-bin-smear case raw `|d|`
    cannot see. This does NOT repeat the previously-tried, and falsified,
    "raw 2-lag difference" idea: that compared a raw (non-detrended) 2-bin difference to
    the same threshold, which scales with slope and produces false positives on any
    sloped sample with no jump at all. Detrending the K>=2 case is what avoids that.

    `jump_distance` dedupes candidates the same way find_peaks always has (nearest
    neighbor wins, closer competitors dropped) -- returned separately as `suppressed`
    purely for visibility; this does not change that pre-existing behavior.

    Returns (peaks, spans, suppressed):
    - peaks: sorted int array of span start indices into `d`/`phase_good` (i.e. the jump
      sits between phase_good[p] and phase_good[p + K]), same indexing convention
      _repair_phase_jumps already uses for a single-bin peak.
    - spans: {p: (p, p + K)} for each p in peaks, K in [1, max_smear_bins].
    - suppressed: sorted list of anchor frequencies (float, in `f_good`) for candidates
      that cleared jump_height but were dropped by the jump_distance dedupe against a
      taller neighbor -- diagnostic only, never corrected, never touches n_ulim.
    """
    d = np.diff(phase_good)
    n = len(d)
    if n == 0:
        return np.array([], dtype=int), {}, []

    df = np.median(np.diff(f_good))
    win = max(int(round(slope_window_ghz / df)), 4 * max_smear_bins + 1)
    if win % 2 == 0:
        win += 1
    s_hat = median_filter(d, size=win, mode="nearest")
    r = d - s_hat

    # Two phases, run in a fixed order, so a clean single-bin jump is ALWAYS resolved
    # by phase A alone and never re-examined by a wider K -- taking an element-wise
    # "best magnitude across all K" instead (the first version of this code) creates a
    # spurious tie: a lone spike considered as a 2-bin window (itself + an adjacent zero
    # residual) has the SAME magnitude as the same spike alone at K=1, so find_peaks'
    # plateau resolution can shift the reported anchor by a bin and silently break
    # legacy equivalence. Claiming bins phase-by-phase (smallest K first) avoids that.
    covered = np.zeros(n, dtype=bool)
    candidates = []  # (start_idx, K, magnitude)

    # K=1 uses the RAW statistic (not `r`) -- see docstring above for why this must be
    # raw to keep legacy-equivalence a structural guarantee rather than a zero-slope
    # coincidence.
    abs_d = np.abs(d)
    peaks1, _ = find_peaks(abs_d, height=jump_height, distance=jump_distance)
    for p in peaks1:
        candidates.append((int(p), 1, float(abs_d[p])))
        covered[p] = True

    csum = np.concatenate([[0.0], np.cumsum(r)])
    for k in range(2, max_smear_bins + 1):
        if k > n:
            break
        starts = np.arange(n - k + 1)
        window_bad = np.zeros(len(starts), dtype=bool)
        for shift in range(k):
            window_bad |= covered[starts + shift]
        s = np.abs(csum[starts + k] - csum[starts])
        s_masked = np.where(window_bad, 0.0, s)
        peaks_k, _ = find_peaks(s_masked, height=jump_height, distance=jump_distance)
        for p in peaks_k:
            candidates.append((int(p), k, float(s_masked[p])))
            covered[p:p + k] = True

    # A final distance-based pass across the COMBINED candidate set: two candidates of
    # differing K can still land within jump_distance of each other without their bins
    # overlapping (so neither excluded the other above). Tallest wins, same priority
    # find_peaks itself uses; the loser is reported as 'suppressed', not silently
    # dropped (see the documented limitation in get_phase_diff).
    candidates.sort(key=lambda c: -c[2])
    accepted: list = []
    suppressed_idx: list = []
    for start, k, _mag in candidates:
        if any(abs(start - a[0]) < jump_distance for a in accepted):
            suppressed_idx.append(start)
            continue
        accepted.append((start, k))

    accepted.sort(key=lambda a: a[0])
    peaks = np.array([a[0] for a in accepted], dtype=int)
    spans = {a[0]: (a[0], a[0] + a[1]) for a in accepted}
    suppressed = sorted(float(f_good[i]) for i in suppressed_idx)
    return peaks, spans, suppressed

def _repair_phase_jumps(
    freq: np.ndarray,
    phase_diff: np.ndarray,
    jump_range: Tuple[float, Optional[float]],
    jump_height: float,
    jump_distance: int,
    jump_exclusion_ghz: float,
    residual_threshold: float,
    default_n_ulim: float,
    boundary_height: float,
    slope_window_ghz: float,
    max_smear_bins: int,
    max_jump_cycles: int,
):
    """Find and repair mid-spectrum 2*pi slips. See get_phase_diff for the rationale.

    Modifies `phase_diff` IN PLACE (a fresh array the caller owns; never the caller's
    original theta/theta_base). `freq` is not modified.

    Two thresholds do two different jobs, DELIBERATELY kept separate:
    - `boundary_height` (low, sensitive) finds every point where the data's slope
      changes enough to call it a segment break -- used ONLY to bound how far the
      before/after line fits below extend, i.e. to keep them local.
    - `jump_height` (the caller's threshold, meant to be conservative) decides which
      of those breaks is actually treated as a candidate 2*pi slip worth correcting
      or flagging.
      Both are necessary as otherwise the fit windows can balloon out hundreds of GHz and 
      pick up unrelated curvature. Segmenting on a separately low, sensitive 
      `boundary_height` (independent of however conservative `jump_height` is) keeps every 
      real candidate's fit window naturally local regardless.

    n_ulim tracks "how far into the spectrum this phase_diff should be trusted": it is
    `default_n_ulim` if every candidate reconciled cleanly, or otherwise the LOWEST
    frequency among all 'untrusted' candidates -- a simple minimum, independent of scan
    order. A peak that reconciles ('corrected' or 'clean') never raises n_ulim back up
    past an earlier untrusted one.

    This replaced an earlier "last-write-wins" version (walk candidates in frequency
    order, overwriting n_ulim at each: set to that peak's own frequency on 'untrusted',
    reset back to default_n_ulim on a clean 'corrected') that could let a later clean
    correction erase an earlier untrusted flag -- fragile and order-dependent by
    construction, and measured this session to add little the sample's own amplitude
    mask (`mask_bounds`, applied by the caller) doesn't already cover: of several real
    jump-detection fixes found this session, most landed inside already-masked bands, so
    n_ulim's dynamic range was mostly being spent on unbounded-jump_range noise-floor
    artifacts already harmless via masking, not on catching real problems the mask
    misses. The minimum-based version is a conservative, monotonic proxy for "where does
    phase info become untrustworthy" without that fragility.

    Candidate jumps come from `_detect_jump_candidates` (detrended, multi-bin-aware --
    see its docstring), which can propose a jump spanning `K >= 1` adjacent bins (a
    genuine slip landing between two grid points splits its magnitude across a few
    bins). `p` below is always the span's START index; for K=1 this is identical to the
    single-bin peak this function used before detrending was introduced. A span's
    INTERIOR points (strictly between its start and end) belong to neither the "before"
    nor the "after" line and are hard-excluded from both fits by INDEX, not by
    `jump_exclusion_ghz` distance -- a floating point/grid-spacing comparison is not
    guaranteed to catch every interior point of a multi-bin span. Any `boundary_height`
    peak landing inside a span (e.g. on the taller of two smeared bins) is likewise
    excluded from bounding that span's fit windows, for the same reason a peak can't
    bound itself: otherwise the "after" window collapses to nothing and a real jump
    crossing `jump_height` gets spuriously reported 'skipped'. The correction step, when
    applied, starts at the span's END (the first point where the branch has settled),
    which coincides with `p + 1` for K=1.

    The step tried against each candidate is whichever multiple of 2*pi in
    `+-max_jump_cycles` minimizes the before/after residual, not just {-2*pi, 0, +2*pi}
    -- verified on real data that a single slip can skip more than one cycle at once:
    a masked, heavily-absorptive band (near-total signal loss) can let theta wrap
    unpredictably, and a genuine +-1-cycle-only search picks the least-bad of three
    wrong options, leaving a large, avoidable residual (observed: -13.1 rad, misflagged
    'untrusted') when the true fix was a clean 2-cycle step (residual ~0.4 rad). This
    only changes which discrete step an ALREADY-actionable candidate (one that already
    crossed `jump_height`) gets corrected by; it can't create a new false detection.

    Returns (n_ulim, jumps); `jumps` holds one dict per detected peak with keys
    'f', 'status' ('corrected' | 'clean' | 'untrusted' | 'skipped' | 'suppressed'),
    'step', 'residual' ('skipped' entries also carry 'n_before'/'n_after'; 'suppressed'
    entries -- a candidate that crossed jump_height but was dropped by the
    jump_distance dedupe against a taller neighbor -- are diagnostic only: never
    corrected, never touch n_ulim, kept exactly as before this detector existed).
    """
    lo, hi = jump_range
    good = freq >= lo
    if hi is not None:
        good &= freq <= hi
    n_ulim = default_n_ulim
    jumps: list = []

    good_idx = np.where(good)[0]  # maps compressed index -> index into phase_diff
    if len(good_idx) < 3:
        return n_ulim, jumps
    f_good = freq[good_idx]
    phase_good = phase_diff[good_idx]  # fancy indexing copies; kept in sync below
    dtheta = np.abs(np.diff(phase_good))

    boundary_peaks, _ = find_peaks(dtheta, height=boundary_height, distance=jump_distance)
    peaks, spans, suppressed = _detect_jump_candidates(
        f_good, phase_good, jump_height, jump_distance, slope_window_ghz, max_smear_bins)
    for f_s in suppressed:
        jumps.append({'f': f_s, 'status': 'suppressed', 'step': 0.0, 'residual': float('nan')})
    if len(peaks) == 0:
        return n_ulim, jumps

    # Points within +-jump_exclusion_ghz of ANY actionable jump's anchor are left out of
    # the before/after line fits: the samples straddling a slip belong to neither line.
    peak_freqs = f_good[peaks]
    fit_ok = ~np.any(np.abs(f_good[:, None] - peak_freqs[None, :]) <= jump_exclusion_ghz, axis=1)
    # Hard-exclude every INTERIOR point of a multi-bin span by index (K=1 spans have no
    # interior points, so this is a no-op there -- see docstring above).
    for p in peaks:
        b = spans[int(p)][1]
        if b > p + 1:
            fit_ok[p + 1:b] = False
    step_options = np.arange(-max_jump_cycles, max_jump_cycles + 1) * 2 * np.pi

    for p in peaks:
        b = spans[int(p)][1]
        # Bound this peak's fit windows by the nearest OTHER boundary_height peaks
        # (excluding every boundary peak inside this span, which would otherwise bound
        # the span to an empty window) -- not by the nearest other ACTIONABLE peak. See
        # the docstring above for why that distinction matters.
        others = boundary_peaks[(boundary_peaks < p) | (boundary_peaks >= b)]
        before_others = others[others <= p]
        lo_idx = int(before_others.max()) + 1 if len(before_others) else 0
        after_others = others[others >= b]
        hi_idx = int(after_others.min()) + 1 if len(after_others) else len(phase_good)

        before = slice(lo_idx, p + 1)
        after = slice(b, hi_idx)
        before_ok, after_ok = fit_ok[before], fit_ok[after]
        peak_f = float(f_good[p])

        if before_ok.sum() < 2 or after_ok.sum() < 2:
            jumps.append({
                'f': peak_f, 'status': 'skipped', 'step': 0.0, 'residual': float('nan'),
                'n_before': int(before_ok.sum()), 'n_after': int(after_ok.sum()),
            })
            continue

        m_b, B_b = np.polyfit(f_good[before][before_ok], phase_good[before][before_ok], 1)
        m_a, B_a = np.polyfit(f_good[after][after_ok], phase_good[after][after_ok], 1)
        # Compare the two fits AT peak_f (where continuity is actually required), not
        # their y-intercepts at f=0 -- verified on real data that this distinction
        # matters whenever the before/after slopes differ substantially: extrapolating
        # both lines back to f=0 amplifies any slope mismatch by however far peak_f is
        # from zero, which can swamp the actual step decision. A real case (a masked,
        # heavily-absorptive foam band ~1720 GHz into a ~5000-bin spectrum, m_b/m_a
        # differing by ~3.4x) had a step that reconciled the two fits almost perfectly
        # AT peak_f (residual 0.5 rad) but looked like the WORST option of five by the
        # f=0-intercept comparison (extrapolation error there: 35-60 rad, dwarfing the
        # true ~0-25 rad spread at peak_f) -- so the old criterion picked one of the
        # worse steps outright. Comparing at peak_f is a strict improvement: it reduces
        # to the same choice as the intercept comparison whenever m_a == m_b (the two
        # extrapolation errors are then identical for every step and cancel out).
        step = step_options[np.argmin(np.abs((m_a * peak_f + B_a + step_options) - (m_b * peak_f + B_b)))]

        if step != 0:
            # Apply from the span's end onward in the FULL array (including bins outside
            # jump_range), and keep the compressed copy in sync so later peaks in this
            # call see the correction.
            phase_diff[good_idx[b]:] += step
            phase_good[b:] += step

        residual = (m_a * peak_f + B_a + step) - (m_b * peak_f + B_b)
        if abs(residual) > residual_threshold:
            status = 'untrusted'
        elif step != 0:
            status = 'corrected'
        else:
            status = 'clean'
        jumps.append({'f': peak_f, 'status': status, 'step': float(step), 'residual': float(residual)})

    untrusted_freqs = [j['f'] for j in jumps if j['status'] == 'untrusted']
    if untrusted_freqs:
        n_ulim = min(untrusted_freqs)
    jumps.sort(key=lambda j: j['f'])
    return n_ulim, jumps

def get_phase_diff(
    self,
    *,
    unwrap: bool = False,
    absolute: bool = False,
    offset: Optional[float] = None,
    fit_range: Tuple[float, float] = (200.0, 500.0),
    max_branch_cycles: int = 10,
    jump_range: Tuple[float, Optional[float]] = (150.0, None),
    jump_height: float = 5.2,
    boundary_height: float = 0.5,
    jump_distance: int = 10,
    jump_exclusion_ghz: float = 5.0,
    residual_threshold: float = 1.0,
    default_n_ulim: float = 2000.0,
    slope_window_ghz: float = 100.0,
    max_smear_bins: int = 1,
    max_jump_cycles: int = 2,
    verbose: bool = False,
    label: Optional[str] = None,
    return_info: bool = False,
):
    """
    Sample-minus-reference phase difference with its 2*pi ambiguity resolved, for this
    instance vs. its reference (`self.base_freq, self.samp_phase, self.base_phase`).

    Shared by TeraFlashAnalyzer and TeraScanAnalyzer (attached the same way get_spectra
    etc. are), since the algorithm below only needs freq/theta/theta_base and doesn't
    care which instrument produced them -- both classes have the same attributes.

    Two independent steps:

    1. Branch pick: `theta - theta_base` is only known up to a whole number of extra
       2*pi cycles (the "branch"). Pick the multiple of 2*pi (within
       +-max_branch_cycles) that makes a straight-line fit over `fit_range` extrapolate
       closest to zero at f=0 therefore calibrates the phase information.
    2. Jump repair: a detrended, multi-bin-aware detector (`_detect_jump_candidates`)
       finds mid-spectrum 2*pi slips within `jump_range`, then reconciles each with a
       -2*pi/0/+2*pi step using LOCAL line fits on either side (excluding
       +-jump_exclusion_ghz around, and every interior point of, a detected jump, since
       points straddling a slip belong to neither line). "Local" here specifically
       means bounded by the nearest OTHER point crossing `boundary_height` -- a second,
       more sensitive threshold used only to keep these fits from extending into
       unrelated parts of the spectrum (see the `boundary_height` parameter below for
       why this is a separate knob from `jump_height`, not the same one reused). A jump
       whose fits still disagree by more than `residual_threshold` rad after the best
       step is 'untrusted'.

    Why the jump thresholds are shaped this way: raw `|diff(phase_diff)|` per frequency
    bin is the phase's own SLOPE (2*pi*n*L*df/c for a sample of index n, thickness L)
    plus measurement ripple, not ripple alone, and a genuine 2*pi slip can land between
    two grid points, splitting its ~6.28 rad magnitude across up to a few adjacent bins. 
    Comparing this raw quantity to one global threshold conflates two things with no 
    fixed ratio. `_detect_jump_candidates` removes a CENTERED ROLLING MEDIAN local-slope 
    estimate (window set by `slope_window_ghz`, same GHz-to-bins convention as `fill_dips`) 
    from the SIGNED `diff(phase_diff)` before thresholding, then sums the detrended residual 
    over `1..max_smear_bins` adjacent bins to recover a smeared slip's full magnitude. A low
    `jump_height` (e.g. 0.5) can flip the apparent slope of the whole curve. `jump_height=5.2` 
    sits above the largest ripple used in the test data (~4.9 rad, raw) and below a genuine 
    ~6.28 rad slip; because detrending removes slope rather than adding noise, this 
    raw-calibrated value stays conservative post-detrending too, but re-check it 
    (`return_info=True`, `verbose=True`) against any very different sample or instrument rather 
    than assume it transfers unchanged. `jump_height` also caps how much per-bin phase 
    slope ESTIMATION ERROR (via `slope_window_ghz`) this detector can tolerate before it can no
    longer distinguish a real slip from the sample's own dispersion -- thicker/
    higher-index samples, a coarser frequency grid, or a shorter `slope_window_ghz` may
    need a stricter `jump_height`.

    `jump_range`'s upper bound defaults to None (unbounded) DELIBERATELY, not out of
    laziness: a correction only ever applies FORWARD from the frequency it's detected
    at (see _repair_phase_jumps), so a spurious "jump" found deep in a sample's noise
    floor -- SNR/model validity commonly collapses well past the physically meaningful
    band, and find_peaks will fire on that pure noise -- can only corrupt phase_diff
    *above* itself, never at or below the last genuine feature ONE step *before* it.
    Verified against real meteorite data: capping this at 1500 GHz (this threshold's
    original tuning band) vs. leaving it unbounded produces byte-identical phase_diff
    everywhere below 1500 GHz, for every sample tested -- there, no jump of any kind
    occurs in the trusted band under either setting, so this holds trivially. It is
    NOT sufficient on its own once real jumps are involved, though (see
    `boundary_height` below for the failure mode this originally caused and how it's
    avoided) -- a hard per-dataset cap is also actively harmful on its own terms: it
    silently missed a genuine, correctable 2*pi slip that happened to sit just past an
    arbitrarily chosen 1500 GHz ceiling in unrelated foam-sample data, while an
    equally-plausible 2000 GHz ceiling would have caught that one but clipped a
    different sample's real feature elsewhere. Leaving it unbounded needs no
    per-dataset tuning; the tradeoff is a possibly-uninformative `n_ulim` deep in the
    noise floor (see default_n_ulim) -- harmless as long as nothing downstream reads
    phase/n from a frequency nobody actually trusts anyway. Narrow `jump_range` (or
    raise `jump_height`) only if you have a specific reason to stop the search short of
    the full spectrum; do not narrow it "just in case" -- that reintroduces the exact
    per-dataset fragility this default avoids.

    `boundary_height` is a SEPARATE, more sensitive threshold from `jump_height`, doing
    a different job: `jump_height` decides which detected break is treated as an actual
    candidate 2*pi slip; `boundary_height` decides where the before/after line fits for
    THAT candidate are allowed to extend to, regardless of how far away the next
    candidate happens to be. Reusing `jump_height` for both jobs (the obvious, naive
    design) creates a hidden coupling: raising `jump_height` to suppress distant false
    positives also removes the nearby low-level segment breaks that were incidentally
    keeping a REAL jump's fit windows local. Once those neighbors are gone, the window
    can balloon hundreds of GHz in one direction, absorb unrelated dispersion
    curvature, and flip the sign of the correction outright -- verified on real foam
    data: a genuine, cleanly-explained +2*pi correction (residual ~0.4 rad) became a
    wrongly-signed "correction" with a -38 rad residual once `jump_range` was widened
    without also keeping boundary detection separately sensitive. `boundary_height` at
    a low, ripple-scale value (this default matches the height once used, incorrectly,
    for the actionable threshold itself) restores locality unconditionally rather than
    as a side effect of how many other things happen to cross `jump_height`. Since
    `boundary_height` never itself triggers a correction, setting it low costs
    essentially nothing -- worst case, a real jump's window gets slightly too tight and
    it's reported 'skipped' (safe) rather than incorrectly resolved.

    These defaults are validated against two real instruments' data (mm-thick
    meteorite slabs and a family of foam dielectrics), including cases from each where
    naive versions of this algorithm produced wrong answers (documented above); a very
    different sample or instrument should still re-check them (see `return_info=True`
    and `verbose=True`) rather than assume they transfer unchanged.

    Parameters
    ----------
    unwrap : bidirectionally double-unwrap (`np.unwrap(np.unwrap(theta) -
        np.unwrap(theta_base))`) instead of a plain difference, for the rare case where
        theta wraps faster than theta_base. Caller decides; nothing here inspects
        sample names.
    absolute : take `abs()` of the difference before branch-picking. Ignored if
        `unwrap=True` (matches this codebase's existing conventions either way).
    offset : pin a branch (radians, e.g. `4*np.pi`) you've independently verified,
        skipping the auto search in step 1 entirely.
    boundary_height : rad; low, sensitive threshold used only to keep each candidate
        jump's before/after fit windows local (see the long note above) -- distinct
        from `jump_height`, which decides whether a break is actionable at all. Should
        normally stay well below `jump_height`; lowering it further essentially never
        hurts (see above).
    default_n_ulim : the "fully trusted" n_ulim value returned when every detected
        candidate reconciles cleanly (no 'untrusted' status anywhere); otherwise n_ulim
        is the lowest frequency among all 'untrusted' candidates, full stop (see
        _repair_phase_jumps for the exact, deliberately simple, order-independent
        semantics -- no later clean correction can raise n_ulim back past an earlier
        untrusted one).
    slope_window_ghz : rad; width (in GHz, converted to bins the same way `fill_dips`
        does) of the centered rolling median used to estimate local phase slope before
        jump detection (see `_detect_jump_candidates`). Wider tolerates noisier data but
        reacts more slowly to genuine curvature; narrower risks the jump itself biasing
        its own baseline estimate.
    max_smear_bins : largest number of adjacent bins a single 2*pi slip is allowed to be
        split across (see `_detect_jump_candidates`). Defaults to 1 (i.e. matches the
        single-bin detector this package used before smeared-jump detection existed) --
        NOT because smeared jumps are rare (one caused a real, missed correction in
        production foam data) but because whether K>1 is SAFE depends on how strongly
        autocorrelated a dataset's own ripple is, which this package cannot know in
        advance. Measured directly: a family of meteorite slab measurements has ripple
        correlated enough that its K=2 noise ceiling (~5.76 rad, worst case across 9
        samples) lands within 0.05 rad of a real foam sample's K=2 jump floor (~5.72
        rad) -- these are NOT separable by any fixed jump_height, on this real data, not
        a hypothetical edge case. K=3 is worse still (~8.1 rad ceiling on the same
        meteorite data, exceeding a slip's fixed ~6.28 rad total outright). Foam data
        measured so far has a much lower ceiling (~2.3 rad worst case) and safely
        supports `max_smear_bins=2` -- but this is a property of THAT dataset's own
        ripple, not something to assume transfers. Before raising this for a new
        dataset, measure its own noise ceiling (max detrended K-bin sum outside any
        known real jump) against `jump_height` with margin on both sides, the same way
        (see IRBF's main.ipynb, which passes `max_smear_bins=2` explicitly with this
        same justification -- not a magic per-dataset number, a documented, measured
        override, the same as its per-batch `unwrap=` decision).
    max_jump_cycles : largest number of whole 2*pi cycles a single detected candidate
        may be corrected by (searched as `+-max_jump_cycles`, picking whichever
        reconciles the before/after fits best -- see `_repair_phase_jumps`). Defaults to
        2, unlike `max_smear_bins`: this only changes which discrete step an ALREADY-
        actionable candidate gets corrected by, so it can't manufacture a new false
        detection the way a lower `jump_height` or higher `max_smear_bins` could --
        verified on real data (a masked, heavily-absorptive foam sample band) where the
        true fix was a clean 2-cycle step and a +-1-cycle-only search left a large,
        avoidable residual instead. Still bounded (not unbounded like
        `max_branch_cycles`) because a large search range applied to a genuinely
        untrustworthy region (heavy noise, near-total signal loss) could occasionally
        find some large multiple that spuriously minimizes residual by coincidence
        rather than because it's correct.
    verbose : print the branch offset chosen and each jump found/corrected/skipped, in
        the same format this logic has printed in past notebook implementations.
    label : name used in printed messages (verbose=True only); defaults to this
        instance's sample stem (`self.samp`) if not given.
    return_info : if True, also return a dict with 'offset', 'offset_source' ('auto' or
        'manual'), 'intercept', 'offset_at_edge', and 'jumps' (see _repair_phase_jumps).

    Returns
    -------
    (phase_diff, n_ulim) normally, or (phase_diff, n_ulim, info) if `return_info=True`.
    n_ulim (GHz) is the frequency up to which phase_diff (and any n derived from it)
    should be trusted. Never modifies `theta`/`theta_base`.
    """
    freq, theta, theta_base = self.base_freq, self.samp_phase, self.base_phase

    if unwrap:
        phase_diff = np.unwrap(np.unwrap(theta) - np.unwrap(theta_base))
    else:
        phase_diff = np.asarray(theta, dtype=float) - np.asarray(theta_base, dtype=float)
        if absolute:
            phase_diff = np.abs(phase_diff)

    name = label if label is not None else self.samp
    if offset is None:
        offset, intercept, at_edge = _pick_branch_offset(
            freq, phase_diff, fit_range, max_branch_cycles)
        if at_edge:
            print(f"Warning: {name}: branch offset {offset / np.pi:.1f}*pi is at the edge of the "
                  f"+-{max_branch_cycles}-cycle search; increase max_branch_cycles to check whether "
                  f"a further branch fits better.")
        if verbose:
            print(f"{name}: branch offset = {offset / np.pi:.1f}*pi (auto, intercept={intercept:.1f})")
    else:
        intercept, at_edge = None, False
        if verbose:
            print(f"{name}: branch offset = {offset / np.pi:.1f}*pi (manual)")
    phase_diff = phase_diff + offset  # new array: never aliases theta/theta_base

    n_ulim, jumps = _repair_phase_jumps(
        freq, phase_diff, jump_range, jump_height, jump_distance,
        jump_exclusion_ghz, residual_threshold, default_n_ulim, boundary_height,
        slope_window_ghz, max_smear_bins, max_jump_cycles)

    if verbose:
        for j in jumps:
            if j['status'] == 'skipped':
                print(f"{name}: jump near {j['f']:.0f} GHz SKIPPED "
                      f"({j['n_before']} pts before / {j['n_after']} pts after "
                      f"the {jump_exclusion_ghz:g} GHz exclusion -- not enough unmasked data to fit)")
            elif j['status'] == 'suppressed':
                print(f"{name}: jump candidate near {j['f']:.0f} GHz SUPPRESSED "
                      f"(within jump_distance of a taller neighbor -- diagnostic only, "
                      f"not corrected, does not affect n_ulim)")
            else:
                print(f"{name}: jump near {j['f']:.0f} GHz, best_step={j['step'] / np.pi:.1f}*pi, "
                      f"residual={j['residual']:.2f} rad ({j['status']})")

    if return_info:
        info = {
            'offset': float(offset),
            'offset_source': 'manual' if intercept is None else 'auto',
            'intercept': intercept,
            'offset_at_edge': at_edge,
            'jumps': jumps,
        }
        return phase_diff, n_ulim, info
    return phase_diff, n_ulim

