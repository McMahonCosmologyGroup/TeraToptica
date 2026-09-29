# src/TeraToptica/tf.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, peak_widths
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter, maximum_filter1d

from . import utils as _utils
from .utils import (
    build_mask_from_bounds,
    convolve_1d,
    compute_boxnum_from_window_size,
    rolling_std_error,
)

@dataclass(frozen=True)
class FFTConfig:
    """FFT-from-pulse settings (LabVIEW-like).

    The TF LabVIEW program applies a modified Blackman apodization (see
    `_modblackmanwindow`) before computing an rFFT magnitude. If `dfreq` is
    provided (in GHz), we symmetrically zero-pad the time series to reach an
    approximate target frequency resolution.
    """

    dfreq: float | None = None  # GHz
    tmin_ps: float | None = None  # optional relative time-domain cut start (ps)
    tmax_ps: float | None = None  # optional relative time-domain cut end (ps)
    rel_start: float = 0.01
    rel_end: float = 0.01
    fft_norm: Literal["ortho"] | None = "ortho"

@dataclass(frozen=True)
class TeraFlashConfig:
    """
    Configuration for TF5 reflection/transmission reduction.

    window is the smoothing window type.
    window_size is in GHz. It is converted to an equivalent standard deviation for 'gaussian'
    mask_bounds are frequency ranges (GHz) to mask, e.g. [(557, 560), ...].
    """
    window: Literal["boxcar", "bartlett", "blackman", "gaussian", "hanning", "hamming", "median"] = "boxcar"
    window_size: float = 15.0
    mask_bounds: Optional[Sequence[Tuple[float, float]]] = None
    include_mes_err: bool = True

    # If True, find and smooth over atmospheric dips
    atm_correction: bool = False
    atm_threshold: float = 0.2  # fractional depth below local envelope to flag as a atm dip
    atm_envelope_window_ghz: float = 150.0  # rolling-max envelope window for atm dip finding
    atm_prominence_db: float = 2.0  # min peak prominence (dB) for the shape check
    atm_max_width_ghz: float = 300.0  # max peak width (GHz) for the shape check

    # how to interpret TF5 export format
    layout: Literal["separate_files", "reference_included"] = "separate_files"

    # Optional open-subtraction: ((sample-open)/(base-open))^2
    open_stem: Optional[str] = None

    # Optional: build spectra from the pulse via FFT (LabVIEW-like)
    use_FFT: bool = False
    fft: FFTConfig = FFTConfig()

    # noise-floor estimate region (GHz) used for measurement error term
    noise_floor_min_ghz: float = 5500.0


class TeraFlashAnalyzer:
    """
    TeraFlash Reflection/Transmission analysis using TF-5D-RC-Host exported CSVs:
      - <name>.spectr.csv  (frequency-domain amplitude/current + phase)
      - <name>.pulse.csv   (time-domain pulse for normalization)

    Notes
    -----
    Internal variable name uses `samp`; public methods use `sample`.
    Implementing dynamic column selection based on header rows is coming soon.
    """

    def __init__(
        self,
        path_measurement: str | Path,
        sample: str | None = None,
        base: str | None = None,
        *,
        # for the "both in one file" export (single stem)
        config: TeraFlashConfig = TeraFlashConfig(),
    ):
        self.path_measurement = Path(path_measurement)
        self.config = config

        # Normalize inputs:
        # - separate_files: base+sample stems required
        # - both_in_one_file: fn required
        if config.layout == "separate_files":
            if base is None or sample is None:
                raise ValueError("For layout='separate_files', you must pass base=... and sample=... Perhaps you meant layout='reference_included'?")
            self.samp = sample
            self.base = base
        else:
            if sample is None:
                raise ValueError("For layout='reference_included, you must pass sample=...")
            if self.config.open_stem is not None:
                raise ValueError("Open subtraction is not compatible with layout='reference_included.")
            
            self.samp = sample
            self.base = None

        self._analyze_data()

    def _analyze_data(self) -> None:
        # 1) Load time-domain pulses
        if self.config.layout == "separate_files":
            samp_pulse = self._pulse_path(self.samp)
            base_pulse = self._pulse_path(self.base)
            # TF5: (time, signal) in cols (0,1)
            self.samp_time, self.samp_signal = self._load_pulse_csv(samp_pulse, cols=(0, 1))
            self.base_time, self.base_signal = self._load_pulse_csv(base_pulse, cols=(0, 1))

            if self.config.open_stem is not None:
                open_pulse = self._pulse_path(self.config.open_stem)
                self.open_time, self.open_signal = self._load_pulse_csv(open_pulse, cols=(0, 1))

        else:
            pulse = self._pulse_path(self.samp)
            # “both-in-one” pulse export: your original used base=(0,2) and sample=(0,1)
            self.samp_time, self.samp_signal = self._load_pulse_csv(pulse, cols=(0, 1))
            self.base_time, self.base_signal = self._load_pulse_csv(pulse, cols=(0, 2))

        # 2) Load or generate frequency-domain and phase and normalize if needed 
        # use_FFT does not require normalization, but averages in time-domain which is less accurate
        if not self.config.use_FFT:
            if self.config.layout == "separate_files":
                samp_spec = self._spectr_path(self.samp)
                base_spec = self._spectr_path(self.base)
                # (freq, pc, phase) in (0, 1,2)
                self.samp_freq, self.samp_amp, self.samp_phase = self._load_spectr_csv(
                    samp_spec, cols=(0, 1, 2)
                )
                self.base_freq, self.base_amp, self.base_phase = self._load_spectr_csv(
                    base_spec, cols=(0, 1, 2)
                )
                if self.config.open_stem is not None:
                    open_spec = self._spectr_path(self.config.open_stem)
                    self.open_freq, self.open_amp, self.open_phase = self._load_spectr_csv(
                        open_spec, cols=(0, 1, 2)
                    )

            elif self.config.layout == "reference_included":
                spec = self._spectr_path(self.samp)
                # sample: (freq, pc, phase) in (0, 1,2); sample: (0, 3, 4)
                self.samp_freq, self.samp_amp, self.samp_phase = self._load_spectr_csv(
                    spec, cols=(0, 1, 2)
                )
                self.base_freq, self.base_amp, self.base_phase = self._load_spectr_csv(
                    spec, cols=(0, 3, 4)
                )
            else:
                raise ValueError(f"Unsupported layout: {self.config.layout} Layout must be 'separate_files' or 'reference_included'.")

        else:
            if (self.config.fft.tmin_ps is not None) or (self.config.fft.tmax_ps is not None):
                tmin = self.config.fft.tmin_ps+np.min(self.samp_time) if self.config.fft.tmin_ps is not None else np.min(self.samp_time)
                tmax = self.config.fft.tmax_ps+np.min(self.samp_time) if self.config.fft.tmax_ps is not None else np.max(self.samp_time)
                
                samp_mask = (self.samp_time >= tmin) & (self.samp_time <= tmax)
                base_mask = (self.base_time >= tmin) & (self.base_time <= tmax)

                if samp_mask.sum() < 2 or base_mask.sum() < 2:
                    raise ValueError("Time cut results in fewer than 2 samples for FFT; adjust tmin_ps and tmax_ps in the FFTConfig.")

                self.samp_time = self.samp_time[samp_mask]
                self.samp_signal = self.samp_signal[samp_mask]

                self.base_time = self.base_time[base_mask]
                self.base_signal = self.base_signal[base_mask]

                if self.config.open_stem is not None:
                    open_mask = (self.open_time >= tmin) & (self.open_time <= tmax)
                    self.open_time = self.open_time[open_mask]
                    self.open_signal = self.open_signal[open_mask]

            self.samp_freq, self.samp_amp, self.samp_phase = self._spectr_from_pulse(
                self.samp_time,
                self.samp_signal,
                dfreq=self.config.fft.dfreq,
                rel_start=self.config.fft.rel_start,
                rel_end=self.config.fft.rel_end,
                norm=self.config.fft.fft_norm,
            )
            self.base_freq, self.base_amp, self.base_phase = self._spectr_from_pulse(
                self.base_time, 
                self.base_signal, 
                dfreq=self.config.fft.dfreq,
                rel_start=self.config.fft.rel_start,
                rel_end=self.config.fft.rel_end,
                norm=self.config.fft.fft_norm,
            )

            self.C = 1.0 # not used in FFT mode

            if self.config.open_stem is not None:
                self.open_freq, self.open_amp, self.open_phase = self._spectr_from_pulse(
                    self.open_time,
                    self.open_signal,
                    dfreq=self.config.fft.dfreq,
                    rel_start=self.config.fft.rel_start,
                    rel_end=self.config.fft.rel_end,
                    norm=self.config.fft.fft_norm,
                )

                self.C_open = 1.0 # not used in FFT mode

        # Check lengths and frequencies match; if not, interpolate sample (and open) to base
        if (len(self.base_amp) != len(self.samp_amp)) or (not np.allclose(self.base_freq, self.samp_freq)):
            amp = interp1d(self.samp_freq, self.samp_amp, bounds_error=False, fill_value='extrapolate')
            ph = interp1d(self.samp_freq, self.samp_phase, bounds_error=False, fill_value='extrapolate')

            self.samp_amp   = amp(self.base_freq)
            self.samp_phase = ph(self.base_freq)
            self.samp_freq  = self.base_freq.copy()

            print(f'Warning: file {self.samp} had mismatched lengths or values. Interpolated amplitude and phase to match reference and discarding raw sample frequencies.')

        if self.config.open_stem is not None:
            if (len(self.open_amp) != len(self.base_amp)) or (not np.allclose(self.open_freq, self.base_freq)):
                amp = interp1d(self.open_freq, self.open_amp, bounds_error=False, fill_value='extrapolate')
                ph = interp1d(self.open_freq, self.open_phase, bounds_error=False, fill_value='extrapolate')

                self.open_amp   = amp(self.base_freq)
                self.open_phase = ph(self.base_freq)
                self.open_freq  = self.base_freq.copy()

                print(f'Warning: file {self.config.open_stem} had mismatched lengths or values. Interpolated amplitude and phase to match reference and discarding raw open frequencies.')

        # Normalize amplitudes to time-domain energy if not using FFT mode (normalization is automatic in FFT mode)
        if not self.config.use_FFT:
            samp_freq_E = np.sum(np.abs(self.samp_amp) ** 2)
            base_freq_E = np.sum(np.abs(self.base_amp) ** 2)

            samp_time_E = np.sum(np.abs(self.samp_signal) ** 2)
            base_time_E = np.sum(np.abs(self.base_signal) ** 2)

            samp_scalar = samp_time_E / samp_freq_E
            base_scalar = base_time_E / base_freq_E

            self.C = samp_scalar / base_scalar
            self.samp_amp = np.sqrt(samp_scalar) * self.samp_amp
            self.base_amp = np.sqrt(base_scalar) * self.base_amp

            if self.config.open_stem is not None:
                open_freq_E = np.sum(np.abs(self.open_amp) ** 2)
                open_time_E = np.sum(np.abs(self.open_signal) ** 2)

                open_scalar = open_time_E / open_freq_E

                self.C_open = open_scalar / base_scalar
                self.open_amp = np.sqrt(open_scalar) * self.open_amp

        self.mask = build_mask_from_bounds(self.base_freq, self.config.mask_bounds)

        if self.config.atm_correction:
            base_flags = self.find_atm_mask(
                self.base_freq, self.base_amp,
                threshold=self.config.atm_threshold,
                envelope_window_ghz=self.config.atm_envelope_window_ghz,
                prominence_db=self.config.atm_prominence_db,
                max_width_ghz=self.config.atm_max_width_ghz,
            )
            samp_flags = self.find_atm_mask(
                self.samp_freq, self.samp_amp,
                threshold=self.config.atm_threshold,
                envelope_window_ghz=self.config.atm_envelope_window_ghz,
                prominence_db=self.config.atm_prominence_db,
                max_width_ghz=self.config.atm_max_width_ghz,
            )

            # A dip flagged in only one channel must still be corrected in
            # BOTH: each channel's mask is found independently, so they can
            # disagree at a given frequency (noise trips one channel's shape
            # check but not the other's). Correcting only the flagged channel
            # then leaves the other raw at that exact frequency -- dividing a
            # corrected value by a raw one (or vice versa) fabricates a new
            # ratio spike/dip that was never in the raw data. Union the two
            # masks *before* filling (not after), so each channel's fill is
            # computed against the same shared mask rather than its own,
            # possibly narrower one.
            self.atm_mask = base_flags | samp_flags
            self.samp_atm_mask = self.atm_mask

            self.atm_envelope = self.fill_dips(
                self.base_freq, self.base_amp, self.atm_mask,
                envelope_window_ghz=self.config.atm_envelope_window_ghz,
            )
            self.samp_atm_envelope = self.fill_dips(
                self.samp_freq, self.samp_amp, self.samp_atm_mask,
                envelope_window_ghz=self.config.atm_envelope_window_ghz,
            )

            self.base_amp[self.atm_mask] = self.atm_envelope[self.atm_mask]
            self.samp_amp[self.samp_atm_mask] = self.samp_atm_envelope[self.samp_atm_mask]

        # 3) Spectra (unsmoothed + smoothed)

        if self.config.open_stem is None:
            self.normed_unsmoothed = (self.samp_amp / self.base_amp) ** 2
        else:
            denom = (self.base_amp - self.open_amp)
            numer = (self.samp_amp - self.open_amp)
            self.normed_unsmoothed = (numer / denom) ** 2

        self.boxnum = compute_boxnum_from_window_size(self.base_freq, self.config.window_size)

        self.bcbase = convolve_1d(self.base_amp, self.boxnum, window=self.config.window)
        self.bcsamp = convolve_1d(self.samp_amp, self.boxnum, window=self.config.window)

        if self.config.open_stem is None:
            self.normed = (self.bcsamp / self.bcbase) ** 2
        else:
            self.bcopen = convolve_1d(self.open_amp, self.boxnum, window=self.config.window)
            self.normed = ((self.bcsamp - self.bcopen) / (self.bcbase - self.bcopen)) ** 2

        # 4) Errors
        self.smoothing_err = rolling_std_error(self.normed_unsmoothed, self.boxnum)

        if self.config.include_mes_err:
            # noise estimate in high-frequency region
            hf = self.base_freq > self.config.noise_floor_min_ghz
            if np.any(hf):
                noise_amp = np.average(self.base_amp[hf])
            else:
                raise ValueError("No frequency points found above noise_floor_min_ghz for noise estimation.")
            
            if self.config.open_stem is None:
                self.mes_err = 2*self.bcsamp*noise_amp/(self.bcbase**2)*np.sqrt(1+(self.bcsamp/self.bcbase)**2)
            else:
                # Define R = N / D = (S-O)/(B-O)
                D = (self.bcbase - self.bcopen)
                N = (self.bcsamp - self.bcopen)
                R = N / D

                dR_dS = 1.0 / D
                dR_dB = -R / D
                dR_dO = (self.bcsamp - self.bcbase) / (D**2)

                var_R = (noise_amp**2) * (dR_dS**2 + dR_dB**2 + dR_dO**2)
                self.mes_err = 2.0 * np.abs(R) * np.sqrt(var_R)
        
            self.err = np.sqrt(self.smoothing_err**2 + self.mes_err**2)
        else:
            self.err = self.smoothing_err

    # ----------------------------
    # I/O helpers (internal)
    # ----------------------------
    def _spectr_path(self, stem: str) -> Path:
        return self.path_measurement / f"{stem}.spectr.csv"

    def _pulse_path(self, stem: str) -> Path:
        return self.path_measurement / f"{stem}.pulse.csv"

    @staticmethod
    def _load_spectr_csv(path: Path, cols: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        freq, pc, phase = np.genfromtxt(
            path, delimiter=",", skip_header=1, usecols=cols
        ).T
        m = ~np.isnan(freq)
        return freq[m], pc[m], phase[m]

    @staticmethod
    def _load_pulse_csv(path: Path, cols: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        t, sig = np.genfromtxt(
            path, delimiter=",", skip_header=1, usecols=cols
        ).T
        m = ~np.isnan(t)
        return t[m], sig[m]


    # ----------------------------
    # Atmospheric dip detection and interpolating
    # ----------------------------

    @staticmethod
    def _depth_mask(freq: np.ndarray, amp: np.ndarray, threshold: float, envelope_window_ghz: float):
        """Rolling-max envelope + fractional-depth threshold. See find_atm_mask."""
        df = np.median(np.diff(freq))
        win = max(int(round(envelope_window_ghz / df)), 3)
        if win % 2 == 0:
            win += 1
        envelope = maximum_filter1d(amp, size=win, mode="nearest")
        envelope = np.where(envelope > 0, envelope, np.nan)
        depth = 1.0 - amp / envelope
        depth = np.nan_to_num(depth, nan=1.0)
        return depth > threshold, envelope, win

    @staticmethod
    def _shape_mask(freq: np.ndarray, amp: np.ndarray, prominence_db: float, max_width_ghz: float, rel_height: float = 0.6):
        """
        Peak-finder-based dip shape check: atmospheric dips are sudden photocurrent drops
        followed by an immediate rise. Runs scipy.signal.find_peaks on
        -10*log10(amp) with a prominence floor (in dB) and a width cap (in GHz),
        so real features (fringes, harmonic passbands, filter cutoffs) are excluded.
        """
        df = np.median(np.diff(freq))
        neg_db = -10 * np.log10(np.maximum(amp, 1e-300))
        max_width_samples = max_width_ghz / df
        peaks, _ = find_peaks(neg_db, prominence=prominence_db, width=(0, max_width_samples))
        mask = np.zeros_like(amp, dtype=bool)
        if len(peaks) == 0:
            return mask
        _, _, left_ips, right_ips = peak_widths(neg_db, peaks, rel_height=rel_height)
        for l, r in zip(left_ips, right_ips):
            lo, hi = int(np.floor(l)), int(np.ceil(r)) + 1
            mask[max(lo, 0):min(hi, len(mask))] = True
        return mask

    @staticmethod
    def find_atm_mask(
        freq: np.ndarray,
        amp: np.ndarray,
        threshold: float = 0.2,
        envelope_window_ghz: float = 150.0,
        prominence_db: float = 2.0,
        max_width_ghz: float = 300.0,
    ) -> np.ndarray:
        """
        Data-driven line/dip finder combining two independent checks, both must
        agree for a bin to be flagged:

        1. Depth: `amp` drops more than `threshold` (fractional) below a rolling-max
        envelope over `envelope_window_ghz`
        2. Shape: >=2 dB prominent, narrower than 300 GHz i.e. *sudden drop, sudden rise*). 
        Requiring both matters: depth alone would also flag broad, gradual real features 
        (interference fringes, harmonic passbands) The shape check structurally protects these real features.

        This only detects -- see fill_dips() for computing replacement values.
        Detection and filling are deliberately separate: when correcting two
        channels (e.g. sample and reference) that must agree on *where* a dip is
        (see tf.py's dip_correction, which unions each channel's independently
        detected mask before filling either), filling must happen after that
        union, using each channel's own data at the shared mask -- not each
        channel's own, possibly narrower, mask.

        Returns a boolean mask, same shape as `amp`.
        """
        depth_mask, envelope, win = TeraFlashAnalyzer._depth_mask(freq, amp, threshold, envelope_window_ghz)
        shape_mask = TeraFlashAnalyzer._shape_mask(freq, amp, prominence_db, max_width_ghz)
        return depth_mask & shape_mask

    @staticmethod
    def fill_dips(
        freq: np.ndarray,
        amp: np.ndarray,
        mask: np.ndarray,
        envelope_window_ghz: float = 150.0,
    ) -> np.ndarray:
        """
        Compute replacement values for the bins flagged in `mask` (see
        find_atm_mask). Filling is two steps. First, bracket-interpolate log(amp) between
        the nearest unflagged ("good") neighbors on either side of every flagged
        point, giving an unbiased but locally noisy curve (in a dense cluster of
        many nearby lines, "good" data survives only in narrow, scattered islands,
        so which two points happen to bracket a given gap -- and therefore the
        interpolated value -- can jump around). Second, that interpolated curve is
        passed through a rolling *median* (robust to the flagged points still
        mixed in, unlike a mean) over `envelope_window_ghz`, which damps that
        jumpiness without reintroducing the max's upward bias.

        Returns `fill`, same shape as `amp`. `fill` equals `amp` at every point
        except where interpolation was possible; the caller applies it as
        `amp[mask] = fill[mask]`. Flagged points with no good data on one side
        (e.g. at the edge of the measured range) are left as measured.
        """
        df = np.median(np.diff(freq))
        win = max(int(round(envelope_window_ghz / df)), 3)
        if win % 2 == 0:
            win += 1

        fill = amp.copy()
        good = ~mask
        if np.any(mask) and np.any(good):
            log_interp = interp1d(
                freq[good], np.log(np.maximum(amp[good], 1e-300)),
                kind="linear", bounds_error=False, fill_value=np.nan,
            )
            interp_vals = log_interp(freq)
            fillable = np.isfinite(interp_vals)

            log_curve = np.where(fillable, interp_vals, np.log(np.maximum(amp, 1e-300)))
            log_curve_smooth = median_filter(log_curve, size=win, mode="nearest")

            mask_fillable = mask & fillable
            fill[mask_fillable] = np.exp(log_curve_smooth[mask_fillable])

        return fill

    # ----------------------------
    # Phase-difference cleaning (2*pi branch pick + mid-spectrum jump repair)
    # ----------------------------

    @staticmethod
    def _pick_branch_offset(
        freq: np.ndarray,
        phase_diff: np.ndarray,
        fit_range: Tuple[float, float],
        max_branch_cycles: int,
    ):
        """Zero-intercept 2*pi branch pick. See clean_phase_diff for the rationale.

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

    @staticmethod
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
        # dropped (see the documented limitation in clean_phase_diff).
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

    @staticmethod
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
        """Find and repair mid-spectrum 2*pi slips. See clean_phase_diff for the rationale.

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
        peaks, spans, suppressed = TeraFlashAnalyzer._detect_jump_candidates(
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

    @staticmethod
    def clean_phase_diff(
        freq: np.ndarray,
        theta: np.ndarray,
        theta_base: np.ndarray,
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
        Sample-minus-reference phase difference with its 2*pi ambiguity resolved.

        Pure array function (works for any freq/theta/theta_base triple, including
        TeraScanAnalyzer data); `get_phase_diff()` is the TeraFlashAnalyzer instance
        shortcut that feeds it `self.base_freq, self.samp_phase, self.base_phase`.

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
        label : name used in printed messages (defaults to "phase_diff" if not given; the
            `get_phase_diff()` wrapper passes the sample's file stem).
        return_info : if True, also return a dict with 'offset', 'offset_source' ('auto' or
            'manual'), 'intercept', 'offset_at_edge', and 'jumps' (see _repair_phase_jumps).

        Returns
        -------
        (phase_diff, n_ulim) normally, or (phase_diff, n_ulim, info) if `return_info=True`.
        n_ulim (GHz) is the frequency up to which phase_diff (and any n derived from it)
        should be trusted. Never modifies `theta`/`theta_base`.
        """
        if unwrap:
            phase_diff = np.unwrap(np.unwrap(theta) - np.unwrap(theta_base))
        else:
            phase_diff = np.asarray(theta, dtype=float) - np.asarray(theta_base, dtype=float)
            if absolute:
                phase_diff = np.abs(phase_diff)

        name = label if label is not None else "phase_diff"
        if offset is None:
            offset, intercept, at_edge = TeraFlashAnalyzer._pick_branch_offset(
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

        n_ulim, jumps = TeraFlashAnalyzer._repair_phase_jumps(
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

    # ----------------------------
    # FFT helper (LabVIEW-like)
    # ----------------------------

    @staticmethod
    def _modblackmanwindow(n: int, relativewidth_start: float, relativewidth_end: float, alpha: float = 0.16) -> np.ndarray:
        """Modified Blackman window matching the TF LabVIEW-style implementation."""
        w = np.ones(n, dtype=float)

        width_start = int(np.floor(relativewidth_start * n))
        width_end = int(np.floor(relativewidth_end * n))

        if width_start >= 1:
            nrel = np.arange(0, width_start) / (2 * width_start - 1)
            w[:width_start] = 0.5 * (
                1 - alpha - np.cos(2 * np.pi * nrel) + alpha * np.cos(4 * np.pi * nrel)
            )

        if width_end >= 1:
            nrel = np.arange(width_end, 2 * width_end) / (2 * width_end - 1)
            w[-width_end:] = 0.5 * (
                1 - alpha - np.cos(2 * np.pi * nrel) + alpha * np.cos(4 * np.pi * nrel)
            )

        return w

    @classmethod
    def _spectr_from_pulse(
        cls,
        time: np.ndarray,
        signal: np.ndarray,
        *,
        dfreq: float | None,
        rel_start: float,
        rel_end: float,
        norm: Literal["ortho"] | None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute (freq_GHz, amplitude) from a time-domain pulse.

        Applies LabVIEW-like modified Blackman apodization and (optionally) symmetric
        zero-padding to reach a target frequency resolution.
        """
        if len(time) != len(signal):
            raise ValueError("time and signal must have the same length")
        if len(time) < 2:
            raise ValueError("Need at least 2 time samples to compute FFT")

        dt = time[1] - time[0]
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Non-finite or non-positive time step")

        x = np.asarray(signal, dtype=float)
        win = cls._modblackmanwindow(len(x), rel_start, rel_end, alpha=0.16)
        x = x * win

        if dfreq is not None:
            # Native frequency resolution in GHz (legacy definition)
            dfreq0 = 1000.0 / dt / len(x)
            scalar = dfreq0 / dfreq
            if scalar < 1:
                scalar = 1
            pad_each = int((scalar - 1) * len(x) / 2)
            if pad_each > 0:
                x = np.pad(x, pad_each, mode="constant", constant_values=0.0)

        freq = 1000.0 * np.fft.rfftfreq(len(x), dt)
        X = np.fft.rfft(x, norm=norm)
        amp = np.abs(X)
        phase = np.unwrap(np.angle(X))
        return freq, amp, phase
    
    # ----------------------------
    # TeraFlash specific getters for TD data
    # ----------------------------

    def get_base_pulse(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.base_time, self.base_signal

    def get_sample_pulse(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.samp_time, self.samp_signal

    def get_phases(self):
        return self.samp_phase, self.base_phase

    def get_phase_diff(self, **kwargs):
        """(phase_diff, n_ulim) [, info] for this sample vs. its reference.

        Thin instance wrapper over clean_phase_diff(self.base_freq, self.samp_phase,
        self.base_phase, **kwargs) -- see that staticmethod for every keyword argument,
        its default, and the rationale behind the defaults. `label` defaults to this
        instance's sample stem if not given.
        """
        kwargs.setdefault('label', self.samp)
        return self.clean_phase_diff(self.base_freq, self.samp_phase, self.base_phase, **kwargs)

    def get_norm_factor(self):
        return self.C
    
    def get_open_norm_factor(self):
        if self.config.open_stem is None:
            raise ValueError("Open normalization factor is not available when open_stem is None.")
        return self.C_open

TeraFlashAnalyzer.get_spectra = _utils.get_spectra
TeraFlashAnalyzer.get_smoothed_spectra = _utils.get_smoothed_spectra
TeraFlashAnalyzer.get_sample_current = _utils.get_sample_current
TeraFlashAnalyzer.get_base_current = _utils.get_base_current
TeraFlashAnalyzer.get_open_current = _utils.get_open_current
TeraFlashAnalyzer.get_mask = _utils.get_mask