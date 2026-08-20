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