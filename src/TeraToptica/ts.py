# src/TeraToptica/ts.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import hilbert

from . import utils as _utils
from .utils import (
    build_mask_from_bounds,
    convolve_1d,
    compute_boxnum_from_window_size,
    rolling_std_error,
)

@dataclass(frozen=True)
class FourierCutConfig:
    """Legacy-style Fourier cut on the spectrum.

    This reproduces your original behavior:
      1) interpolate through masked values,
      2) take an IFFT of the spectrum (treated as a function of frequency),
      3) zero "time" components beyond a cutoff,
      4) FFT back.

    Units note
    ----------
    Your spectrum is sampled on a frequency grid in GHz. The conjugate axis of an FFT
    over frequency has units of 1/GHz, i.e. nanoseconds (ns), because 1 GHz = 1/ns.

    So a 200 ps cutoff corresponds to 0.2 ns.
    """

    enabled: bool = False
    t_cut_ps: float = 200.0  # ps (legacy choice)


@dataclass(frozen=True)
class TeraScanConfig:
    """Configuration for TeraScan reflection/transmission reduction."""

    # Smoothing
    window: Literal[
        "boxcar", "bartlett", "blackman", "gaussian", "hanning", "hamming", "median"
    ] = "hanning"
    window_size: float = 1.0  # GHz

    # Masking
    mask_bounds: Optional[Sequence[Tuple[float, float]]] = None  # GHz ranges

    # Error model
    include_mes_err: bool = True
    i_dc: float = 14e-3  # same units as the signal column (legacy 14 pA/nA-scale term)

    # Whether the fiber stretcher (StrMod) was used (i.e., amplitude/phase already separated)
    fiber_stretcher: bool = True

    # Optional open-subtraction: ((sample-open)/(base-open))^2
    open_stem: Optional[str] = None

    # Optional ripple suppression
    fourier_cut: FourierCutConfig = FourierCutConfig()

    # Header names (dynamic column selection)
    header_freq_act: str = "Frequency Act (GHz)"
    header_freq_set: str = "Frequency Set (GHz)"
    header_pc: str = "THz Photocurrent (nA)"
    header_amp: str = "Amplitude (via StrMod)"
    header_phase: str = "Phase (via StrMod)"

# ----------------------------
# Analyzer
# ----------------------------

class TeraScanAnalyzer:
    """TeraScan reflection/transmission analyzer.

    This refactor unifies the legacy classes:
      - rt_str        -> fiber_stretcher=True
      - rt_unstr      -> fiber_stretcher=False (Hilbert amplitude from photocurrent)
      - ro_class      -> open_stem provided
      - ts_fourier    -> fourier_cut.enabled=True (200 ps cut)

    Naming
    ------
    Matches tf.py conventions:
      - base frequency axis: `self.base_freq`
      - sample frequency axis: `self.samp_freq`
      - base "current/amplitude" array: `self.base_pc`
      - sample "current/amplitude" array: `self.samp_pc`
      - phases: `self.base_phase`, `self.samp_phase`

    Internal variable name uses `samp`; public API uses `sample`.
    """

    def __init__(
        self,
        path_measurement: str | Path,
        sample: str,
        base: str,
        *,
        config: TeraScanConfig = TeraScanConfig(),
    ):
        self.path_measurement = Path(path_measurement)
        self.samp = sample
        self.base = base
        self.config = config

        self._analyze_data()

    def _analyze_data(self) -> None:
        # 1) Load arrays + map columns dynamically from header
        samp_path = self._data_path(self.samp)
        base_path = self._data_path(self.base)

        samp_header = self._read_header(samp_path)
        base_header = self._read_header(base_path)

        samp_arr = self._load_txt(samp_path)
        base_arr = self._load_txt(base_path)

        # Select columns based on whether the fiber stretcher was used.
        if self.config.fiber_stretcher:
            samp_fcol, samp_ampcol, samp_phcol = self._select_columns(samp_header)
            base_fcol, base_ampcol, base_phcol = self._select_columns(base_header)

            self.samp_freq = samp_arr[:, samp_fcol]
            self.base_freq = base_arr[:, base_fcol]

            self.samp_amp = samp_arr[:, samp_ampcol]
            self.base_amp = base_arr[:, base_ampcol]

            self.samp_phase = samp_arr[:, samp_phcol]
            self.base_phase = base_arr[:, base_phcol]

            self.samp_pc = self.samp_amp * np.cos(self.samp_phase)            
            self.base_pc = self.base_amp * np.cos(self.base_phase)
        else:
            samp_fcol, samp_pccol = self._select_columns(samp_header)
            base_fcol, base_pccol = self._select_columns(base_header)

            self.samp_freq = samp_arr[:, samp_fcol]
            self.base_freq = base_arr[:, base_fcol]

            self.samp_pc = samp_arr[:, samp_pccol]
            self.base_pc = base_arr[:, base_pccol]

        # Optional open (RO-style normalization)
        if self.config.open_stem is not None:
            open_path = self._data_path(self.config.open_stem)
            open_header = self._read_header(open_path)
            open_arr = self._load_txt(open_path)
            if self.config.fiber_stretcher:
                open_fcol, open_ampcol, open_phcol = self._select_columns(open_header)

                self.open_freq = open_arr[:, open_fcol]
                self.open_amp = open_arr[:, open_ampcol]
                self.open_phase = open_arr[:, open_phcol]

                self.open_pc = self.open_amp * np.cos(self.open_phase)
            else:
                open_fcol, open_pccol = self._select_columns(open_header)

                self.open_freq = open_arr[:, open_fcol]
                self.open_pc = open_arr[:, open_pccol]

        # 2) Choose a common frequency grid (use base as reference)
        #    and interpolate sample/open/phase if needed.
        if (len(self.samp_freq) != len(self.base_freq)) or (not np.allclose(self.samp_freq, self.base_freq)):            
            if self.config.fiber_stretcher:
                amp = interp1d(self.samp_freq, self.samp_amp, bounds_error=False, fill_value="extrapolate")
                ph  = interp1d(self.samp_freq, self.samp_phase, bounds_error=False, fill_value='extrapolate')
                pc  = interp1d(self.samp_freq, self.samp_pc, bounds_error=False, fill_value='extrapolate')

                self.samp_amp   = amp(self.base_freq)
                self.samp_pc    = pc(self.base_freq)
                self.samp_phase = ph(self.base_freq)
            else:
                pc = interp1d(self.samp_freq, self.samp_pc, bounds_error=False, fill_value="extrapolate")
                self.samp_pc    = pc(self.base_freq)

            self.samp_freq = self.base_freq.copy()
            print(f"Warning: file {self.samp} had mismatched lengths or values. Interpolated to match reference and discarding raw sample frequencies.")

        if self.config.open_stem is not None:
            if (len(self.open_freq) != len(self.base_freq)) or (not np.allclose(self.open_freq, self.base_freq)):
                if self.config.fiber_stretcher:
                    amp = interp1d(self.open_freq, self.open_amp, bounds_error=False, fill_value='extrapolate')
                    ph  = interp1d(self.open_freq, self.open_phase, bounds_error=False, fill_value='extrapolate')
                    pc  = interp1d(self.open_freq, self.open_pc, bounds_error=False, fill_value='extrapolate')

                    self.open_amp   = amp(self.base_freq)
                    self.open_pc    = pc(self.base_freq)
                    self.open_phase = ph(self.base_freq)
                else:
                    pc = interp1d(self.open_freq, self.open_pc, bounds_error=False, fill_value="extrapolate")
                    self.open_pc    = pc(self.base_freq)

                self.open_freq = self.base_freq.copy()
                print(f"Warning: file {self.config.open_stem} had mismatched lengths or values. Interpolated to match reference and discarding raw open frequencies.")

        # 3) Convert raw signal -> amplitude + phase depending on stretcher
        if not self.config.fiber_stretcher:
            N = len(self.base_freq)
            base_padded = np.pad(self.base_pc, N, mode="reflect")
            samp_padded = np.pad(self.samp_pc, N, mode="reflect")

            base_analytic = hilbert(base_padded)[N:-N]
            samp_analytic = hilbert(samp_padded)[N:-N]

            self.base_amp = np.abs(base_analytic)
            self.samp_amp = np.abs(samp_analytic)

            self.base_phase = np.unwrap(np.angle(base_analytic))
            self.samp_phase = np.unwrap(np.angle(samp_analytic))

            if self.config.open_stem is not None:
                open_padded = np.pad(self.open_pc, N, mode="reflect")
                open_analytic = hilbert(open_padded)[N:-N]

                self.open_amp = np.abs(open_analytic)
                self.open_phase = np.unwrap(np.angle(open_analytic))

        # 4) Mask
        self.mask = build_mask_from_bounds(self.base_freq, self.config.mask_bounds)

        # 5) Build unsmoothed normalized spectrum
        if self.config.open_stem is None:
            self.normed_unsmoothed = (self.samp_amp / self.base_amp) ** 2
        else:
            denom = (self.base_amp - self.open_amp)
            numer = (self.samp_amp - self.open_amp)
            self.normed_unsmoothed = (numer / denom) ** 2

        # 6) Smooth
        self.boxnum = compute_boxnum_from_window_size(self.base_freq, self.config.window_size)

        bcbase = convolve_1d(self.base_amp, self.boxnum, window=self.config.window)
        bcsamp = convolve_1d(self.samp_amp, self.boxnum, window=self.config.window)

        if self.config.open_stem is None:
            self.normed = (bcsamp / bcbase) ** 2
        else:
            bcopen = convolve_1d(self.open_amp, self.boxnum, window=self.config.window)
            self.normed = ((bcsamp - bcopen) / (bcbase - bcopen)) ** 2

        # 7) Optional Fourier cut (200 ps)
        if self.config.fourier_cut.enabled:
            self.normed = self._fourier_cut(self.base_freq, self.normed, mask=self.mask, t_cut_ps=self.config.fourier_cut.t_cut_ps)

        # 8) Error estimates
        smoothing_err = rolling_std_error(self.normed_unsmoothed, self.boxnum)

        if self.config.include_mes_err:
            if self.config.open_stem is None:
                # y = (S/B)^2 with sigma_S=sigma_B=i_dc
                mes_err = 2*bcsamp*self.config.i_dc/(bcbase**2)*np.sqrt(1+(bcsamp/bcbase)**2)
            else:
                # Clean propagation with Sigma_S = Sigma_B = Sigma_O = i_dc
                # Define R = N / D = (S-O)/(B-O)
                D = (bcbase - bcopen)
                N = (bcsamp - bcopen)
                R = N / D

                dR_dS = 1.0 / D
                dR_dB = -R / D
                dR_dO = (bcsamp - bcbase) / (D**2)

                var_R = (self.config.i_dc**2) * (dR_dS**2 + dR_dB**2 + dR_dO**2)
                mes_err = 2.0 * np.abs(R) * np.sqrt(var_R)

            self.err = np.sqrt(smoothing_err**2 + mes_err**2)
        else:
            self.err = smoothing_err

    # ----------------------------
    # Internal helpers
    # ----------------------------

    def _data_path(self, stem: str) -> Path:
        """Resolve a TeraScan file path.

        Standardize on the `.txt` extension.
        - If caller passes a name without suffix, append `.txt`.
        - If caller passes a full filename with suffix, keep it.
        """
        p = self.path_measurement / stem
        if p.suffix == "":
            p = p.with_suffix(".txt")
        return p

    @staticmethod
    def _read_header(path: Path) -> list[str]:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            line = f.readline().strip("\n\r")
        # TeraScan exports are tab-separated
        return [c.strip() for c in line.split("\t") if c.strip() != ""]

    @staticmethod
    def _load_txt(path: Path) -> np.ndarray:
        # Your legacy code used skiprows=1 and tab separation.
        return np.loadtxt(path, skiprows=1, delimiter="\t")

    def _select_columns(self, header: list[str]):
        """Select required columns based on `fiber_stretcher`.

        Returns
        -------
        If fiber_stretcher=True:
            (freq_col, amp_col, phase_col)
        If fiber_stretcher=False:
            (freq_col, pc_col)

        This is intentionally strict: missing or unexpected headers should raise.
        """

        # Frequency: always prefer actual frequency
        if self.config.header_freq_act in header:
            freq_col = header.index(self.config.header_freq_act)
        else:
            freq_col = header.index(self.config.header_freq_set)

        if self.config.fiber_stretcher:
            amp_col = header.index(self.config.header_amp)
            phase_col = header.index(self.config.header_phase)
            return freq_col, amp_col, phase_col

        pc_col = header.index(self.config.header_pc)
        return (freq_col, pc_col)

    @staticmethod
    def _fourier_cut(
        freq_ghz: np.ndarray,
        y: np.ndarray,
        *,
        mask: np.ndarray,
        t_cut_ps: float,
    ) -> np.ndarray:
        """Reproduce the legacy Fourier cut with a time-domain threshold.

        Implementation:
        - Interpolate masked points in frequency space.
        - IFFT over frequency samples.
        - Zero components with |t| > t_cut.
        - FFT back and return the real part.

        Here, `t` is the conjugate axis of frequency (in ns).
        """
        y = np.asarray(y, dtype=float)
        freq_ghz = np.asarray(freq_ghz, dtype=float)

        df = np.median(np.diff(freq_ghz))
        if not np.isfinite(df) or df <= 0:
            return y

        x = freq_ghz
        good = (~mask) & np.isfinite(y) & np.isfinite(x)
        if np.sum(good) < 2:
            return y

        interp_func = interp1d(x[good], y[good], bounds_error=False, fill_value="extrapolate")
        y_interp = interp_func(x)

        Yt = np.fft.ifft(y_interp)
        t_ns = np.fft.fftfreq(len(y_interp), d=df)  # 1/GHz = ns

        t_cut_ns = float(t_cut_ps) * 1e-3  # ps -> ns
        Yt[np.abs(t_ns) > t_cut_ns] = 0.0

        return np.real(np.fft.fft(Yt))

    # ----------------------------
    # Minimal native getters (TS-specific)
    # ----------------------------

    def get_mod_currents(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (samp_pc, base_pc) which includes phase modulation.
        If fiber_stretcher=False, this is the raw signal. 
        If fiber_stretcher=True, this is amplitude*cos(phase).
        """
        return self.samp_pc, self.base_pc

    def get_currents(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (sample_amp, base_amp).
        If fiber_stretcher=True, amplitude is read from file.
        If fiber_stretcher=False, amplitude is derived from the analytic Hilbert signal.
        """
        return self.samp_amp, self.base_amp

    def get_phase(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (sample_phase, base_phase).
        If fiber_stretcher=True and phase column exists, phase is read from file.
        If fiber_stretcher=False, phase is derived from the analytic Hilbert signal.
        """
        return self.samp_phase, self.base_phase

# Attach legacy-style getters from utils for backwards compatibility
TeraScanAnalyzer.get_spectra = _utils.get_spectra
TeraScanAnalyzer.get_smoothed_spectra = _utils.get_smoothed_spectra
TeraScanAnalyzer.get_sample_current = _utils.get_sample_current
TeraScanAnalyzer.get_base_current = _utils.get_base_current
TeraScanAnalyzer.get_open_current = _utils.get_open_current
TeraScanAnalyzer.get_mask = _utils.get_mask