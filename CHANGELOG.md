# Changelog

Each release lists the changes that can alter results ("Results change") separately from
everything else. Record the TeraToptica version with your data products (`TeraToptica.__version__`).

## [0.1.0] - 2026-10-15

First tagged release. Everything below changed since the package was first published (0.0.0, 2026-01-30).

### Results change
- **Late-pulse de-aliasing**: a pulse arriving more than 1/(2Δf) after the start of its record (100 ps on a 5 GHz
  grid) advances the phase by more than π per bin. Both phase sources then unwrap it with the wrong slope. When
  the slope of the phase difference misses 2πτ by more than half an alias step, `get_phase_diff` now re-unwraps
  it about the delay model 2πfτ and reports `info['dealiased']`. Correctly unwrapped data is unchanged: of 110
  project measurements on both phase paths, none were affected. The bundled alumina example was aliased.
- **Phase orientation** (`0597bdd`): `get_phase_diff` orients the phase difference by the measured pulse delay
  instead of taking `abs()`. Thin, low-index samples (e.g. HD30 foams) no longer come out mirrored to n < 1.
  `absolute=True` is deprecated: it is ignored on the pulse path and raises a `FutureWarning`.
- **Branch pick** (`a74b961`): on TeraFlash data, the 2π branch of the phase difference is set by the measured
  pulse delay (`branch_method='pulse'`, now the default) rather than by extrapolating a line to f = 0. It adds
  the diagnostics `offset_frac`, `offset_spread`, `offset_ambiguous` and `offset_inconsistent` (with
  `return_info=True`), and `get_pulse_delay()`. TeraScan still uses `branch_method='linear'`.
- **FFT phase sign** (`97d3087`): in `use_FFT=True` mode the phase now has the same sign convention as the
  `.spectr.csv` export. Before, n from FFT-mode phase was mirrored about 1. The default mode is unchanged.
- **Atmospheric-line correction** (`03c8fdb`): dips are detected on the reference beam only, from a smoothed copy,
  with an SNR gate and a two-sided envelope. New option `atm_restrict_to_mask_bounds`.
- **Gaussian smoothing** (`913558e`): `window_size` is now the FWHM for `window='gaussian'`, as for the other
  windows. Before, it was used as a standard deviation, which smoothed over a width about 2.4 times larger.

### Added
- `get_phase_diff` with jump repair and the trusted-band limit `n_ulim`, shared by both instruments (`c2b3339`, `d5d4d91`).
- TeraFlash time cuts before the FFT (`FFTConfig.tmin_ps`, `tmax_ps`), open-beam subtraction (`open_stem`), and
  atmospheric-dip interpolation (`atm_correction`).
- TeraScan trims sample, base and open measurements to their common frequency range.
- A pytest suite (`tests/`), GitHub Actions CI, Sphinx documentation, `CITATION.cff`.

### Changed
- Packaging moved to `pyproject.toml`. The unused `numba` dependency was dropped. Install as before with `pip install -e .`.

### Fixed
- TeraScan `get_phase_diff()` raised with its defaults: the linear pick's default `fit_range=(200, 500)` GHz lies
  above the TeraScan band. `fit_range=None` (the new default) uses (200, 500) when the data covers it and the data's
  own band otherwise. An explicit `fit_range` behaves as before.
- `get_open_current` returned the base current (`a79241f`).
- TeraScan `get_phase` renamed to `get_phases`, consistent with TeraFlash (`7348ce1`).
