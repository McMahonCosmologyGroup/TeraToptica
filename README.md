# TeraToptica

[![CI](https://github.com/McMahonCosmologyGroup/TeraToptica/actions/workflows/ci.yml/badge.svg)](https://github.com/McMahonCosmologyGroup/TeraToptica/actions/workflows/ci.yml)
[![Docs](https://readthedocs.org/projects/teratoptica/badge/?version=latest)](https://teratoptica.readthedocs.io)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18438202.svg)](https://doi.org/10.5281/zenodo.18438202)

A Python analysis package for reflection and transmission data from Toptica Photonics' **TeraFlash** (pulsed,
time-domain) and **TeraScan** (CW, frequency-domain) THz spectrometers. It reads the instruments' exported files,
normalizes a sample measurement by a reference (an open beam for transmission, a mirror plate for reflection), and
returns the power ratio with smoothing, masking, error bars and phase.

The authors of this code are not affiliated with Toptica Photonics.

Documentation: <https://teratoptica.readthedocs.io>. It includes this guide, a guide to phase and refractive index, and the API reference.

## Installation

```bash
git clone https://github.com/McMahonCosmologyGroup/TeraToptica.git
cd TeraToptica
pip install -e .
```

Requires Python ≥ 3.9 with numpy, scipy, matplotlib and astropy. With an editable install (`-e`), `git pull`
updates the package in place.

Results can change between versions (see [Changes that affect results](#changes-that-affect-results)), so record
`TeraToptica.__version__` with your data products. To reproduce or pin a result, install a tagged release:

```bash
pip install "git+https://github.com/McMahonCosmologyGroup/TeraToptica.git@v0.1.0"
```

## Quick start

`example/examples.ipynb` runs both instruments on the bundled data in `example/data/`.

### TeraFlash

TeraFlash measurements are TF5 exports, read as pairs of files sharing a stem: `<stem>.pulse.csv` (time domain)
and `<stem>.spectr.csv` (frequency domain). Pass stems, not file names.

```python
from TeraToptica import TeraFlashAnalyzer as TFA, TeraFlashConfig as TFC

mask_bounds = ((0, 150), (545, 570), (740, 765), (1500, 10000))  # GHz ranges to flag

cfg = TFC(window='boxcar', window_size=15, mask_bounds=mask_bounds, atm_correction=True)

tfa = TFA(path_measurement='example/data/TeraFlash',
          sample='al_LF_575-775ps_1000avgs',
          base='open_575-775ps_1000avgs',      # open beam (transmission) or plate (reflection)
          config=cfg)

f, t, ut, mask, err, boxnum = tfa.get_smoothed_spectra()
```

| Returned | Meaning |
|---|---|
| `f` | frequency [GHz], on the reference's grid |
| `t` | smoothed power ratio (sample/base)², i.e. transmittance or reflectance |
| `ut` | unsmoothed power ratio |
| `mask` | `True` inside `mask_bounds` |
| `err` | 1σ error: rolling-std smoothing error and noise-floor measurement error, in quadrature |
| `boxnum` | smoothing window in samples. `f[::boxnum]` gives independent bins. |

Other accessors:
- `get_spectra()` returns `(f, ut)`.
- `get_sample_current()` / `get_base_current()` return `(f, photocurrent)`.
- `get_phases()` returns `(theta_sample, theta_base)`.
- `get_sample_pulse()` / `get_base_pulse()` return `(t, signal)`.
- `get_pulse_delay()` returns the sample-minus-reference delay in ps.
- `get_norm_factor()` returns the energy normalization.

The amplitudes are rescaled so their frequency-domain energy matches the time-domain pulse energy. If the sample's
frequency grid differs from the reference's, the sample is interpolated onto the reference grid.

### TeraScan

TeraScan measurements are `.txt` exports. The sample and base stems are positional.

```python
from TeraToptica import TeraScanAnalyzer as TSA, TeraScanConfig as TSC

cfg = TSC(window='boxcar', window_size=2, mask_bounds=mask_bounds, fiber_stretcher=True)
ts = TSA('example/data/TeraScan', 'LPE_5o7cm^-1_90-180ghz', 'open_90-180ghz', config=cfg)

f, t, ut, mask, err, boxnum = ts.get_smoothed_spectra()
amp_sample, amp_base = ts.get_currents()
```

Columns are selected from the file header, and the header names can be changed in `TeraScanConfig`. Set
`fiber_stretcher=False` for data taken without the fiber stretcher; the amplitude then comes from a Hilbert
transform of the photocurrent. `fourier_cut=FourierCutConfig(enabled=True)` suppresses ripple.

## Configuration

Both config classes are frozen dataclasses. Build a new one to change a setting.

**`TeraFlashConfig`**

| Field | Default | Purpose |
|---|---|---|
| `window`, `window_size` | `'boxcar'`, `15.0` GHz | Smoothing kernel and its width. For `'gaussian'` the width is a FWHM. |
| `mask_bounds` | `None` | `((lo, hi), ...)` GHz ranges to flag in `mask`, e.g. water lines and unreliable band edges. |
| `include_mes_err` | `True` | Add the noise-floor measurement error to `err`. |
| `noise_floor_min_ghz` | `5500.0` | Frequencies above this value estimate the noise floor. The data must extend past it. |
| `atm_correction` | `False` | Detect atmospheric absorption dips and fill them in both channels at the same points. |
| `atm_restrict_to_mask_bounds` | `False` | Only fill dips inside `mask_bounds`. This prevents false positives elsewhere, but misses lines outside the masks. |
| `atm_threshold`, `atm_prominence_db`, `atm_max_width_ghz`, `atm_envelope_window_ghz`, `atm_min_snr` | see `tf.py` | Detection gates. Detection runs on the reference beam only, so a real feature of the sample is never filled. |
| `layout` | `'separate_files'` | `'separate_files'`: sample and base are separate stems. `'reference_included'`: one stem whose reference channel is the base. |
| `open_stem` | `None` | Optional open-subtraction, `((sample − open)/(base − open))²`. Only with `'separate_files'`. |
| `use_FFT`, `fft` | `False`, `FFTConfig()` | Build spectra by FFT of the pulse, with a modified Blackman apodization, zero-padding (`dfreq`) and time cuts (`tmin_ps`, `tmax_ps`), instead of reading `.spectr.csv`. |

**`TeraScanConfig`** defaults to `window='hanning'` and `window_size=1.0` GHz. It also has `mask_bounds`,
`include_mes_err`, `i_dc` (the dark-current error term, `14e-3`), `fiber_stretcher`, `open_stem`, `fourier_cut` and
the header-name fields.

## Phase and refractive index

`get_phase_diff()` returns `(phase_diff, n_ulim)`: the sample-minus-reference phase with its 2π ambiguity resolved,
and the frequency [GHz] up to which it can be trusted.

```python
import numpy as np, scipy.constants as const

f, _ = tfa.get_spectra()  # phase_diff is on the reference's frequency grid
phase_diff, n_ulim = tfa.get_phase_diff()
thickness = 7e-3  # m, the sample's measured thickness
n = 1 + phase_diff * const.c / (2 * np.pi * f * 1e9) / thickness
good = f < n_ulim
```

- **Branch choice:** on TeraFlash data the 2π branch is set by the measured pulse delay (`branch_method='pulse'`).
  TeraScan uses a linear fit (`'linear'`) over `fit_range`. By default that is (200, 500) GHz, or the data's own band
  when the data doesn't reach it (TeraScan's ~90–180 GHz).
- **Phase source:** `use_FFT=True` takes the phase from the FFT of the averaged pulse rather than the instrument's
  exported spectrum. The exported phase is built from a running average of per-trace phases, and can carry a
  constant offset on low-signal data. Prefer `use_FFT=True` for n.
- **Late pulses:** a pulse arriving more than 1/(2Δf) after the start of its time window (100 ps on the usual
  5 GHz grid) aliases bin-by-bin phase unwrapping. `get_phase_diff` detects this, re-unwraps the phase about the
  pulse delay, and reports `info['dealiased'] = True`.
- **Diagnostics:** pass `return_info=True` and `verbose=True` to check the branch choice on unfamiliar samples.
- **Deprecated:** `absolute=True` is ignored on the pulse path. Don't take `abs()` of the phase difference yourself.

## Changes that affect results

The full list, by version, is in [CHANGELOG.md](CHANGELOG.md). Re-reduce data products made with older
versions if these changes matter to them. These are the changes in 0.1.0:

| Commit | Date | Change |
|---|---|---|
| v0.1.0 | 2026-10-15 | Phases of pulses late in their time window (> 1/(2Δf), 100 ps on a 5 GHz grid) are de-aliased. These used to give the wrong slope; all other data is unchanged. TeraScan `get_phase_diff()` works with the default `fit_range`. |
| `0597bdd` | 2026-10-15 | The phase difference is oriented by the pulse delay instead of `abs()`. Thin, low-index samples no longer come out with n < 1. |
| `a74b961` | 2026-10-15 | Pulse-delay branch pick for `get_phase_diff`; adds `get_pulse_delay()`. |
| `97d3087` | 2026-10-13 | FFT-mode phase sign flipped to match the `.spectr.csv` convention. The default `use_FFT=False` path is unchanged. |
| `03c8fdb` | 2026-10-02 | Atmospheric-dip detection runs on the reference beam only, adds an SNR gate, and adds `atm_restrict_to_mask_bounds`. |
| `913558e` | 2026-08-19 | Gaussian smoothing width fixed: `window_size` is now a FWHM, not a standard deviation. |

## Tests

```bash
pip install -e ".[test]"
pytest
```

The suite takes a few seconds. It covers the array helpers, both analyzers on the bundled example data, and
synthetic slabs of known index on both phase paths. That includes regression tests for the FFT phase sign, the
phase orientation (the old `abs()` failure) and the branch diagnostics. GitHub Actions runs the suite on Python
3.10–3.13, and builds the docs, on every push and pull request.

## Contributing and support

Report bugs and questions as [GitHub issues](https://github.com/McMahonCosmologyGroup/TeraToptica/issues); a
minimal example (the stems and config you used) helps. For changes, open a pull request against `main`:

- add a test for any bug you fix;
- run `pytest` before pushing;
- add an entry to `CHANGELOG.md`, under "Results change" if outputs can differ.

Releases are tagged `vX.Y.Z`. The version is bumped for every change that alters results.

Build the docs locally with `pip install -e ".[docs]"` and `sphinx-build docs docs/_build/html`.

## Citing

If you use TeraToptica in a publication, please cite it via Zenodo:
[10.5281/zenodo.18438202](https://doi.org/10.5281/zenodo.18438202). `CITATION.cff` has the citation metadata, and GitHub
shows it under "Cite this repository". Please also give the version you used.

## License

MIT. See `LICENSE`.
