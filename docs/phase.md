# Phase and refractive index

`get_phase_diff()` returns the sample-minus-reference phase difference with its 2π ambiguity resolved, and
`n_ulim`, the frequency [GHz] up to which it can be trusted. For a slab of thickness `w` [m]:

```python
import numpy as np, scipy.constants as const

f, _ = tfa.get_spectra()
phase_diff, n_ulim, info = tfa.get_phase_diff(return_info=True)
n = 1 + phase_diff * const.c / (2 * np.pi * f * 1e9 * w)
good = (f >= 150) & (f < n_ulim) & ~tfa.get_mask()
```

The group index needs no phase at all, and is a useful cross-check: `n_g = 1 + const.c * tfa.get_pulse_delay() * 1e-12 / w`.
For weakly dispersive samples, `n` and `n_g` should agree to a few times 10⁻³.

## How the branch is chosen

The phase difference is only known up to whole cycles. On TeraFlash data, `get_phase_diff` uses the measured
pulse delay τ to choose them (`branch_method='pulse'`):

1. **Orient** the difference so it follows the delay. This is normally a no-op, since both phase sources use the
   same sign convention, but it protects against data with the opposite convention. It replaces the old
   `absolute=True` workaround. Don't take `abs()` yourself: it folds a small or offset phase, and thin low-index
   samples come out with n < 1.
2. **De-alias** if needed (see below), then **repair** mid-band 2π slips (jump repair), and set `n_ulim` below any
   jump it can't reconcile.
3. **Pick the branch:** take the whole number of cycles between the phase and 2πfτ, as a power-weighted median.

TeraScan has no pulses, so it uses `branch_method='linear'`: the branch that makes a straight-line fit over
`fit_range` extrapolate closest to zero at f = 0. The default (`fit_range=None`) is (200, 500) GHz when the data
covers it, and otherwise the data's own usable band, e.g. about 90–180 GHz for TeraScan.

## Diagnostics

With `return_info=True`, check these on any unfamiliar sample:

| Key | Meaning | Action |
|---|---|---|
| `offset_frac` | Part of a cycle left over after rounding to a branch. Near 0 for clean samples (≤ 0.12 on HDPE and foams). | — |
| `offset_ambiguous` | `abs(offset_frac) > 0.35`: the neighbouring branch fits almost as well. | Compare n with `n_g`. Try `use_FFT=True` (see below). |
| `offset_spread` / `offset_inconsistent` | Spread of the residual across the band. Above 0.5 cycles, no single branch fits everywhere. | Usually a real 2π slip through a low-transmission band. Trust n only where the signal is strong. |
| `orientation` | `-1` if the difference was flipped. | Expected only for data with the opposite sign convention. |
| `dealiased` | `True` if the phase was re-unwrapped because the pulse sat late in its record (see below). | Nothing needed; check `n` against `n_g` as usual. |
| `jumps` | Each repaired, untrusted or skipped mid-band jump. | Untrusted jumps set `n_ulim`. |

## Exported phase or FFT phase?

A TeraFlash measurement is saved twice: as the averaged pulse (`.pulse.csv`), and as the instrument's spectrum
(`.spectr.csv`). The instrument builds the spectrum from a running average of per-trace phases. Averaging phases
isn't linear: when some traces slip a branch in noisy bins, the exported phase picks up a constant offset, and
sometimes fractional steps at water lines. The FFT of the averaged pulse (`use_FFT=True`) is exact up to whole
cycles.

A comparison of 385 traces across meteorite, foam, filter and lens measurements (October 2026) found:

- **Bright, transparent samples** (HDPE, foams, most filters) carry no measurable export offset. Both sources give
  the same n to about 10⁻⁴.
- **The offset follows the signal-to-noise ratio at 20–60 GHz**, where it is set:

  | SNR at 20–60 GHz | Median offset | Traces over 0.05π |
  |---|---|---|
  | 100–300 | 0.12π | two-thirds |
  | 300–1000 | 0.04π | 41% |
  | above 1000 | 0.002π | 4% |

  Lossy samples, apertured beams and short averages are most at risk. An offset φ₀ biases n by c·φ₀/(2πfw), which
  is largest for thin samples at low frequency.
- **The FFT was never worse,** and it removed offsets of up to about π. The two Pallasite 180° scans were examples:
  flagged ambiguous on the export, clean on the FFT.
- **Amplitudes agree to about 1% in band.** The exported noise floor is 1.3–1.8 times higher, so the FFT also gains
  dynamic range.

Recommendation: use `use_FFT=True` for phase and n. It costs nothing on clean data and protects lossy or
low-signal data. Either source is fine for transmittance.

## Late pulses: phase aliasing

On a frequency grid with spacing Δf, a pulse arriving a time t after the start of its window advances the phase
by 2πΔf·t per bin. Once t > 1/(2Δf), which is 100 ps on the usual 5 GHz grid, that step exceeds π. Bin-by-bin
unwrapping then aliases it, and the phase difference comes out with a slope that is wrong by a multiple of 2π/Δf.
The instrument export and the FFT path both do this, and zero-padding makes it worse.

`get_phase_diff` detects this case by comparing the slope of the phase difference with 2πτ. When it finds it, it
re-unwraps the phase about the delay model 2πfτ and sets `info['dealiased'] = True`. Correctly unwrapped data is
left untouched; on 110 project measurements none were aliased and every result is unchanged. The bundled alumina
example (pulse about 150 ps into a 200 ps window) is aliased, and is now recovered: its Δφ slope matches 2πτ.
`get_phases()` still returns the instrument's absolute phases, so if you build your own phase difference from it, a
late pulse will alias the same way.

Things the FFT doesn't fix:
- **Low-signal slips:** real 2π slips through low-transmission bands. These are flagged `offset_inconsistent` on
  both paths.
- **Truncated pulses:** pulses that haven't decayed by the end of the time window. Use a longer window.
- **Deep water lines in lossy samples:** these can produce an untrusted jump. Keep them in `mask_bounds`.
