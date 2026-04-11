# Changelog

This file records notable additions and changes in feature branches relative to
`BLAST-WarpX/warpx:development`.

---

## `feature/prescribed-current-injection`

**New feature: file-driven prescribed current injection**

Adds support for injecting a time-varying current waveform directly into one or more
rectangular faces of the simulation domain.  Intended for macroscopic EM simulations of
conductors (coils, transmission lines) driven by an external circuit.

### New input parameters

| Parameter | Type | Description |
|---|---|---|
| `warpx.current_injection` | bool | Enable the feature (requires `algo.em_solver_medium = macroscopic`). |
| `warpx.current_injection.file` | string | Two-column plain-text file: `t [s]  I [A]`. Linear interpolation. |
| `warpx.current_injection.n_pairs` | int | Number of drive/return face pairs (≥ 1). |
| `warpx.current_injection.pair_N.drive.{xlo,xhi,ylo,yhi,zlo,zhi}` | float | Bounding box of the drive face [m]. |
| `warpx.current_injection.pair_N.drive.A` | float | Cross-section area [m²] for `J = I(t)/A`. |
| `warpx.current_injection.pair_N.drive.dir` | int (0/1/2) | Current component direction (x/y/z). |
| `warpx.current_injection.pair_N.return.{xlo,...,A,dir}` | float/int, optional | Return face (omit when conductor is in domain via `sigma_function`). |

### Modified files

- `Source/WarpX.H` — `CIFace`, `CIPair` structs; member arrays and `InjectPrescribedCurrent` declaration.
- `Source/WarpX.cpp` — Input parsing for all `warpx.current_injection.*` keys.
- `Source/Evolve/WarpXEvolve.cpp` — `InjectPrescribedCurrent` call each step, before the macroscopic E-solve.
- `Docs/source/usage/parameters.rst` — New section *Maxwell solver: prescribed current injection*.
