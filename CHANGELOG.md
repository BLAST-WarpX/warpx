# Changelog

This file records notable additions and changes in feature branches relative to
`BLAST-WarpX/warpx:development`.

---

## `feature/prescribed-current-injection`

**New feature: file-driven prescribed current injection**

Adds support for injecting a time-varying current waveform into one or more rectangular
boxes of the simulation domain, to drive a macroscopic conductor (e.g. a coil) from an
external circuit while keeping WarpX agnostic to the circuit model.

The current is deposited by an artificial antenna species
(`PrescribedCurrentParticleContainer`) through the standard particle current-deposition
path, rather than written directly into `current_fp`.  Consequently the imposed current
works with the explicit **and** implicit field solvers and with the `vacuum` and
`macroscopic` media, and it shows up in the `jx/jy/jz` diagnostics.

### New input parameters

| Parameter | Type | Description |
|---|---|---|
| `warpx.current_injection` | bool | Enable the feature. |
| `warpx.current_injection.file` | string, optional | Global two-column file `t [s]  I [A]` (used by faces without their own file). |
| `warpx.current_injection.n_pairs` | int | Number of drive faces (≥ 1). |
| `warpx.current_injection.pair_N.file` | string, optional | Per-face waveform file, overriding the global file. |
| `warpx.current_injection.pair_N.drive.{xlo,xhi,ylo,yhi,zlo,zhi}` | float | Bounding box of the drive face [m]. |
| `warpx.current_injection.pair_N.drive.A` | float | Cross-section area [m²] for `J = I(t)/A`. |
| `warpx.current_injection.pair_N.drive.dir` | int (0/1/2), optional | Current direction (shared by all faces). |
| `warpx.current_injection.pair_N.drive.sign` | int (±1), optional | Current sign; `-1` gives a return face. |

Requires `algo.particle_shape >= 1` and `algo.current_deposition` (`direct` or a
charge-conserving scheme).

### Modified files

- `Source/Particles/PrescribedCurrentParticleContainer.{H,cpp}` — new antenna species that parses `warpx.current_injection.*` and deposits `J = I(t)/A`.
- `Source/Particles/MultiParticleContainer.cpp` — creates the container when `warpx.current_injection = 1`.
- `Source/WarpX.cpp` — sets the particle shape when current injection is enabled.
- `Docs/source/usage/parameters.rst` — section *Maxwell solver: prescribed current injection*.
