# Opt-in face-exact RZ photon absorption

Status, 13 September 2026: implemented and bounded CPU/CUDA qualification passes.
This extends the geometric packet marcher; it does not enable moving-frame
radiation, direct recoil with conversion, or arbitrary material-wall handling.

`radiation_transport.require_cell_interface_exact_streaming=1` now supports
RZ absorption with an axis-containing domain, no EB, and no momentum coupling
or packet/diffusion conversion. Select `photon_boundary=absorbing` to keep
material walls reflecting while photons leave. Alternatively, inherited
particle faces must be Open radially and Open or periodic axially. Unsupported
combinations retain explicit rejection. Legacy RZ remains unchanged when the
option is false; existing RCYLINDER segment bounds are retained.

Each bounded straight Cartesian photon path is split at positive-radius
circular faces and axial planes. The nearest event owns the segment; corner
events update both cell indices. The coordinate axis is not an absorbing or
reflecting face. Zero-length departures update ownership without discarding
remaining path length. Periodic axial crossings use one neighboring ghost
cell until the normal boundary sum and particle redistribution. Open exits
are recorded at the face, including an exact endpoint, before deleting packet
ownership. Cross-cell material-energy scatter is atomic on CPU and GPU.

The marcher calculates cell path lengths, not exact integrals of an arbitrary
smooth opacity inside a cell. Continuous opacity expressions still use a
segment-midpoint sample and require path-substep refinement. Streaming parsers
receive Cartesian x/y/z; radial profiles must use `sqrt(x*x+y*y)`.

## Bounded checks

- Independent full-ray circle/plane intersections and Beer-Lambert integration
  check every cell's deposited energy, remaining packet energy and escaped
  energy. The cases cover an oblique ray, an exact corner, axis traversal,
  both axial exits, an outer radial exit, and periodic axial transport. A
  two-rank oblique case crosses FAB/rank boundaries.
- The primitive additionally checks exact departures, tangency, nearly
  transverse propagation, invalid axial data, and 1e-12/1/1e12 geometry scales
  on the device. It retains all prior radial-face tests.
- The CPU selection passes 18/18 stages in 2.95 seconds. The combined local
  CUDA face-streaming/pressure-work-rejection selection passes 22/22 in 88.84
  seconds; that timing includes the pressure-work checkpoint fixture.
- Single-precision fields with double-precision particles also pass 18/18
  stages in 3.03 seconds. The field/particle-precision-aware analysis bounds
  remain fixed; this does not enable double-only material momentum coupling.
- Initial oblique/axis reference tests correctly failed because the new input
  mistakenly used Cartesian x as radius in its opacity expression. Correcting
  the input to the explicitly radial profile made the independent per-cell
  reference pass. No assertion was weakened. The original failed log remains
  `build-demo/rz-face-tests.log`; corrected results are in
  `build-demo/rz-face-oracle-tests.log` and `build-demo/rz-face-cuda-tests.log`.

This is an interface-accuracy feature gate, not a dynamic-hohlraum run or a
full application accuracy claim. Packet/diffusion conversion with this new
RZ policy, coupled momentum ownership and moving spectral transport remain
separate feature increments.
