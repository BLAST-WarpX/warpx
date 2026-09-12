# Moving reflecting-cavity accuracy qualification

Status (2026-09-09): CPU spatial and temporal refinement passes. This qualifies
the low-level gray radiation solver with prescribed tangential material drift,
not native particle-wall handling, multigroup transport or full production use.
No material-fluid closure is substituted for hybrid PIC. No new FLASH run is
claimed: the independent reference here is an analytic radiation diffusion mode.

## Physical question and reference

Does a closed optical cavity retain radiation energy while diffusing a
nonuniform radiation field at the correct rate in moving material? A zero-leakage
identity alone would also pass for a solver that simply froze the radiation.

The Cartesian interval has L=1, fixed reflecting optical walls, constant proper
scattering opacity kappa L=10000, zero absorption and tangential beta=0.002.
Material velocity is externally prescribed; this test does not accelerate
native particles. The radiation starts from a cell-center sampled, boosted
isotropic rest-frame cosine mode. The final time satisfies c T/L=1000.

For R=E-beta*q_t, gamma=(1-beta^2)^(-1/2), the gray diffusion limit gives

    D_normal = c/(3*kappa*gamma)
    R(x,t) = 1 + 0.1 exp(-D_normal*pi^2*t/L^2) cos(pi*x/L).

The normal energy flux vanishes at both walls. The cosine amplitude decreases
by approximately 28%, so this is an appreciable transport test. The reference
is an asymptotic diffusion solution, not an exact finite-opacity M1 solution.
Mesh and timestep errors must decrease before reaching that modeling-error floor.

## CPU results

Errors below are fractions, normalized by the final analytic perturbation
amplitude, not by the much larger uniform background. Full-profile L1 uses every
cell, independently of the cosine-amplitude projection.

Spatial refinement, with 6400 fixed timesteps:

| Cells | Relative amplitude error | Full-profile L1 / amplitude |
| --- | --- | --- |
| 32 | 2.743183352e-4 | 1.747064131e-4 |
| 64 | 7.535107756e-5 | 4.797469462e-5 |
| 128 | 2.538891816e-5 | 1.616343990e-5 |

Both errors decrease by approximately 3.64 and 2.97 between successive meshes.
The finest full-profile error is 0.00162% of the perturbation amplitude.

Temporal refinement, with 128 fixed cells:

| Steps | Measured cosine amplitude | Full-profile L1 / amplitude |
| --- | --- | --- |
| 200 | 0.07198596866 | 1.828614339e-4 |
| 400 | 0.07197624734 | 9.686250095e-5 |
| 800 | 0.07197138318 | 5.383204890e-5 |

Successive amplitude self-differences have ratio 1.9985609, consistent with
first-order backward Euler. Across the six runs, maximum reported global energy
and momentum residuals are below 5e-16 and 5e-15 respectively. Fixed walls export
exactly zero energy and tangential momentum at every accepted step.

The CI-sized study uses 16/32/64 cells with 1600 fixed steps, and 100/200/400
steps on 64 cells. It passes in 6.81 seconds on CPU. Its mesh-ratio gates exceed
2 for both error measures; its temporal-ratio gate is 1.8–2.2. The finer study
additionally requires both finest errors below 5e-5. Existing periodic transport
refinement still passes; the combined CPU CTest pair took 18.64 seconds.

## Reproduction and visualization

From the worktree root:

```sh
cmake --build build-production-cpu -j 8 --target test_implicit_moment_transport
ctest --test-dir build-production-cpu -R test_reflecting_cavity_refinement --output-on-failure
mkdir -p build-production-cpu/reflecting-cavity-production
cd build-production-cpu/reflecting-cavity-production
python ../../Examples/Tests/radiation_transport/analysis_reflecting_cavity.py ../bin/test_implicit_moment_transport --production
```

The script preserves per-case logs, writes `reflecting_cavity_results.json`,
and plots both refinement sequences in `reflecting_cavity_refinement.png`.
The production-resolution run and visualization have been executed locally.

The local measured-refinement plot is generated at
`build-production-cpu/reflecting-cavity-production/reflecting_cavity_refinement.png`;
it is a qualification artifact, not bundled with the source documentation.

The same CI-sized refinement gate passes on Tesla P40 CUDA (103.97 seconds).
The six reported cosine amplitudes agree with CPU at the executable's printed
precision; both temporal ratios are 1.997124145. This is not a bitwise field
comparison. The finer 32/64/128-cell, 6400-step study above was run on CPU only.
The CUDA report is preserved as
`build-production-cpu/reflecting-cavity-production/p40-ci-results.json`.

## Remaining native boundary work

The optical-wall pressure/flux is only one part of native boundary support.
Current guards also protect particle-owned carry, shape-weight deposition/gather
at physical edges, native caloric boundary volumes, and coupled-interval rollback.
The existing native particle reflector changes velocity components, but does not
transform radiation carry or export its boundary momentum. A stationary specular
reflection must transform both momentum and deferred momentum consistently;
thermal reemission and particle loss need different energy/ownership rules.
Do not enable those operations by removing the periodic guards alone.
