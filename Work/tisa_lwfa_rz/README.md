# Ti:Sa LWFA — RZ Quasi-3D

LWFA in cylindrical (RZ) geometry with 2 azimuthal Fourier modes.
Physically more correct than 2D slab: proper cylindrical diffraction,
correct relativistic self-focusing threshold, axisymmetric bubble structure.

## Why RZ instead of 2D or full 3D

- Mode 0: axisymmetric plasma bubble — exact in RZ
- Mode 1: linearly polarized laser — exact with cos/sin decomposition
- ~20x fewer grid points than equivalent 3D Cartesian
- Particles still move in 3D (r, θ, z); only fields use Fourier modes
- Laser diffraction is cylindrical (∝ 1/z) vs 2D slab (∝ 1/√z)

## Input files

- inputs_rz      — r_max = 20 µm, Nr=64  (original; beam hits boundary ~4 zR)
- inputs_rz_r40  — r_max = 40 µm, Nr=128 (recommended; laser fits within domain)

At a0=4, w0=5µm, zR=98µm: after 1520 fs the beam width reaches ~19 µm.
r_max=40µm provides clearance to prevent PEC boundary reflections.

## Key parameters (inputs_rz_r40)

- geometry.dims = RZ, warpx.n_rz_azimuthal_modes = 2
- r ∈ [0, 40 µm], z ∈ [-56, 12] µm, Nr=128, Nz=512
- boundary: none/pec (r=0/rmax), pec (z boundaries)
- warpx.filter_npass_each_dir = 0 1  (no r-filtering for RZ FDTD)
- 4000 steps × 0.38 fs = 1520 fs ≈ 19 plasma periods

Resource usage (RTX 4060 8GB):
- inputs_rz:     VRAM 52 MB,  ~5.7 min
- inputs_rz_r40: VRAM 98 MB,  ~10 min

## How to Run

```bash
cd /mnt/d/Dev/warpx/Work/tisa_lwfa_rz
/mnt/d/Dev/warpx/build_cuda/bin/warpx.rz.MPI.CUDA.DP.PDP.OPMD.EB.QED inputs_rz_r40
```

## Fields in output

RZ uses cylindrical components: Er, Et (theta), Ez instead of Ex, Ey, Ez.
- Et mode 1 (cos) ≈ Ey at θ=0  →  laser field on axis
- Ez mode 0       →  axisymmetric wakefield
- rho mode 0      →  bubble electron density

The animation mirrors mode-0 fields symmetrically and mode-1 antisymmetrically
to reconstruct the full RZ cross-section.

## Note on self-focusing

At these parameters P ≈ 13 TW << P_crit ≈ 152 TW (for n_e=2×10^23 m^-3).
The laser diffracts without self-guiding. For guided propagation, either
increase power or add a preformed parabolic plasma channel.
