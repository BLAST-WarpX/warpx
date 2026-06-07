# Ti:Sa LWFA — Clean Reference Run

2D simulation of Laser Wakefield Acceleration in the bubble/blowout regime.
This is the primary clean reference: AMR off, no test beam, a0=4.

## Physics

An 800 nm Ti:Sapphire laser (a0=4) propagates through underdense plasma
(n_e = 2×10^23 m^-3, λ_p ≈ 23.6 µm) with a moving window.
At a0=4 the laser is in the nonlinear bubble/blowout regime: electrons are
completely expelled from the laser axis, forming a nearly field-free cavity
(bubble) behind the pulse. Ez inside the bubble provides a strong
accelerating field.

Key parameters:
- a0 = 4, λ = 800 nm, w0 = 5 µm, τ = 15 fs
- n_e = 2×10^23 m^-3, λ_p = 23.6 µm, λ_p/c = 79 fs
- Domain: x ∈ [-30, 30] µm, z ∈ [-56, 12] µm (moving window)
- Grid: 64 × 512, dx = 937 nm, dz = 133 nm
- 4000 steps × 0.44 fs/step = 1754 fs ≈ 22 plasma periods

Resource usage (RTX 4060 8GB):
- VRAM: 18 MB
- Runtime: ~7 min

## How to Run

```bash
cd /mnt/d/Dev/warpx/Work/tisa_lwfa_clean
/mnt/d/Dev/warpx/build_cuda/bin/warpx.2d.MPI.CUDA.DP.PDP.OPMD.EB.QED inputs_2d
```

## Visualize

```bash
cd /mnt/d/Dev/warpx/Work/animations
python3 make_animations.py   # generates lwfa_clean.mp4
```

Fields: Ey (laser), Ez (wakefield), rho (bubble density).
Line profiles at x=0 are shown below each colormap.
