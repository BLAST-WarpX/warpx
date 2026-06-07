# Ti:Sa LWFA — 3D Cartesian (machine limit test)

Short 3D run to measure VRAM consumption and step rate on RTX 4060 8GB.
Not intended as a production simulation (1 ppc is very noisy).

## Grid

- 64 × 64 × 512 = 2,097,152 cells
- dx = dy = 625 nm, dz = 133 nm
- Domain: x,y ∈ [-20, 20] µm, z ∈ [-56, 12] µm

## Measured resource usage (RTX 4060 8GB)

- VRAM: 714 MB (Arena used)
- Step rate: 0.47 s/step
- 500 steps in 236 s ≈ 4 min → t_max = 212 fs = 2.7 λ_p/c

## Scaling estimates for longer 3D runs

| steps | ppc (3D) | VRAM est. | time est. |
|-------|----------|-----------|-----------|
| 4000  | 1×1×1    | ~714 MB   | ~31 min   |
| 4000  | 2×2×2    | ~2.5 GB   | ~90 min   |
| 4000  | 4×4×4    | ~8 GB     | limit     |

RTX 4060 8GB hard limit: ~16M cells or ~8M cells at 4ppc.

## How to Run

```bash
cd /mnt/d/Dev/warpx/Work/tisa_lwfa_3d_test
/mnt/d/Dev/warpx/build_cuda/bin/warpx.3d.MPI.CUDA.DP.PDP.OPMD.EB.QED inputs_3d
```

## Note

For physically accurate LWFA, prefer RZ (tisa_lwfa_rz/) which gives correct
cylindrical geometry at 1/20 the cost of 3D. Full 3D at research resolution
(dz=80nm) would require ~212M cells — beyond this GPU's capacity.
