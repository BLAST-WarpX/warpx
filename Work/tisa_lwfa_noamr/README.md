# Ti:Sa LWFA — AMR Off (comparison run)

Same as tisa_lwfa_amr but with AMR disabled. Used to demonstrate that the
rectangular artifact in the AMR version is caused by the fixed refinement box,
not by the physics.

## Key difference from tisa_lwfa_amr

- amr.max_level = 0  (uniform grid)
- dt = 0.44 fs  (vs 0.22 fs for AMR)
- 2000 steps → 877 fs ≈ 11 plasma periods
- No rectangular artifact in density or Ez plots

## How to Run

```bash
cd /mnt/d/Dev/warpx/Work/tisa_lwfa_noamr
/mnt/d/Dev/warpx/build_cuda/bin/warpx.2d.MPI.CUDA.DP.PDP.OPMD.EB.QED inputs_base_2d
```

Resource usage (RTX 4060 8GB): VRAM 18 MB, ~4 min.

Note: inputs_base_2d still includes the test beam species (100 electrons,
q=-1pC, uz=500). For a clean run without the beam see tisa_lwfa_clean/.
