# 3D Quartz_Boundary Test

This directory contains a minimal 3D input file for testing the Quartz (dielectric) boundary condition in WarpX.

**Note:** In `ProjectionDivCleaner.cpp`, the divergence cleaner (divcleaner) is disabled for quartz boundary conditions. This allows the simulation to run with quartz boundaries, but means field divergence is not actively corrected.

## File Description
- `inputs_test_3d_quartz_field`  
  Tests electromagnetic wave reflection and transmission at a quartz boundary in 3D, verifying the Quartz boundary condition.
- `analysis_quartz.py`  
  Analysis script to visualize Ey and Bx along z, and check field amplitude and variation.

## How to Run
1. Build the WarpX executable (e.g., `warpx`).
2. Run the simulation from the main directory:
   ```bash
   ./build/bin/warpx.3d.MPI.OMP.DP.PDP.OPMD.EB.QED Examples/Tests/Quartz_Boundary/inputs_test_3d_quartz_field
   ```

## Viewing and Analyzing Results
- Output (e.g., `diags/diag1000080`) will be generated after running.
- To analyze and visualize the results, run:
   ```bash
   python Examples/Tests/Quartz_Boundary/analysis_quartz.py diags/diag1000080
   ```
- This will generate `quartz_boundary_fields_z.png` and print field amplitude statistics.

## How to Verify Correctness
1. **Theoretical Reflection/Transmission:**
   - For a plane wave normally incident on a dielectric (quartz, εᵣ=3.8), the theoretical reflection and transmission coefficients are:
     - Reflection:  
       \( R = \frac{n_1 - n_2}{n_1 + n_2} \),  Transmission:  \( T = \frac{2 n_1}{n_1 + n_2} \)
       where \( n_1 = 1 \) (vacuum), \( n_2 = \sqrt{3.8} \approx 1.95 \)
     - So \( R \approx -0.32 \), \( T \approx 0.68 \)
   - In the analysis plot, compare the reflected and transmitted field amplitudes to the incident amplitude, and check if the ratios match the theory.
2. **Field Continuity:**
   - At the quartz boundary, the tangential electric field should be continuous, and the normal component should satisfy the dielectric jump condition.
3. **Parameter Variation:**
   - Change `epsilon_r` in the input file and observe the change in reflection/transmission, which should follow the theoretical formula.

## Physical Background
- The Quartz boundary condition simulates a dielectric interface, with typical relative permittivity εᵣ = 3.8.
- Useful for plasma, microwave, and other electromagnetic simulations involving dielectric boundaries.
- Divergence cleaning is disabled for quartz boundaries in this test.

---
For more advanced tests or analysis scripts, please refer to the official WarpX documentation or contact the developers. 