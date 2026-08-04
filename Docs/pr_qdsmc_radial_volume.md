# QDSMC radial volume weighting in RZ

## Summary

QDSMC electron-energy markers carry an extensive electron count
`N_e = n_e V_phys` in RZ. The marker volume is

\[
V_\mathrm{phys}(r) =
\begin{cases}
(\Delta r\,\Delta z)\,2\pi r, & r > 0,\\
(\Delta r\,\Delta z)\,\pi\Delta r f_\mathrm{axis}, & r = \Delta r/2\;\text{(first cell)}.
\end{cases}
\]

The second line applies to the first radial cell, whose marker is at
`r = Delta r / 2`. Here `f_axis` is the same axis-volume factor used by
WarpX's cylindrical charge-density scaling: `1/3` when the Verboncoeur axis
correction is enabled, and `1/4` otherwise.

The marker also carries the extensive entropy `K_e N_e`. After advection,
QDSMC recovers the entropy variable from the deposited ratio

\[
K_e^{n+1} = \frac{\mathcal{D}[K_e N_e]}{\mathcal{D}[N_e]},
\]

where `D` denotes the same nodal deposition operator for both quantities.
This keeps the entropy transport consistent with the cylindrical measure.

Cartesian behavior is unchanged because `V_phys` remains the ordinary cell
volume there.

## Implementation

- `QdsmcParticleContainer::SetK` computes `N_e = n_e V_phys`.
- `DepositField` deposits `N_e` without dividing by the Cartesian cell volume.
- `QDSMCUpdateTe` uses the deposited extensive-count ratio and then applies the
  existing polytropic temperature relation.

The change is limited to the RZ geometry path; no spherical geometry behavior
is changed.
