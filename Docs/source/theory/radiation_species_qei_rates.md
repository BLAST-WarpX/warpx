# Material-specific electron–ion relaxation rates

Status, 13 September 2026: implemented with focused two-material CPU checks.
This removes the requirement to use one effective relaxation expression for
all materials. It does not provide calibrated tungsten/foam rates or replace
the hybrid particle thermal closure with FLASH fluid hydrodynamics.

## Input contract

PICMI users can supply `electron_ion_relaxation_rate_species` as a dictionary
of species names to numeric rates or expressions on `HybridPICSolver`.
It generates the species list and named parsers below, including consistent
renaming of solver keyword constants. The input-only contract test also
covers numeric fallback/zero rates when a constant name is already in use.

```text
hybrid_pic_model.electron_ion_relaxation_rate(rho,Te,Ti,t) = 1.e9
hybrid_pic_model.electron_ion_relaxation_species = shell foam
hybrid_pic_model.electron_ion_relaxation_rate_shell(rho_s,rho,Te,Ti,t) = 2.e9
hybrid_pic_model.electron_ion_relaxation_rate_foam(rho_s,rho,Te,Ti,t) = 5.e8
```

These numbers illustrate syntax, not recommended physical coefficients.
Each listed depositing, massive, fixed-charge positive ion species overrides
the global rate; unlisted species use the global fallback, zero if omitted.
The evolved electron-energy equation is required. Both densities are charge
densities in C/m3, both temperatures are eV, time is seconds, and the result is
a finite nonnegative rate in 1/s with the existing Qei normalization. Zero is
an exact exchange no-op for the selected species, not an additive correction.

The same expression is consumed by the electron pair exchange and the ion
stochastic proposal. The local ion projection still pays the realized electron
energy ledger while preserving cell ion momentum. Existing thermal-support
and available-energy requirements remain gates. No unsupported cooling request
is silently moved to another cell.

`HybridSpeciesQeiRates.txt` records the global fallback and sorted species/name
expressions. Restart rejects a missing, added or changed contract. The manifest
compares expression text; external parser constants must be preserved by the
caller. Rate input does not introduce charge-state evolution or a Coulomb-log
prescription.

## Boundary fixes and bounded evidence

The new density-dependent test exposed an RZ finite-volume inconsistency:
species deposits omitted physical reflecting-wall contributions while total
charge included them. A true 1/2 species charge fraction appeared as 1/4 on
walls and 1/8 at corners. The finite-volume path now applies matching physical
charge/current boundary operators. The coordinate-axis fold remains in radial
inverse-volume scaling; the older non-FV entropy/drag deposit convention is
unchanged.

The older, non-resolved Qei source remap also integrated a finite-volume
electron source using geometric corner volumes instead of the native electron
axis/wall measure. Recipient partition weights remain geometric, but source
integration now uses the native caloric node volume for RZ FV states.

The two-material test runs 50 finite exchanges, not a few PIC smoke steps.
Equal charge fractions use different ion masses and charges. The light species
gains about 7.3% of the initial total energy; a zero heavy-species override must
leave every heavy velocity bitwise unchanged. Specific-rate and global-fallback
paths pass with both resolved and older cell-support exchange. The native
energy residual improves from 4.44e-4 before the remap correction to below
9e-16, and the species fraction is 1/2 at every tested wall/axis node. Invalid
rates are rejected before the ion update instead of propagating NaN. Tests
also exercise the real checkpoint read/write contract and two-rank execution.

These checks establish input semantics, local ownership and conservation. They
do not establish a calibrated equilibration time for high-Z material, nor do
they justify reducing particle thermal support to make an application run.
