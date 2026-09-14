Conservative hybrid electron-ion exchange
==========================================

This one-step, spatially uniform test exercises the complete conservative
``Q_ei`` path: exact finite-heat-capacity electron-ion temperature exchange,
the electron source ledgers, nodal-to-cell ion request mapping, and the local
ion moment projection.  The analysis compares the electron temperature with
the closed-form two-temperature solution and requires the realized ion
kinetic-energy change to cancel the electron energy change while preserving
the initial ion momentum.

The ions are fixed in position so that transport and field work cannot enter
the energy balance.  A deterministic seed and 256 ions per cell provide local
thermal variance without making this regression expensive.
