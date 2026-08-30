.. _examples-hohlraum-radiation:

Hohlraum-Like Radiation Transport
=================================

This RZ demonstrator exercises the hybrid particle/diffusion radiation model in a
hohlraum-like geometry. A dense cylindrical wall and endcaps surround a low-density
cavity with two axial laser entrance holes. Hot wall electrons emit LTE grey radiation;
radiation diffuses inside the optically thick wall and becomes streaming photon packets
when it reaches the thin cavity. Packets that return to the wall are converted back to
the diffusion representation.

Grey and four-group variants are supplied for both material representations:

* ``inputs_rz_hybrid`` couples radiation to the QDSMC hybrid-electron energy equation.
* ``inputs_rz_hybrid_multigroup`` adds four photon-energy groups to that model.
* ``inputs_rz_kinetic`` couples radiation to a fully kinetic electron species.
* ``inputs_rz_kinetic_multigroup`` adds the same groups to the kinetic model.

Run
---

From this directory, run one of:

.. code-block:: bash

   warpx.rz inputs_rz_hybrid
   warpx.rz inputs_rz_hybrid_multigroup
   warpx.rz inputs_rz_kinetic
   warpx.rz inputs_rz_kinetic_multigroup

For an MPI run, prefix the command with the launcher appropriate to the system. The
input files write ``radiation_material_energy`` and either the grey
``radiation_diffusion_energy`` field or the suffixed multigroup components, plus
``ParticleEnergy`` and ``RadiationEnergy`` reduced diagnostics.
The latter records streaming, diffusion, total-radiation, signed current-step material
exchange, boundary escape and restart-safe cumulative values in one file. Total
radiation plus cumulative material exchange plus cumulative boundary loss is the direct
conservation check.

After a hybrid multigroup run, render a mirrored RZ overview of the electron
temperature, total radiation energy density and four spectral groups with:

.. code-block:: bash

   python plot_demo.py diags/diag1000020 --output hohlraum_multigroup.png

Modeling notes
--------------

The numerical opacity values and the representative 300 eV photon energy are deliberately
simple placeholders. A predictive hohlraum calculation must replace them with suitable
Planck, Rosseland and energy-dependent opacity data, and should perform resolution,
packet-threshold and timestep convergence studies. These may be supplied as parser
expressions, or as structured SI-unit tables through the ``*_table_file`` options
documented under :ref:`running-cpp-parameters`. The multigroup decks demonstrate parser
opacities in four broad bands; their values are still illustrative rather than material
data. The transport remains explicit and single-level and does not yet model within-group
line transport, material ionization, scattering or photon momentum deposition.

Radiation/material exchange is applied at the beginning of each explicit PIC step. In
the hybrid deck it updates the QDSMC electron temperature before entropy transport; in
the kinetic deck it updates electron particle energies before their usual push and
collision operators. The supplied decks use the cell-local implicit-temperature LTE
solver, which avoids frozen-temperature over-emission when the opacity-times-timestep
is large. Its effective electron heat capacity is held constant within the radiation
operator. The outer radiation/material split remains conservative but first order in
time.

The kinetic deck freezes charge deposition, gathering and ordinary particle pushing so
the compact example isolates radiation/material exchange without pretending that its
mesh resolves a dense kinetic plasma. Remove those flags only in a configuration that
also resolves the kinetic plasma scales. The hybrid deck evolves its QDSMC electron
temperature and is the less expensive starting point for long radiation-hydrodynamic
experiments, but ion motion and hydrodynamic closure still need application-specific
validation.
