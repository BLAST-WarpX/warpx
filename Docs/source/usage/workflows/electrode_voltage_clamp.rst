.. _usage-electrode-voltage-clamp:

Holding Embedded Electrodes at a Prescribed Voltage (RZ, Yee)
=============================================================

In an electromagnetic run, WarpX sets embedded-boundary (EB) potentials only in the initial electrostatic solve.
Afterwards nothing maintains them: the field does work on charge moving between the electrodes (for example after emission or before collection), and with no source to replace that energy, the electrode voltages change; a current-carrying diode discharges within about one transit time (the time an electron needs to cross the gap from cathode to anode).
``StaircaseBiasCorrector`` acts as an ideal voltage source for one or more separate EB conductors.
After every step it restores each conductor to its target voltage without changing :math:`\boldsymbol{B}` or any frozen edge.

Method
------

WarpX's explicit Yee solver represents an EB conductor as a *staircase*: E edges whose stencil touches a cut or covered cell are frozen (not updated).
The nodes at the ends of frozen edges are the conductor's *fixed nodes*.
The clamp works on exactly this representation.

Notation, for conductors :math:`k = 1, \dots, n`.
Scalars are italic, vectors bold italic (both field vectors and vectors with one entry per grid node), matrices upright bold:

* :math:`\boldsymbol{E}`, :math:`\boldsymbol{B}`: the electric and magnetic fields;
* :math:`\boldsymbol{s}_k`: indicator of the fixed nodes of conductor *k* (1 there, 0 elsewhere);
* :math:`\mathbf{V}`: diagonal matrix of node volumes (the control volume of each node);
* :math:`\boldsymbol{\rho}`, :math:`\boldsymbol{q} = \mathbf{V} \boldsymbol{\rho}`: deposited particle charge density and charge per node;
* :math:`\nabla_h`: WarpX's discrete Yee gradient and divergence;
* :math:`\boldsymbol{\psi}_k`: unit potential of conductor *k*, and :math:`\boldsymbol{E}_k = -\nabla_h \boldsymbol{\psi}_k` its unit field;
* :math:`Q_k`, :math:`U_k`, :math:`U^*_k`: charge, voltage and target voltage of conductor *k*, collected in the vectors :math:`\boldsymbol{Q}`, :math:`\boldsymbol{U}`, :math:`\boldsymbol{U}^*`; :math:`\mathbf{C}`: the :math:`n \times n` capacitance matrix.

The clamp has four parts:

* **Actuator.** At setup, one harmonic potential :math:`\boldsymbol{\psi}_k` per conductor is solved on the staircase (1 on the fixed nodes of conductor *k*, 0 on those of the other conductors and on the grounded PEC wall).
  Its field :math:`\boldsymbol{E}_k` uses the ordinary Yee gradient, so :math:`\nabla_h \times \boldsymbol{E}_k = 0`, and it is zero on frozen edges: adding it changes neither :math:`\boldsymbol{B}` nor the frozen edges.
* **Observer.** The charge on conductor *k* is read from WarpX's own nodal divergence on the conductor's fixed nodes, minus the particle charge there: :math:`Q_k = \varepsilon_0\, \boldsymbol{s}_k^T \mathbf{V}\, \nabla_h\!\cdot \boldsymbol{E} - \boldsymbol{s}_k^T \boldsymbol{q}`.
* **Plasma contribution.** The charge the plasma induces on the conductors when all of them are grounded follows from the unit potentials, by the Shockley-Ramo theorem :cite:p:`vc-ShockleyJAP1938,vc-RamoPIRE1939`: :math:`Q_{g,k} = -\boldsymbol{\psi}_k^T \boldsymbol{q}`.
  ``voltage_clamp_solve_grounded`` solves this grounded problem directly and gives an independent check at any step.
* **Feedback.** The capacitance matrix is measured with the observer itself, :math:`C_{jk} = \varepsilon_0\, \boldsymbol{s}_j^T \mathbf{V}\, \nabla_h\!\cdot \boldsymbol{E}_k`, so that reading back a field built from the unit fields is exact.
  The voltages are :math:`\boldsymbol{U} = \mathbf{C}^{-1}(\boldsymbol{Q} - \boldsymbol{Q}_g)`, and the correction :math:`\boldsymbol{E} \mathrel{+}= \sum_k (U^*_k - U_k)\, \boldsymbol{E}_k` restores the targets exactly.

These identities hold only if the deposited :math:`\boldsymbol{\rho}` and :math:`\boldsymbol{J}` satisfy the discrete continuity equation next to the conductors.
The clamp therefore requires Esirkepov current deposition, no current or charge filter, and (for now) linear particles: with higher shapes, the charge deposition must use the same reduced shape next to the EB as Esirkepov (`#7421 <https://github.com/BLAST-WarpX/warpx/pull/7421>`__).

Supported configurations
------------------------

* RZ, azimuthal mode 0, one mesh level, explicit staggered Yee solver, laboratory frame.
* Linear particles (``particle_shape = 1``) with Esirkepov deposition.
  No current/charge filter.
  Energy-conserving field gathering, and no particle E field read from file.
* Radial boundary: the axis and a PEC outer wall, which is the grounded potential reference.
  Axial boundaries: both periodic or both PEC.
* Listed conductors must be separate bodies.
* A conductor that touches a PEC domain boundary is part of the ground: the staircase ties it to the wall, so it cannot hold another voltage.
  Such conductors (for example a curved outer wall) are found at setup and must not be listed; listing one aborts.
  Touching is decided on the staircase, which joins a conductor to the wall across a gap smaller than one cell.
* List every connected body as its own conductor, even if two bodies should share a voltage.
  The staircase keeps each connected body an equipotential, so the clamp holds each listed conductor exactly; for disconnected pieces listed as one conductor it holds only their combined charge, and the pieces can drift apart.
* With the Verboncoeur axis correction (``boundary.verboncoeur_axis_correction``, on by default), charge on axis nodes that do not belong to a conductor aborts the run: that case is not verified yet.
  Either keep the axis inside a conductor, or set ``boundary.verboncoeur_axis_correction = 0``.
* Particles absorbed at a PEC domain boundary need ``particles.crop_on_PEC_boundary = 1``; otherwise their current breaks the discrete Gauss law next to the wall and biases the voltage reading (requires `#7435 <https://github.com/BLAST-WarpX/warpx/pull/7435>`__ for Esirkepov).
* Emitted particles must be created inside the conductor's fixed-node shell (between the geometric surface and the represented surface).
  Particles created in free vacuum carry charge without a current, which the field cannot see.
  Emission with ``inject_from_embedded_boundary`` creates particles on the EB surface, but then advances them by a random fraction of a step without depositing current; it is not verified with the clamp yet.

Other configurations abort with a message.

Usage
-----

.. code-block:: python

   from pywarpx import callbacks
   from pywarpx.staircase_bias_corrector import StaircaseBiasCorrector

   sim.initialize_inputs()
   sim.initialize_warpx()

   clamp = StaircaseBiasCorrector(
       sim,
       correction_interval=1,
       electrodes=[
           {"name": "cathode", "region": "(x<0.03)", "potential": -1000.0},
           {"name": "anode", "region": "(x>0.03)", "potential": 0.0},
       ],
   )
   clamp.setup_after_init()        # unit potentials, capacitance matrix
   clamp.initialize_vacuum_bias()  # new run; on restart: clamp.resume_from_checkpoint()
   callbacks.installafterstep(clamp.correct_after_step)  # after particle scraping

   sim.step(n_steps)

A ``region`` is a parser expression in ``(x, y, z)`` (``x`` is the radius, ``y = 0``) that is 1 on one whole conductor and 0 on the others.
It is evaluated only on the fixed nodes, so it does not need to follow the conductor surface: any expression that separates the conductors works, however complex their (e.g. STL) geometry.
Products act as "and", sums of non-overlapping terms as "or".
For example, a rod and two end rings, split at the midplane:

.. code-block:: python

   electrodes = [
       {"name": "rod", "region": "(x<0.01)", "potential": -1000.0},
       {"name": "ring_lo", "region": "(x>0.04)*(z<0)", "potential": -500.0},
       {"name": "ring_hi", "region": "(x>0.04)*(z>=0)", "potential": -500.0},
   ]

Every conductor must be listed, grounded ones with potential 0, except conductors that touch a PEC domain boundary: they are part of the ground and are not listed.
A misassigned or missing node aborts at setup with a message.
``measure_voltage_state()`` returns the observer voltages and charges.

What it does not do
-------------------

* The clamp holds the voltages of the *represented* conductors.
  Next to the metal the staircase leaves field-free strips in vacuum, so near-surface forces carry the usual staircase error.
  The reported voltage is the clamp's own charge coordinate; an independent check is the line integral of :math:`E_r` between the conductors.
  That check holds only where the line integral is path-independent: a changing magnetic flux through the area between two paths (:math:`\oint \boldsymbol{E} \cdot d\boldsymbol{l} = -d\Phi_B/dt`) makes them differ, so compare along a path where this flux is negligible.
* There is no circuit model (no floating electrodes, no series impedance) and no dielectric charging.
* It is available from Python only; there is no inputs-file interface yet.
* Cartesian geometries (XZ, 3D), PMC and open (absorbing) field boundaries are not supported yet.
* Each correction costs one charge deposition, one nodal divergence and a few reductions; the unit potentials are solved once, at setup.

Examples and tests
------------------

``Examples/Tests/voltage_clamp`` contains the regression tests:

* vacuum: capacitance matrix against the discrete coaxial formula, static clamped field;
* a coaxial space-charge diode with and without the clamp, including the grounded-charge cross-check;
* the diode on two ranks with the anode joined to the outer wall, so that only the cathode is listed;
* checkpoint/restart during emission.

.. bibliography::
    :keyprefix: vc-
