.. _examples-ohm-solver-electron-energy-eq:

Ohm solver: Electron energy equation
====================================

In these examples the electron temperature :math:`T_e` used for the electron
pressure term of the generalized Ohm's law is evolved with the electron energy
equation

.. math::

    \frac{\partial U_e}{\partial t} + \nabla\cdot(U_e \mathbf{V}_e) + P_e \nabla\cdot\mathbf{V}_e = \eta J^2 - Q_{ei},

where :math:`U_e = n_e k_B T_e/(\gamma_e - 1)` is the electron internal energy
density, solved with the QDSMC kinetic-enslaving scheme of
:cite:t:`ex-Belyaev2024` (``hybrid_pic_model.solve_electron_energy_equation``).
Each of the three tests below isolates one piece of the equation with an exact
analytic solution: the transport terms on the left-hand side (adiabatic
compression), the Joule-heating source (force-free field decay), and the
electron-ion temperature-relaxation sink :math:`Q_{ei}`.

Adiabatic compression
---------------------

With all sources off, entropy-conserving transport of an initially uniform
entropy requires the pointwise adiabat

.. math::

    T_e(x, t) = T_{e0} \left( \frac{n(x,t)}{n_0} \right)^{\gamma_e - 1}

at every cell and time, independent of the flow. A uniform, unmagnetized,
zero-resistivity plasma is given a sinusoidal ion velocity perturbation which
drives an electron-pressure ion-acoustic compression/rarefaction wave, and the
measured :math:`T_e` is compared against the adiabat.

Run
^^^

.. dropdown:: Script ``inputs_test_2d_ohm_solver_electron_energy_picmi.py``

   .. literalinclude:: inputs_test_2d_ohm_solver_electron_energy_picmi.py
      :language: python3
      :caption: You can copy this file from ``Examples/Tests/ohm_solver_electron_energy_eq/inputs_test_2d_ohm_solver_electron_energy_picmi.py``. One script covers all three cases; select this one with ``--case adiabat``.

Execute:

.. code-block:: bash

   python3 inputs_test_2d_ohm_solver_electron_energy_picmi.py --case adiabat

Analyze
^^^^^^^

.. dropdown:: Script ``analysis_adiabat.py``

   .. literalinclude:: analysis_adiabat.py
      :language: python3
      :caption: You can copy this file from ``Examples/Tests/ohm_solver_electron_energy_eq/analysis_adiabat.py``.

.. figure:: https://github.com/user-attachments/assets/1e0e11ea-e54e-4c35-8f92-47b8373fcb70
   :alt: Electron temperature collapsing onto the adiabat
   :width: 90%

   Measured :math:`T_e` profiles against the analytic adiabat (left) and the
   collapse of all cells and times onto :math:`T_e/T_{e0} = (n/n_0)^{\gamma_e-1}`
   (right).

Joule heating
-------------

A linear force-free field :math:`\mathbf{B}(x) = B_0[0, \sin kx, \cos kx]`
satisfies :math:`\nabla\times\mathbf{B} = k\mathbf{B}`, so the current is
parallel to the field (no :math:`\mathbf{J}\times\mathbf{B}` force) with
uniform magnitude :math:`|J| = k B_0/\mu_0`. Nothing moves, the transport
terms vanish identically, and the electron temperature obeys the pure
Joule-heating ramp

.. math::

    \frac{d T_e}{d t} = (\gamma_e - 1)\, \frac{\eta J^2}{n_e k_B},

while the magnetic field energy decays resistively as
:math:`E_B(t) = E_B(0)\, e^{-2t/\tau_R}` with :math:`\tau_R = \mu_0/(\eta k^2)`.
The analysis fits the input resistivity from both signatures independently.

Run
^^^

.. dropdown:: Script ``inputs_test_2d_ohm_solver_electron_energy_picmi.py``

   .. literalinclude:: inputs_test_2d_ohm_solver_electron_energy_picmi.py
      :language: python3
      :caption: You can copy this file from ``Examples/Tests/ohm_solver_electron_energy_eq/inputs_test_2d_ohm_solver_electron_energy_picmi.py``. One script covers all three cases; select this one with ``--case joule``.

Execute:

.. code-block:: bash

   python3 inputs_test_2d_ohm_solver_electron_energy_picmi.py --case joule --eta-scale 20

Analyze
^^^^^^^

.. dropdown:: Script ``analysis_joule.py``

   .. literalinclude:: analysis_joule.py
      :language: python3
      :caption: You can copy this file from ``Examples/Tests/ohm_solver_electron_energy_eq/analysis_joule.py``.

Execute:

.. code-block:: bash

   python3 analysis_joule.py --eta-scale 20

.. figure:: https://github.com/user-attachments/assets/cfb5874a-0181-4bf0-9b9c-58b0aeafefbc
   :alt: Joule heating energy budget and electron temperature ramp
   :width: 90%

   Cumulative energy budget (left): the electron thermal gain tracks the
   magnetic-field loss plus a small ion kinetic drain, with the total
   conserved. Density-weighted mean electron temperature rise against the
   analytic Joule-heating ramp, with the resistive current decay folded in
   (right).

Electron-ion temperature relaxation
-----------------------------------

A uniform, unmagnetized, zero-resistivity plasma with hot electrons
(:math:`T_{e0} \gg T_{i0}`) relaxes purely through the electron-ion
thermal-equilibration exchange
:math:`Q_{ei} = 3 n_e k_B \nu_{ei} (T_e - T_i)`, which cools the electron
fluid and heats the kinetic ions by exactly the same amount. For a constant
:math:`\nu_{ei}` the temperature difference decays exponentially at the rate
:math:`[3(\gamma_e - 1) + 2]\,\nu_{ei}` while the total thermal energy is
conserved; for :math:`\gamma_e = 5/3` both species meet at
:math:`(T_{e0} + T_{i0})/2`.

Run
^^^

.. dropdown:: Script ``inputs_test_2d_ohm_solver_electron_energy_picmi.py``

   .. literalinclude:: inputs_test_2d_ohm_solver_electron_energy_picmi.py
      :language: python3
      :caption: You can copy this file from ``Examples/Tests/ohm_solver_electron_energy_eq/inputs_test_2d_ohm_solver_electron_energy_picmi.py``. One script covers all three cases; select this one with ``--case qei``.

Execute:

.. code-block:: bash

   python3 inputs_test_2d_ohm_solver_electron_energy_picmi.py --case qei

Analyze
^^^^^^^

.. dropdown:: Script ``analysis_qei.py``

   .. literalinclude:: analysis_qei.py
      :language: python3
      :caption: You can copy this file from ``Examples/Tests/ohm_solver_electron_energy_eq/analysis_qei.py``.

Execute:

.. code-block:: bash

   python3 analysis_qei.py --nu-ei 1e6

.. figure:: https://github.com/user-attachments/assets/cebd3552-0782-4489-9b69-bf25f92f6535
   :alt: Electron-ion temperature relaxation
   :width: 90%

   Electron and ion temperatures relaxing to the common equilibrium value
   (left), the exponential decay of the temperature difference against the
   analytic rate (center), and the drift of the total thermal energy (right).
