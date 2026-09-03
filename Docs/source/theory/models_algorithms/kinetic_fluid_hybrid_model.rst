.. _theory-kinetic-fluid-hybrid-model:

Ampere's law coupled with Ohm's law (a.k.a. "hybrid PIC")
=========================================================

Many problems in plasma physics fall in a class where both electron kinetics and electromagnetic waves do not
play a critical role in the solution. Examples of such situations include the
study of collisionless magnetic reconnection and instabilities driven by ion
temperature anisotropy, to mention only two. For these kinds of problems the
computational cost of resolving the electron dynamics can be avoided by modeling
the electrons as a neutralizing fluid rather than kinetic particles. By further
using Ohm's law to compute the electric field rather than evolving it with the
Maxwell-Faraday equation, light waves can be stepped over. The simulation resolution
can then be set by the ion time and length scales (commonly the ion cyclotron
period :math:`1/\Omega_i` and ion skin depth :math:`l_i`, respectively), which
can reduce the total simulation time drastically compared to a simulation that
has to resolve the electron Debye length and CFL-condition based on the speed
of light.

Many authors have described variations of the kinetic ion & fluid electron model,
generally referred to as particle-fluid hybrid or just hybrid-PIC models. The
implementation in WarpX is described in detail in :cite:t:`kfhm-Groenewald2023`.
The "Model derivation" section below gives a detailed description of the model
that follows mostly from the above reference, but succinctly, the model
entails the following:

The magnetic field is advanced in time using Faraday's law,

    .. math::

        \frac{\partial\boldsymbol{B}}{\partial t} = -\boldsymbol{\nabla}\times\boldsymbol{E},

where the electric field is calculated from Ohm's law which involves the currents,
the magnetic field, and the electron pressure (for which an additional closure is required,
see :ref:`here <theory-hybrid-model-elec-temp>`),

    .. math::

        \boldsymbol{E} = -\frac{1}{en_e}\left( \boldsymbol{J}_e\times\boldsymbol{B} + \boldsymbol{\nabla} P_e \right)+\eta\boldsymbol{J}-\eta_h \nabla^2\boldsymbol{J}.

The electron current is in turn obtained by subtracting the ion current (obtained from
kinetic ion macro-particles) from the total current (obtained from Ampere's law):

    .. math::

        \boldsymbol{J}_e = \boldsymbol{J} - \sum_{s\neq e}\boldsymbol{J}_s - \boldsymbol{J}_{ext}

where

    .. math::

        \mu_0\boldsymbol{J} = \boldsymbol{\nabla}\times\boldsymbol{B}.

Algorithm details
-----------------

.. note::

    Various verification tests of the hybrid model implementation can be found in
    the :ref:`examples section <examples-hybrid-model>`.

The kinetic-fluid hybrid extension mostly uses the same routines as the standard electromagnetic
PIC algorithm with the only exception that the E-field is calculated from Ohm's law
rather than it being updated from the full Maxwell-Ampere equation. The E-field update occurs
after particle pushing and deposition (charge and current density) has been completed. Therefore, based
on the usual time-staggering in the PIC algorithm, when the E-field is updated
at timestep :math:`t=t_n`, the quantities :math:`\rho^n`, :math:`\rho^{n+1}`, :math:`\boldsymbol{J}_i^{n-1/2}`
and  :math:`\boldsymbol{J}_i^{n+1/2}` are all known.

Field update
^^^^^^^^^^^^

The field update is done in three steps as described below.

First half step
"""""""""""""""

Firstly the E-field at :math:`t=t_n` is calculated for which the current density needs to
be interpolated to the correct time, using :math:`\boldsymbol{J}_i^n = 1/2(\boldsymbol{J}_i^{n-1/2}+ \boldsymbol{J}_i^{n+1/2})`.
The electron pressure is simply calculated using :math:`\rho^n` and the B-field is also already
known at the correct time since it was calculated for :math:`t=t_n` at the end of the last step.
Once :math:`\boldsymbol{E}^n` is calculated, it is used to push :math:`\boldsymbol{B}^n` forward in time
(using the Maxwell-Faraday equation) to :math:`\boldsymbol{B}^{n+1/2}`.

Second half step
""""""""""""""""

Next, the E-field is recalculated to get :math:`\boldsymbol{E}^{n+1/2}`. This is done
using the known fields :math:`\boldsymbol{B}^{n+1/2}`, :math:`\boldsymbol{J}_i^{n+1/2}` and
interpolated charge density :math:`\rho^{n+1/2}=1/2(\rho^n+\rho^{n+1})` (which is
also used to calculate the electron pressure). Similarly as before, the B-field
is then pushed forward to get :math:`\boldsymbol{B}^{n+1}` using the newly calculated
:math:`\boldsymbol{E}^{n+1/2}` field.

Extrapolation step
""""""""""""""""""

Obtaining the E-field at timestep :math:`t=t_{n+1}` is a well documented issue for
the hybrid model. Currently the approach in WarpX is to simply extrapolate
:math:`\boldsymbol{J}_i` forward in time, using

    .. math::

        \boldsymbol{J}_i^{n+1} = \frac{3}{2}\boldsymbol{J}_i^{n+1/2} - \frac{1}{2}\boldsymbol{J}_i^{n-1/2}.

With this extrapolation all fields required to calculate :math:`\boldsymbol{E}^{n+1}`
are known and the simulation can proceed.

Sub-stepping
^^^^^^^^^^^^

It is also well known that hybrid PIC routines require the B-field to be
updated with a smaller timestep than needed for the particles. A 4th order
Runge-Kutta scheme is used to update the B-field. The RK scheme is repeated a
number of times during each half-step outlined above. The number of sub-steps
used can be specified by the user through a runtime simulation parameter
(see :ref:`input parameters section <running-cpp-parameters-hybrid-model>`).

.. _theory-hybrid-model-elec-temp:

Electron pressure
^^^^^^^^^^^^^^^^^

The electron pressure is assumed to be a scalar quantity and calculated using the given
input parameters, :math:`T_{e0}`, :math:`n_0` and :math:`\gamma` using

    .. math::

        P_e = n_0T_{e0}\left( \frac{n_e}{n_0} \right)^\gamma.

The isothermal limit is given by :math:`\gamma = 1` while :math:`\gamma = 5/3`
(default) produces the adiabatic limit.

Alternatively, the electron temperature entering the pressure can be evolved
in space and time with the electron energy equation, as described in the next
section.

.. _theory-hybrid-model-electron-energy-eq:

Electron energy equation
^^^^^^^^^^^^^^^^^^^^^^^^

Instead of evaluating the polytropic closure with the constant reference state
:math:`(n_0, T_{e0})`, WarpX can evolve the electron temperature
:math:`T_e(\vec{x}, t)` with the electron internal-energy equation
(``hybrid_pic_model.solve_electron_energy_equation``),

    .. math::

        \frac{\partial U_e}{\partial t} + \nabla\cdot(U_e \vec{V}_e) + P_e \nabla\cdot\vec{V}_e = S_e,

where :math:`U_e = n_e k_B T_e/(\gamma - 1)` is the electron internal energy
density, :math:`\vec{V}_e = \vec{J}_e/(-e n_e)` is the electron fluid velocity
and :math:`S_e` collects the source and sink terms. The local electron
pressure :math:`P_e = n_e k_B T_e` then feeds back into Ohm's law.

The homogeneous part of the equation (the left-hand side) is solved with the
QDSMC kinetic-enslavement scheme of :cite:t:`kfhm-Belyaev2024`: the electron
entropy function :math:`K_e = T_e\, n_e^{1-\gamma}`, which the transport terms
conserve along electron-fluid characteristics, is advected by fictitious
Lagrangian markers. Each PIC step one marker is initialized at every cell
center carrying the local :math:`K_e N_e` and :math:`N_e` (with :math:`N_e`
the electron count of the cell), is pushed by one timestep with
:math:`\vec{V}_e` interpolated at its position, and both quantities are
deposited back to the grid with the standard (linear) particle shape factors.
The updated temperature is recovered from the deposited quantities and the
ion-derived density as

    .. math::

        T_e = \frac{\sum K_e N_e}{\sum N_e}\, n_e^{\gamma - 1}.

Since the scheme only advects the electron entropy, thermal conduction is
neglected (:math:`\nabla\cdot\vec{q}_e = 0`).

Two source terms can be enabled on the right-hand side. The first is the Joule
(Ohmic) heating consistent with the resistive friction in Ohm's law
(``hybrid_pic_model.include_joule_heating``), applied per ion species
:math:`s`:

    .. math::

        \frac{d T_e}{d t} = (\gamma - 1) \sum_s \frac{Z_s e^2\, \eta_{s,\mathrm{eff}}\, n_s |\Delta\vec{V}|^2}{k_B},

where :math:`\Delta\vec{V} = \vec{J}/(e n_e)` is the electron-ion relative
drift, :math:`Z_s` the charge state and :math:`\eta_{s,\mathrm{eff}} = \eta`
the Ohm's-law resistivity. For a single species this reduces
exactly to the familiar :math:`dT_e/dt = (\gamma - 1)\,\eta J^2/(n_e k_B)`.
Above a user-set electron temperature threshold the heat can optionally be
redirected to the kinetic ions instead of the electron fluid
(``hybrid_pic_model.joule_redirect_Te_threshold``), which is useful to model
regimes where the electrons radiate strongly.

The second source is the electron-ion temperature relaxation, enabled by
specifying the rate ``hybrid_pic_model.electron_ion_relaxation_rate``,

    .. math::

        Q_{ei} = \sum_s 3\, n_s k_B\, \nu_{ei}\, (T_e - T_{i,s}),

with the rate :math:`\nu_{ei}(\rho, T_e, T_i, t)` given by a user expression.
The sink on the electron fluid is paired with a matching thermal-velocity
kick on the ion macro-particles of each species so that the exchange
conserves energy exactly.

Verification tests of the transport terms (adiabatic compression, and slab
transport through a below-floor halo), the Joule source (force-free field
decay) and the :math:`Q_{ei}` exchange are described in the
:ref:`examples section <examples-ohm-solver-electron-energy-eq>`.

Electron current
^^^^^^^^^^^^^^^^

WarpX's displacement current diagnostic can be used to output the electron current in
the kinetic-fluid hybrid model since in the absence of kinetic electrons, and under
the assumption of zero displacement current, that diagnostic simply calculates the
hybrid model's electron current.

Model derivation
----------------

The basic justification for the hybrid model is that the system to which it is
applied is dominated by ion kinetics, with ions moving much slower than electrons
and photons. In this scenario two critical approximations can be made, namely,
neutrality (:math:`n_e=n_i`) and the Maxwell-Ampere equation can be simplified by
neglecting the displacement current term :cite:p:`kfhm-Nielson1976`, giving,

    .. math::

        \mu_0\boldsymbol{J} = \boldsymbol{\nabla}\times\boldsymbol{B},

where :math:`\boldsymbol{J} = \sum_{s\neq e}\boldsymbol{J}_s + \boldsymbol{J}_e + \boldsymbol{J}_{ext}` is the total electrical current,
i.e. the sum of electron and ion currents as well as any external current (not captured through plasma
particles). Since ions are treated in the regular
PIC manner, the ion current, :math:`\sum_{s\neq e}\boldsymbol{J}_s`, is known during a simulation. Therefore,
given the magnetic field, the electron current can be calculated.

The electron momentum transport equation (obtained from multiplying the Vlasov equation by mass and
integrating over velocity), also called the generalized Ohm's law, is given by:

    .. math::

        en_e\boldsymbol{E} = \frac{m}{e}\frac{\partial \boldsymbol{J}_e}{\partial t} + \frac{m}{e}\left( \boldsymbol{U}_e\cdot\boldsymbol{\nabla} \right) \boldsymbol{J}_e - \boldsymbol{\nabla}\cdot {\overleftrightarrow P}_e - \boldsymbol{J}_e\times\boldsymbol{B}+\boldsymbol{R}_e

where :math:`\boldsymbol{U}_e = \boldsymbol{J}_e/(en_e)` is the electron fluid velocity,
:math:`{\overleftrightarrow P}_e` is the electron pressure tensor and
:math:`\boldsymbol{R}_e` is the drag force due to collisions between electrons and ions.
Applying the above momentum equation to the Maxwell-Faraday equation (:math:`\frac{\partial\boldsymbol{B}}{\partial t} = -\boldsymbol{\nabla}\times\boldsymbol{E}`)
and substituting in :math:`\boldsymbol{J}` calculated from the Maxwell-Ampere equation, gives,

    .. math::

        \frac{\partial\boldsymbol{J}_e}{\partial t} = -\frac{1}{\mu_0}\boldsymbol{\nabla}\times\left(\boldsymbol{\nabla}\times\boldsymbol{E}\right) - \frac{\partial\boldsymbol{J}_{ext}}{\partial t} - \sum_{s\neq e}\frac{\partial\boldsymbol{J}_s}{\partial t}.

Plugging this back into the generalized Ohm's law gives:

    .. math::

        \left(en_e +\frac{m}{e\mu_0}\boldsymbol{\nabla}\times\boldsymbol{\nabla}\times\right)\boldsymbol{E} =&
        - \frac{m}{e}\left( \frac{\partial\boldsymbol{J}_{ext}}{\partial t} + \sum_{s\neq e}\frac{\partial\boldsymbol{J}_s}{\partial t} \right) \\
        &+ \frac{m}{e}\left( \boldsymbol{U}_e\cdot\boldsymbol{\nabla} \right) \boldsymbol{J}_e - \boldsymbol{\nabla}\cdot {\overleftrightarrow P}_e - \boldsymbol{J}_e\times\boldsymbol{B}+\boldsymbol{R}_e.

If we now further assume electrons are inertialess (i.e. :math:`m=0`), the above equation simplifies to,

    .. math::

        en_e\boldsymbol{E} = -\boldsymbol{J}_e\times\boldsymbol{B}-\boldsymbol{\nabla}\cdot{\overleftrightarrow P}_e+\boldsymbol{R}_e.

Making the further simplifying assumptions that the electron pressure is isotropic and that
the electron drag term can be written using a simple resistivity (:math:`\eta`) and hyper-resistivity (:math:`\eta_h`)
i.e. :math:`\boldsymbol{R}_e = en_e(\eta-\eta_h \nabla^2)\boldsymbol{J}`, brings us to the implemented form of
Ohm's law:

    .. math::

        \boldsymbol{E} = -\frac{1}{en_e}\left( \boldsymbol{J}_e\times\boldsymbol{B} + \boldsymbol{\nabla} P_e \right)+\eta\boldsymbol{J}-\eta_h \nabla^2\boldsymbol{J}.

Lastly, if an electron temperature is given from which the electron pressure can
be calculated, the model is fully constrained and can be evolved given initial
conditions.

.. bibliography::
    :keyprefix: kfhm-
