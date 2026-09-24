.. _theory-darwin-solver:

Darwin Solver
=============

Many problems in plasma physics involve self-consistent magnetic fields and inductive electric fields,
but no electromagnetic radiation. Examples include low-frequency Alfvén and whistler waves,
magnetized plasma instabilities, and the self-fields of slowly moving particle beams.
For these problems, an electromagnetic PIC simulation must still resolve the propagation of light waves
(e.g., through the CFL condition :math:`c\Delta t \lessapprox \Delta x`), even though they play no role
in the physics of interest. Simpler electrostatic PIC simulations, on the other hand, cannot capture the
magnetic and inductive effects.

The *Darwin model* :cite:p:`dw-Nielson1976` sits in between these two approaches: it removes light waves from
Maxwell's equations while retaining the low-frequency (magnetoinductive) physics. It does so by
decomposing the electric field into an irrotational (curl-free) and a solenoidal (divergence-free) component,

.. math::

    \boldsymbol{E} = \boldsymbol{E}_{irr} + \boldsymbol{E}_{sol} \qquad \boldsymbol{E}_{irr} = -\boldsymbol{\nabla}\phi
    \qquad \boldsymbol{E}_{sol} = -\frac{\partial\boldsymbol{A}}{\partial t} \qquad \boldsymbol{B} = \boldsymbol{\nabla}\times\boldsymbol{A},

where the vector potential :math:`\boldsymbol{A}` is taken in the Coulomb gauge (:math:`\boldsymbol{\nabla}\cdot\boldsymbol{A} = 0`),
and by dropping the displacement current associated with the solenoidal component from the Maxwell-Ampere equation:

.. math::

    \boldsymbol{\nabla}\times\boldsymbol{B} = \mu_0\boldsymbol{j} + \frac{1}{c^2}\frac{\partial\boldsymbol{E}_{irr}}{\partial t}.

The other Maxwell equations are unchanged. In terms of the potentials, the fields are then obtained from:

.. math::

    \boldsymbol{\nabla}^2 \phi = - \rho/\epsilon_0 \qquad
    -\boldsymbol{\nabla}^2 \left(\boldsymbol{\nabla}\times\boldsymbol{A}\right) = \mu_0\boldsymbol{\nabla}\times\boldsymbol{j}

(where the second equation is the curl of the modified Maxwell-Ampere equation, which eliminates the :math:`\boldsymbol{E}_{irr}` term).
Because there is no longer a time derivative of :math:`\boldsymbol{E}_{sol}` in the field equations, light waves are
removed from the system and the time step is not constrained by the speed of light.
The Darwin model is accurate to first order in :math:`v^2/c^2`, and is therefore valid when the particles and the relevant
phase velocities are non-relativistic, and when radiation plays no significant role.

For details of the possible input parameters, see :pp:param:`algo.evolve_scheme` = ``semi_implicit_darwin``.

Algorithm details
-----------------

.. note::

    A verification test of the Darwin solver (the dispersion of Alfvén modes in a magnetized plasma) can be found in
    the :ref:`examples section <examples-magnetized-plasma-modes>`.

The Darwin solver in WarpX is *semi-implicit*: at each time step, the electrostatic field is computed explicitly
(with any of the electrostatic solvers described in :ref:`theory-electrostatic-pic`), while the inductive field is obtained
from a single *linear* implicit solve that includes the response of the plasma. Particle positions :math:`\boldsymbol{x}` and
the potential :math:`\phi` are defined at integer time steps, while particle momenta :math:`\boldsymbol{u}`,
the vector potential :math:`\boldsymbol{A}` and the magnetic field :math:`\boldsymbol{B}` are defined at half-integer time steps.

Magnetoinductive equation
^^^^^^^^^^^^^^^^^^^^^^^^^

To guarantee the Coulomb gauge by construction, the change of the vector potential over one time step is
written in terms of an auxiliary field :math:`\boldsymbol{Z}^n`:

.. math::

    \boldsymbol{A}^{n+1/2} - \boldsymbol{A}^{n-1/2} = \boldsymbol{\nabla}\times\boldsymbol{Z}^n
    \qquad \boldsymbol{E}_{sol}^n = -\frac{\boldsymbol{\nabla}\times\boldsymbol{Z}^n}{\Delta t}.

Evaluating the equation for :math:`\boldsymbol{A}` at time :math:`t_n` with :math:`\boldsymbol{A}^n = \boldsymbol{A}^{n-1/2} + \boldsymbol{\nabla}\times\boldsymbol{Z}^n/2`
then gives a fourth-order equation for :math:`\boldsymbol{Z}^n`:

.. math::

    \boldsymbol{\nabla}^4\boldsymbol{Z}^n = 2\mu_0\boldsymbol{\nabla}\times\boldsymbol{j}^n + 2\boldsymbol{\nabla}^2\boldsymbol{B}^{n-1/2}.

This equation is implicit, since the current :math:`\boldsymbol{j}^n` is deposited with the time-centered momenta
:math:`(\boldsymbol{u}^{n+1/2} + \boldsymbol{u}^{n-1/2})/2`, which themselves depend on :math:`\boldsymbol{E}_{sol}^n`
and therefore on :math:`\boldsymbol{Z}^n`.

Plasma response and mass matrices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The momentum update (with :math:`\boldsymbol{B}^{n-1/2}` held fixed) is linear in the electric field. The current :math:`\boldsymbol{j}^n`
can therefore be split exactly into a known part :math:`\hat{\boldsymbol{j}}^n`, deposited from momenta that were pushed with
:math:`\boldsymbol{E}_{irr}^n` only, and a part that is linear in the unknown :math:`\boldsymbol{E}_{sol}^n`.
The latter is expressed with the *mass matrices*, i.e., the linearized response of the deposited current to the electric field,
which are deposited from the particles together with :math:`\hat{\boldsymbol{j}}^n` (the same mass matrices are used by the
:ref:`implicit electromagnetic PIC <theory-implicit-em-pic>` solvers).
Substituting this into the equation above yields the linear magnetoinductive equation:

.. math::

    \boldsymbol{\nabla}^4\boldsymbol{Z}^n + \boldsymbol{\nabla}\times\left(\chi\,\boldsymbol{\nabla}\times\boldsymbol{Z}^n\right)
    = 2\mu_0\boldsymbol{\nabla}\times\hat{\boldsymbol{j}}^n + 2\boldsymbol{\nabla}^2\boldsymbol{B}^{n-1/2},

where :math:`\chi` denotes the mass matrices scaled by :math:`2\mu_0/\Delta t`.
This equation is solved with a matrix-free GMRES solver, optionally preconditioned by a multigrid (MLMG) solve of the approximate
operator :math:`\boldsymbol{\nabla}^2\left(\boldsymbol{\nabla}^2 + \chi\right)`, in which :math:`\chi` is replaced by a local, diagonal
approximation of the mass matrices.
Since the equation is linear in :math:`\boldsymbol{Z}^n`, a single linear solve per time step is sufficient, and no nonlinear
(Picard or Newton) iteration is required.

Time step
^^^^^^^^^

In summary, each time step performs the following operations:

   - The particle momenta are pushed from :math:`\boldsymbol{u}^{n-1/2}` with the electrostatic field :math:`\boldsymbol{E}_{irr}^n` only (predictor push).
   - The current :math:`\hat{\boldsymbol{j}}^n` and the mass matrices are deposited, using the time-centered momenta.
   - The magnetoinductive equation is solved for :math:`\boldsymbol{Z}^n`, and the inductive field :math:`\boldsymbol{E}_{sol}^n = -\boldsymbol{\nabla}\times\boldsymbol{Z}^n/\Delta t` is computed.
   - The momenta are corrected with the acceleration due to :math:`\boldsymbol{E}_{sol}^n` to obtain :math:`\boldsymbol{u}^{n+1/2}`, and the positions are pushed to :math:`\boldsymbol{x}^{n+1}`.
   - The magnetic field is advanced to :math:`\boldsymbol{B}^{n+1/2}` with :math:`\partial\boldsymbol{B}/\partial t = -\boldsymbol{\nabla}\times\boldsymbol{E}_{sol}`,
     which is equivalent to updating :math:`\boldsymbol{A}` and taking its curl, without storing :math:`\boldsymbol{A}`.
   - The electrostatic potential :math:`\phi^{n+1}` is computed from the charge density at the new positions.

Note that :math:`\boldsymbol{\nabla}^4\boldsymbol{Z}` equals :math:`-\boldsymbol{\nabla}^2\left(\boldsymbol{\nabla}\times\boldsymbol{\nabla}\times\boldsymbol{Z}\right)`
only if :math:`\boldsymbol{\nabla}\cdot\boldsymbol{Z} = 0`, which is not explicitly enforced. The update of :math:`\boldsymbol{A}` itself
remains divergence-free by construction.

.. bibliography::
    :keyprefix: dw-
