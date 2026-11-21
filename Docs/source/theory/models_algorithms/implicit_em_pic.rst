.. _theory-implicit-em-pic:

Implicit electromagnetic PIC
============================

Theta-implicit PML
------------------

For exterior vacuum PML, the transverse split fields are advanced with the
same time centering as the regular fields. For a split field component
:math:`u` with conductivity :math:`\sigma`, the stage equation is

.. math::

    u^{n+\theta} = u^n + \theta\Delta t
        \left(C^{n+\theta} - \sigma u^{n+\theta}\right),

where :math:`C` is the appropriate directional curl contribution, including
:math:`c^2` for the electric field. The existing Yee PML curl stencils and
conductivity profiles are used.

The split electric fields are included in the nonlinear solver vector.
The magnetic stage is eliminated locally as

.. math::

    B_s^{n+\theta} = \frac{B_s^n + \theta\Delta t C_s(E^{n+\theta})}
                              {1 + \theta\Delta t\sigma_s}.

Each residual evaluation reconstructs this stage from the saved old-time
magnetic field. The PML and regular grid exchange fields at the stage time;
regular-grid nodes shared with the PML interface are counted only once in
solver norms. After convergence, both fields are advanced using
:math:`u^{n+1} = [u^{n+\theta}-(1-\theta)u^n]/\theta`.
The explicit solver's separate exponential PML damping step is not applied.
The supported configurations are listed under :pp:param:`algo.evolve_scheme`.
