.. _theory-electrostatic-pic:

Electrostatic PIC
=================

In the *electrostatic particle-in-cell method* only the electric field is
self-consistently updated with the particle motion. This approach uses the
Poisson equation to obtain the electrostatic potential from the charge density
(which is obtained directly from the simulation macro-particle). There are a
few different variations of the electrostatic PIC method implemented in WarpX
as outlined below. For details of the possible input parameters for each case
see :ref:`here <param-electrostatic-pic>`.

.. _theory-electrostatic-pic-labframe:

Labframe
--------

Poisson's equation is solved in the lab frame with
the charge density of all species combined. More specifically, the code solves:

.. math::

    \boldsymbol{\nabla}^2 \phi = - \rho/\epsilon_0 \qquad \boldsymbol{E} = - \boldsymbol{\nabla}\phi

.. _theory-electrostatic-pic-electromagnetostatic:

Electromagnetostatic
--------------------

Poisson's equation is solved in the lab frame with
the charge density of all species combined.  Additionally the 3-component vector potential
is solved in the Coulomb Gauge with the current density of all species combined
to include self magnetic fields. More specifically, the code solves:

.. math::

    \boldsymbol{\nabla}^2 \phi = - \rho/\epsilon_0 \qquad \boldsymbol{E} = - \boldsymbol{\nabla}\phi \\
    \boldsymbol{\nabla}^2 \boldsymbol{A} = - \mu_0 \boldsymbol{j} \qquad \boldsymbol{B} = \boldsymbol{\nabla}\times\boldsymbol{A}

.. _theory-electrostatic-pic-effective-potential:

Effective Potential
-------------------

Poisson's equation is solved with a modified dielectric function
(resulting in an "effective potential") to create a semi-implicit scheme which is robust to the
numerical instability seen in explicit electrostatic PIC when :math:`\Delta t \omega_{pe} > 2`.
If this option is used the additional parameter ``warpx.effective_potential_factor`` can also be
specified to set the value of :math:`C_{EP}` (default 4). The method is stable for :math:`C_{EP} \geq 1`
regardless of :math:`\Delta t`, however, the larger :math:`C_{EP}` is set, the lower the numerical plasma
frequency will be and therefore care must be taken to not set it so high that the plasma mode
hybridizes with other modes of interest.
Details of the method can be found in Appendix A of :cite:t:`es-Barnes2021` (note that in that paper
the method is referred to as "semi-implicit electrostatic" but here it has been renamed to "effective potential"
to avoid confusion with the semi-implicit method of Chen et al.).
In short, the code solves:

.. math::

    \boldsymbol{\nabla}\cdot\left(1+\frac{C_{EP}}{4}\sum_{s \in \text{species}}(\omega_{ps}\Delta t)^2 \right)\boldsymbol{\nabla} \phi = - \rho/\epsilon_0 \qquad \boldsymbol{E} = - \boldsymbol{\nabla}\phi

Relativistic
------------

Poisson's equation is solved **for each species**
in their respective rest frame. The corresponding field
is mapped back to the simulation frame and will produce both E and B
fields. More specifically, in the simulation frame, this is equivalent to solving **for each species**

.. math::

    \boldsymbol{\nabla}^2 - (\boldsymbol{\beta}\cdot\boldsymbol{\nabla})^2\phi = - \rho/\epsilon_0 \qquad
    \boldsymbol{E} = -\boldsymbol{\nabla}\phi + \boldsymbol{\beta}(\boldsymbol{\beta} \cdot \boldsymbol{\nabla}\phi)
    \qquad \boldsymbol{B} = -\frac{1}{c}\boldsymbol{\beta}\times\boldsymbol{\nabla}\phi

where :math:`\boldsymbol{\beta}` is the average (normalized) velocity of the considered species (which can be relativistic).
See, e.g., :cite:t:`es-Vaypop2008` for more information.

.. bibliography::
    :keyprefix: es-
