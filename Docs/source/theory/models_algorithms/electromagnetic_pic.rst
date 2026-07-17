.. _theory-em-pic:

Electromagnetic PIC
===================

In the *electromagnetic Particle-In-Cell method* :cite:p:`pt-Birdsalllangdon,pt-HockneyEastwoodBook`,
the fields are updated using Maxwell's equations:

.. math::
   \frac{\partial \boldsymbol{B}}{\partial t} = -\nabla\times \boldsymbol{E}
   :label: Faraday-1

.. math::
   \frac{1}{c^2}\frac{\partial \boldsymbol{E}}{\partial t} = \nabla\times \boldsymbol{B}-\mu_0 \boldsymbol{j}
   :label: Ampere-1

where :math:`\boldsymbol{E}` and :math:`\boldsymbol{B}` are the electric and magnetic field
components, and :math:`\boldsymbol{j}` is the current density.

Because the electromagnetic PIC method retains the full Maxwell equations,
this method can capture the **physics of the electromagnetic waves**,
including their propagation and self-consistent interaction with particles.

The electromagnetic PIC method can be run either with an explicit or implicit time integration scheme:

   - In the **explicit integration scheme**, the particles and fields are updated sequentially at each time step
     (see :ref:`theory-explicit-em-pic`). This integration scheme is simple, but requires a small enough time step
     size :math:`\Delta t` to ensure the stability of the simulation (e.g., CFL condition :math:`c\Delta t \lessapprox \Delta x`,
     need to resolve the plasma frequency :math:`\omega_p \Delta t \leq 2` :cite:p:`pt-Birdsalllangdon,pt-HockneyEastwoodBook`).

   - In the **implicit integration scheme**, the particles and fields are updated simultaneously at each time step, using
     an iterative solver (see :ref:`theory-implicit-em-pic`). While this integration scheme is more complex, it can use
     larger time step sizes :math:`\Delta t` and still retain the stability of the simulation. In addition, the implicit
     integration scheme is exactly energy conserving.

For more details, see the sections below:

.. toctree::
   :maxdepth: 1

   explicit_em_pic
   implicit_em_pic
