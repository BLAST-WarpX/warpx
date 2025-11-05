Fully electromagnetic PIC
=========================

In the *fully electromagnetic particle-in-cell method* :cite:p:`pt-Birdsalllangdon,pt-HockneyEastwoodBook`,
the fields are updated using Maxwell's equations.

.. math::
   \frac{\mathbf{\partial B}}{\partial t} = -\nabla\times\mathbf{E}
   :label: Faraday-1

.. math::
   \frac{\mathbf{\partial E}}{\partial t} = \nabla\times\mathbf{B}-\mathbf{J}
   :label: Ampere-1

where :math:`\mathbf{E}` and :math:`\mathbf{B}` are the electric and magnetic field
components, and :math:`\mathbf{J}` is the current density.

.. warning::

   TODO: Mention current deposition ; introduce different method.
   TODO: Mention what physics can be captured, at which cost
   TODO: Change units

.. toctree::
   :maxdepth: 1

   explicit_em_pic
   implicit_em_pic