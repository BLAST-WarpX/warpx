.. _theory-implicit-em-pic:

Implicit electromagnetic PIC
============================

Theta-implicit refined grids and exterior PML
---------------------------------------------

For Cartesian staggered Yee grids, the theta-implicit nonlinear state contains
all fine-patch, coarse-patch, and exterior-PML electric fields. Each residual
evaluation restores every magnetic patch to its step-start state before computing
its value at :math:`n+\theta`. Refined particle gathers reconstruct auxiliary
fields from that trial state.

In macroscopic media the electric residual uses

.. math::

   \boldsymbol{E}^{n+\theta} - \boldsymbol{E}^{n}
   = \frac{\theta\,\Delta t}{\epsilon}
     \left(\nabla\times\boldsymbol{H}^{n+\theta}
       -\boldsymbol{J}^{n+1/2}-\sigma\boldsymbol{E}^{n+\theta}\right),
   \qquad \boldsymbol{H}=\boldsymbol{B}/\mu.

The exterior PML retains the Cartesian split-field formulation. Electric split
components contribute :math:`-\theta\Delta t\,\sigma_d E_d^{n+\theta}` to the
residual. Magnetic split components solve their local damping term implicitly,
using :math:`(B_d^n-\theta\Delta t\,(\nabla\times E)_d)/(1+\theta\Delta t\,\sigma_d)`.
Shared regular/PML boundary nodes belong to the regular grid and are excluded
from PML solver norms. PML cells retain the vacuum material model.

The accepted fine, coarse, and PML fields are extrapolated from :math:`n+\theta`
to :math:`n+1`. Current-controlled ports are then projected at the accepted time.
Checkpoints retain the physical fields and PML split components; coarse/PML
initial guesses are reconstructed from the saved current fields after restart.

This configuration requires Newton with native AMReX GMRES,
``jacobian.pc_type = none`` and no mass
matrices, filtering, divergence cleaning, moving window, or dynamic load balancing.
