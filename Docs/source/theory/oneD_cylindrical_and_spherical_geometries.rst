.. _theory-grid-cylindrical_rz:

Cylindrical and Spherical Geometries
=================

Coordinates
-------------

.. _fig-axisymmetric_coordinates:

.. figure:: Cyl_and_Sph_Coords.png
   :alt: figure not found

   Cylindrical (left) and spherical (right) coordinate systems. The azimuthal angle :math:`\theta` is the same in both systems and ranges from :math:`0` to :math:`2\pi`. The elevation angle :math:`\phi` in the spherical system ranges from :math:`-\pi/2` to :math:`\pi/2`.

1D axisymmetric spherical coordinates
-------------

The general form of the differential divergence, gradient, and curl operators in a spherical coordinate system are

.. math::

   \begin{aligned}
   \nabla\cdot\mathbf{F}     & = \frac{1}{r^2}\frac{\partial}{\partial r}\left(r^2F_r\right) + \frac{1}{r\cos\phi}\frac{\partial F_{\theta}}{\partial\theta} + \frac{1}{r\cos\phi}\frac{\partial}{\partial\phi}\left(\cos\phi F_{\phi}\right)
   \\
   \nabla\times\mathbf{F}    & = \frac{1}{r}\left[\frac{\partial F_{\phi}}{\partial r} - \frac{\partial}{\partial}\left(\cos\phi F_{\theta}\right)\right]\hat{\mathbf{r}}
   \\
   \end{aligned}

where :math:`\mathbf{t}=\left(q\Delta t/2m\right)\mathbf{B}^{i}/\bar{\gamma}^{i}` and where

Jacobian
-------------

.. bibliography::
    :keyprefix: kp-
