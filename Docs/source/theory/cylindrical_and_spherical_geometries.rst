.. _theory-grid-cylindrical-spherical:

Cylindrical and Spherical Geometries
====================================

.. _fig_cyl_and_sph_coords:

.. figure:: Cyl_and_Sph_Coords.png
   :alt: Figure not found

   Cylindrical (left) and spherical (right) coordinate systems. The azimuthal angle :math:`\theta` is the same in both systems and ranges from :math:`0` to :math:`2\pi`. The elevation angle :math:`\phi` in the spherical system ranges from :math:`-\pi/2` to :math:`\pi/2`.

The cylindrical and spherical coordinate systems are shown in :numref:`fig_cyl_and_sph_coords` (see :cite:t:`cyl_sph-Angusjcp2024` for further details).

Cylindrical coordinates
--------------------------

The cylindrical coordinate system is shown in the left panel of :numref:`fig_cyl_and_sph_coords`. General expressions for the divergence and curl operators in this coordinate system are

.. math::

   \begin{aligned}
   \nabla\cdot\mathbf{F}     & = \frac{1}{r}\frac{\partial}{\partial r}\left(rF_r\right) + \frac{1}{r}\frac{\partial F_{\theta}}{\partial\theta} + \frac{\partial F_z}{\partial z},
   \\
   \nabla\times\mathbf{F}    & = \left[\frac{1}{r}\frac{\partial F_z}{\partial \theta} - \frac{\partial F_{\theta}}{\partial z}\right]\hat{\mathbf{r}} + \left[\frac{\partial F_r}{\partial z} - \frac{\partial F_z}{\partial r}\right]\hat{\boldsymbol\theta} + \frac{1}{r}\left[\frac{\partial}{\partial r}\left(rF_{\theta}\right) - \frac{1}{r}\frac{\partial F_r}{\partial\theta}\right]\hat{\mathbf{z}}.
   \end{aligned}

WarpX supports two cylindrical geometries: A 2D-RZ one with optional azimuthal modes and a 1D axisymmetric one.

1D cylindrical coordinates
^^^^^^^^^^^^^^^^^^^^^^^^^^

All vector fields :math:`\mathbf{F}=F_r\hat{\mathbf{r}} + F_{\theta}\hat{\boldsymbol\theta} + F_{z}\hat{\mathbf{z}}` and scalar fields :math:`f` are assumed to be *cylindrically symmetric* (:math:`\partial/\partial\theta = 0`) and *axially uniform* (:math:`\partial/\partial z = 0`) in the one-dimensional cylindrical coordinate system. That is, :math:`\mathbf{F}` is invariant under rotations about the origin and translations in the Z-direction, and :math:`f=f(r)`. Under these assumptions, the differential operators reduce to

.. math::

   \begin{aligned}
   \nabla\cdot\mathbf{F}   &= \frac{1}{r}\frac{\partial}{\partial r}\left(rF_r\right),
   \\
   \nabla\times\mathbf{F}  &= 0\hat{\mathbf{r}} - \frac{\partial F_z}{\partial r}\hat{\boldsymbol\theta} + \frac{1}{r}\frac{\partial}{\partial r}\left(rF_{\theta}\right)\hat{\mathbf{z}}.
   \end{aligned}

Spherical coordinates
------------------------

The spherical coordinate system is shown in the right panel of :numref:`fig_cyl_and_sph_coords`. General expressions for the divergence and curl operators in this coordinate system are

.. math::

   \begin{aligned}
   \nabla\cdot\mathbf{F}     & = \frac{1}{r^2}\frac{\partial}{\partial r}\left(r^2F_r\right) + \frac{1}{r\cos\phi}\frac{\partial F_{\theta}}{\partial\theta} + \frac{1}{r\cos\phi}\frac{\partial}{\partial\phi}\left(\cos\phi F_{\phi}\right),
   \\
   \nabla\times\mathbf{F}    & = \frac{1}{r}\left[\frac{\partial F_{\phi}}{\partial r} - \frac{\partial}{\partial\phi}\left(\cos\phi F_{\theta}\right)\right]\hat{\mathbf{r}} + \frac{1}{r}\left[\frac{\partial F_r}{\partial\phi} - \frac{\partial}{\partial r}\left(rF_{\phi}\right)\right]\hat{\boldsymbol\theta} + \frac{1}{r}\left[\frac{\partial}{\partial r}\left(rF_{\theta}\right) - \frac{1}{\cos\phi}\frac{\partial F_r}{\partial\theta}\right]\hat{\boldsymbol\phi}.
   \end{aligned}

For spherical coordinates, WarpX only supports a 1D axisymmetric version.

1D spherical coordinates
^^^^^^^^^^^^^^^^^^^^^^^^

All vector fields :math:`\mathbf{F}=F_r\hat{\mathbf{r}} + F_{\theta}\hat{\boldsymbol\theta} + F_{\phi}\hat{\boldsymbol\phi}` and scalar fields :math:`f` are assumed to be *spherically symmetric* in the one-dimensional spherical coordinate system. That is, :math:`\mathbf{F}` is invariant under all rotations about the origin and :math:`f=f(r)`. Mathematically, this implies

.. math::

   F_{\theta} = F_{\phi} = 0, \qquad \frac{\partial f}{\partial\theta} = \frac{\partial f}{\partial\phi} = 0,

where :math:`f` is any scalar field. Under these assumptions, the differential operators reduce to

.. math::

   \nabla\cdot\mathbf{F} = \frac{1}{r^2}\frac{\partial}{\partial r}\left(r^2F_r\right), \qquad \nabla\times\mathbf{F} = 0.

.. bibliography::
    :keyprefix: cyl_sph-
