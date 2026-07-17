.. _theory-kinetic-particles:

Kinetic Particles
=================

.. _theory-kinetic-particles-push:

Particle push
-------------

A centered finite-difference discretization of the Newton-Lorentz
equations of motion is given by

.. math::
   \frac{\mathbf{x}^{i+1}-\mathbf{x}^{i}}{\Delta t} = \mathbf{v}^{i+1/2},
   :label: leapfrog_x

.. math::
   \frac{\gamma^{i+1/2}\mathbf{v}^{i+1/2}-\gamma^{i-1/2}\mathbf{v}^{i-1/2}}{\Delta t} = \frac{q}{m}\left(\mathbf{E}^{i}+\mathbf{\bar{v}}^{i}\times\mathbf{B}^{i}\right).
   :label: leapfrog_v

In order to close the system, :math:`\bar{\mathbf{v}}^{i}` must be
expressed as a function of the other quantities. The two implementations that have become the most popular are presented below.

.. _theory-kinetic-particles-push-boris:

Boris relativistic velocity rotation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The solution proposed by Boris :cite:p:`kp-BorisICNSP70` is given by

.. math::
   \mathbf{\bar{v}}^{i} = \frac{\gamma^{i+1/2}\mathbf{v}^{i+1/2}+\gamma^{i-1/2}\mathbf{v}^{i-1/2}}{2\bar{\gamma}^{i}}
   :label: boris_v

where :math:`\bar{\gamma}^{i}` is defined by :math:`\bar{\gamma}^{i} \equiv (\gamma^{i+1/2}+\gamma^{i-1/2} )/2`.

The system (:eq:`leapfrog_v`, :eq:`boris_v`) is solved very
efficiently following Boris’ method, where the electric field push
is decoupled from the magnetic push. Setting :math:`\mathbf{u}=\gamma\mathbf{v}`, the
velocity is updated using the following sequence:

.. math::

   \begin{aligned}
   \mathbf{u^{-}}     & = \mathbf{u}^{i-1/2}+\left(q\Delta t/2m\right)\mathbf{E}^{i}
   \\
   \mathbf{u'}        & = \mathbf{u}^{-}+\mathbf{u}^{-}\times\mathbf{t}
   \\
   \mathbf{u}^{+}     & = \mathbf{u}^{-}+\mathbf{u'}\times2\mathbf{t}/(1+\mathbf{t}^{2})
   \\
   \mathbf{u}^{i+1/2} & = \mathbf{u}^{+}+\left(q\Delta t/2m\right)\mathbf{E}^{i}
   \end{aligned}

where :math:`\mathbf{t}=\left(q\Delta t/2m\right)\mathbf{B}^{i}/\bar{\gamma}^{i}` and where
:math:`\bar{\gamma}^{i}` can be calculated as :math:`\bar{\gamma}^{i}=\sqrt{1+(\mathbf{u}^-/c)^2}`.

The Boris implementation is second-order accurate, time-reversible and fast. Its implementation is very widespread and used in the vast majority of PIC codes.

.. _theory-kinetic-particles-push-vay:

Vay Lorentz-invariant formulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It was shown in :cite:t:`kp-Vaypop2008` that the Boris formulation is
not Lorentz invariant and can lead to significant errors in the treatment
of relativistic dynamics. A Lorentz invariant formulation is obtained
by considering the following velocity average

.. math::
   \mathbf{\bar{v}}^{i} = \frac{\mathbf{v}^{i+1/2}+\mathbf{v}^{i-1/2}}{2}.
   :label: new_v

This gives a system that is solvable analytically (see :cite:t:`kp-Vaypop2008`
for a detailed derivation), giving the following velocity update:

.. math::
   \mathbf{u^{*}} = \mathbf{u}^{i-1/2}+\frac{q\Delta t}{m}\left(\mathbf{E}^{i}+\frac{\mathbf{v}^{i-1/2}}{2}\times\mathbf{B}^{i}\right),
   :label: pusher_gamma

.. math::
   \mathbf{u}^{i+1/2} = \frac{\mathbf{u^{*}}+\left(\mathbf{u^{*}}\cdot\mathbf{t}\right)\mathbf{t}+\mathbf{u^{*}}\times\mathbf{t}}{1+\mathbf{t}^{2}},
   :label: pusher_upr

where

.. math::

   \begin{align}
   \mathbf{t} & = \boldsymbol{\tau}/\gamma^{i+1/2},
   \\
   \boldsymbol{\tau} & = \left(q\Delta t/2m\right)\mathbf{B}^{i},
   \\
   \gamma^{i+1/2} & = \sqrt{\sigma+\sqrt{\sigma^{2}+\left(\boldsymbol{\tau}^{2}+w^{2}\right)}},
   \\
   w & = \mathbf{u^{*}}\cdot\boldsymbol{\tau},
   \\
   \sigma & = \left(\gamma'^{2}-\boldsymbol{\tau}^{2}\right)/2,
   \\
   \gamma' & = \sqrt{1+(\mathbf{u}^{*}/c)^{2}}.
   \end{align}

This Lorentz invariant formulation
is particularly well suited for the modeling of ultra-relativistic
charged particle beams, where the accurate account of the cancellation
of the self-generated electric and magnetic fields is essential, as
shown in :cite:t:`kp-Vaypop2008`.

.. bibliography::
    :keyprefix: kp-
