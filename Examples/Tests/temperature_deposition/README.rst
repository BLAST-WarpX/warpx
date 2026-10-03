Weighted particle temperature deposition
========================================

These native RZ and 3D tests keep a small set of ions at fixed positions and
deposit their temperatures through the hybrid solver. Disabling particle push
and gather isolates the estimator from field evolution.

For two independent samples with positive effective weights, the unbiased
weighted variance is exactly half the squared difference of the two velocities.
The analysis uses this identity, independently of the implementation's weight
moments. It checks equal and unequal macroparticle weights, unequal shape weights
at equal macroparticle weight, and invariance under a common weight scale from
``1.e-25`` to ``1.e25``. In RZ it samples the axis, the adjacent cell and the
interior. A particle pair straddles box boundaries so that two-rank runs exercise
the communication between the two deposition passes. Empty, single-sample and
numerically single-effective-sample regions must have finite zero temperature.

The temperature fields in these tests are the shape-matched vector temperatures
used by the hybrid solver, in kelvin. They are distinct from the scalar NGP
``T_<species>`` diagnostic. Filtering is disabled so the checks apply directly to
the deposited estimator.
