# Homogeneous directed-radiation acceleration

This is an additional radiation-specific verification, not a replacement for
the failed cold-contact controls or a new FLASH closure. The existing FLASH
MGD reference uses gradient-driven diffusion rather than an independently
initialized homogeneous flux; this case therefore uses an independent ODE.

Let e=E/E0 and q=F/(c E0) along the initial beam. Define epsilon=E0/(rho c^2),
x=u/c, gamma=sqrt(1+x^2) and beta=x/gamma. In a homogeneous periodic medium,
mass density stays constant and spatial divergences vanish. For coherent gray
scattering in the material frame, G'0=0 and G'z=kappa q'. The tensor boost gives

    q'/E0 = gamma^2 [(1+beta^2)q - beta(e+P/E0)]
    P/E0 = chi(q/e) e
    chi(f) = (3+4 f^2)/(5+2 sqrt(4-3 f^2)).

With tau=c kappa t, Lorentz transformation of the four-force and the independent
energy/momentum invariants reduce the reference to a scalar ODE:

    x(q) = x0 + epsilon (1-q)
    e(q) = 1 + (gamma0-gamma(q))/epsilon
    dq/dtau = -gamma^3 [(1+beta^2)q - beta (1+chi) e].

The electron caloric inventory is constant. Radiation energy lost equals actual
ion kinetic work, and the radiation momentum decrease equals actual ion impulse.
Use these identities independently of the runtime's conservation ledger.

The input uses beta0=0.002, epsilon=0.005, q0=e0=1 and tau_end=10. It keeps
the entire trajectory below the supported beta=0.01 envelope while producing
appreciable work relative to radiation energy. The 10-micrometre periodic
domain makes particle drift/migration substantial during the physical interval.
Opacity is prescribed to verify the gray equations, not claimed as a validated
microphysical table for a particular plasma.

Run a temporal sequence over this fixed physical duration, compare energy,
momentum and proper velocity against the independently integrated scalar ODE,
and check spatial uniformity, electron caloric constancy, particle-carried
inventories, actual travel and changed-rank restart. The full source iteration
and native PIC update remain active. No stationary-particle helper is substituted.
## Executed CPU qualification

The 200/400/800-step sequence uses 21 equally spaced physical-time outputs,
including the rapid initial transient. All cases pass independent actual
particle/radiation/carry conservation, caloric constancy and uniformity gates.
Halving dt reduces the sampled maximum q, energy/work, proper-velocity and
travel errors by factors 1.9797-1.9823 and then 1.9897-1.9910: first-order
convergence, not a claim of a higher-order integrator.

At 400 steps the particles travel 0.646116 domain lengths and gain 0.443521%
of initial radiation energy. The final E/E0 is 0.9955647853 and q is
0.00927805835; the independent ODE gives 0.99556474665 and 0.00927250210.
The nonzero late-time flux is important: additional endpoint gates reject a
stationary-source approximation even when its transient errors could fit a
percent-level bound. Initial beta=0.002 increases to approximately 0.0069535,
within the declared low-beta envelope.

The registered 400-step CI case and independent analysis pass in 6.79 seconds
on two CPU ranks. Its time-refinement gate is available by passing the three
case directories to `analysis_moving_moment_beam.py` with `--output`; every
reported convergence ratio must exceed 1.8. The same 400-step case also passes
on two P40 GPUs under MPI. Its final e, q, u/c and travel agree with the CPU
values to the displayed digits; maximum electron-temperature drift remains
about 4.2e-14 relative. This is full native moving-PIC execution, not only a
CUDA primitive-kernel check. Separately, the MPI/CUDA trapped-pulse and
two-to-one-rank restart selection passes all four checks in 232.59 seconds.
The cold-interface failures remain recorded separately and are not waived.
