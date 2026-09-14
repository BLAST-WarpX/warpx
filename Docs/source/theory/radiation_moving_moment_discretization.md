# Moving gray moment transport: discrete design and remaining proof obligations

Status: the candidate flux, prescribed-material implicit transport and a private
native-material feedback prototype are implemented. They are not a production
radiation/material mode.
The source and private-interval tests alone do not establish spatial properties.

## Variables and physical diffusion target

Use lab-frame radiation energy density E, q = F/c, and M1 pressure P. Radiation
momentum density is q/c. The conservative equations are

    E_t + c div(q) = -c G0
    q_t + c div(P) = -c G

Here G is momentum gained by material per volume/time, and c G0 is its total
energy gain. The native kinetic/caloric source stage supplies the equal opposite
material transfer; this does not introduce FLASH's fluid closure into WarpX.

The following derivation is for constant prescribed beta = v/c and coherent
isotropic gray scattering with proper inverse mean free path kappa. It is a
necessary moving-diffusion test, not a proof for variable velocity, absorption,
material acceleration, interfaces, or finite frequency groups.

Let gamma = (1-|beta|^2)^(-1/2), r = E-beta.q, and s = q-P.beta. Coherent
scattering obeys G0=beta.G, so

    r_t + c div(s) = 0.

Transforming the comoving scattering force (0,kappa*q_comoving) gives the exact
identity

    s = beta*r + (I-beta beta^T) G/(gamma*kappa).

At leading isotropic comoving equilibrium,

    E_eq = gamma^2 (1+|beta|^2/3) r
    q_eq = (4/3) gamma^2 beta r
    P_eq = (r/3) I + (4/3) gamma^2 beta beta^T r.

Consequently s_eq=beta*r. Substituting the leading advection equation into the
momentum equation gives G=-grad(r)/3 at first diffusive order. Thus

    r_t + c beta.grad(r) = div(D grad(r))
    D = c (I-beta beta^T)/(3*kappa*gamma).

For motion along a 1D mesh, D=c/(3*kappa*gamma^3). Transverse diffusion is
c/(3*kappa*gamma), with off-diagonal terms for oblique motion. At equilibrium
r equals the comoving radiation energy. These provide independently measurable
advection speed and diffusion tensor, not merely a conserved global ledger.

The [ARK-RT paper](https://arxiv.org/html/2011.13926#S6.SS2) explicitly notes that
its stationary asymptotic correction does not preserve the moving-fluid limit.
Copying a stationary correction onto the entire energy flux would suppress
trapped-radiation transport. A new discretization must meet the target above.

## Candidate conservative face construction

At a face use one shared material beta, pressure closure on both adjacent
radiation states, and normal n. Form both projected states with this same face
beta: r_L/R=E_L/R-beta.q_L/R and s_L/R=q_L/R-P_L/R.beta.

One candidate blends the *projected* slow flux, not just lab energy:

    alpha_n = 1/[1 + 3*kappa*gamma*dx_n/(2*(1-beta_n^2))]
    Fq = c [average(P.n) - alpha_n * jump(q)/2]
    Fr = c [beta_n*average(r)
            -(1-alpha_n)*abs(beta_n)*jump(r)/2
            +alpha_n*(average(s.n)-beta_n*average(r)-jump(r)/2)]
         -(1-alpha_n)^2 sum_{t != n} D_nt * grad_t(r)_face
    FE = Fr + beta.Fq.

The extra transverse term is zero in 1D. It must be implemented with one shared
face gradient and audited for monotonicity in multiple dimensions. Its squared
blend switches it off in the transparent limit without a kappa=0 division.

For kappa*dx -> 0, this reduces to ordinary light-speed local Lax-Friedrichs
fluxes for E and q. For kappa*dx -> infinity, the slow flux retains material
advection, its normal numerical diffusion approaches D_nn, and the transverse
term supplies D_nt. Material-scale upwinding remains in the limiting advection
discretization; it must be separated from physical pulse broadening in tests.

Correcting lab energy alone is insufficient: the uncorrected pressure flux then
leaves an O(beta^2/kappa) contribution to the projected diffusion coefficient.
Defining FE through Fr+beta.Fq avoids that mismatch for constant beta.

This is a candidate flux, not a positivity theorem. In particular, transverse
terms, opacity jumps and time-dependent beta still need analysis and numerical
qualification. No flux-vector clipping or radiation-energy floor is permitted
to hide a failure of realizability or conservation.

## Integration obligations

1. Implement shared energy/momentum face fluxes and independent limiting-flux
   tests before connecting them to the runtime solver.
2. Couple transport and stiff sources in the implicit residual. A split
   hyperbolic predictor can be non-realizable even when the final source-coupled
   state is valid; the existing source helper's physical-input guard must not be
   silently removed to accommodate such a predictor.
3. Derive a forced-source or monolithic residual with a realizable iterate,
   native caloric response and private kinetic state. Reject the entire interval
   on failed convergence, invalid final state or failed physical ledgers.
4. Test prescribed moving trapped pulses against the advection/diffusion tensor
   above, then finite-mass material acceleration and variable-velocity work.
   Require temporal and spatial refinement, not only equilibrium convergence.
5. Add incoming/outgoing boundary flux inventories, interface tests, runtime
   diagnostics and restart schema before claiming a usable transport mode.
6. Keep finite-frequency group transforms, packet conversion and radial geometry
   under their existing guards until their separate physical gates are passed.

The material adapter currently shares impulse as a common proper-velocity
increment among selected ions. Its single bulk frame is an explicit modeling
assumption; it does not supply species-resolved opacity/frame physics for
counterstreaming mixtures. Mixture force allocation and frame validity require
their own qualification before broader production claims.

## Initial implementation evidence

`MovingMomentFlux.H` implements the projected face construction. Its 270 CPU/GPU
cases compare independent equilibrium/transparent fluxes, all normal directions,
both drift signs, transverse diffusion terms, and optical depths including an
overflowing kappa*dx product. The weak nonequilibrium flux is checked separately
from the much larger advective flux. Worst CPU flux error is 1.028e-15 against
the 1e-10 gate. Invalid states are rejected, not clipped.

`ImplicitMomentTransport` solves transport and prescribed gray sources together
with an analytic M1 pressure-Jacobian Newton iteration, a matrix-free GMRES correction solve and
a local four-component preconditioner. Krylov vectors may be signed, but every
physical trial is reclosed and checked for realizability. The source-only
helper's physical-input guard is not removed. The linear solve must target an
increment: its first full-state-RHS implementation stopped too early on a weak
transport change and failed the equation gate at step 4. The correction solve
resolved that failure without changing the gate.

For the coupled implementation, a realizable local implicit-source solution now
provides the initial radiation guess. It is exact for the spatially uniform
prescribed-material case and avoids an unnecessary ill-conditioned constant-mode
solve. This is not a split evolution step: the complete transport/source
residual and all final ledgers still determine acceptance. A failed optional
prediction leaves that cell's old physical state as its initial guess.

The initial 1D suite comprises eight runs and 2,200 timesteps. For the trapped
mode, kappa=1e4 per domain length, beta=3e-4 and total c*t/L=1000: material moves
0.3 domain lengths while the Fourier amplitude diffuses. At a fixed 400 steps,
64/128/256 cells give relative Fourier errors 0.0892073/0.0470909/0.0253661.
The finest error passes the predeclared 0.035 bound. At 256 cells, the independent
100/200/400-step difference ratio is 1.98037, within the 1.6--2.4 temporal gate.
Transparent-beam errors at 64/128/256 cells and 100/200/400 steps are
0.0322699/0.0162726/0.0081704; the finest bound is 0.01. Both spatial sequences
must improve by a factor greater than 1.6 per refinement.

Every step checks raw prescribed-material energy/momentum exchange. Forced
nonlinear-budget failure preserves all supplied radiation/transfer fields and
ghosts. A two-rank 128-cell/200-step trapped run reproduces the one-rank Fourier
error. The 2D CUDA executable, with a profile uniform in its transverse direction,
completes 100-step trapped and beam runs with the same errors as CPU. This is a
backend check, not an oblique 2D-interface production benchmark.

Remaining work includes finite-mass/native-electron spatial coupling, genuinely
multidimensional gradients, nonperiodic flux inventories, interface/vacuum
robustness, runtime/restart integration, and the later group/conversion/radial
gates. First-order upwind broadening is still significant on coarse grids; these
initial successes do not establish production accuracy or positivity in general.

## Spatial native-material feedback: qualification in progress

The private coupled stage can now include spatial moment transport within each
material trial. The temperature callback still evaluates the native nodal
electron caloric response, and the force is staged on kinetic ions with their
particle-owned numerical carry. The source equations are re-evaluated using the
actual candidate temperature and finite-kick secant velocity, including a fresh
spatial flux divergence. The interval driver can include these stages without
changing its all-or-nothing private-state commit contract. It still does not
drift particles or refresh density/derived pressure within an interval.

Three numerical issues exposed by this integration are distinct:

- The linear solve must use the actual nonlinear residual and the full analytic
  pressure derivative, not reconstruct a full-state residual from a frozen
  Eddington tensor. Physical trials still use the exact M1 closure.
- Near LTE, a rounded radiation state is insufficient to recover a small source
  by multiplying its imbalance by a large opacity. The driven local source
  retains a separately refined increment. Its algebraic RHS may be signed, but
  its initial physical guess and accepted states must remain realizable. The
  original source-only API continues to reject nonphysical old states.
- Small per-step periodic zero-mode errors can accumulate into a measurable
  long-time momentum error. The solver projects the radiation state onto the
  assigned total transfer, then rechecks realizability, local equations and raw
  global inventories. This changes actual state, not a diagnostic reservoir;
  a projected state that fails any gate is not accepted.

The periodic correction is formed by compensated cellwise balances before a
global sum. Subtracting two large global inventories to obtain the small
correction loses precisely the exchange being conserved. The coupled acceptance
also checks local projected caloric force directly against assigned heat, with
source-arithmetic uncertainty only: the larger spatial-flux rounding allowance
must not hide an inconsistent material heat source.

Finalization chooses a numerically conditioned source representation. For
c*dt*(kappa_a+kappa_s) <= 1, it evaluates the force directly on the solved stored
radiation state; reconstructing that small source by subtracting the transport
RHS is less accurate. For larger rates it retains the independently solved
source increment, avoiding amplification of rounded LTE moments by stiffness.
Both branches pass the same final local and global checks. A periodic correction
may be attempted once local equations pass even if the uncorrected global
balance does not: repairing that zero mode is its purpose. No state is accepted
until the post-correction global and local checks all pass.

Material feedback can now validate the nonlinear trial temperature itself
against the native old-Cv-weighted nodal caloric equation. The ordinary native
caloric inverse provides a Newton search direction, but its rounded output need
not replace a converged nonlinear variable and be multiplied by stiffness again.
The prescribed-temperature evaluation returns nodal residual, source scale and
arithmetic uncertainty, and separately records actual old-to-new EOS energy at
cell corners. It preserves the 1e-11 nodal equation gate and raw conservation
checks; it does not add an energy reservoir or substitute a fluid EOS.

Flux rounding bounds use absolute Jacobian-weighted face operands, not the
magnitude of the already cancelled net flux. Independent Python reconstruction
of the face flux and Lorentz four-force retains its original 1e-10 gates.
For spatial tests, a transverse momentum inventory with near-zero net momentum
is normalized by the local absolute momentum and actual particle changes. A
ratio of two cancelling global sums is not a meaningful conservation metric.
The raw residual and the 1e-10 gate are unchanged.

The 400-step moving thermal test passed in 1D and 2D on one and two MPI ranks
after the conservative projection. The prescribed transport refinement suite
also passed again. Stiff nonuniform native-material feedback remains under
active qualification: a temperature Newton--Krylov proposal resolves the former
fixed-point stall, but final independent accuracy and backend checks must pass
before this stage is recorded as qualified. These results are not FLASH-backed
moving-interface, nonperiodic, finite-group, radial or full-runtime evidence.
