Native RZ pressure and magnetic work
===================================

Scope
-----

``HybridPICModel::tryBfieldEvolveNativePressure`` is a private callable pressure
channel in the actual native ``BfieldEvolve`` / ``FieldPush`` / ``EvolveB`` path.
It evolves mode-0 toroidal magnetic field, nodal electron internal energy and
temperature together. The charge density is frozen. The channel admits only
double-precision ideal electrons without material tables or latent transitions,
an axis-containing Yee or collocated mesh, a stationary outer PEC boundary, periodic axial
boundaries, and zero poloidal magnetic fields. It excludes AMR, EB, PML,
external fields/current, filtering and single-precision communication.

The entry takes independently owned energy, temperature, charge and caloric
volume fields. Charge is copied and its halo is filled privately; the input is
read-only. Positive finite charge and energy are required, with no floor. The
configured thermodynamics executor supplies pressure and the inverse U-to-T
relation at every trial state. The caller's caloric volume is explicit: these
tests use the existing native FV axis volume pi*dr**2*dz/3 and outer half-volume.
They do not replace it by magnetic quadrature.

The private callable entry does not remove the existing production RZ
zero-Ampere guard or the collocated particle-work guard.
No particles move or gather the new electric field in this scope. Hall,
resistive, ion-current, external and hyper-resistive work do not enter this
channel. Computing a nonzero Ampere current does not qualify those other terms.

Discrete work identity
----------------------

Let G be the forward nodal-to-edge gradient. The native Yee toroidal curl is

::

    (C E)[i,j] = (Ez[i+1,j] - Ez[i,j])/dr
               - (Er[i,j+1] - Er[i,j])/dz.

The electric field is ``E = -M A^-1 G P``, where M zeros the tangential Ez edge
at the outer PEC wall. There is no Ez mask at the coordinate axis. A is the
logarithmic mean of the two positive charge values at an edge:

::

    A(a,b) = (b-a)/(log(b)-log(a)),     A(a,a) = a.

This is a new pressure discretization. For smooth positive charge,
``A^-1 = rho_mid^-1 + O(dx**2)``, so it is second-order consistent with
``-grad(P)/rho``. At constant temperature ``P=c*rho``, it reduces to the gradient
of ``-c*log(rho)`` to arithmetic precision; at constant charge it reduces to a
scaled pressure gradient. The native curl therefore vanishes for both controls
when the potential is compatible with the PEC wall. The fixtures make both
charge and temperature axially constant at that wall. A PEC-incompatible
potential would represent real boundary forcing, not a null test.

For toroidal Yee B, the existing FieldEnergy quadrature is the annular cell
volume ``W[i] = 2*pi*(i+1/2)*dr**2*dz``. There is no nodal-axis half-weight in B.
The conjugate work current is ``M^T C^T W B/mu0``. It differs from the physical
Ampere current. The caloric equation is

::

    V*dU/dt = P * G^T A^-1 M^T C^T W B/mu0.

The dot product of this equation with the constant nodal vector exactly
cancels ``B^T W C E/mu0``. Native axis/PEC fills are applied to the actual E/B
state before its field push. The work uses a gather stencil, so it has no
scatter races on CPU or GPU.

On a collocated grid C uses centered radial and axial differences. B_theta is
zero at the coordinate axis; odd PEC Ez ghosts give the outer radial curl
``-Ez[N-1]/dr``. Its FieldEnergy weights are
``W[i] = 2*pi*i*dr**2*dz``, with half weight at the outer wall. G uses centered
differences internally, zero Er at the axis, and the one-sided normal gradient
``(P[N]-P[N-1])/dr`` at the wall. The normal electric field is unconstrained
there. This one-sided gradient preserves the compatible-potential null even
when the axial derivative of that potential is nonzero next to the wall.
Tangential Ez is still zero at the wall. Both native transposes include these
axis and doubled PEC coefficients. The collocated quadratic magnetic defect
is subtracted from the caloric inventory at the same node. These are scoped
field/caloric stencils: a collocated particle/wall pairing is not implied.

Runge--Kutta transaction
-----------------------

The bounded branch uses the native RK4 and Fehlberg RKF45 tableaux. Each trial
combines B and U with the same coefficients, reconstructs T/P from trial U,
computes the weighted-transpose work, calculates the actual Ampere current, and
invokes the native Faraday push with the isolated pressure edges.

Explicit RK does not preserve a quadratic invariant exactly. Keep the RK sum
of local caloric work, then compute the remaining magnetic quadrature defect
independently at every magnetic cell:

::

    D_cell = W/mu0 * [ (B1-B0)*(B1+B0)/2
                     - sum_s b_s * B_s * deltaB_s ].

The opposite defect is partitioned equally to that cell's four nodal energy
inventories and divided by each node's own V. This is a local signed correction,
with no global scaling, floor or total-energy redistribution. The secant product
includes the quadratic magnetic cross term. The quadrature defect is O(dt**5)
for RK4 and leaves its fourth-order convergence intact. Replacing the entire
thermal endpoint by separately averaged pressure and magnetic midpoint would
instead produce second-order local thermal evolution, despite conserving the
global invariant; the independent ODE gate detected that failure during design.

RKF45 compares the corrected low/high B/U candidates. A rejected trial restores
B/U/T before retry. A failed RK4 trial restarts the entire half-step in RKF45,
including already accepted caloric work. An exhausted budget restores every
allocated valid and ghost cell of B/E/plasma-current/U/T, plus the adaptive
substep count. Pressure scratch is scoped and cannot survive a rejected call.
Accepted E and Ampere caches are rebuilt from the accepted state.

Qualification
-------------

The tests use the actual ``FieldEnergy::ComputeNorm2`` and a separately summed
native caloric inventory. The independent host ODE assembles the curl and
gradient transposes by scattering complete cells/edges, stores periodic nodes
once, and integrates with a separately written RK4. It calls none of the native
work, pressure-stencil or RK helpers.

The analytic spatial control measures the actual native magnetic increment
against continuum ``curl(-grad(P)/rho)`` on 16, 32 and 64 cell meshes. The nulls
include uniform T with varying charge and uniform charge with varying T. Other
controls exercise multi-step nonzero B/Ampere current, time refinement, full
RK4 restart rejection, finite RKF45 rejection and successful retry, rejection
from one rank's invalid thermal state, and decomposition into one, four and
sixteen boxes on one/two ranks, for both Yee and collocated grids. The collocated
tests also reject nonzero toroidal field at the axis without changing input
state. The host oracle evaluates the inverse logarithmic mean with log1p so
that independently sampled symmetric endpoints differing by one ulp remain
well-conditioned; it does not use the native near-equality series.

The VisMF control serializes the accepted B/U/T bundle and explicitly saves
the substep count, then replays the next scoped increment. It is **bundle
serialization/replay**. Normal WarpX checkpoint/restart is covered separately
by the isolated ordinary channel below.

Remaining scope
---------------

The matching particle/electron pairing for the new stencil,
moving charge/particles, mixed Ohm work ledgers and GPU backend qualification
remain separate gates. The existing admission guards
continue to reject those unsupported production combinations. Tables, vacuum,
open boundaries, AMR and poloidal/multimode fields are also unqualified.

Isolated ordinary native channel
-------------------------------

``hybrid_pic_model.field_evolution_channel = pressure_only`` explicitly selects
the fixed-charge pressure/Faraday channel in ordinary ``WarpX::Evolve``. The
default ``full`` selector retains the existing complete Ohm evolution. The
isolated channel requires positive, massive depositing ions and every species
to have ``do_not_push=1`` and ``do_not_gather=1``. Native deposited charge must
be strictly positive without floors. It supports ideal electrons with gamma
greater than one and no tables, and excludes the legacy electron-energy
transport, ionization, heat conduction, sources, radiation, collisions,
resistivity/hyper-resistivity, external fields/current and filtering. Therefore
``solve_electron_energy_equation`` remains zero; the selected channel owns its
own caloric evolution. Its initial T may use ``initial_elec_temp(x,y,z)`` in eV.

Three registered native fields own U, frozen charge and caloric volume. The
initialization deposits actual frozen carriers once, seeds U from T and the
configured ideal EOS, and constructs the pressure E and Ampere caches. The two
ordinary magnetic half-steps use the coupled native B/U/T transaction. A failure
in the second half restores the entire step, including all B/E/Ampere/U/T/Pe
valid and ghost cells and the adaptive controller; the frozen charge, volume,
ion currents and density history are unchanged. No algebraic closure or legacy
Ohm solve overwrites the accepted caloric state.

The registered U/charge/volume and existing temperature fields are required
normal WarpX checkpoint state. ``HybridNativePressure.txt`` records the exact
stencil/grid, EOS, closed-boundary and RK control contract and adaptive substep
count. Restart validates this contract before reading potentially incompatible
field layouts. Missing owners or a changed channel/stencil/BC are rejected;
no legacy reconstruction of the isolated caloric owner is permitted.

``test_native_rz_pressure_runtime`` calls the real native initialization and
``WarpX::Evolve`` and uses the normal checkpoint diagnostic. Its host energy
audit independently assembles caloric volumes and magnetic quadrature, checks
them against actual FieldEnergy, and verifies zero wall tangential E and frozen
particle kinetic energy. The runner compares U/T/B/E/frozen charge/volume and
the nontrivial adaptive controller exactly at restart and after continuation,
including a one-rank checkpoint restarted on two ranks. A separate bytewise
full-allocation test forces acceptance of the first half followed by rejection
of the second half, then verifies full-step rollback. This qualification is
distinct from the private VisMF bundle replay described above.
