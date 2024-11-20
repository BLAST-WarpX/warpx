.. _theory-cold-fluid-model:

Cold Relativistic Fluid Model
=============================

An alternate to the representation of the plasma as macroparticles, is the cold relativistic fluid model.
The cold relativistic fluid model is typically faster to compute than
particles and useful to replace particles when kinetic effects are negligible. This
can be done for certain parts of the plasma, such as the background plasma, while still
representing particle beams as a group of macroparticles. The two models then couple through
Maxwell's equations.

In the cold limit (zero internal temperature and pressure) of a relativistic plasma, the Maxwell-Fluid
equations govern the plasma evolution. The fluid equations per species, ``s``, are given by,

.. math::

   \frac{\partial N_s}{\partial t} + \nabla \cdot (N_s\mathbf{V}_s) &= 0 \\
   \frac{\partial (N\mathbf{U})_s}{\partial t} + \nabla \cdot ((N\mathbf{U})_s\mathbf{V}_s) &= \frac{q_sN_s}{m_s}(\mathbf{E}_s + \mathbf{V}_s \times \mathbf{B}_s).

Where the fields are updated via Maxwell's equations,

.. math::

   \nabla \cdot \mathbf{E} &= \frac{\rho}{\varepsilon_0} \\
   \nabla \cdot \mathbf{B} &= 0 \\
   \nabla \times \mathbf{E} &= -\frac{\partial \mathbf{B}}{\partial t} \\
   \nabla \times \mathbf{B} &= \mu_0 \mathbf{J} + \mu_0 \varepsilon_0 \frac{\partial \mathbf{E}}{\partial t}.

The fluids are coupled to the fields through,

.. math::

   \rho &= \rho_{ptcl}+\sum_s q_sN_s \\
   \mathbf{J} &= \mathbf{J}_{ptcl}+\sum_s q_sN_s\mathbf{V}_s \\
   \mathbf{V}_s &= \frac{ \mathbf{U}_s }{ \sqrt{ 1 + \mathbf{U}_s^2/c^2} } \\
   (N\mathbf{U})_s &= N_s\mathbf{U}_s

where the particle quantities are calculated by the PIC algorithm.


Implementation details
----------------------

.. _fig_fluid_loop:

.. figure:: https://github.com/ECP-WarpX/WarpX/assets/69021085/dcbcc0e4-7899-43e4-b580-f57eb359b457
   :alt: Figure showing fluid Loop embedded within the overall PIC loop.

   Fluid Loop embedded within the overall PIC loop.

The fluid timeloop is embedded inside the standard PIC timeloop and consists of
the following steps: 1. Higuera and Cary push of the momentum 2. Non-inertial (momentum source)
terms (only in cylindrical geometry) 3. boundary conditions and MPI Communications 4. MUSCL
scheme for advection terms 5. Current and Charge Deposition. :numref:`fig_fluid_loop` gives
a visual representation of these steps, and we describe each of these in more detail.

Step 0: **Preparation**
    Before the fluid loop begins, it is assumed that the program is in the state where fields :math:`\mathbf{E}`
    and :math:`\mathbf{B}` are available integer timestep. The
    fluids themselves are represented by arrays of fluid quantities (density and
    momentum density, :math:`\mathbf{Q} \equiv \{ N, NU_x, NU_y, NU_z \}`) known
    on a nodal grid and at half-integer timestep.

Step 1: **Higuera and Cary Push**
    The time staggering of the fields is used by the momentum source term, which is solved with a
    Higuera and Cary push :cite:p:`cfm-HigueraPOP2017`. We do not adopt spatial
    grid staggering, all discretized fluid quantities exist on the nodal grid. External fields
    can be included at this step.

Step 2: **Non-inertial Terms**
    In RZ, the divergence of the flux terms has additional non-zero elements outside of the
    derivatives. These terms are Strang split and are time integrated via equation 2.18 from :cite:t:`cfm-ShuJCP1988`,
    which is the SSP-RK3 integrator.

Step 3: **Boundary Conditions and Communications**
    At this point, the code applies boundary conditions (assuming Neumann boundary conditions
    for the fluid quantities) and exchanges guard cells between
    MPI ranks in preparation of derivative terms in the next step.

Step 4: **Advective Push**
    For the advective term, a MUSCL scheme with a low-diffusion minmod slope
    limiting is used. We further simplify the conservative equations in terms of primitive
    variables, :math:`\{ N, U_x, U_y, U_z \}`. Which we found to be
    more stable than conservative variables for the MUSCL reconstruction. Details of
    the scheme can be found in :cite:t:`cfm-VanLeerBookChapter1997`.

Step 5: **Current and Charge Deposition**
    Once this series of steps is complete and the fluids have been evolved by an entire
    timestep, the current and charge is deposited onto the grid and added to the total current and charge
    densities.

.. note::
   The algorithm is safe with zero fluid density.

   It also implements a positivity limiter on the density to prevent negative density regions from forming.

   There is currently no ability to perform azimuthal mode decomposition in RZ.

   Mesh refinement is not supported for the fluids.

   The implemented MUSCL scheme has a simplified slope averaging, see the extended writeup for details.

   More details on the precise implementation are available here, `WarpX_Cold_Rel_Fluids.pdf`_.
.. _WarpX_Cold_Rel_Fluids.pdf: https://github.com/ECP-WarpX/WarpX/files/12886437/WarpX_Cold_Rel_Fluids.pdf

.. warning::
      If using the fluid model with the Kinetic-Fluid Hybrid model or the electrostatic solver, there is a known
      issue that the fluids deposit at a half-timestep offset in the charge-density.

.. bibliography::
   :keyprefix: cfm-
