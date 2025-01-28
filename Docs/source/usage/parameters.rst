.. _running-cpp-parameters:

Parameters: Inputs File
=======================

This documents on how to use WarpX with an inputs file (e.g., ``warpx.3d input_3d``).

Complete example input files can be found in :ref:`the examples section <usage-examples>`.

.. note::

   WarpX input options are read via AMReX `ParmParse <https://amrex-codes.github.io/amrex/docs_html/Basics.html#parmparse>`__.

.. note::

   The AMReX parser (see :ref:`running-cpp-parameters-parser`) is used for the right-hand-side of all input parameters that consist of one or more integers or floats, so expressions like ``<species_name>.density_max = "2.+1."`` and/or using user-defined constants are accepted.

.. _running-cpp-parameters-overall:

Overall simulation parameters
-----------------------------

* ``authors`` (`string`: e.g. ``"Jane Doe <jane@example.com>, Jimmy Joe <jimmy@example.com>"``)
    Authors of an input file / simulation setup.
    When provided, this information is added as metadata to (openPMD) output files.

* ``max_step`` (`integer`)
    The number of PIC cycles to perform.

* ``stop_time`` (`float`; in seconds)
    The maximum physical time of the simulation. Can be provided instead of ``max_step``. If both
    ``max_step`` and ``stop_time`` are provided, both criteria are used and the simulation stops
    when the first criterion is hit.

    Note: in boosted-frame simulations, ``stop_time`` refers to the time in the boosted frame.

* ``warpx.used_inputs_file`` (`string`; default: ``warpx_used_inputs``)
    Name of a file that WarpX writes to archive the used inputs.
    The context of this file will contain an exact copy of all explicitly and implicitly used inputs parameters, including those :ref:`extended and overwritten from the command line <usage_run>`.

* ``warpx.gamma_boost`` (`float`)
    The Lorentz factor of the boosted frame in which the simulation is run.
    (The corresponding Lorentz transformation is assumed to be along ``warpx.boost_direction``.)

    When using this parameter, the input parameters are interpreted as in the
    lab-frame and automatically converted to the boosted frame.
    (See the corresponding documentation of each input parameters for exceptions.)

* ``warpx.boost_direction`` (string: ``x``, ``y`` or ``z``)
    The direction of the Lorentz-transform for boosted-frame simulations
    (The direction ``y`` cannot be used in 2D simulations.)

* ``warpx.zmax_plasma_to_compute_max_step`` (`float`) optional
    Can be useful when running in a boosted frame. If specified, automatically
    calculates the number of iterations required in the boosted frame for the
    lower `z` end of the simulation domain to reach
    ``warpx.zmax_plasma_to_compute_max_step`` (typically the plasma end,
    given in the lab frame). The value of ``max_step`` is overwritten, and
    printed to standard output. Currently only works if the Lorentz boost and
    the moving window are along the z direction.

* ``warpx.compute_max_step_from_btd`` (`integer`; 0 by default) optional
    Can be useful when computing back-transformed diagnostics.  If specified,
    automatically calculates the number of iterations required in the boosted
    frame for all back-transformed diagnostics to be completed. If ``max_step``,
    ``stop_time``, or ``warpx.zmax_plasma_to_compute_max_step`` are not specified,
    or the current values of ``max_step`` and/or ``stop_time`` are too low to fill
    all BTD snapshots, the values of ``max_step`` and/or ``stop_time`` are
    overwritten with the new values and printed to standard output.

* ``warpx.random_seed`` (`string` or `int` > 0) optional
    If provided ``warpx.random_seed = random``, the random seed will be determined
    using `std::random_device` and `std::clock()`,
    thus every simulation run produces different random numbers.
    If provided ``warpx.random_seed = n``, and it is required that `n > 0`,
    the random seed for each MPI rank is `(mpi_rank+1) * n`,
    where `mpi_rank` starts from 0.
    `n = 1` and ``warpx.random_seed = default``
    produce the default random seed.
    Note that when GPU threading is used,
    one should not expect to obtain the same random numbers,
    even if a fixed ``warpx.random_seed`` is provided.

* ``algo.evolve_scheme`` (`string`, default: `explicit`)
    Specifies the evolve scheme used by WarpX.

    * ``explicit``: Use an explicit solver, such as the standard FDTD or PSATD

    * ``theta_implicit_em``: Use a fully implicit electromagnetic solver with a time-biasing parameter theta bound between 0.5 and 1.0. Exact energy conservation is achieved using theta = 0.5. Maximal damping of high-k modes is obtained using theta = 1.0. Choices for the nonlinear solver include a Picard iteration scheme and particle-suppressed (PS) JFNK.
      The algorithm itself is numerical stable for large time steps. That is, it does not require time steps that resolve the plasma period or the CFL condition for light waves. However, the practicality of using a large time step depends on the nonlinear solver. Note that the Picard solver is for demonstration only. It is inefficient and will most like not converge when
      :math:`\omega_{pe} \Delta t` is close to or greater than one or when the CFL condition for light waves is violated. The PS-JFNK method must be used in order to use large time steps. However, the current implementation of PS-JFNK is still inefficient because the JFNK solver is not preconditioned and there is no use of the mass matrices to minimize the cost of a linear iteration. The time step is limited by how many cells a particle can cross in a time step (MPI-related) and by the need to resolve the relevant physics.
      The Picard method is described in `Angus et al., On numerical energy conservation for an implicit particle-in-cell method coupled with a binary Monte-Carlo algorithm for Coulomb collisions <https://doi.org/10.1016/j.jcp.2022.111030>`__.
      The PS-JFNK method is described in `Angus et al., An implicit particle code with exact energy and charge conservation for electromagnetic studies of dense plasmas <https://doi.org/10.1016/j.jcp.2023.112383>`__ . (The version implemented in WarpX is an updated version that includes the relativistic gamma factor for the particles.) Also see `Chen et al., An energy- and charge-conserving, implicit, electrostatic particle-in-cell algorithm. <https://doi.org/10.1016/j.jcp.2011.05.031>`__ .
      Exact energy conservation requires that the interpolation stencil used for the field gather match that used for the current deposition. ``algo.current_deposition = direct`` must be used with ``interpolation.galerkin_scheme = 0``, and ``algo.current_deposition = Esirkepov`` must be used with ``interpolation.galerkin_scheme = 1``. If using ``algo.current_deposition = villasenor``, the corresponding field gather routine will automatically be selected and the ``interpolation.galerkin_scheme`` flag does not need to be specified. The Esirkepov and villasenor deposition schemes are charge-conserving.

    * ``strang_implicit_spectral_em``: Use a fully implicit electromagnetic solver. All of the comments for ``theta_implicit_em``
      above apply here as well (except that theta is fixed to 0.5 and that charge will not be conserved).
      In this version, the advance is Strang split, with a half advance of the source free Maxwell's equation (with a spectral solver), a full advance of the particles plus longitudinal E field, and a second half advance of the source free Maxwell's equations.
      The advantage of this method is that with the Spectral advance of the fields, it is dispersionless.
      Note that exact energy convergence is achieved only with one grid block and ``psatd.periodic_single_box_fft == 1``. Otherwise,
      the energy convservation is spoiled because of the inconsistency of the periodic assumption of the spectral solver and the
      non-periodic behavior of the individual blocks.

    * ``semi_implicit_em``: Use an approximately energy conserving semi-implicit electromagnetic solver. Choices for the nonlinear solver include a Picard iteration scheme and particle-suppressed JFNK.
      Note that this method has the CFL limitation :math:`\Delta t < c/\sqrt( \sum_i 1/\Delta x_i^2 )`. The Picard solver for this method can only be expected to work well when :math:`\omega_{pe} \Delta t` is less than one.
      The method is described in `Chen et al., A semi-implicit, energy- and charge-conserving particle-in-cell algorithm for the relativistic Vlasov-Maxwell equations <https://doi.org/10.1016/j.jcp.2020.109228>`__.
      Exact energy conservation requires that the interpolation stencil used for the field gather match that used for the current deposition. ``algo.current_deposition = direct`` must be used with ``interpolation.galerkin_scheme = 0``, and ``algo.current_deposition = Esirkepov`` must be used with ``interpolation.galerkin_scheme = 1``. If using ``algo.current_deposition = villasenor``, the corresponding field gather routine will automatically be selected and the ``interpolation.galerkin_scheme`` flag does not need to be specified. The Esirkepov and villasenor deposition schemes are charge-conserving.

* ``implicit_evolve.theta`` (`float`, default: 0.5)
    When `algo.evolve_scheme = theta_implicit_em`, the fields used on the RHS of the equations for the implicit advance
    are computed as (1-theta)*E_{n} + theta*E_{n+1}. theta is bound between 0.5 and 1. The default value of theta = 0.5
    is needed for exact energy conservation. For theta > 0.5, high-k modes will be damped and the method will not be
    exactly energy conserving, but the solver may perform better.

* ``implicit_evolve.nonlinear_solver`` (`string`, default: None)
    When `algo.evolve_scheme` is either `theta_implicit_em`, `strang_implicit_spectral_em`, or `semi_implicit_em`, this sets the nonlinear solver used
    to advance the field-particle system in time. Options are `picard` or `newton`.

* ``implicit_evolve.max_particle_iterations`` (`integer`, default: 21)
    When `algo.evolve_scheme` is either `theta_implicit_em`, `strang_implicit_spectral_em`, or `semi_implicit_em` and `implicit_evolve.nonlinear_solver = newton`
    , this sets the maximum number of iterations for the method used to obtain a self-consistent update of the particles at
    each iteration in the JFNK process.

* ``implicit_evolve.particle_tolerance`` (`float`, default: 1.e-10)
    When `algo.evolve_scheme` is either `theta_implicit_em`, `strang_implicit_spectral_em`, or `semi_implicit_em` and `implicit_evolve.nonlinear_solver = newton`
    , this sets the relative tolerance for the iterative method used to obtain a self-consistent update of the particles at
    each iteration in the JFNK process.

* ``picard.verbose`` (`bool`, default: 1)
    When `implicit_evolve.nonlinear_solver = picard`, this sets the verbosity of the Picard solver. If true, then information
    on the nonlinear error are printed to screen at each nonlinear iteration.

* ``picard.require_convergence`` (`bool`, default: 1)
    When `implicit_evolve.nonlinear_solver = picard`, this sets whether the Picard method is required to converge at each
    time step. If it is required, an abort is raised if it does not converge and the code then exits. If not, then a warning
    is issued and the calculation continues.

* ``picard.maximum_iterations`` (`int`, default: 100)
    When `implicit_evolve.nonlinear_solver = picard`, this sets the maximum iterations used by the Picard method. If
    `picard.require_convergence = false`, the solution is considered converged if the iteration count reaches this value,
    but a warning is issued. If `picard.require_convergence = true`, then an abort is raised if the iteration count reaches
    this value.

* ``picard.relative_tolerance`` (`float`, default: 1.0e-6)
    When `implicit_evolve.nonlinear_solver = picard`, this sets the relative tolerance used by the Picard method for determining
    convergence. The absolute error for the Picard method is the L2 norm of the difference of the solution vector between
    two successive iterations. The relative error is the absolute error after iteration k > 1 divided by the absolute error
    after the first iteration. The Picard method is considered converged when the relative error is below the relative tolerance.
    This is the preferred means of determining convergence.

* ``picard.absolute_tolerance`` (`float`, default: 0.0)
    When `implicit_evolve.nonlinear_solver = picard`, this sets the absolute tolerance used by the Picard method for determining
    convergence. The default value is 0.0, which means that the absolute tolerance is not used to determine converence. The
    solution vector in the nonlinear solvers are in physical units rather than normalized ones. Thus, the absolute scale
    of the problem can vary over many orders and magnitude depending on the problem. The relative tolerance is the preferred
    means of determining convergence.

* ``newton.verbose`` (`bool`, default: 1)
    When `implicit_evolve.nonlinear_solver = newton`, this sets the verbosity of the Newton solver. If true, then information
    on the nonlinear error are printed to screen at each nonlinear iteration.

* ``newton.require_convergence`` (`bool`, default: 1)
    When `implicit_evolve.nonlinear_solver = newton`, this sets whether the Newton method is required to converge at each
    time step. If it is required, an abort is raised if it does not converge and the code then exits. If not, then a warning
    is issued and the calculation continues.

* ``newton.maximum_iterations`` (`int`, default: 100)
    When `implicit_evolve.nonlinear_solver = newton`, this sets the maximum iterations used by the Newton method. If
    `newton.require_convergence = false`, the solution is considered converged if the iteration count reaches this value,
    but a warning is issued. If `newton.require_convergence = true`, then an abort is raised if the iteration count reaches
    this value.

* ``newton.relative_tolerance`` (`float`, default: 1.0e-6)
    When `implicit_evolve.nonlinear_solver = newton`, this sets the relative tolerance used by the Newton method for determining
    convergence. The absolute error for the Newton method is the L2 norm of the residual vector. The relative error is the
    absolute error divided by the L2 norm of the initial residual associated with the initial guess. The Newton method is
    considered converged when the relative error is below the relative tolerance. This is the preferred means of determining
    convergence.

* ``newton.absolute_tolerance`` (`float`, default: 0.0)
    When `implicit_evolve.nonlinear_solver = newton`, this sets the absolute tolerance used by the Newton method for determining
    convergence. The default value is 0.0, which means that the absolute tolerance is not used to determine converence. The
    residual vector in the nonlinear solvers are in physical units rather than normalized ones. Thus, the absolute scale
    of the problem can vary over many orders and magnitude depending on the problem. The relative tolerance is the preferred
    means of determining convergence.

* ``gmres.verbose_int`` (`int`, default: 2)
    When `implicit_evolve.nonlinear_solver = newton`, this sets the verbosity of the AMReX::GMRES linear solver. The default
    value of 2 gives maximumal verbosity and information about the residual are printed to the screen at each GMRES iteration.

* ``gmres.restart_length`` (`int`, default: 30)
    When `implicit_evolve.nonlinear_solver = newton`, this sets the iteration number at which to do a restart in AMReX::GMRES.
    This parameter is used to save memory on building the Krylov subspace basis vectors for linear systems that are ill-conditioned
    and require many iterations to converge.

* ``gmres.relative_tolerance`` (`float`, default: 1.0e-4)
    When `implicit_evolve.nonlinear_solver = newton`, this sets the relative tolerance used to determine convergence of the
    AMReX::GMRES linear solver used to compute the Newton step in the JNFK process. The absolute error is the L2 norm of the
    residual vector. The relative error is the absolute error divided by the L2 norm of the initial residual (typically equal
    to the norm of the nonlinear residual from the end of the previous Newton iteration). The linear solver is considered
    converged when the relative error is below the relative tolerance. This is the preferred means of determining convergence.

* ``gmres.absolute_tolerance`` (`float`, default: 0.0)
    When `implicit_evolve.nonlinear_solver = newton`, this sets the absolute tolerance used to determine converence of the
    GMRES linear solver used to compute the Newton step in the JNFK process. The default value is 0.0, which means that the
    absolute tolerance is not used to determine converence. The residual vector in the nonlinear/linear solvers are in physical
    units rather than normalized ones. Thus, the absolute scale of the problem can vary over many orders and magnitude depending
    on the problem. The relative tolerance is the preferred means of determining convergence.

* ``gmres.maximum_iterations`` (`int`, default: 1000)
    When `implicit_evolve.nonlinear_solver = newton`, this sets the maximum iterations used by the GMRES linear solver. The
    solution to the linear system is considered converged if the iteration count reaches this value.

* ``warpx.do_electrostatic`` (`string`) optional (default `none`)
    Specifies the electrostatic mode. When turned on, instead of updating
    the fields at each iteration with the full Maxwell equations, the fields
    are recomputed at each iteration from the Poisson equation.
    There is no limitation on the timestep in this case, but
    electromagnetic effects (e.g. propagation of radiation, lasers, etc.)
    are not captured. There are several options:

    * ``labframe``: Poisson's equation is solved in the lab frame with
      the charge density of all species combined. More specifically, the code solves:

      .. math::

        \boldsymbol{\nabla}^2 \phi = - \rho/\epsilon_0 \qquad \boldsymbol{E} = - \boldsymbol{\nabla}\phi

    * ``labframe-electromagnetostatic``: Poisson's equation is solved in the lab frame with
      the charge density of all species combined.  Additionally the 3-component vector potential
      is solved in the Coulomb Gauge with the current density of all species combined
      to include self magnetic fields. More specifically, the code solves:

      .. math::

        \boldsymbol{\nabla}^2 \phi = - \rho/\epsilon_0 \qquad \boldsymbol{E} = - \boldsymbol{\nabla}\phi \\
        \boldsymbol{\nabla}^2 \boldsymbol{A} = - \mu_0 \boldsymbol{j} \qquad \boldsymbol{B} = \boldsymbol{\nabla}\times\boldsymbol{A}

    * ``labframe-effective-potential``: Poisson's equation is solved with a modified dielectric function
      (resulting in an "effective potential") to create a semi-implicit scheme which is robust to the
      numerical instability seen in explicit electrostatic PIC when :math:`\Delta t \omega_{pe} > 2`.
      If this option is used the additional parameter ``warpx.effective_potential_factor`` can also be
      specified to set the value of :math:`C_{EP}` (default 4). The method is stable for :math:`C_{EP} \geq 1`
      regardless of :math:`\Delta t`, however, the larger :math:`C_{EP}` is set, the lower the numerical plasma
      frequency will be and therefore care must be taken to not set it so high that the plasma mode
      hybridizes with other modes of interest.
      Details of the method can be found in Appendix A of :cite:t:`param-Barnes2021` (note that in that paper
      the method is referred to as "semi-implicit electrostatic" but here it has been renamed to "effective potential"
      to avoid confusion with the semi-implicit method of Chen et al.).
      In short, the code solves:

      .. math::

        \boldsymbol{\nabla}\cdot\left(1+\frac{C_{EP}}{4}\sum_{s \in \text{species}}(\omega_{ps}\Delta t)^2 \right)\boldsymbol{\nabla} \phi = - \rho/\epsilon_0 \qquad \boldsymbol{E} = - \boldsymbol{\nabla}\phi

    * ``relativistic``: Poisson's equation is solved **for each species**
      in their respective rest frame. The corresponding field
      is mapped back to the simulation frame and will produce both E and B
      fields. More specifically, in the simulation frame, this is equivalent to solving **for each species**

      .. math::

        \boldsymbol{\nabla}^2 - (\boldsymbol{\beta}\cdot\boldsymbol{\nabla})^2\phi = - \rho/\epsilon_0 \qquad
        \boldsymbol{E} = -\boldsymbol{\nabla}\phi + \boldsymbol{\beta}(\boldsymbol{\beta} \cdot \boldsymbol{\nabla}\phi)
        \qquad \boldsymbol{B} = -\frac{1}{c}\boldsymbol{\beta}\times\boldsymbol{\nabla}\phi

      where :math:`\boldsymbol{\beta}` is the average (normalized) velocity of the considered species (which can be relativistic).
      See, e.g., :cite:t:`param-Vaypop2008` for more information.

* ``warpx.poisson_solver`` (`string`) optional (default `multigrid`)

    * ``multigrid``: Poisson's equation is solved using an iterative multigrid (MLMG) solver.
        See the `AMReX documentation <https://amrex-codes.github.io/amrex/docs_html/LinearSolvers.html#>`__
        for details of the MLMG solver (the default solver used with electrostatic
        simulations). The default behavior of the code is to check whether there is
        non-zero charge density in the system and if so force the MLMG solver to
        use the solution max norm when checking convergence. If there is no charge
        density, the MLMG solver will switch to using the initial guess max norm
        error when evaluating convergence and an absolute error tolerance of
        :math:`10^{-6}` :math:`\mathrm{V/m}^2` will be used (unless a different
        non-zero value is specified by the user via
        ``warpx.self_fields_absolute_tolerance``).

    * ``fft``: Poisson's equation is solved using an Integrated Green Function method (which requires FFT calculations).
        See these references for more details :cite:t:`param-QiangPhysRevSTAB2006`, :cite:t:`param-QiangPhysRevSTAB2006err`.
        It only works in 3D and it requires the compilation flag ``-DWarpX_FFT=ON``.
        If mesh refinement is enabled, this solver only works on the coarsest level.
        On the refined patches, the Poisson equation is solved with the multigrid solver.
        In electrostatic mode, this solver requires open field boundary conditions (``boundary.field_lo,hi = open``).
        In electromagnetic mode, this solver can be used to initialize the species' self fields
        (``<species_name>.initialize_self_fields=1``) provided that the field BCs are PML (``boundary.field_lo,hi = PML``).

          * ``warpx.use_2d_slices_fft_solver`` (`bool`) optional (default: 0): Select the type of Integrated Green Function solver.
            If 0, solve Poisson equation in full 3D geometry.
            If 1, solve Poisson equation in a quasi 3D geometry, neglecting the :math:`z` derivatives in the Laplacian of the Poisson equation.
            In practice, in this case, the code performes many 2D Poisson solves on all :math:`(x,y)` slices, each slice at a given :math:`z`.
            This is often a good approximation for ultra-relativistic beams propagating along the :math:`z` direction, with the relativistic solver.
            As a consequence, this solver does not need to do an FFT along the :math:`z` direction,
            and instead uses only transverse FFTs (along :math:`x` and :math:`y`) at each :math:`z` position (or :math:`z` "slice").

          * ``ablastr.nprocs_igf_fft`` (`int`) optional (default: number of MPI ranks): Number of MPI ranks used to parallalelize the FFT solver.
            This can be less or equal than then number of MPI ranks that are used to run the overall simulation.
            It can be useful if the auxiliary simulation boxes fit within a single process, so to avoid extra communications.
            The auxiliary boxes are extended boxes in real and spectral space that are used to perform the necessary FFTs.
            The extended simulation box size in real space is :math:`2n_x-1, 2n_y-1, 2n_z-1` with the 3D solver, :math:`2n_x-1, 2n_y -1, n_z` with the 2D solver.
            The extended simulation box size in spectral space is :math:`n_x, 2n_y-1, 2n_z-1` with the 3D solver, :math:`n_x, 2n_y-1, n_z` with the 2D solver.

* ``warpx.self_fields_required_precision`` (`float`, default: 1.e-11)
    The relative precision with which the electrostatic space-charge fields should
    be calculated. More specifically, the space-charge fields are
    computed with an iterative Multi-Level Multi-Grid (MLMG) solver.
    This solver can fail to reach the default precision within a reasonable time.
    This only applies when warpx.do_electrostatic = labframe.

* ``warpx.self_fields_absolute_tolerance`` (`float`, default: 0.0)
    The absolute tolerance with which the space-charge fields should be
    calculated in units of :math:`\mathrm{V/m}^2`. More specifically, the acceptable
    residual with which the solution can be considered converged. In general
    this should be left as the default, but in cases where the simulation state
    changes very little between steps it can occur that the initial guess for
    the MLMG solver is so close to the converged value that it fails to improve
    that solution sufficiently to reach the ``self_fields_required_precision``
    value.

* ``warpx.self_fields_max_iters`` (`integer`, default: 200)
    Maximum number of iterations used for MLMG solver for space-charge
    fields calculation. In case if MLMG converges but fails to reach the desired
    ``self_fields_required_precision``, this parameter may be increased.
    This only applies when warpx.do_electrostatic = labframe.

* ``warpx.self_fields_verbosity`` (`integer`, default: 2)
    The verbosity used for MLMG solver for space-charge fields calculation. Currently
    MLMG solver looks for verbosity levels from 0-5. A higher number results in more
    verbose output.

* ``amrex.abort_on_out_of_gpu_memory``  (``0`` or ``1``; default is ``1`` for true)
    When running on GPUs, memory that does not fit on the device will be automatically swapped to host memory when this option is set to ``0``.
    This will cause severe performance drops.
    Note that even with this set to ``1`` WarpX will not catch all out-of-memory events yet when operating close to maximum device memory.
    `Please also see the documentation in AMReX <https://amrex-codes.github.io/amrex/docs_html/GPU.html#inputs-parameters>`__.

* ``amrex.the_arena_is_managed``  (``0`` or ``1``; default is ``0`` for false)
    When running on GPUs, device memory that is accessed from the host will automatically be transferred with managed memory.
    This is useful for convenience during development, but has sometimes severe performance and memory footprint implications if relied on (and sometimes vendor bugs).
    For all regular WarpX operations, we therefore do explicit memory transfers without the need for managed memory and thus changed the AMReX default to false.
    `Please also see the documentation in AMReX <https://amrex-codes.github.io/amrex/docs_html/GPU.html#inputs-parameters>`__.

* ``amrex.omp_threads``  (``system``, ``nosmt`` or positive integer; default is ``nosmt``)
    An integer number can be set in lieu of the ``OMP_NUM_THREADS`` environment variable to control the number of OpenMP threads to use for the ``OMP`` compute backend on CPUs.
    By default, we use the ``nosmt`` option, which overwrites the OpenMP default of spawning one thread per logical CPU core, and instead only spawns a number of threads equal to the number of physical CPU cores on the machine.
    If set, the environment variable ``OMP_NUM_THREADS`` takes precedence over ``system`` and ``nosmt``, but not over integer numbers set in this option.


Signal Handling
^^^^^^^^^^^^^^^

WarpX can handle Unix (Linux/macOS) `process signals <https://en.wikipedia.org/wiki/Signal_(IPC)>`__.
This can be useful to configure jobs on HPC and cloud systems to shut down cleanly when they are close to reaching their allocated walltime or to steer the simulation behavior interactively.

Allowed signal names are documented in the `C++ standard <https://en.cppreference.com/w/cpp/utility/program/SIG_types>`__ and `POSIX <https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/signal.h.html>`__.
We follow the same naming, but remove the ``SIG`` prefix, e.g., the WarpX signal configuration name for ``SIGINT`` is ``INT``.

* ``warpx.break_signals`` (array of `string`, separated by spaces) optional
    A list of signal names or numbers that the simulation should
    handle by cleanly terminating at the next timestep

* ``warpx.checkpoint_signals`` (array of `string`, separated by spaces) optional
    A list of signal names or numbers that the simulation should
    handle by outputting a checkpoint at the next timestep. A
    diagnostic of type `checkpoint` must be configured.

.. note::

   Certain signals are only available on specific platforms, please see the links above for details.
   Typically supported on Linux and macOS are ``HUP``, ``INT``, ``QUIT``, ``ABRT``, ``USR1``, ``USR2``, ``TERM``, ``TSTP``, ``URG``, and ``IO`` among others.

   Signals to think about twice before overwriting in *interactive simulations*:
   Note that ``INT`` (interupt) is the signal that ``Ctrl+C`` sends on the terminal, which most people use to abort a process; once overwritten you need to abort interactive jobs with, e.g., ``Ctrl+\`` (``QUIT``) or sending the ``KILL`` signal.
   The ``TSTP`` (terminal stop) command is sent interactively from ``Ctrl+Z`` to temporarily send a process to sleep (until send in the background with commands such as ``bg`` or continued with ``fg``), overwriting it would thus disable that functionality.
   The signals ``KILL`` and ``STOP`` cannot be used.

   The ``FPE`` and ``ILL`` signals should not be overwritten in WarpX, as they are `controlled by AMReX <https://amrex-codes.github.io/amrex/docs_html/Debugging.html#breaking-into-debuggers>`__ for :ref:`debug workflows that catch invalid floating-point operations <debugging_warpx>`.
.. tip::

   For example, the following logic can be added to `Slurm batch scripts <https://docs.gwdg.de/doku.php?id=en:services:application_services:high_performance_computing:running_jobs_slurm:signals>`__ (`signal name to number mapping here <https://en.wikipedia.org/wiki/Signal_(IPC)#Default_action>`__) to gracefully shut down 6 min prior to walltime.
   If you have a checkpoint diagnostics in your inputs file, this automatically will write a checkpoint due to the default ``<diag_name>.dump_last_timestep = 1`` option in WarpX.

   .. code-block:: bash

      #SBATCH --signal=1@360

      srun ...                   \
        warpx.break_signals=HUP  \
        > output.txt

   For `LSF batch systems <https://www.ibm.com/docs/en/spectrum-lsf/10.1.0?topic=options-wa>`__, the equivalent job script lines are:

   .. code-block:: bash

      #BSUB -wa 'HUP' -wt '6'

      jsrun ...                  \
        warpx.break_signals=HUP  \
        > output.txt

.. _running-cpp-parameters-box:

Setting up the field mesh
-------------------------

* ``amr.n_cell`` (`2 integers in 2D`, `3 integers in 3D`)
    The number of grid points along each direction (on the **coarsest level**)

* ``amr.max_level`` (`integer`, default: ``0``)
    When using mesh refinement, the number of refinement levels that will be used.

    Use 0 in order to disable mesh refinement.
    Note: currently, ``0`` and ``1`` are supported.

* ``amr.ref_ratio`` (`integer` per refined level, default: ``2``)
    When using mesh refinement, this is the refinement ratio per level.
    With this option, all directions are fined by the same ratio.

* ``amr.ref_ratio_vect`` (`3 integers for x,y,z per refined level`)
    When using mesh refinement, this can be used to set the refinement ratio per direction and level, relative to the previous level.

    Example: for three levels, a value of ``2 2 4 8 8 16`` refines the first level by 2-fold in x and y and 4-fold in z compared to the coarsest level (level 0/mother grid); compared to the first level, the second level is refined 8-fold in x and y and 16-fold in z.

* ``geometry.dims`` (`string`)
    The dimensions of the simulation geometry.
    Supported values are ``1``, ``2``, ``3``, ``RZ``.
    For ``3``, a cartesian geometry of ``x``, ``y``, ``z`` is modeled.
    For ``2``, the axes are ``x`` and ``z`` and all physics in ``y`` is assumed to be translation symmetric.
    For ``1``, the only axis is ``z`` and the dimensions ``x`` and ``y`` are translation symmetric.
    For ``RZ``, we apply an azimuthal mode decomposition, with ``warpx.n_rz_azimuthal_modes`` providing further control.

    Note that this value has to match the :ref:`WarpX_DIMS <building-cmake-options>` compile-time option.
    If you installed WarpX from a :ref:`package manager <install-users>`, then pick the right executable by name.

* ``warpx.n_rz_azimuthal_modes`` (`integer`; 1 by default)
    When using the RZ version, this is the number of azimuthal modes.
    The default is ``1``, which corresponds to a perfectly axisymmetric simulation.

* ``geometry.prob_lo`` and ``geometry.prob_hi`` (`2 floats in 2D`, `3 floats in 3D`; in meters)
    The extent of the full simulation box. This box is rectangular, and thus its
    extent is given here by the coordinates of the lower corner (``geometry.prob_lo``) and
    upper corner (``geometry.prob_hi``). The first axis of the coordinates is x
    (or r with cylindrical) and the last is z.

* ``warpx.do_moving_window`` (`integer`; 0 by default)
    Whether to use a moving window for the simulation

* ``warpx.moving_window_dir`` (either ``x``, ``y`` or ``z``)
    The direction of the moving window.

* ``warpx.moving_window_v`` (`float`)
    The speed of moving window, in units of the speed of light
    (i.e. use ``1.0`` for a moving window that moves exactly at the speed of light)

* ``warpx.start_moving_window_step`` (`integer`; 0 by default)
    The timestep at which the moving window starts.

* ``warpx.end_moving_window_step`` (`integer`; default is ``-1`` for false)
    The timestep at which the moving window ends.

* ``warpx.fine_tag_lo`` and ``warpx.fine_tag_hi`` (`2 floats in 2D`, `3 floats in 3D`; in meters) optional
    **When using static mesh refinement with 1 level**, the extent of the refined patch.
    This patch is rectangular, and thus its extent is given here by the coordinates
    of the lower corner (``warpx.fine_tag_lo``) and upper corner (``warpx.fine_tag_hi``).

* ``warpx.ref_patch_function(x,y,z)`` (`string`) optional
    A function of `x`, `y`, `z` that defines the extent of the refined patch when
    using static mesh refinement with ``amr.max_level``>0. Note that the function can be used
    to define distinct regions for refinement, however, the refined regions should be such that
    the pml layer surrounding the patches should not overlap. For this reason, when defining
    distinct patches, please ensure that they are sufficiently separated.

* ``warpx.refine_plasma`` (`integer`) optional (default `0`)
    Increase the number of macro-particles that are injected "ahead" of a mesh
    refinement patch in a moving window simulation.

    Note: in development; only works with static mesh-refinement, specific
    to moving window plasma injection, and requires a single refined level.

* ``warpx.n_current_deposition_buffer`` (`integer`)
    When using mesh refinement: the particles that are located inside
    a refinement patch, but within ``n_current_deposition_buffer`` cells of
    the edge of this patch, will deposit their charge and current to the
    lower refinement level, instead of depositing to the refinement patch
    itself. See the :ref:`mesh-refinement section <theory-amr>` for more details.
    If this variable is not explicitly set in the input script,
    ``n_current_deposition_buffer`` is automatically set so as to be large
    enough to hold the particle shape, on the fine grid

* ``warpx.n_field_gather_buffer`` (`integer`, optional)
    Default: ``warpx.n_field_gather_buffer = n_current_deposition_buffer + 1`` (one cell larger than ``n_current_deposition_buffer`` on the fine grid).

    When using mesh refinement, particles that are located inside a refinement patch, but within ``n_field_gather_buffer`` cells of the edge of the patch, gather the fields from the lower refinement level, instead of gathering the fields from the refinement patch itself.
    This avoids some of the spurious effects that can occur inside the refinement patch, close to its edge.
    See the section :ref:`Mesh refinement <theory-amr>` for more details.

* ``warpx.do_single_precision_comms`` (`integer`; 0 by default)
    Perform MPI communications for field guard regions in single precision.
    Only meaningful for ``WarpX_PRECISION=DOUBLE``.

* ``particles.deposit_on_main_grid`` (`list of strings`)
    When using mesh refinement: the particle species whose name are included
    in the list will deposit their charge/current directly on the main grid
    (i.e. the coarsest level), even if they are inside a refinement patch.

* ``particles.gather_from_main_grid`` (`list of strings`)
    When using mesh refinement: the particle species whose name are included
    in the list will gather their fields from the main grid
    (i.e. the coarsest level), even if they are inside a refinement patch.

.. _running-cpp-parameters-bc:

Domain Boundary Conditions
--------------------------

* ``boundary.field_lo`` and ``boundary.field_hi`` (`2 strings` for 2D, `3 strings` for 3D, `pml` by default)
    Boundary conditions applied to fields at the lower and upper domain boundaries.
    Options are:

    * ``Periodic``: This option can be used to set periodic domain boundaries. Note that if the fields for lo in a certain dimension are set to periodic, then the corresponding upper boundary must also be set to periodic. If particle boundaries are not specified in the input file, then particles boundaries by default will be set to periodic. If particles boundaries are specified, then they must be set to periodic corresponding to the periodic field boundaries.

    * ``pml`` (default): This option can be used to add Perfectly Matched Layers (PML) around the simulation domain. See the :ref:`PML theory section <theory-bc-PML>` for more details.
      Additional pml algorithms can be explored using the parameters ``warpx.do_pml_in_domain``, ``warpx.pml_has_particles``, and ``warpx.do_pml_j_damping``.

    * ``absorbing_silver_mueller``: This option can be used to set the Silver-Mueller absorbing boundary conditions. These boundary conditions are simpler and less computationally expensive than the pml, but are also less effective at absorbing the field. They only work with the Yee Maxwell solver.

    * ``damped``: This is the recommended option in the moving direction when using the spectral solver with moving window (currently only supported along z). This boundary condition applies a damping factor to the electric and magnetic fields in the outer half of the guard cells, using a sine squared profile. As the spectral solver is by nature periodic, the damping prevents fields from wrapping around to the other end of the domain when the periodicity is not desired. This boundary condition is only valid when using the spectral solver.

    * ``pec``: This option can be used to set a Perfect Electric Conductor at the simulation boundary. Please see the :ref:`PEC theory section <theory-bc-pec>` for more details. Note that PEC boundary is invalid at `r=0` for the RZ solver. Please use ``none`` option. This boundary condition does not work with the spectral solver.

    * ``pec_insulator``: This option specifies a mixed perfect electric conductor and insulator boundary, where some part of the
      boundary is PEC and some is insulator. In the insulator portion, the normal fields are extrapolated and the tangential fields
      are either set to the specified value or extrapolated. The region that is insulator is specified using a spatially dependent expression with the insulator being in the area where the value of the expression is greater than zero.
      The expressions are given for the low and high boundary on each axis, as listed below. The tangential fields are specified as
      expressions that can depend on the location and time. The tangential fields are in two pairs, the electric fields and the
      magnetic fields. In each pair, if one is specified, the other will be set to zero if not also specified.

      * ``insulator.area_x_lo(y,z)``: For the lower x (or r) boundary, expression specifying the insulator location

      * ``insulator.area_x_hi(y,z)``: For the upper x (or r) boundary, expression specifying the insulator location

      * ``insulator.area_y_lo(x,z)``: For the lower y boundary, expression specifying the insulator location

      * ``insulator.area_y_hi(x,z)``: For the upper y boundary, expression specifying the insulator location

      * ``insulator.area_z_lo(x,y)``: For the lower z boundary, expression specifying the insulator location

      * ``insulator.area_z_hi(x,y)``: For the upper z boundary, expression specifying the insulator location

      * ``insulator.Ey_x_lo(y,z,t)``, ``insulator.Ez_x_lo(y,z,t)``, ``insulator.By_x_lo(y,z,t)``, ``insulator.Bz_x_lo(y,z,t)``: expressions of the tangential field values for the lower x (or r) boundary

      * ``insulator.Ey_x_hi(y,z,t)``, ``insulator.Ez_x_hi(y,z,t)``, ``insulator.By_x_hi(y,z,t)``, ``insulator.Bz_x_hi(y,z,t)``: expressions of the tangential field values for the upper x (or r) boundary

      * ``insulator.Ex_y_lo(x,z,t)``, ``insulator.Ez_y_lo(x,z,t)``, ``insulator.Bx_y_lo(x,z,t)``, ``insulator.Bz_y_lo(x,z,t)``: expressions of the tangential field values for the lower y boundary

      * ``insulator.Ex_y_hi(x,z,t)``, ``insulator.Ez_y_hi(x,z,t)``, ``insulator.Bx_y_hi(x,z,t)``, ``insulator.Bz_y_hi(x,z,t)``: expressions of the tangential field values for the upper y boundary

      * ``insulator.Ex_z_lo(x,y,t)``, ``insulator.Ey_z_lo(x,y,t)``, ``insulator.Bx_z_lo(x,y,t)``, ``insulator.By_z_lo(x,y,t)``: expressions of the tangential field values for the lower z boundary

      * ``insulator.Ex_z_hi(x,y,t)``, ``insulator.Ey_z_hi(x,y,t)``, ``insulator.Bx_z_hi(x,y,t)``, ``insulator.By_z_hi(x,y,t)``: expressions of the tangential field values for the upper z boundary

    * ``none``: No boundary condition is applied to the fields with the electromagnetic solver. This option must be used for the RZ-solver at `r=0`.

    * ``neumann``: For the electrostatic multigrid solver, a Neumann boundary condition (with gradient of the potential equal to 0) will be applied on the specified boundary.

    * ``open``: For the electrostatic Poisson solver based on a Integrated Green Function method.

* ``boundary.potential_lo_x/y/z`` and ``boundary.potential_hi_x/y/z`` (default `0`)
    Gives the value of the electric potential, in Volts, at the boundaries, for ``pec`` boundaries. With electrostatic solvers
    (i.e., with ``warpx.do_electrostatic = ...``), this is used in order to compute the potential
    in the simulation volume at each timestep. When using other solvers (e.g. Maxwell solver),
    setting these variables will trigger an electrostatic solve at ``t=0``, to compute the initial
    electric field produced by the boundaries.

* ``boundary.particle_lo`` and ``boundary.particle_hi`` (`2 strings` for 2D, `3 strings` for 3D, `absorbing` by default)
    Options are:

    * ``Absorbing``: Particles leaving the boundary will be deleted.

    * ``Periodic``: Particles leaving the boundary will re-enter from the opposite boundary. The field boundary condition must be consistently set to periodic and both lower and upper boundaries must be periodic.

    * ``Reflecting``: Particles leaving the boundary are reflected from the boundary back into the domain.
      When ``boundary.reflect_all_velocities`` is false, the sign of only the normal velocity is changed, otherwise the sign of all velocities are changed.

    * ``Thermal``: Particles leaving the boundary are reflected from the boundary back into the domain
      and their velocities are thermalized. The tangential velocity components are sampled from ``gaussian`` distribution
      and the component normal to the boundary is sampled from ``gaussian flux`` distribution.
      The standard deviation for these distributions should be provided for each species using
      ``boundary.<species>.u_th``. The same standard deviation is used to sample all components.

    * ``None``: No boundary conditions are applied to the particles.
      When using RZ, this option must be used for the lower radial boundary, the first value of ``boundary.particle_lo``.
      This should not be used in any other cases.

* ``boundary.reflect_all_velocities`` (`bool`) optional (default `false`)
    For a reflecting boundary condition, this flags whether the sign of only the normal velocity is changed or all velocities.

* ``boundary.verboncoeur_axis_correction`` (`bool`) optional (default `true`)
    Whether to apply the Verboncoeur correction on the charge and current density on axis when using RZ.
    For nodal values (rho and Jz), the cell volume for values on axis is calculated as :math:`\pi*\Delta r^2/4`.
    In :cite:t:`param-VerboncoeurJCP2001`, it is shown that using
    :math:`\pi*\Delta r^2/3` instead will give a uniform density if the particle density is uniform.

Additional PML parameters
-------------------------

* ``warpx.pml_ncell`` (`int`; default: 10)
    The depth of the PML, in number of cells.

* ``do_similar_dm_pml`` (`int`; default: 1)
    Whether or not to use an amrex::DistributionMapping for the PML grids that is `similar` to the mother grids, meaning that the
    mapping will be computed to minimize the communication costs between the PML and the mother grids.

* ``warpx.pml_delta`` (`int`; default: 10)
    The characteristic depth, in number of cells, over which
    the absorption coefficients of the PML increases.

* ``warpx.do_pml_in_domain`` (`int`; default: 0)
    Whether to create the PML inside the simulation area or outside. If inside,
    it allows the user to propagate particles in PML and to use extended PML

* ``warpx.pml_has_particles`` (`int`; default: 0)
    Whether to propagate particles in PML or not. Can only be done if PML are in simulation domain,
    i.e. if `warpx.do_pml_in_domain = 1`.

* ``warpx.do_pml_j_damping`` (`int`; default: 0)
    Whether to damp current in PML. Can only be used if particles are propagated in PML,
    i.e. if `warpx.pml_has_particles = 1`.

* ``warpx.v_particle_pml`` (`float`; default: 1)
    When ``warpx.do_pml_j_damping = 1``, the assumed velocity of the particles to be absorbed in the PML, in units of the speed of light `c`.

* ``warpx.do_pml_dive_cleaning`` (`bool`)
    Whether to use divergence cleaning for E in the PML region.
    The value must match ``warpx.do_pml_divb_cleaning`` (either both false or both true).
    This option seems to be necessary in order to avoid strong Nyquist instabilities in 3D simulations with the PSATD solver, open boundary conditions and PML in all directions. 2D simulations and 3D simulations with open boundary conditions and PML only in one direction might run well even without divergence cleaning.
    This option is implemented only for the Cartesian PSATD solver; it is turned on by default in this case.

* ``warpx.do_pml_divb_cleaning`` (`bool`)
    Whether to use divergence cleaning for B in the PML region.
    The value must match ``warpx.do_pml_dive_cleaning`` (either both false or both true).
    This option seems to be necessary in order to avoid strong Nyquist instabilities in 3D simulations with the PSATD solver, open boundary conditions and PML in all directions. 2D simulations and 3D simulations with open boundary conditions and PML only in one direction might run well even without divergence cleaning.
    This option is implemented only for the Cartesian PSATD solver; it is turned on by default in this case.

.. _running-cpp-parameters-eb:

Embedded Boundary Conditions
----------------------------

In WarpX, the embedded boundary can be defined in either of two ways:

    - **From an analytical function:**
        In that case, you will need to set the following parameter in the input file.

        * ``warpx.eb_implicit_function`` (`string`)
            A function of `x`, `y`, `z` that defines the surface of the embedded
            boundary. That surface lies where the function value is 0 ;
            the physics simulation area is where the function value is negative ;
            the interior of the embedded boundary is where the function value is positive.

    - **From an STL file:**
        In that case, you will need to set the following parameters in the input file.

        * ``eb2.stl_file`` (`string`)
            The path to an `STL file <https://en.wikipedia.org/wiki/STL_(file_format)>`__.
            In addition, you also need to set ``eb2.geom_type = stl``, in order for the file to be read by WarpX.
            `See the AMReX documentation for more details <https://amrex-codes.github.io/amrex/docs_html/EB.html>`__.

Whether the embedded boundary is defined with an analytical function or an STL file, you can
additionally define the electric potential at the embedded boundary with an analytical function:

* ``warpx.eb_potential(x,y,z,t)`` (`string`)
    Gives the value of the electric potential, in Volts, at the surface of the embedded boundary,
    as a function of  `x`, `y`, `z` and `t`. With electrostatic solvers (i.e., with
    ``warpx.do_electrostatic = ...``), this is used in order to compute the potential
    in the simulation volume at each timestep. When using other solvers (e.g. Maxwell solver),
    setting this variable will trigger an electrostatic solve at ``t=0``, to compute the initial
    electric field produced by the boundaries. Note that this function is also evaluated
    inside the embedded boundary. For this reason, it is important to define
    this function in such a way that it is constant inside the embedded boundary.

.. _running-cpp-parameters-parallelization:

Distribution across MPI ranks and parallelization
-------------------------------------------------

* ``warpx.numprocs`` (`2 ints` for 2D, `3 ints` for 3D) optional (default `none`)
    This optional parameter can be used to control the domain decomposition on the
    coarsest level. The domain will be chopped into the exact number of pieces in each
    dimension as specified by this parameter. If it's not specified, the domain
    decomposition will be determined by the parameters that will be discussed below.  If
    specified, the product of the numbers must be equal to the number of MPI processes.

* ``amr.max_grid_size`` (`integer`) optional (default `128`)
    Maximum allowable size of each **subdomain**
    (expressed in number of grid points, in each direction).
    Each subdomain has its own ghost cells, and can be handled by a
    different MPI rank ; several OpenMP threads can work simultaneously on the
    same subdomain.

    If ``max_grid_size`` is such that the total number of subdomains is
    **larger** that the number of MPI ranks used, than some MPI ranks
    will handle several subdomains, thereby providing additional flexibility
    for **load balancing**.

    When using mesh refinement, this number applies to the subdomains
    of the coarsest level, but also to any of the finer level.

* ``algo.load_balance_intervals`` (`string`) optional (default `0`)
    Using the `Intervals parser`_ syntax, this string defines the timesteps at which
    WarpX should try to redistribute the work across MPI ranks, in order to have
    better load balancing.
    Use 0 to disable load_balancing.

    When performing load balancing, WarpX measures the wall time for
    computational parts of the PIC cycle. It then uses this data to decide
    how to redistribute the subdomains across MPI ranks. (Each subdomain
    is unchanged, but its owner is changed in order to have better performance.)
    This relies on each MPI rank handling several (in fact many) subdomains
    (see ``max_grid_size``).

* ``algo.load_balance_efficiency_ratio_threshold`` (`float`) optional (default `1.1`)
    Controls whether to adopt a proposed distribution mapping computed during a load balance.
    If the the ratio of the proposed to current distribution mapping *efficiency* (i.e.,
    average cost per MPI process; efficiency is a number in the range [0, 1]) is greater
    than the threshold value, the proposed distribution mapping is adopted.  The suggested
    range of values is ``algo.load_balance_efficiency_ratio_threshold >= 1``, which ensures
    that the new distribution mapping is adopted only if doing so would improve the load
    balance efficiency. The higher the threshold value, the more conservative is the criterion
    for adoption of a proposed distribution; for example, with
    ``algo.load_balance_efficiency_ratio_threshold = 1``, the proposed distribution is
    adopted *any* time the proposed distribution improves load balancing; if instead
    ``algo.load_balance_efficiency_ratio_threshold = 2``, the proposed distribution is
    adopted only if doing so would yield a 100% to the load balance efficiency (with this
    threshold value, if the  current efficiency is ``0.45``, the new distribution would only be
    adopted if the proposed efficiency were greater than ``0.9``).

* ``algo.load_balance_with_sfc`` (`0` or `1`) optional (default `0`)
    If this is `1`: use a Space-Filling Curve (SFC) algorithm in order to
    perform load-balancing of the simulation.
    If this is `0`: the Knapsack algorithm is used instead.

* ``algo.load_balance_knapsack_factor`` (`float`) optional (default `1.24`)
    Controls the maximum number of boxes that can be assigned to a rank during
    load balance when using the 'knapsack' policy for update of the distribution
    mapping; the maximum is
    `load_balance_knapsack_factor*(average number of boxes per rank)`.
    For example, if there are 4 boxes per rank and `load_balance_knapsack_factor=2`,
    no more than 8 boxes can be assigned to any rank.

* ``algo.load_balance_costs_update`` (``heuristic`` or ``timers``) optional (default ``timers``)
    If this is `heuristic`: load balance costs are updated according to a measure of
    particles and cells assigned to each box of the domain.  The cost :math:`c` is
    computed as

    .. math::

       c = n_{\text{particle}} \cdot w_{\text{particle}} + n_{\text{cell}} \cdot w_{\text{cell}},

    where
    :math:`n_{\text{particle}}` is the number of particles on the box,
    :math:`w_{\text{particle}}` is the particle cost weight factor (controlled by ``algo.costs_heuristic_particles_wt``),
    :math:`n_{\text{cell}}` is the number of cells on the box, and
    :math:`w_{\text{cell}}` is the cell cost weight factor (controlled by ``algo.costs_heuristic_cells_wt``).

    If this is `timers`: costs are updated according to in-code timers.

* ``algo.costs_heuristic_particles_wt`` (`float`) optional
    Particle weight factor used in `Heuristic` strategy for costs update; if running on GPU,
    the particle weight is set to a value determined from single-GPU tests on Summit,
    depending on the choice of solver (FDTD or PSATD) and order of the particle shape.
    If running on CPU, the default value is `0.9`. If running on GPU, the default value is

    +----------+-----------------------+
    |          | Particle shape factor |
    +----------+-------+-------+-------+
    |          | 1     | 2     | 3     |
    +==========+=======+=======+=======+
    | FDTD/CKC | 0.599 | 0.732 | 0.855 |
    +----------+-------+-------+-------+
    | PSATD    | 0.425 | 0.595 | 0.75  |
    +----------+-------+-------+-------+

* ``algo.costs_heuristic_cells_wt`` (`float`) optional
    Cell weight factor used in `Heuristic` strategy for costs update; if running on GPU,
    the cell weight is set to a value determined from single-GPU tests on Summit,
    depending on the choice of solver (FDTD or PSATD) and order of the particle shape.
    If running on CPU, the default value is `0.1`. If running on GPU, the default value is

    +----------+-----------------------+
    |          | Particle shape factor |
    +----------+-------+-------+-------+
    |          | 1     | 2     | 3     |
    +==========+=======+=======+=======+
    | FDTD/CKC | 0.401 | 0.268 | 0.145 |
    +----------+-------+-------+-------+
    | PSATD    | 0.575 | 0.405 | 0.25  |
    +----------+-------+-------+-------+

* ``warpx.do_dynamic_scheduling`` (`0` or `1`) optional (default `1`)
    Whether to activate OpenMP dynamic scheduling.

* ``warpx.roundrobin_sfc`` (`0` or `1`) optional (default `0`)
    Whether to use AMReX's RRSFS strategy for making DistributionMapping to
    override the default space filling curve (SFC) strategy. If this is
    enabled, the round robin method is used to distribute Boxes ordered by
    SFC. This could potentially mitigate the load imbalance issue during
    initialization by avoiding putting neighboring boxes on the same
    process.

.. _running-cpp-parameters-parser:

Math parser and user-defined constants
--------------------------------------

WarpX uses AMReX's math parser that reads expressions in the input file.
It can be used in all input parameters that consist of one or more integers or floats.
Integer input expecting boolean, 0 or 1, are not parsed.
Note that when multiple values are expected, the expressions are space delimited.
For integer input values, the expressions are evaluated as real numbers and the final result rounded to the nearest integer.
See `this section <https://amrex-codes.github.io/amrex/docs_html/Basics.html#parser>`__ of the AMReX documentation for a complete list of functions supported by the math parser.

WarpX constants
^^^^^^^^^^^^^^^

WarpX provides a few pre-defined constants, that can be used for any parameter that consists of one or more floats.

======== ===================
q_e      elementary charge
m_e      electron mass
m_p      proton mass
m_u      unified atomic mass unit (Dalton)
epsilon0 vacuum permittivity
mu0      vacuum permeability
clight   speed of light
kb       Boltzmann's constant (J/K)
pi       math constant pi
======== ===================

See ``Source/Utils/WarpXConst.H`` for the values.

User-defined constants
^^^^^^^^^^^^^^^^^^^^^^

Users can define their own constants in the input file.
These constants can be used for any parameter that consists of one or more integers or floats.
User-defined constant names can contain only letters, numbers and the character ``_``.
The name of each constant has to begin with a letter. The following names are used
by WarpX, and cannot be used as user-defined constants: ``x``, ``y``, ``z``, ``X``, ``Y``, ``t``.
The values of the constants can include the predefined WarpX constants listed above as well as other user-defined constants.
For example:

* ``my_constants.a0 = 3.0``
* ``my_constants.z_plateau = 150.e-6``
* ``my_constants.n0 = 1.e22``
* ``my_constants.wp = sqrt(n0*q_e**2/(epsilon0*m_e))``

Coordinates
^^^^^^^^^^^

Besides, for profiles that depend on spatial coordinates (the plasma momentum distribution or the laser field, see below `Particle initialization` and `Laser initialization`), the parser will interpret some variables as spatial coordinates. These are specified in the input parameter, i.e., ``density_function(x,y,z)`` and ``field_function(X,Y,t)``.

The parser reads python-style expressions between double quotes, for instance
``"a0*x**2 * (1-y*1.e2) * (x>0)"`` is a valid expression where ``a0`` is a
user-defined constant (see above) and ``x`` and ``y`` are spatial coordinates. The names are case sensitive. The factor
``(x>0)`` is ``1`` where ``x>0`` and ``0`` where ``x<=0``. It allows the user to
define functions by intervals.
Alternatively the expression above can be written as ``if(x>0, a0*x**2 * (1-y*1.e2), 0)``.

.. _running-cpp-parameters-particle:

Particle initialization
-----------------------

* ``particles.species_names`` (`strings`, separated by spaces)
    The name of each species. This is then used in the rest of the input deck ;
    in this documentation we use `<species_name>` as a placeholder.

* ``particles.photon_species`` (`strings`, separated by spaces)
    List of species that are photon species, if any.
    **This is required when compiling with QED=TRUE.**

* ``particles.use_fdtd_nci_corr`` (`0` or `1`) optional (default `0`)
    Whether to activate the FDTD Numerical Cherenkov Instability corrector.
    Not currently available in the RZ configuration.

* ``particles.rigid_injected_species`` (`strings`, separated by spaces)
    List of species injected using the rigid injection method. The rigid injection
    method is useful when injecting a relativistic particle beam in boosted-frame
    simulations; see the :ref:`input-output section <boosted_frame-io>` for more details.
    For species injected using this method, particles are translated along the `+z`
    axis with constant velocity as long as their ``z`` coordinate verifies
    ``z<zinject_plane``. When ``z>zinject_plane``,
    particles are pushed in a standard way, using the specified pusher.
    (see the parameter ``<species_name>.zinject_plane`` below)

* ``particles.do_tiling`` (`bool`) optional (default `false` if WarpX is compiled for GPUs, `true` otherwise)
    Controls whether tiling ('cache blocking') transformation is used for particles.
    Tiling should be on when using OpenMP and off when using GPUs.

* ``<species_name>.species_type`` (`string`) optional (default `unspecified`)
    Type of physical species.
    Currently, the accepted species are
    ``"electron"``, ``"positron"``, ``"muon"``, ``"antimuon"``, ``"photon"``, ``"neutron"``, ``"proton"`` , ``"alpha"``,
    ``"hydrogen1"`` (a.k.a. ``"protium"``), ``"hydrogen2"`` (a.k.a. ``"deuterium"``), ``"hydrogen3"`` (a.k.a. ``"tritium"``),
    ``"helium"``, ``"helium3"``, ``"helium4"``,
    ``"lithium"``, ``"lithium6"``, ``"lithium7"``, ``"beryllium"``, ``"beryllium9"``, ``"boron"``, ``"boron10"``, ``"boron11"``,
    ``"carbon"``, ``"carbon12"``, ``"carbon13"``, ``"carbon14"``, ``"nitrogen"``, ``"nitrogen14"``, ``"nitrogen15"``,
    ``"oxygen"``, ``"oxygen16"``, ``"oxygen17"``, ``"oxygen18"``, ``"fluorine"``, ``"fluorine19"``, ``"neon"``, ``"neon20"``,
    ``"neon21"``, ``"neon22"``, ``"aluminium"``, ``"argon"``, ``"copper"``, ``"xenon"`` and ``"gold"``.
    The difference between ``"proton"`` and ``"hydrogen1"`` is that the mass of the latter includes also the mass
    of the bound electron (same for ``"alpha"`` and ``"helium4"``). When only the name of an element is specified, the mass
    is a weighted average of the masses of the stable isotopes. For all the elements with ``Z < 11`` we provide
    also the stable isotopes as an option for ``species_type`` (e.g., ``"helium3"`` and ``"helium4"``).
    Either ``species_type`` or both ``mass`` and ``charge`` have to be specified.

* ``<species_name>.charge`` (`float`) optional (default `NaN`)
    The charge of one `physical` particle of this species.
    If ``species_type`` is specified, the charge will be set to the physical value and ``charge`` is optional.
    When ``<species>.do_field_ionization = 1``, the physical particle charge is equal to ``ionization_initial_level * charge``, so latter parameter should be equal to q_e (which is defined in WarpX as the elementary charge in coulombs).

* ``<species_name>.mass`` (`float`) optional (default `NaN`)
    The mass of one `physical` particle of this species.
    If ``species_type`` is specified, the mass will be set to the physical value and ``mass`` is optional.

* ``<species_name>.xmin,ymin,zmin`` and ``<species_name>.xmax,ymax,zmax`` (`float`) optional (default unlimited)
    When ``<species_name>.xmin`` and ``<species_name>.xmax`` are set, they delimit the region within which particles are injected.
    If periodic boundary conditions are used in direction ``i``, then the default (i.e. if the range is not specified) range will be the simulation box, ``[geometry.prob_hi[i], geometry.prob_lo[i]]``.

* ``<species_name>.injection_sources`` (``list of strings``) optional
    Names of additional injection sources. By default, WarpX assumes one injection source per species, hence all of the input
    parameters below describing the injection are parameters directly of the species. However, this option allows
    additional sources, the names of which are specified here. For each source, the name of the source is added to the
    input parameters below. For instance, with ``<species_name>.injection_sources = source1 source2`` there can be the two input
    parameters ``<species_name>.source1.injection_style`` and ``<species_name>.source2.injection_style``.
    For the parameters of each source, the parameter with the name of the source will be used.
    If it is not given, the value of the parameter without the source name will be used. This allows parameters used for all
    sources to be specified once. For example, if the ``source1`` and ``source2`` have the same value of ``uz_m``, then it can be
    set using ``<species_name>.uz_m`` instead of setting it for each source.
    Note that since by default ``<species_name>.injection_style = none``, all injection sources can be input this way.
    Note that if a moving window is used, the bulk velocity of all of the sources must be the same since it is used when updating the window.

* ``<species_name>.injection_style`` (`string`; default: ``none``)
    Determines how the (macro-)particles will be injected in the simulation.
    The number of particles per cell is always given with respect to the coarsest level (level 0/mother grid), even if particles are immediately assigned to a refined patch.

    The options are:

    * ``NUniformPerCell``: injection with a fixed number of evenly-spaced particles per cell.
      This requires the additional parameter ``<species_name>.num_particles_per_cell_each_dim``.

    * ``NRandomPerCell``: injection with a fixed number of randomly-distributed particles per cell.
      This requires the additional parameter ``<species_name>.num_particles_per_cell``.

    * ``SingleParticle``: Inject a single macroparticle.
      This requires the additional parameters:

      * ``<species_name>.single_particle_pos`` (`3 doubles`, particle 3D position [meter])

      * ``<species_name>.single_particle_u`` (`3 doubles`, particle 3D normalized momentum, i.e. :math:`\gamma \beta`)

      * ``<species_name>.single_particle_weight`` ( `double`, macroparticle weight, i.e. number of physical particles it represents)

    * ``MultipleParticles``: Inject multiple macroparticles.
      This requires the additional parameters:

      * ``<species_name>.multiple_particles_pos_x`` (list of `doubles`, X positions of the particles [meter])

      * ``<species_name>.multiple_particles_pos_y`` (list of `doubles`, Y positions of the particles [meter])

      * ``<species_name>.multiple_particles_pos_z`` (list of `doubles`, Z positions of the particles [meter])

      * ``<species_name>.multiple_particles_ux`` (list of `doubles`, X normalized momenta of the particles, i.e. :math:`\gamma \beta_x`)

      * ``<species_name>.multiple_particles_uy`` (list of `doubles`, Y normalized momenta of the particles, i.e. :math:`\gamma \beta_y`)

      * ``<species_name>.multiple_particles_uz`` (list of `doubles`, Z normalized momenta of the particles, i.e. :math:`\gamma \beta_z`)

      * ``<species_name>.multiple_particles_weight`` (list of `doubles`, macroparticle weights, i.e. number of physical particles each represents)

    * ``gaussian_beam``: Inject particle beam with gaussian distribution in
      space in all directions. This requires additional parameters:

      * ``<species_name>.q_tot`` (beam charge),

      * ``<species_name>.npart`` (number of macroparticles in the beam),

      * ``<species_name>.x/y/z_m`` (average position in `x/y/z`),

      * ``<species_name>.x/y/z_rms`` (standard deviation in `x/y/z`),

      There are additional optional parameters:

      * ``<species_name>.x/y/z_cut`` (optional, particles with ``abs(x-x_m) > x_cut*x_rms`` are not injected, same for y and z. ``<species_name>.q_tot`` is the charge of the un-cut beam, so that cutting the distribution is likely to result in a lower total charge),
      * ``<species_name>.do_symmetrize`` (optional, whether to symmetrize the beam)

      * ``<species_name>.symmetrization_order`` (order of symmetrization, default is 4, can be 4 or 8).

      If ``<species_name>.do_symmetrize`` is 0, no symmetrization occurs.  If ``<species_name>.do_symmetrize`` is 1,
      then the beam is symmetrized according to the value of ``<species_name>.symmetrization_order``.
      If set to 4, symmetrization is in the x and y direction, (x,y) (-x,y) (x,-y) (-x,-y).
      If set to 8, symmetrization is also done with x and y exchanged, (y,x), (-y,x), (y,-x), (-y,-x)).

      * ``<species_name>.focal_distance`` (optional, distance between the beam centroid and the position of the focal plane of the beam, along the direction of the beam mean velocity; space charge is ignored in the initialization of the particles)

      If ``<species_name>.focal_distance`` is specified, ``x_rms``, ``y_rms`` and ``z_rms`` are the sizes of the beam in the focal plane. Since the beam is not necessarily initialized close to its focal plane, the initial size of the beam will differ from ``x_rms``, ``y_rms``, ``z_rms``.

      Usually, in accelerator physics the operative quantities are the normalized emittances :math:`\epsilon_{x,y}` and beta functions :math:`\beta_{x,y}`.
      We assume that the beam travels along :math:`z` and we mark the quantities evaluated at the focal plane with a :math:`*`.
      Therefore, the normalized transverse emittances and beta functions are related to the focal distance :math:`f = z - z^*`, the beam sizes :math:`\sigma_{x,y}` (which in the code are ``x_rms``, ``y_rms``), the beam relativistic Lorentz factor :math:`\gamma`, and the normalized momentum spread :math:`\Delta u_{x,y}` according to the equations below (:cite:t:`param-Wiedemann2015`).

      .. math::

          \Delta u_{x,y} &= \frac{\epsilon^*_{x,y}}{\sigma^*_{x,y}},

          \sigma*_{x, y} &= \sqrt{ \frac{ \epsilon^*_{x,y} \beta^*_{x,y} }{\gamma}},

          \sigma_{x,y}(z) &= \sigma^*_{x,y} \sqrt{1 + \left( \frac{z - z^*}{\beta^*_{x,y}} \right)^2}


    * ``external_file``: Inject macroparticles with properties (mass, charge, position, and momentum - :math:`\gamma \beta m c`) read from an external openPMD file.
      With it users can specify the additional arguments:

      * ``<species_name>.injection_file`` (`string`) openPMD file name and

      * ``<species_name>.charge`` (`double`) optional (default is read from openPMD file) when set this will be the charge of the physical particle represented by the injected macroparticles.

      * ``<species_name>.mass`` (`double`) optional (default is read from openPMD file) when set this will be the charge of the physical particle represented by the injected macroparticles.

      * ``<species_name>.z_shift`` (`double`) optional (default is no shift) when set this value will be added to the longitudinal, ``z``, position of the particles.

      * ``<species_name>.impose_t_lab_from_file`` (`bool`) optional (default is false) only read if warpx.gamma_boost > 1., it allows to set t_lab for the Lorentz Transform as being the time stored in the openPMD file.

      Warning: ``q_tot!=0`` is not supported with the ``external_file`` injection style. If a value is provided, it is ignored and no re-scaling is done.
      The external file must include the species ``openPMD::Record`` labeled ``position`` and ``momentum`` (`double` arrays), with dimensionality and units set via ``openPMD::setUnitDimension`` and ``setUnitSI``.
      If the external file also contains ``openPMD::Records`` for ``mass`` and ``charge`` (constant `double` scalars) then the species will use these, unless overwritten in the input file (see ``<species_name>.mass``, ``<species_name>.charge`` or ``<species_name>.species_type``).
      The ``external_file`` option is currently implemented for 2D, 3D and RZ geometries, with record components in the cartesian coordinates ``(x,y,z)`` for 3D and RZ, and ``(x,z)`` for 2D.
      For more information on the `openPMD format <https://github.com/openPMD>`__ and how to build WarpX with it, please visit :ref:`the install section <install-developers>`.

    * ``NFluxPerCell``: Continuously inject a flux of macroparticles from a surface. The emitting surface can be chosen to be either a plane
      defined by the user (using some of the parameters listed below), or the embedded boundary (see :ref:`Embedded Boundary Conditions <running-cpp-parameters-eb>`).
      This requires the additional parameters:

      * ``<species_name>.flux_profile`` (see the description of this parameter further below)

      * ``<species_name>.inject_from_embedded_boundary`` (`0` or `1`, default `0` ; whether to inject from the embedded boundary or from a user-specified plane.
        When injecting from the embedded boundary, the momentum distribution specified by the user along ``z`` (see e.g. ``uz_m``, ``uz_th`` below) is interpreted
        as the momentum distribution along the local normal to the embedded boundary.)

      * ``<species_name>.surface_flux_pos`` (only used when injecting from a plane, `double`, location of the injection plane [meter])

      * ``<species_name>.flux_normal_axis`` (only used when injecting from a plane, `x`, `y`, or `z` for 3D, `x` or `z` for 2D, or `r`, `t`, or `z` for RZ. When `flux_normal_axis` is `r` or `t`, the `x` and `y` components of the user-specified momentum distribution are interpreted as the `r` and `t` components respectively)

      * ``<species_name>.flux_direction`` (only used when injecting from a plane, `-1` or `+1`, direction of flux relative to the plane)

      * ``<species_name>.num_particles_per_cell`` (`double`)

      * ``<species_name>.flux_tmin`` (`double`, Optional time at which the flux will be turned on. Ignored when negative.)

      * ``<species_name>.flux_tmax`` (`double`, Optional time at which the flux will be turned off. Ignored when negative.)

    * ``none``: Do not inject macro-particles (for example, in a simulation that starts with neutral, ionizable atoms, one may want to create the electrons species -- where ionized electrons can be stored later on -- without injecting electron macro-particles).

* ``<species_name>.num_particles_per_cell_each_dim`` (`3 integers in 3D and RZ, 2 integers in 2D`)
    With the NUniformPerCell injection style, this specifies the number of particles along each axis
    within a cell. Note that for RZ, the three axis are radius, theta, and z and that the recommended
    number of particles per theta is at least two times the number of azimuthal modes requested.
    (It is recommended to do a convergence scan of the number of particles per theta)

* ``<species_name>.random_theta`` (`bool`) optional (default `1`)
    When using RZ geometry, whether to randomize the azimuthal position of particles.
    This is used when ``<species_name>.injection_style = NUniformPerCell``.

* ``<species_name>.do_splitting`` (`bool`) optional (default `0`)
    Split particles of the species when crossing the boundary from a lower
    resolution domain to a higher resolution domain.

    Currently implemented on CPU only.

* ``<species_name>.do_continuous_injection`` (`0` or `1`)
    Whether to inject particles during the simulation, and not only at
    initialization. This can be required with a moving window and/or when
    running in a boosted frame.

* ``<species_name>.initialize_self_fields`` (`0` or `1`)
    Whether to calculate the space-charge fields associated with this species
    at the beginning of the simulation.
    The fields are calculated for the mean gamma of the species.

* ``<species_name>.self_fields_required_precision`` (`float`, default: 1.e-11)
    The relative precision with which the initial space-charge fields should
    be calculated. More specifically, the initial space-charge fields are
    computed with an iterative Multi-Level Multi-Grid (MLMG) solver.
    For highly-relativistic beams, this solver can fail to reach the default
    precision within a reasonable time ; in that case, users can set a
    relaxed precision requirement through ``self_fields_required_precision``.

* ``<species_name>.self_fields_absolute_tolerance`` (`float`, default: 0.0)
    The absolute tolerance with which the space-charge fields should be
    calculated in units of :math:`\mathrm{V/m}^2`. More specifically, the acceptable
    residual with which the solution can be considered converged. In general
    this should be left as the default, but in cases where the simulation state
    changes very little between steps it can occur that the initial guess for
    the MLMG solver is so close to the converged value that it fails to improve
    that solution sufficiently to reach the ``self_fields_required_precision``
    value.

* ``<species_name>.self_fields_max_iters`` (`integer`, default: 200)
    Maximum number of iterations used for MLMG solver for initial space-charge
    fields calculation. In case if MLMG converges but fails to reach the desired
    ``self_fields_required_precision``, this parameter may be increased.

* ``<species_name>.profile`` (`string`)
    Density profile for this species. The options are:

    * ``constant``: Constant density profile within the box, or between ``<species_name>.xmin``
      and ``<species_name>.xmax`` (and same in all directions). This requires additional
      parameter ``<species_name>.density``. i.e., the plasma density in :math:`m^{-3}`.

    * ``predefined``: Predefined density profile.
      This requires additional parameters ``<species_name>.predefined_profile_name`` and ``<species_name>.predefined_profile_params``.
      Currently, only a parabolic channel density profile is implemented.

    * ``parse_density_function``: the density is given by a function in the input file.
      It requires additional argument ``<species_name>.density_function(x,y,z)``, which is a
      mathematical expression for the density of the species, e.g.
      ``electrons.density_function(x,y,z) = "n0+n0*x**2*1.e12"`` where ``n0`` is a
      user-defined constant, see above. WARNING: where ``density_function(x,y,z)`` is close to zero, particles will still be injected between ``xmin`` and ``xmax`` etc., with a null weight. This is undesirable because it results in useless computing. To avoid this, see option ``density_min`` below.

* ``<species_name>.flux_profile`` (`string`)
    Defines the expression of the flux, when using ``<species_name>.injection_style=NFluxPerCell``

    * ``constant``: Constant flux. This requires the additional parameter ``<species_name>.flux``.
      i.e., the injection flux in :math:`m^{-2}.s^{-1}`.

    * ``parse_flux_function``: the flux is given by a function in the input file.
      It requires the additional argument ``<species_name>.flux_function(x,y,z,t)``, which is a
      mathematical expression for the flux of the species.

* ``<species_name>.density_min`` (`float`) optional (default `0.`)
    Minimum plasma density. No particle is injected where the density is below this value.

* ``<species_name>.density_max`` (`float`) optional (default `infinity`)
    Maximum plasma density. The density at each point is the minimum between the value given in the profile, and `density_max`.

* ``<species_name>.radially_weighted`` (`bool`) optional (default `true`)
    Whether particle's weight is varied with their radius. This only applies to cylindrical geometry.
    The only valid value is true.

* ``<species_name>.momentum_distribution_type`` (`string`)
    Distribution of the normalized momentum (`u=p/mc`) for this species. The options are:

    * ``at_rest``: Particles are initialized with zero momentum.

    * ``constant``: constant momentum profile. This can be controlled with the additional parameters
      ``<species_name>.ux``, ``<species_name>.uy`` and ``<species_name>.uz``, the normalized
      momenta in the x, y and z direction respectively, which are all ``0.`` by default.

    * ``uniform``: uniform probability distribution between a minimum and a maximum value.
      The x, y and z directions are sampled independently and the final momentum space is a cuboid.
      The parameters that control the minimum and maximum domain of the distribution
      are ``<species_name>.u<x,y,z>_min`` and ``<species_name>.u<x,y,z>_max`` in each
      direction respectively (e.g., ``<species_name>.uz_min = 0.2`` and ``<species_name>.uz_max = 0.4``
      to control the generation along the ``z`` direction).
      All the parameters default to ``0``.

    * ``gaussian``: gaussian momentum distribution in all 3 directions. This can be controlled with the
      additional arguments for the average momenta along each direction
      ``<species_name>.ux_m``, ``<species_name>.uy_m`` and ``<species_name>.uz_m`` as
      well as standard deviations along each direction ``<species_name>.ux_th``,
      ``<species_name>.uy_th`` and ``<species_name>.uz_th``.
      These 6 parameters are all ``0.`` by default.

    * ``gaussianflux``: Gaussian momentum flux distribution, which is Gaussian in the plane and v*Gaussian normal to the plane.
      It can only be used when ``injection_style = NFluxPerCell``.
      This can be controlled with the additional arguments to specify the plane's orientation, ``<species_name>.flux_normal_axis`` and
      ``<species_name>.flux_direction``, for the average momenta along each direction
      ``<species_name>.ux_m``, ``<species_name>.uy_m`` and ``<species_name>.uz_m``, as
      well as standard deviations along each direction ``<species_name>.ux_th``,
      ``<species_name>.uy_th`` and ``<species_name>.uz_th``.
      ``ux_m``, ``uy_m``, ``uz_m``, ``ux_th``, ``uy_th`` and ``uz_th`` are all ``0.`` by default.

    * ``maxwell_boltzmann``: Maxwell-Boltzmann distribution that takes a dimensionless
      temperature parameter :math:`\theta` as an input, where :math:`\theta = \frac{k_\mathrm{B} \cdot T}{m \cdot c^2}`,
      :math:`T` is the temperature in Kelvin, :math:`k_\mathrm{B}` is the Boltzmann constant, :math:`c` is the speed of light, and :math:`m` is the mass of the species.
      Theta is specified by a combination of ``<species_name>.theta_distribution_type``, ``<species_name>.theta``, and ``<species_name>.theta_function(x,y,z)`` (see below).
      For values of :math:`\theta > 0.01`, errors due to ignored relativistic terms exceed 1%.
      Temperatures less than zero are not allowed.
      The plasma can be initialized to move at a bulk velocity :math:`\beta = v/c`.
      The speed is specified by the parameters ``<species_name>.beta_distribution_type``, ``<species_name>.beta``, and ``<species_name>.beta_function(x,y,z)`` (see below).
      :math:`\beta` can be positive or negative and is limited to the range :math:`-1 < \beta < 1`.
      The direction of the velocity field is given by ``<species_name>.bulk_vel_dir = (+/-) 'x', 'y', 'z'``, and must be the same across the domain.
      Please leave no whitespace
      between the sign and the character on input. A direction without a sign will be treated as
      positive. The MB distribution is initialized in the drifting frame by sampling three Gaussian
      distributions in each dimension using, the Box Mueller method, and then the distribution is
      transformed to the simulation frame using the flipping method. The flipping method can be
      found in Zenitani 2015 section III. B. (Phys. Plasmas 22, 042116).
      By default, ``beta`` is equal to ``0.`` and ``bulk_vel_dir`` is ``+x``.

      Note that though the particles may move at relativistic speeds in the simulation frame,
      they are not relativistic in the drift frame. This is as opposed to the Maxwell Juttner
      setting, which initializes particles with relativistic momentums in their drifting frame.

    * ``maxwell_juttner``: Maxwell-Juttner distribution for high temperature plasma that takes a dimensionless temperature parameter :math:`\theta` as an input, where :math:`\theta = \frac{k_\mathrm{B} \cdot T}{m \cdot c^2}`,
      :math:`T` is the temperature in Kelvin, :math:`k_\mathrm{B}` is the Boltzmann constant, and :math:`m` is the mass of the species.
      Theta is specified by a combination of ``<species_name>.theta_distribution_type``, ``<species_name>.theta``, and ``<species_name>.theta_function(x,y,z)`` (see below).
      The Sobol method used to generate the distribution will not terminate for :math:`\theta \lesssim 0.1`, and the code will abort if it encounters a temperature below that threshold.
      The Maxwell-Boltzmann distribution is recommended for temperatures in the range :math:`0.01 < \theta < 0.1`.
      Errors due to relativistic effects can be expected to approximately between 1% and 10%.
      The plasma can be initialized to move at a bulk velocity :math:`\beta = v/c`.
      The speed is specified by the parameters ``<species_name>.beta_distribution_type``, ``<species_name>.beta``, and ``<species_name>.beta_function(x,y,z)`` (see below).
      :math:`\beta` can be positive or negative and is limited to the range :math:`-1 < \beta < 1`.
      The direction of the velocity field is given by ``<species_name>.bulk_vel_dir = (+/-) 'x', 'y', 'z'``, and must be the same across the domain.
      Please leave no whitespace
      between the sign and the character on input. A direction without a sign will be treated as
      positive. The MJ distribution will be initialized in the moving frame using the Sobol method,
      and then the distribution will be transformed to the simulation frame using the flipping method.
      Both the Sobol and the flipping method can be found in Zenitani 2015 (Phys. Plasmas 22, 042116).
      By default, ``beta`` is equal to ``0.`` and ``bulk_vel_dir`` is ``+x``.

      Please take notice that particles initialized with this setting can be relativistic in two ways.
      In the simulation frame, they can drift with a relativistic speed beta. Then, in the drifting
      frame they are still moving with relativistic speeds due to high temperature. This is as opposed
      to the Maxwell Boltzmann setting, which initializes non-relativistic plasma in their relativistic
      drifting frame.

    * ``radial_expansion``: momentum depends on the radial coordinate linearly. This
      can be controlled with additional parameter ``u_over_r`` which is the slope (``0.`` by default).

    * ``parse_momentum_function``: the momentum :math:`u = (u_{x},u_{y},u_{z})=(\gamma v_{x}/c,\gamma v_{y}/c,\gamma v_{z}/c)` is given by a function in the input
      file. It requires additional arguments ``<species_name>.momentum_function_ux(x,y,z)``,
      ``<species_name>.momentum_function_uy(x,y,z)`` and ``<species_name>.momentum_function_uz(x,y,z)``,
      which gives the distribution of each component of the momentum as a function of space.

    * ``gaussian_parse_momentum_function``: Gaussian momentum distribution where the mean and the standard deviation are given by functions of position in the input file.
      Both are assumed to be non-relativistic.
      The mean is the normalized momentum, :math:`u_m = \gamma v_m/c`.
      The standard deviation is normalized, :math:`u_{th} = v_{th}/c`.
      For example, this might be ``u_th = sqrt(T*q_e/mass)/clight`` given the temperature (in eV) and mass.
      It requires the following arguments:

      * ``<species_name>.momentum_function_ux_m(x,y,z)``: mean :math:`u_{x}`
      * ``<species_name>.momentum_function_uy_m(x,y,z)``: mean :math:`u_{y}`
      * ``<species_name>.momentum_function_uz_m(x,y,z)``: mean :math:`u_{z}`
      * ``<species_name>.momentum_function_ux_th(x,y,z)``: standard deviation of :math:`u_{x}`
      * ``<species_name>.momentum_function_uy_th(x,y,z)``: standard deviation of :math:`u_{y}`
      * ``<species_name>.momentum_function_uz_th(x,y,z)``: standard deviation of :math:`u_{z}`

* ``<species_name>.theta_distribution_type`` (`string`) optional (default ``constant``)
    Only read if ``<species_name>.momentum_distribution_type`` is ``maxwell_boltzmann`` or ``maxwell_juttner``.
    See documentation for these distributions (above) for constraints on values of theta. Temperatures less than zero are not allowed.

    * If ``constant``, use a constant temperature, given by the required float parameter ``<species_name>.theta``.

    * If ``parser``, use a spatially-dependent analytic parser function, given by the required parameter ``<species_name>.theta_function(x,y,z)``.

* ``<species_name>.beta_distribution_type`` (`string`) optional (default ``constant``)
    Only read if ``<species_name>.momentum_distribution_type`` is ``maxwell_boltzmann`` or ``maxwell_juttner``.
    See documentation for these distributions (above) for constraints on values of beta.

    * If ``constant``, use a constant speed, given by the required float parameter ``<species_name>.beta``.

    * If ``parser``, use a spatially-dependent analytic parser function, given by the required parameter ``<species_name>.beta_function(x,y,z)``.

* ``<species_name>.zinject_plane`` (`float`)
    Only read if  ``<species_name>`` is in ``particles.rigid_injected_species``.
    Injection plane when using the rigid injection method.
    See ``particles.rigid_injected_species`` above.

* ``<species_name>.rigid_advance`` (`bool`)
    Only read if ``<species_name>`` is in ``particles.rigid_injected_species``.

    * If ``false``, each particle is advanced with its
      own velocity ``vz`` until it reaches ``zinject_plane``.

    * If ``true``, each particle is advanced with the average speed of the species
      ``vzbar`` until it reaches ``zinject_plane``.

* ``species_name.predefined_profile_name`` (`string`)
    Only read if ``<species_name>.profile`` is ``predefined``.

    * If ``parabolic_channel``, the plasma profile is a parabolic profile with
      cosine-like ramps at the beginning and the end of the profile.
      The density is given by

      .. math::

          n = n_0 n(x,y) n(z-z_0)

      with

      .. math::

          n(x,y) = 1 + 4\frac{x^2+y^2}{k_p^2 R_c^4}

      where :math:`k_p` is the plasma wavenumber associated with density :math:`n_0`.
      Here, with :math:`z_0` as the start of the plasma, :math:`n(z-z_0)` is a cosine-like up-ramp from :math:`0` to :math:`L_{ramp,up}`,
      constant to :math:`1` from :math:`L_{ramp,up}` to :math:`L_{ramp,up} + L_{plateau}`
      and a cosine-like down-ramp from :math:`L_{ramp,up} + L_{plateau}` to
      :math:`L_{ramp,up} + L_{plateau}+L_{ramp,down}`. All parameters are given
      in ``predefined_profile_params``.

* ``<species_name>.predefined_profile_params`` (list of `float`)
    Parameters for the predefined profiles.

    * If ``species_name.predefined_profile_name`` is ``parabolic_channel``,
      ``predefined_profile_params`` contains a space-separated list of the
      following parameters, in this order: :math:`z_0` :math:`L_{ramp,up}` :math:`L_{plateau}`
      :math:`L_{ramp,down}` :math:`R_c` :math:`n_0`

* ``<species_name>.do_backward_propagation`` (`bool`)
    Inject a backward-propagating beam to reduce the effect of charge-separation
    fields when running in the boosted frame. See examples.

* ``<species_name>.split_type`` (`int`) optional (default `0`)
    Splitting technique. When `0`, particles are split along the simulation
    axes (4 particles in 2D, 6 particles in 3D). When `1`, particles are split
    along the diagonals (4 particles in 2D, 8 particles in 3D).

* ``<species_name>.do_not_deposit`` (`0` or `1` optional; default `0`)
    If `1` is given, both charge deposition and current deposition will
    not be done, thus that species does not contribute to the fields.

* ``<species_name>.do_not_gather`` (`0` or `1` optional; default `0`)
    If `1` is given, field gather from grids will not be done,
    thus that species will not be affected by the field on grids.

* ``<species_name>.do_not_push`` (`0` or `1` optional; default `0`)
    If `1` is given, this species will not be pushed
    by any pusher during the simulation.

* ``<species_name>.addIntegerAttributes`` (list of `string`)
    User-defined integer particle attribute for species, ``species_name``.
    These integer attributes will be initialized with user-defined functions
    when the particles are generated.
    If the user-defined integer attribute is ``<int_attrib_name>`` then the
    following required parameter must be specified to initialize the attribute.
    * ``<species_name>.attribute.<int_attrib_name>(x,y,z,ux,uy,uz,t)`` (`string`)
    ``t`` represents the physical time in seconds during the simulation.
    ``x``, ``y``, ``z`` represent particle positions in the unit of meter.
    ``ux``, ``uy``, ``uz`` represent the particle momenta in the unit of
    :math:`\gamma v/c`, where
    :math:`\gamma` is the Lorentz factor,
    :math:`v/c` is the particle velocity normalized by the speed of light.
    E.g. If ``electrons.addIntegerAttributes = upstream``
    and ``electrons.upstream(x,y,z,ux,uy,uz,t) = (x>0.0)*1`` is provided
    then, an integer attribute ``upstream`` is added to all electron particles
    and when these particles are generated, the particles with position less than ``0``
    are assigned a value of ``1``.

* ``<species_name>.addRealAttributes`` (list of `string`)
    User-defined real particle attribute for species, ``species_name``.
    These real attributes will be initialized with user-defined functions
    when the particles are generated.
    If the user-defined real attribute is ``<real_attrib_name>`` then the
    following required parameter must be specified to initialize the attribute.

   * ``<species_name>.attribute.<real_attrib_name>(x,y,z,ux,uy,uz,t)`` (`string`)
     ``t`` represents the physical time in seconds during the simulation.
     ``x``, ``y``, ``z`` represent particle positions in the unit of meter.
     ``ux``, ``uy``, ``uz`` represent the particle momenta in the unit of
     :math:`\gamma v/c`, where
     :math:`\gamma` is the Lorentz factor,
     :math:`v/c` is the particle velocity normalized by the speed of light.

* ``<species>.save_particles_at_xlo/ylo/zlo``, ``<species>.save_particles_at_xhi/yhi/zhi`` and ``<species>.save_particles_at_eb`` (`0` or `1` optional, default `0`)
    If `1` particles of this species will be copied to the scraped particle
    buffer for the specified boundary if they leave the simulation domain in
    the specified direction. **If USE_EB=TRUE** the ``save_particles_at_eb``
    flag can be set to `1` to also save particle data for the particles of this
    species that impact the embedded boundary.
    The scraped particle buffer can be used to track particle fluxes out of the
    simulation.
    The particle data can be written out by setting up a ``BoundaryScrapingDiagnostic``.
    It is also accessible via the Python interface. The
    function ``get_particle_boundary_buffer``, found in the
    ``picmi.Simulation`` class as
    ``sim.extension.get_particle_boundary_buffer()``, can be
    used to access the scraped particle buffer. An entry is included for every
    particle in the buffer of the timestep at which the particle was scraped.
    This can be accessed by passing the argument ``comp_name="step_scraped"`` to
    the above mentioned function.

    .. note::

       When accessing the data via Python, the scraped particle buffer relies on the user
       to clear the buffer after processing the data. The
       buffer will grow unbounded as particles are scraped and therefore could
       lead to memory issues if not periodically cleared. To clear the buffer
       call ``clear_buffer()``.

* ``<species>.do_field_ionization`` (`0` or `1`) optional (default `0`)
    Do field ionization for this species (using the ADK theory).

* ``<species>.do_adk_correction`` (`0` or `1`) optional (default `0`)
    Whether to apply the correction to the ADK theory proposed by Zhang, Lan and Lu in `Q. Zhang et al. (Phys. Rev. A 90, 043410, 2014) <https://doi.org/10.1103/PhysRevA.90.043410>`__.
    If so, the probability of ionization is modified using an empirical model that should be more accurate in the regime of high electric fields.
    Currently, this is only implemented for Hydrogen, although Argon is also available in the same reference.

* ``<species>.physical_element`` (`string`)
    Only read if `do_field_ionization = 1`. Symbol of chemical element for
    this species. Example: for Helium, use ``physical_element = He``.
    All the elements up to atomic number Z=100 (Fermium) are supported.

* ``<species>.ionization_product_species`` (`string`)
    Only read if `do_field_ionization = 1`. Name of species in which ionized
    electrons are stored. This species must be created as a regular species
    in the input file (in particular, it must be in `particles.species_names`).

* ``<species>.ionization_initial_level`` (`int`) optional (default `0`)
    Only read if `do_field_ionization = 1`. Initial ionization level of the
    species (must be smaller than the atomic number of chemical element given
    in `physical_element`).

* ``<species>.do_classical_radiation_reaction`` (`int`) optional (default `0`)
    Enables Radiation Reaction (or Radiation Friction) for the species. Species
    must be either electrons or positrons. Boris pusher must be used for the
    simulation. If both ``<species>.do_classical_radiation_reaction`` and
    ``<species>.do_qed_quantum_sync`` are enabled, then the classical module
    will be used when the particle's chi parameter is below ``qed_qs.chi_min``,
    the discrete quantum module otherwise.

* ``<species>.do_qed_quantum_sync`` (`int`) optional (default `0`)
    Enables Quantum synchrotron emission for this species.
    Quantum synchrotron lookup table should be either generated or loaded from disk to enable
    this process (see "Lookup tables for QED modules" section below).
    `<species>` must be either an electron or a positron species.
    **This feature requires to compile with QED=TRUE**

* ``<species>.do_qed_breit_wheeler`` (`int`) optional (default `0`)
    Enables non-linear Breit-Wheeler process for this species.
    Breit-Wheeler lookup table should be either generated or loaded from disk to enable
    this process (see "Lookup tables for QED modules" section below).
    `<species>` must be a photon species.
    **This feature requires to compile with QED=TRUE**

* ``<species>.qed_quantum_sync_phot_product_species`` (`string`)
    If an electron or a positron species has the Quantum synchrotron process, a photon product species must be specified
    (the name of an existing photon species must be provided)
    **This feature requires to compile with QED=TRUE**

* ``<species>.qed_breit_wheeler_ele_product_species`` (`string`)
    If a photon species has the Breit-Wheeler process, an electron product species must be specified
    (the name of an existing electron species must be provided)
    **This feature requires to compile with QED=TRUE**

* ``<species>.qed_breit_wheeler_pos_product_species`` (`string`)
    If a photon species has the Breit-Wheeler process, a positron product species must be specified
    (the name of an existing positron species must be provided).
    **This feature requires to compile with QED=TRUE**

* ``<species>.do_resampling`` (`0` or `1`) optional (default `0`)
    If `1` resampling is performed for this species. This means that the number of macroparticles
    will be reduced at specific timesteps while preserving the distribution function as much as
    possible (details depend on the chosen resampling algorithm).
    This can be useful in situations with continuous creation of particles (e.g. with ionization
    or with QED effects). At least one resampling trigger (see below) must be specified to actually
    perform resampling.

* ``<species>.resampling_algorithm`` (`string`) optional (default `leveling_thinning`)
    The algorithm used for resampling:

    * ``leveling_thinning`` This algorithm is defined in :cite:t:`param-MuravievCPC2021`.
      It has one parameter:

        * ``<species>.resampling_algorithm_target_ratio`` (`float`) optional (default `1.5`)
            This **roughly** corresponds to the ratio between the number of particles before and
            after resampling.

    * ``velocity_coincidence_thinning``` The particles are sorted into phase space
      cells and merged, similar to the approach described in :cite:t:`param-Vranic2015`.
      It has three parameters:

        * ``<species>.resampling_algorithm_delta_ur`` (`float`)
            The width of momentum cells used in clustering particles, in m/s.

        * ``<species>.resampling_algorithm_n_theta`` (`int`)
            The number of cell divisions to use in the :math:`\theta` direction
            when clustering the particle velocities.

        * ``<species>.resampling_algorithm_n_phi`` (`int`)
            The number of cell divisions to use in the :math:`\phi` direction
            when clustering the particle velocities.

* ``<species>.resampling_min_ppc`` (`int`) optional (default `1`)
    Resampling is not performed in cells with a number of macroparticles strictly smaller
    than this parameter.

* ``<species>.resampling_trigger_intervals`` (`string`) optional (default `0`)
    Using the `Intervals parser`_ syntax, this string defines timesteps at which resampling is
    performed.

* ``<species>.resampling_trigger_max_avg_ppc`` (`float`) optional (default `infinity`)
    Resampling is performed everytime the number of macroparticles per cell of the species
    averaged over the whole simulation domain exceeds this parameter.


.. _running-cpp-parameters-fluids:

Cold Relativistic Fluid initialization
--------------------------------------

* ``fluids.species_names`` (`strings`, separated by spaces)
    Defines the names of each fluid species. It is a required input to create and evolve fluid species using the cold relativistic fluid equations.
    Most of the parameters described in the section "Particle initialization" can also be used to initialize fluid properties (e.g. initial density distribution).
    For fluid-specific inputs we use `<fluid_species_name>` as a placeholder. Also see external fields
    for how to specify these for fluids as the function names differ.

.. _running-cpp-parameters-laser:

Laser initialization
--------------------

* ``lasers.names`` (list of `string`)
    Name of each laser. This is then used in the rest of the input deck ;
    in this documentation we use `<laser_name>` as a placeholder. The parameters below
    must be provided for each laser pulse.

* ``<laser_name>.position`` (`3 floats in 3D and 2D` ; in meters)
    The coordinates of one of the point of the antenna that will emit the laser.
    The plane of the antenna is entirely defined by ``<laser_name>.position``
    and ``<laser_name>.direction``.

    ``<laser_name>.position`` also corresponds to the origin of the coordinates system
    for the laser tranverse profile. For instance, for a Gaussian laser profile,
    the peak of intensity will be at the position given by ``<laser_name>.position``.
    This variable can thus be used to shift the position of the laser pulse
    transversally.

    .. note::
        In 2D, ``<laser_name>.position`` is still given by 3 numbers,
        but the second number is ignored.

    When running a **boosted-frame simulation**, provide the value of
    ``<laser_name>.position`` in the laboratory frame, and use ``warpx.gamma_boost``
    to automatically perform the conversion to the boosted frame. Note that,
    in this case, the laser antenna will be moving, in the boosted frame.

* ``<laser_name>.polarization`` (`3 floats in 3D and 2D`)
    The coordinates of a vector that points in the direction of polarization of
    the laser. The norm of this vector is unimportant, only its direction matters.

    .. note::
        Even in 2D, all the 3 components of this vectors are important (i.e.
        the polarization can be orthogonal to the plane of the simulation).

* ``<laser_name>.direction`` (`3 floats in 3D`)
    The coordinates of a vector that points in the propagation direction of
    the laser. The norm of this vector is unimportant, only its direction matters.

    The plane of the antenna that will emit the laser is orthogonal to this vector.

    .. warning::

        When running **boosted-frame simulations**, ``<laser_name>.direction`` should
        be parallel to ``warpx.boost_direction``, for now.

* ``<laser_name>.e_max`` (`float` ; in V/m)
    Peak amplitude of the laser field, in the focal plane.

    For a laser with a wavelength :math:`\lambda = 0.8\,\mu m`, the peak amplitude
    is related to :math:`a_0` by:

    .. math::

        E_{max} = a_0 \frac{2 \pi m_e c^2}{e\lambda} = a_0 \times (4.0 \cdot 10^{12} \;V.m^{-1})

    When running a **boosted-frame simulation**, provide the value of ``<laser_name>.e_max``
    in the laboratory frame, and use ``warpx.gamma_boost`` to automatically
    perform the conversion to the boosted frame.

* ``<laser_name>.a0`` (`float` ; dimensionless)
    Peak normalized amplitude of the laser field, in the focal plane (given in the lab frame, just as ``e_max`` above).
    See the description of ``<laser_name>.e_max`` for the conversion between ``a0`` and ``e_max``.
    Either ``a0`` or ``e_max`` must be specified.

* ``<laser_name>.wavelength`` (`float`; in meters)
    The wavelength of the laser in vacuum.

    When running a **boosted-frame simulation**, provide the value of
    ``<laser_name>.wavelength`` in the laboratory frame, and use ``warpx.gamma_boost``
    to automatically perform the conversion to the boosted frame.

* ``<laser_name>.profile`` (`string`)
    The spatio-temporal shape of the laser. The options that are currently
    implemented are:

    - ``"Gaussian"``: The transverse and longitudinal profiles are Gaussian.
    - ``"parse_field_function"``: the laser electric field is given by a function in the
      input file. It requires additional argument ``<laser_name>.field_function(X,Y,t)``, which
      is a mathematical expression , e.g.
      ``<laser_name>.field_function(X,Y,t) = "a0*X**2 * (X>0) * cos(omega0*t)"`` where
      ``a0`` and ``omega0`` are a user-defined constant, see above. The profile passed
      here is the full profile, not only the laser envelope. ``t`` is time and ``X``
      and ``Y`` are coordinates orthogonal to ``<laser_name>.direction`` (not necessarily the
      x and y coordinates of the simulation). All parameters above are required, but
      none of the parameters below are used when ``<laser_name>.parse_field_function=1``. Even
      though ``<laser_name>.wavelength`` and ``<laser_name>.e_max`` should be included in the laser
      function, they still have to be specified as they are used for numerical purposes.
    - ``"from_file"``: the electric field of the laser is read from an external file. Currently both
      the `lasy <https://lasydoc.readthedocs.io/en/latest/>`_ format as well as a custom binary format are supported. It requires to provide
      the name of the file to load setting the additional parameter ``<laser_name>.binary_file_name`` or ``<laser_name>.lasy_file_name`` (`string`).
      It accepts an optional parameter ``<laser_name>.time_chunk_size`` (`int`), supported for both lasy and binary files;
      this allows to read only time_chunk_size timesteps from the file. New timesteps are read as soon as they are needed.

      The default value is automatically set to the number of timesteps contained in the file
      (i.e. only one read is performed at the beginning of the simulation).
      It also accepts the optional parameter ``<laser_name>.delay`` (`float`; in seconds), which allows
      delaying (``delay > 0``) or anticipating (``delay < 0``) the laser by the specified amount of time.

      Details about the usage of the lasy format: lasy can produce either 3D Cartesian files or RZ files.
      WarpX can read both types of files independently of the geometry in which it was compiled (e.g. WarpX
      compiled with ``WarpX_DIMS=RZ`` can read 3D Cartesian lasy files). In the case where WarpX is compiled
      in 2D (or 1D) Cartesian, the laser antenna will emit the field values that correspond to the slice ``y=0``
      in the lasy file (and ``x=0`` in the 1D case). One can generate a lasy file from Python, see an example
      at ``Examples/Tests/laser_injection_from_file``.

      Details about the usage of the binary format: The external binary file should provide E(x,y,t) on a rectangular (necessarily uniform)
      grid. The code performs a bi-linear (in 2D) or tri-linear (in 3D) interpolation to set the field
      values. x,y,t are meant to be in S.I. units, while the field value is meant to be multiplied by
      ``<laser_name>.e_max`` (i.e. in most cases the maximum of abs(E(x,y,t)) should be 1,
      so that the maximum field intensity can be set straightforwardly with ``<laser_name>.e_max``).
      The binary file has to respect the following format:

      * ``flag`` to indicate the grid is uniform (1 byte, 0 means non-uniform, !=0 means uniform) - only uniform is supported
      * ``nt``, number of timesteps (``uint32_t``, must be >=2)
      * ``nx``, number of points along x (``uint32_t``, must be >=2)
      * ``ny``, number of points along y (``uint32_t``, must be 1 for 2D simulations and >=2 for 3D simulations)
      * ``timesteps`` (``double[2]=[t_min,t_max]``)
      * ``x_coords`` (``double[2]=[x_min,x_max]``)
      * ``y_coords`` (``double[1]`` in 2D, ``double[2]=[y_min,y_max]`` in 3D)
      * ``field_data`` (``double[nt x nx * ny]``, with ``nt`` being the slowest coordinate).

      A binary file can be generated from Python, see an example at ``Examples/Tests/laser_injection_from_file``

* ``<laser_name>.profile_t_peak`` (`float`; in seconds)
    The time at which the laser reaches its peak intensity, at the position
    given by ``<laser_name>.position`` (only used for the ``"gaussian"`` profile)

    When running a **boosted-frame simulation**, provide the value of
    ``<laser_name>.profile_t_peak`` in the laboratory frame, and use ``warpx.gamma_boost``
    to automatically perform the conversion to the boosted frame.

* ``<laser_name>.profile_duration`` (`float` ; in seconds)
    The duration of the laser pulse for the ``"gaussian"`` profile, defined as :math:`\tau` below:

    .. math::

        E(\boldsymbol{x},t) \propto \exp\left( -\frac{(t-t_{peak})^2}{\tau^2} \right)

    Note that :math:`\tau` relates to the full width at half maximum (FWHM) of *intensity*, which is closer to pulse length measurements in experiments, as :math:`\tau = \mathrm{FWHM}_I / \sqrt{2\ln(2)}` :math:`\approx \mathrm{FWHM}_I / 1.1774`.

    For a chirped laser pulse (i.e. with a non-zero ``<laser_name>.phi2``), ``profile_duration`` is the Fourier-limited duration of the pulse, not the actual duration of the pulse. See the documentation for ``<laser_name>.phi2`` for more detail.

    When running a **boosted-frame simulation**, provide the value of
    ``<laser_name>.profile_duration`` in the laboratory frame, and use ``warpx.gamma_boost``
    to automatically perform the conversion to the boosted frame.

* ``<laser_name>.profile_waist`` (`float` ; in meters)
    The waist of the transverse Gaussian :math:`w_0`, i.e. defined such that the electric field of the
    laser pulse in the focal plane is of the form:

    .. math::

        E(\boldsymbol{x},t) \propto \exp\left( -\frac{\boldsymbol{x}_\perp^2}{w_0^2} \right)

* ``<laser_name>.profile_focal_distance`` (`float`; in meters)
    The distance from ``laser_position`` to the focal plane.
    (where the distance is defined along the direction given by ``<laser_name>.direction``.)

    Use a negative number for a defocussing laser instead of a focussing laser.

    When running a **boosted-frame simulation**, provide the value of
    ``<laser_name>.profile_focal_distance`` in the laboratory frame, and use ``warpx.gamma_boost``
    to automatically perform the conversion to the boosted frame.

* ``<laser_name>.phi0`` (`float`; in radians) optional (default `0.`)
    The Carrier Envelope Phase, i.e. the phase of the laser oscillation, at the
    position where the laser envelope is maximum (only used for the ``"gaussian"`` profile)

* ``<laser_name>.stc_direction`` (`3 floats`) optional (default `1. 0. 0.`)
    Direction of laser spatio-temporal couplings.
    See definition in :cite:t:`param-AkturkOE2004`.

* ``<laser_name>.zeta`` (`float`; in meters.seconds) optional (default `0.`)
    Spatial chirp at focus in direction ``<laser_name>.stc_direction``. See definition in
    :cite:t:`param-AkturkOE2004`.

* ``<laser_name>.beta`` (`float`; in seconds) optional (default `0.`)
    Angular dispersion (or angular chirp) at focus in direction ``<laser_name>.stc_direction``.
    See definition in :cite:t:`param-AkturkOE2004`.

* ``<laser_name>.phi2`` (`float`; in seconds**2) optional (default `0.`)
    The amount of temporal chirp :math:`\phi^{(2)}` at focus (in the lab frame). Namely, a wave packet
    centered on the frequency :math:`(\omega_0 + \delta \omega)` will reach its peak intensity
    at :math:`z(\delta \omega) = z_0 - c \phi^{(2)} \, \delta \omega`. Thus, a positive
    :math:`\phi^{(2)}` corresponds to positive chirp, i.e. red part of the spectrum in the
    front of the pulse and blue part of the spectrum in the back. More specifically, the electric
    field in the focal plane is of the form:

    .. math::

        E(\boldsymbol{x},t) \propto Re\left[ \exp\left(  -\frac{(t-t_{peak})^2}{\tau^2 + 2i\phi^{(2)}} + i\omega_0 (t-t_{peak}) + i\phi_0 \right) \right]

    where :math:`\tau` is given by ``<laser_name>.profile_duration`` and represents the
    Fourier-limited duration of the laser pulse. Thus, the actual duration of the chirped laser pulse is:

    .. math::

        \tau' = \sqrt{ \tau^2 + 4 (\phi^{(2)})^2/\tau^2 }

    See also the definition in :cite:t:`param-AkturkOE2004`.

* ``<laser_name>.do_continuous_injection`` (`0` or `1`) optional (default `0`).
    Whether or not to use continuous injection.
    If the antenna starts outside of the simulation domain but enters it
    at some point (due to moving window or moving antenna in the boosted
    frame), use this so that the laser antenna is injected when it reaches
    the box boundary. If running in a boosted frame, this requires the
    boost direction, moving window direction and laser propagation direction
    to be along `z`. If not running in a boosted frame, this requires the
    moving window and laser propagation directions to be the same (`x`, `y`
    or `z`)

* ``<laser_name>.min_particles_per_mode`` (`int`) optional (default `4`)
    When using the RZ version, this specifies the minimum number of particles
    per angular mode. The laser particles are loaded into radial spokes, with
    the number of spokes given by min_particles_per_mode*(warpx.n_rz_azimuthal_modes-1).

* ``lasers.deposit_on_main_grid`` (`int`) optional (default `0`)
    When using mesh refinement, whether the antenna that emits the laser
    deposits charge/current only on the main grid (i.e. level 0), or also
    on the higher mesh-refinement levels.

* ``warpx.num_mirrors`` (`int`) optional (default `0`)
    Users can input perfect mirror condition inside the simulation domain.
    The number of mirrors is given by ``warpx.num_mirrors``. The mirrors are
    orthogonal to the `z` direction. The following parameters are required
    when ``warpx.num_mirrors`` is >0.

* ``warpx.mirror_z`` (list of `float`) required if ``warpx.num_mirrors>0``
    ``z`` location of the front of the mirrors.

* ``warpx.mirror_z_width`` (list of `float`) required if ``warpx.num_mirrors>0``
    ``z`` width of the mirrors.

* ``warpx.mirror_z_npoints`` (list of `int`) required if ``warpx.num_mirrors>0``
    In the boosted frame, depending on `gamma_boost`, ``warpx.mirror_z_width``
    can be smaller than the cell size, so that the mirror would not work. This
    parameter is the minimum number of points for the mirror. If
    ``mirror_z_width < dz/cell_size``, the upper bound of the mirror is increased
    so that it contains at least ``mirror_z_npoints``.

External fields
---------------

Applied to the grid
^^^^^^^^^^^^^^^^^^^

The external fields defined with input parameters that start with ``warpx.B_ext_grid_init_`` or ``warpx.E_ext_grid_init_``
are applied to the grid directly. In particular, these fields can be seen in the diagnostics that output the fields on the grid.

    - When using an **electromagnetic** field solver, these fields are applied to the grid at the beginning of the simulation, and serve as initial condition for the Maxwell solver.
    - When using an **electrostatic** or **magnetostatic** field solver, these fields are added to the fields computed by the Poisson solver, at each timestep.

* ``warpx.B_ext_grid_init_style`` (string) optional
    This parameter determines the type of initialization for the external
    magnetic field. By default, the
    external magnetic field (Bx,By,Bz) is initialized to (0.0, 0.0, 0.0).
    The string can be set to "constant" if a constant magnetic field is
    required to be set at initialization. If set to "constant", then an
    additional parameter, namely, ``warpx.B_external_grid`` must be specified.
    If set to ``parse_B_ext_grid_function``, then a mathematical expression can
    be used to initialize the external magnetic field on the grid. It
    requires additional parameters in the input file, namely,
    ``warpx.Bx_external_grid_function(x,y,z)``,
    ``warpx.By_external_grid_function(x,y,z)``,
    ``warpx.Bz_external_grid_function(x,y,z)`` to initialize the external
    magnetic field for each of the three components on the grid.
    Constants required in the expression can be set using ``my_constants``.
    For example, if ``warpx.Bx_external_grid_function(x,y,z)=Bo*x + delta*(y + z)``
    then the constants `Bo` and `delta` required in the above equation
    can be set using ``my_constants.Bo=`` and ``my_constants.delta=`` in the
    input file. For a two-dimensional simulation, it is assumed that the first dimension
    is `x` and the second dimension is `z`, and the value of `y` is set to zero.
    Note that the current implementation of the parser for external B-field
    does not work with RZ and the code will abort with an error message.

    If ``B_ext_grid_init_style`` is set to be ``read_from_file``, an additional parameter,
    indicating the path of an openPMD data file,
    ``warpx.read_fields_from_path`` must be specified,
    from which external B field data can be loaded into WarpX.
    One can refer to input files in ``Examples/Tests/LoadExternalField`` for more information.
    Regarding how to prepare the openPMD data file, one can refer to
    the `openPMD-example-datasets <https://github.com/openPMD/openPMD-example-datasets>`__.

* ``warpx.E_ext_grid_init_style`` (string) optional
    This parameter determines the type of initialization for the external
    electric field. By default, the
    external electric field (Ex,Ey,Ez) to (0.0, 0.0, 0.0).
    The string can be set to "constant" if a constant electric field is
    required to be set at initialization. If set to "constant", then an
    additional parameter, namely, ``warpx.E_external_grid`` must be specified
    in the input file.
    If set to ``parse_E_ext_grid_function``, then a mathematical expression can
    be used to initialize the external electric field on the grid. It
    required additional parameters in the input file, namely,
    ``warpx.Ex_external_grid_function(x,y,z)``,
    ``warpx.Ey_external_grid_function(x,y,z)``,
    ``warpx.Ez_external_grid_function(x,y,z)`` to initialize the external
    electric field for each of the three components on the grid.
    Constants required in the expression can be set using ``my_constants``.
    For example, if ``warpx.Ex_external_grid_function(x,y,z)=Eo*x + delta*(y + z)``
    then the constants `Bo` and `delta` required in the above equation
    can be set using ``my_constants.Eo=`` and ``my_constants.delta=`` in the
    input file. For a two-dimensional simulation, it is assumed that the first
    dimension is `x` and the second dimension is `z`,
    and the value of `y` is set to zero.
    Note that the current implementation of the parser for external E-field
    does not work with RZ and the code will abort with an error message.

    If ``E_ext_grid_init_style`` is set to be ``read_from_file``, an additional parameter,
    indicating the path of an openPMD data file,
    ``warpx.read_fields_from_path`` must be specified,
    from which external E field data can be loaded into WarpX.
    One can refer to input files in ``Examples/Tests/LoadExternalField`` for more information.
    Regarding how to prepare the openPMD data file, one can refer to
    the `openPMD-example-datasets <https://github.com/openPMD/openPMD-example-datasets>`__.
    Note that if both ``B_ext_grid_init_style`` and ``E_ext_grid_init_style`` are set to
    ``read_from_file``, the openPMD file specified by ``warpx.read_fields_from_path``
    should contain both B and E external fields data.

* ``warpx.E_external_grid`` & ``warpx.B_external_grid`` (list of `3 floats`)
    required when ``warpx.E_ext_grid_init_style="constant"``
    and when ``warpx.B_ext_grid_init_style="constant"``, respectively.
    External uniform and constant electrostatic and magnetostatic field added
    to the grid at initialization. Use with caution as these fields are used for
    the field solver. In particular, do not use any other boundary condition
    than periodic.

* ``warpx.maxlevel_extEMfield_init`` (default is maximum number of levels in the simulation)
    With this parameter, the externally applied electric and magnetic fields
    will not be applied for levels greater than ``warpx.maxlevel_extEMfield_init``.
    For some mesh-refinement simulations,
    the external fields are only applied to the parent grid and not the refined patches. In such cases,
    ``warpx.maxlevel_extEMfield_init`` can be set to 0.
    In that case, the other levels have external field values of 0.

Applied to Particles
^^^^^^^^^^^^^^^^^^^^

The external fields defined with input parameters that start with ``warpx.B_ext_particle_init_`` or ``warpx.E_ext_particle_init_``
are applied to the particles directly, at each timestep. As a results, these fields **cannot** be seen in the diagnostics that output the fields on the grid.

* ``particles.E_ext_particle_init_style`` & ``particles.B_ext_particle_init_style`` (string) optional (default "none")
    These parameters determine the type of the external electric and
    magnetic fields respectively that are applied directly to the particles at every timestep.
    The field values are specified in the lab frame.
    With the default ``none`` style, no field is applied.
    Possible values are ``constant``, ``parse_E_ext_particle_function`` or ``parse_B_ext_particle_function``, or
    ``repeated_plasma_lens``.

    * ``constant``: a constant field is applied, given by the input parameters
      ``particles.E_external_particle`` or ``particles.B_external_particle``, which are lists of the field components.

    * ``parse_E_ext_particle_function`` or ``parse_B_ext_particle_function``: the field is specified as an analytic
      expression that is a function of space (x,y,z) and time (t), relative to the lab frame.
      The E-field is specified by the input parameters:

        * ``particles.Ex_external_particle_function(x,y,z,t)``

        * ``particles.Ey_external_particle_function(x,y,z,t)``

        * ``particles.Ez_external_particle_function(x,y,z,t)``

      The B-field is specified by the input parameters:

        * ``particles.Bx_external_particle_function(x,y,z,t)``

        * ``particles.By_external_particle_function(x,y,z,t)``

        * ``particles.Bz_external_particle_function(x,y,z,t)``

      Note that the position is defined in Cartesian coordinates, as a function of (x,y,z), even for RZ.

    * ``read_from_file``: load the external field from an openPMD file.
        An additional parameter, indicating the path of an openPMD data file, ``particles.read_fields_from_path``
        must be specified, from which the external E field data can be loaded into WarpX.
        One can refer to input files in ``Examples/Tests/LoadExternalField`` for more information.
        Regarding how to prepare the openPMD data file, one can refer to
        the `openPMD-example-datasets <https://github.com/openPMD/openPMD-example-datasets>`__.
        Note that if both ``B_ext_particle_init_style`` and ``E_ext_particle_init_style`` are set to
        ``read_from_file``, the openPMD file specified by ``particles.read_fields_from_path``
        should contain both B and E external fields data.

        .. note::

            When using ``read_from_file``, the fields loaded from the file will be interpolated
            to the resolution of the grid used for the simulation.

    * ``repeated_plasma_lens``: apply a series of plasma lenses.
      The properties of the lenses are defined in the lab frame by the input parameters:

        * ``repeated_plasma_lens_period``, the period length of the repeat, a single float number,

        * ``repeated_plasma_lens_starts``, the start of each lens relative to the period, an array of floats,

        * ``repeated_plasma_lens_lengths``, the length of each lens, an array of floats,

        * ``repeated_plasma_lens_strengths_E``, the electric focusing strength of each lens, an array of floats, when
          ``particles.E_ext_particle_init_style`` is set to ``repeated_plasma_lens``.

        * ``repeated_plasma_lens_strengths_B``, the magnetic focusing strength of each lens, an array of floats, when
          ``particles.B_ext_particle_init_style`` is set to ``repeated_plasma_lens``.

      The repeated lenses are only defined for :math:`z > 0`.
      Once the number of lenses specified in the input are exceeded, the repeated lens stops.

      The applied field is uniform longitudinally (along z) with a hard edge,
      where residence corrections are used for more accurate field calculation. On the time step when a particle enters
      or leaves each lens, the field applied is scaled by the fraction of the time step spent within the lens.
      The fields are of the form :math:`E_x = \mathrm{strength} \cdot x`, :math:`E_y = \mathrm{strength} \cdot y`,
      and :math:`E_z = 0`, and
      :math:`B_x = \mathrm{strength} \cdot y`, :math:`B_y = -\mathrm{strength} \cdot x`, and :math:`B_z = 0`.


Applied to Cold Relativistic Fluids
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The external fields defined with input parameters that start with ``warpx.B_ext_init_`` or ``warpx.E_ext_init_``
are applied to the fluids directly, at each timestep. As a results, these fields **cannot** be seen in the diagnostics that output the fields on the grid.

* ``<fluid_species_name>.E_ext_init_style`` & ``<fluid_species_name>.B_ext_init_style`` (string) optional (default "none")
    These parameters determine the type of the external electric and
    magnetic fields respectively that are applied directly to the cold relativistic fluids at every timestep.
    The field values are specified in the lab frame.
    With the default ``none`` style, no field is applied.
    Possible values are ``parse_E_ext_function`` or ``parse_B_ext_function``.

    * ``parse_E_ext_function`` or ``parse_B_ext_function``: the field is specified as an analytic
      expression that is a function of space (x,y,z) and time (t), relative to the lab frame.
      The E-field is specified by the input parameters:

        * ``<fluid_species_name>.Ex_external_function(x,y,z,t)``

        * ``<fluid_species_name>.Ey_external_function(x,y,z,t)``

        * ``<fluid_species_name>.Ez_external_function(x,y,z,t)``

      The B-field is specified by the input parameters:

        * ``<fluid_species_name>.Bx_external_function(x,y,z,t)``

        * ``<fluid_species_name>.By_external_function(x,y,z,t)``

        * ``<fluid_species_name>.Bz_external_function(x,y,z,t)``

      Note that the position is defined in Cartesian coordinates, as a function of (x,y,z), even for RZ.

Accelerator Lattice
^^^^^^^^^^^^^^^^^^^

Several accelerator lattice elements can be defined as described below.
The elements are defined relative to the `z` axis and in the lab frame, starting at `z = 0`.
They are described using a simplified MAD like syntax.
Note that elements of the same type cannot overlap each other.

* ``lattice.elements`` (``list of strings``) optional (default: no elements)
    A list of names (one name per lattice element), in the order that they
    appear in the lattice.

* ``lattice.reverse`` (``boolean``) optional (default: ``false``)
    Reverse the list of elements in the lattice.

* ``<element_name>.type`` (``string``)
    Indicates the element type for this lattice element. This should be one of:

        * ``drift`` for free drift. This requires this additional parameter:

            * ``<element_name>.ds`` (``float``, in meters) the segment length

        * ``quad`` for a hard edged quadrupole.
          This applies a quadrupole field that is uniform within the `z` extent of the element with a sharp cut off at the ends.
          This uses residence corrections, with the field scaled by the amount of time within the element for particles entering
          or leaving it, to increase the accuracy.
          This requires these additional parameters:

            * ``<element_name>.ds`` (``float``, in meters) the segment length

            * ``<element_name>.dEdx`` (``float``, in volts/meter^2) optional (default: 0.) the electric quadrupole field gradient
              The field applied to the particles will be `Ex = dEdx*x` and `Ey = -dEdx*y`.

            * ``<element_name>.dBdx`` (``float``, in Tesla/meter) optional (default: 0.) the magnetic quadrupole field gradient
              The field applied to the particles will be `Bx = dBdx*y` and `By = dBdx*x`.

        * ``plasmalens`` for a field modeling a plasma lens
          This applies a radially directed plasma lens field that is uniform within the `z` extent of the element with
          a sharp cut off at the ends.
          This uses residence corrections, with the field scaled by the amount of time within the element for particles entering
          or leaving it, to increase the accuracy.
          This requires these additional parameters:

            * ``<element_name>.ds`` (``float``, in meters) the segment length

            * ``<element_name>.dEdx`` (``float``, in volts/meter^2) optional (default: 0.) the electric field gradient
              The field applied to the particles will be `Ex = dEdx*x` and `Ey = dEdx*y`.

            * ``<element_name>.dBdx`` (``float``, in Tesla/meter) optional (default: 0.) the magnetic field gradient
              The field applied to the particles will be `Bx = dBdx*y` and `By = -dBdx*x`.

        * ``line`` a sub-lattice (line) of elements to append to the lattice.

            * ``<element_name>.elements`` (``list of strings``) optional (default: no elements)
              A list of names (one name per lattice element), in the order that they appear in the lattice.

            * ``<element_name>.reverse`` (``boolean``) optional (default: ``false``)
              Reverse the list of elements in the line before appending to the lattice.


.. _running-cpp-parameters-collision:

Collision models
----------------

WarpX provides several particle collision models, using varying degrees of approximation.
Details about the collision models can be found in the :ref:`theory section <multiphysics-collisions>`.

* ``collisions.collision_names`` (`strings`, separated by spaces)
    The name of each collision type.
    This is then used in the rest of the input deck;
    in this documentation we use ``<collision_name>`` as a placeholder.

* ``<collision_name>.type`` (`string`) optional
    The type of collision. The types implemented are:

    - ``pairwisecoulomb`` for pair-wise Coulomb collisions, the default if unspecified.
      This provides a pair-wise relativistic elastic Monte Carlo binary Coulomb collision model,
      following the algorithm given by :cite:t:`param-PerezPOP2012`.
      When the RZ mode is used, `warpx.n_rz_azimuthal_modes` must be set to 1 at the moment,
      since the current implementation of the collision module assumes axisymmetry.
    - ``nuclearfusion`` for fusion reactions.
      This implements the pair-wise fusion model by :cite:t:`param-HigginsonJCP2019`.
      Currently, WarpX supports deuterium-deuterium, deuterium-tritium, deuterium-helium and proton-boron fusion.
      When initializing the reactant and product species, you need to use ``species_type`` (see the documentation
      for this parameter), so that WarpX can identify the type of reaction to use.
      (e.g. ``<species_name>.species_type = 'deuterium'``)
    - ``dsmc`` for pair-wise, non-Coulomb collisions between kinetic species.
      This is a "direct simulation Monte Carlo" treatment of collisions between
      kinetic species. See :ref:`DSMC section <multiphysics-collisions-dsmc>`.
    - ``background_mcc`` for collisions between particles and a neutral background.
      This is a relativistic Monte Carlo treatment for particles colliding
      with a neutral background gas. See :ref:`MCC section <multiphysics-collisions-mcc>`.
    - ``background_stopping`` for slowing of ions due to collisions with electrons or ions.
      This implements the approximate formulae as derived in Introduction to Plasma Physics,
      from Goldston and Rutherford, section 14.2.

* ``<collision_name>.species`` (`strings`)
    If using ``dsmc``, ``pairwisecoulomb`` or ``nuclearfusion``, this should be the name(s) of the species,
    between which the collision will be considered. (Provide only one name for intra-species collisions.)
    If using ``background_mcc`` or ``background_stopping`` type this should be the name of the
    species for which collisions with a background will be included.
    In this case, only one species name should be given.

* ``<collision_name>.product_species`` (`strings`)
    Only for ``nuclearfusion``. The name(s) of the species in which to add
    the new macroparticles created by the reaction.

* ``<collision_name>.ndt`` (`int`) optional
    Execute collision every # time steps. The default value is 1.

* ``<collision_name>.CoulombLog`` (`float`) optional
    Only for ``pairwisecoulomb``. A provided fixed Coulomb logarithm of the
    collision type ``<collision_name>``.
    For example, a typical Coulomb logarithm has a form of
    :math:`\ln(\lambda_D/R)`,
    where :math:`\lambda_D` is the Debye length,
    :math:`R\approx1.4A^{1/3}` is the effective Coulombic radius of the nucleus,
    :math:`A` is the mass number.
    If this is not provided, or if a non-positive value is provided,
    a Coulomb logarithm will be computed automatically according to the algorithm in
    :cite:t:`param-PerezPOP2012`.

* ``<collision_name>.fusion_multiplier`` (`float`) optional.
    Only for ``nuclearfusion``.
    Increasing ``fusion_multiplier`` creates more macroparticles of fusion
    products, but with lower weight (in such a way that the corresponding
    total number of physical particle remains the same). This can improve
    the statistics of the simulation, in the case where fusion reactions are very rare.
    More specifically, in a fusion reaction between two macroparticles with weight ``w_1`` and ``w_2``,
    the weight of the product macroparticles will be ``min(w_1,w_2)/fusion_multiplier``.
    (And the weights of the reactant macroparticles are reduced correspondingly after the reaction.)
    See :cite:t:`param-HigginsonJCP2019` for more details.
    The default value of ``fusion_multiplier`` is 1.

* ``<collision_name>.fusion_probability_threshold`` (`float`) optional.
    Only for ``nuclearfusion``.
    If the fusion multiplier is too high and results in a fusion probability
    that approaches 1 (for a given collision between two macroparticles), then
    there is a risk of underestimating the total fusion yield. In these cases,
    WarpX reduces the fusion multiplier used in that given collision.
    ``m_probability_threshold`` is the fusion probability threshold above
    which WarpX reduces the fusion multiplier.

* ``<collision_name>.fusion_probability_target_value`` (`float`) optional.
    Only for ``nuclearfusion``.
    When the probability of fusion for a given collision exceeds
    ``fusion_probability_threshold``, WarpX reduces the fusion multiplier for
    that collisions such that the fusion probability approches ``fusion_probability_target_value``.

* ``<collision_name>.background_density`` (`float`)
    Only for ``background_mcc`` and ``background_stopping``. The density of the background in :math:`m^{-3}`.
    Can also provide ``<collision_name>.background_density(x,y,z,t)`` using the parser
    initialization style for spatially and temporally varying density. With ``background_mcc``, if a function
    is used for the background density, the input parameter ``<collision_name>.max_background_density``
    must also be provided to calculate the maximum collision probability.

* ``<collision_name>.background_temperature`` (`float`)
    Only for ``background_mcc`` and ``background_stopping``. The temperature of the background in Kelvin.
    Can also provide ``<collision_name>.background_temperature(x,y,z,t)`` using the parser
    initialization style for spatially and temporally varying temperature.

* ``<collision_name>.background_mass`` (`float`) optional
    Only for ``background_mcc`` and ``background_stopping``. The mass of the background gas in kg.
    With ``background_mcc``, if not given the mass of the colliding species will be used unless ionization is
    included in which case the mass of the product species will be used.
    With ``background_stopping``, and ``background_type`` set to ``electrons``, if not given defaults to the electron mass. With
    ``background_type`` set to ``ions``, the mass must be given.

* ``<collision_name>.background_charge_state`` (`float`)
    Only for ``background_stopping``, where it is required when ``background_type`` is set to ``ions``.
    This specifies the charge state of the background ions.

* ``<collision_name>.background_type`` (`string`)
    Only for ``background_stopping``, where it is required, the type of the background.
    The possible values are ``electrons`` and ``ions``. When ``electrons``, equation 14.12 from Goldston and Rutherford is used.
    This formula is based on Coulomb collisions with the approximations that :math:`M_b >> m_e` and :math:`V << v_{thermal\_e}`,
    and the assumption that the electrons have a Maxwellian distribution with temperature :math:`T_e`.

    .. math::
        \frac{dV}{dt} = - \frac{2^{1/2}n_eZ_b^2e^4m_e^{1/2}\log\Lambda}{12\pi^{3/2}\epsilon_0M_bT_e^{3/2}}V

    where :math:`V` is each velocity component, :math:`n_e` is the background density, :math:`Z_b` is the ion charge state,
    :math:`e` is the electron charge, :math:`m_e` is the background mass, :math:`\log\Lambda=\log((12\pi/Z_b)(n_e\lambda_{de}^3))`,
    :math:`\lambda_{de}` is the DeBye length, and :math:`M_b` is the ion mass.
    The equation is integrated over a time step, giving :math:`V(t+dt) = V(t)*\exp(-\alpha*{dt})`
    where :math:`\alpha` is the factor multiplying :math:`V`.

    When ``ions``, equation 14.20 is used.
    This formula is based on Coulomb collisions with the approximations that :math:`M_b >> M` and :math:`V >> v_{thermal\_i}`.
    The background ion temperature only appears in the :math:`\log\Lambda` term.

    .. math::
        \frac{dW_b}{dt} = - \frac{2^{1/2}n_iZ^2Z_b^2e^4M_b^{1/2}\log\Lambda}{8\pi\epsilon_0MW_b^{1/2}}

    where :math:`W_b` is the ion energy, :math:`n_i` is the background density,
    :math:`Z` is the charge state of the background ions, :math:`Z_b` is the ion charge state,
    :math:`e` is the electron charge, :math:`M_b` is the ion mass, :math:`\log\Lambda=\log((12\pi/Z_b)(n_i\lambda_{di}^3))`,
    :math:`\lambda_{di}` is the DeBye length, and :math:`M` is the background ion mass.
    The equation is integrated over a time step, giving :math:`W_b(t+dt) = ((W_b(t)^{3/2}) - 3/2\beta{dt})^{2/3}`
    where :math:`\beta` is the term on the r.h.s except :math:`W_b`.

* ``<collision_name>.scattering_processes`` (`strings` separated by spaces)
    Only for ``dsmc`` and ``background_mcc``. The scattering processes that should be
    included. Available options are ``elastic``, ``back`` & ``charge_exchange``
    for ions and ``elastic``, ``excitationX`` & ``ionization`` for electrons.
    Multiple excitation events can be included for electrons corresponding to
    excitation to different levels, the ``X`` above can be changed to a unique
    identifier for each excitation process. For each scattering process specified
    a path to a cross-section data file must also be given. We use
    ``<scattering_process>`` as a placeholder going forward.

* ``<collision_name>.<scattering_process>_cross_section`` (`string`)
    Only for ``dsmc`` and ``background_mcc``. Path to the file containing cross-section data
    for the given scattering processes. The cross-section file must have exactly
    2 columns of data, the first containing equally spaced energies in eV and the
    second the corresponding cross-section in :math:`m^2`. The energy column should
    represent the kinetic energy of the colliding particles in the center-of-mass frame.

* ``<collision_name>.<scattering_process>_energy`` (`float`)
    Only for ``background_mcc``. If the scattering process is either
    ``excitationX`` or ``ionization`` the energy cost of that process must be given in eV.

* ``<collision_name>.ionization_species`` (`float`)
    Only for ``background_mcc``. If the scattering process is ``ionization`` the
    produced species must also be given. For example if argon properties is used
    for the background gas, a species of argon ions should be specified here.

.. _running-cpp-parameters-numerics:

Numerics and algorithms
-----------------------

This section describes the input parameters used to select numerical methods and algorithms for your simulation setup.

Time step
^^^^^^^^^

* ``warpx.cfl`` (`float`) optional (default `0.999`)
    The ratio between the actual timestep that is used in the simulation
    and the Courant-Friedrichs-Lewy (CFL) limit. (e.g. for `warpx.cfl=1`,
    the timestep will be exactly equal to the CFL limit.)
    For some speed v and grid spacing dx, this limits the timestep to `warpx.cfl * dx / v`.
    When used with the electromagnetic solver, `v` is the speed of light.
    For the electrostatic solver, `v` is the maximum speed among all particles in the domain.

* ``warpx.const_dt`` (`float`)
    Allows direct specification of the time step size, in units of seconds.
    When the electrostatic solver is being used, this must be supplied if not using adaptive timestepping.
    This can be used with the electromagnetic solver, overriding ``warpx.cfl``, but
    it is up to the user to ensure that the CFL condition is met.

* ``warpx.dt_update_interval`` (`string`) optional (default `-1`)
    How many iterations pass between timestep adaptations when using the electrostatic solver.
    Must be greater than `0` to use adaptive timestepping, or else ``warpx.const_dt`` must be specified.

* ``warpx.max_dt`` (`float`) optional
    The maximum timestep permitted for the electrostatic solver, when using adaptive timestepping.
    If supplied, also sets the initial timestep for these simulations, before the first timestep update.

Filtering
^^^^^^^^^

* ``warpx.use_filter`` (`0` or `1`)
    Whether to use filtering in the simulation.
    With the explicit evolve scheme, the filtering is turned on by default, except for RZ FDTD.
    With the implicit evolve schemes, the filtering is turned off by default.
    The filtering smoothes the charge and currents on the mesh, after depositing them from the macro-particles.
    With implicit schemes, the electric field is also filtered (to maintain consistency for energy conservation).
    This uses a bilinear filter (see the :ref:`filtering section <theory-pic-filter>`).
    With the RZ PSATD solver, the filtering is done in :math:`k`-space.

    .. warning::

       Known bug: filter currently not working with FDTD solver in RZ geometry (see https://github.com/ECP-WarpX/WarpX/issues/1943).

* ``warpx.filter_npass_each_dir`` (`3 int`) optional (default `1 1 1`)
    Number of passes along each direction for the bilinear filter.
    In 2D simulations, only the first two values are read.

* ``warpx.use_filter_compensation`` (`0` or `1`; default: `0`)
    Whether to add compensation when applying filtering.
    This is only supported with the RZ spectral solver.

Particle push, charge and current deposition, field gathering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``algo.current_deposition`` (`string`, optional)
    This parameter selects the algorithm for the deposition of the current density.
    Available options are: ``direct``, ``esirkepov``, and ``vay``. The default choice
    is ``esirkepov`` for FDTD maxwell solvers but ``direct`` for standard or
    Galilean PSATD solver (i.e. with ``algo.maxwell_solver = psatd``) and
    for the hybrid-PIC solver (i.e. with ``algo.maxwell_solver = hybrid``) and for
    diagnostics output with the electrostatic solvers (i.e., with
    ``warpx.do_electrostatic = ...``).
    Note that ``vay`` is only available for ``algo.maxwell_solver = psatd``.

    1. ``direct``

       The current density is deposited as described in the section :ref:`current_deposition`.
       This deposition scheme does not conserve charge.

    2. ``esirkepov``

       The current density is deposited as described in
       :cite:t:`param-Esirkepovcpc01`.
       This deposition scheme guarantees charge conservation for shape factors of arbitrary order.

    3. ``vay``

       The current density is deposited as described in :cite:t:`param-VayJCP2013` (see section :ref:`current_deposition` for more details).
       This option guarantees charge conservation only when used in combination
       with ``psatd.periodic_single_box_fft=1``, that is, only for periodic single-box
       simulations with global FFTs without guard cells. The implementation for domain
       decomposition with local FFTs over guard cells is planned but not yet completed.

* ``algo.charge_deposition`` (`string`, optional)
    The algorithm for the charge density deposition. Available options are:

     - ``standard``: standard charge deposition algorithm, described in
       the :ref:`particle-in-cell theory section <theory-pic>`.

* ``algo.field_gathering`` (`string`, optional)
    The algorithm for field gathering. Available options are:

     * ``energy-conserving``: gathers directly from the grid points (either staggered
       or nodal grid points depending on ``warpx.grid_type``).
     * ``momentum-conserving``: first average the fields from the grid points to
       the nodes, and then gather from the nodes.


    Default: ``algo.field_gathering = energy-conserving`` with collocated or staggered grids (note that ``energy-conserving`` and ``momentum-conserving`` are equivalent with collocated grids), ``algo.field_gathering = momentum-conserving`` with hybrid grids.

* ``algo.particle_pusher`` (`string`, optional)
    The algorithm for the particle pusher. Available options are:

     - ``boris``: Boris pusher.
     - ``vay``: Vay pusher (see :cite:t:`param-Vaypop2008`)
     - ``higuera``: Higuera-Cary pusher (see :cite:t:`param-HigueraPOP2017`)

     If ``algo.particle_pusher`` is not specified, ``boris`` is the default.

* ``algo.particle_shape`` (`integer`; `1`, `2`, `3`, or `4`)
    The order of the shape factors (splines) for the macro-particles along all spatial directions: `1` for linear, `2` for quadratic, `3` for cubic, `4` for quartic.
    Low-order shape factors result in faster simulations, but may lead to more noisy results.
    High-order shape factors are computationally more expensive, but may increase the overall accuracy of the results. For production runs it is generally safer to use high-order shape factors, such as cubic order.

    Note that this input parameter is not optional and must always be set in all input files provided that there is at least one particle species (set in input as ``particles.species_names``) or one laser species (set in input as ``lasers.names``) in the simulation. No default value is provided automatically.

Maxwell solver
^^^^^^^^^^^^^^

Two families of Maxwell solvers are implemented in WarpX, based on the Finite-Difference Time-Domain method (FDTD) or the Pseudo-Spectral Analytical Time-Domain method (PSATD), respectively.

* ``algo.maxwell_solver`` (`string`, optional)
    The algorithm for the Maxwell field solver.
    Available options are:

     - ``yee``: Yee FDTD solver.
     - ``ckc``: (not available in ``RZ`` geometry) Cole-Karkkainen solver with Cowan
       coefficients (see :cite:t:`param-CowanPRSTAB13`).
     - ``psatd``: Pseudo-spectral solver (see :ref:`theory <theory-pic-mwsolve-psatd>`).
     - ``ect``: Enlarged cell technique (conformal finite difference solver. See :cite:t:`param-XiaoIEEE2005`).
     - ``hybrid``: The E-field will be solved using Ohm's law and a kinetic-fluid hybrid model (see :ref:`theory <theory-kinetic-fluid-hybrid-model>`).
     - ``none``: No field solve will be performed.

     If ``algo.maxwell_solver`` is not specified, ``yee`` is the default.

* ``algo.em_solver_medium`` (`string`, optional)
    The medium for evaluating the Maxwell solver. Available options are :

    - ``vacuum``: vacuum properties are used in the Maxwell solver.
    - ``macroscopic``: macroscopic Maxwell equation is evaluated. If this option is selected, then the corresponding properties of the medium must be provided using ``macroscopic.sigma``, ``macroscopic.epsilon``, and ``macroscopic.mu`` for each case where the initialization style is ``constant``.  Otherwise if the initialization style uses the parser, ``macroscopic.sigma_function(x,y,z)``, ``macroscopic.epsilon_function(x,y,z)`` and/or ``macroscopic.mu_function(x,y,z)`` must be provided using the parser initialization style for spatially varying macroscopic properties.

    If ``algo.em_solver_medium`` is not specified, ``vacuum`` is the default.

Maxwell solver: PSATD method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``psatd.nox``, ``psatd.noy``, ``pstad.noz`` (`integer`) optional (default `16` for all)
    The order of accuracy of the spatial derivatives, when using the code compiled with a PSATD solver.
    If ``psatd.periodic_single_box_fft`` is used, these can be set to ``inf`` for infinite-order PSATD.

* ``psatd.nx_guard``, ``psatd.ny_guard``, ``psatd.nz_guard`` (`integer`) optional
    The number of guard cells to use with PSATD solver.
    If not set by users, these values are calculated automatically and determined *empirically* and
    equal the order of the solver for collocated grids and half the order of the solver for staggered grids.

* ``psatd.periodic_single_box_fft`` (`0` or `1`; default: 0)
    If true, this will *not* incorporate the guard cells into the box over which FFTs are performed.
    This is only valid when WarpX is run with periodic boundaries and a single box.
    In this case, using `psatd.periodic_single_box_fft` is equivalent to using a global FFT over the whole domain.
    Therefore, all the approximations that are usually made when using local FFTs with guard cells
    (for problems with multiple boxes) become exact in the case of the periodic, single-box FFT without guard cells.

* ``psatd.current_correction`` (`0` or `1`; default: `1`, with the exceptions mentioned below)
    If true, a current correction scheme in Fourier space is applied in order to guarantee charge conservation.
    The default value is ``psatd.current_correction=1``, unless a charge-conserving current deposition scheme is used (by setting ``algo.current_deposition=esirkepov`` or ``algo.current_deposition=vay``) or unless the ``div(E)`` cleaning scheme is used (by setting ``warpx.do_dive_cleaning=1``).

    If ``psatd.v_galilean`` is zero, the spectral solver used is the standard PSATD scheme described in :cite:t:`param-VayJCP2013` and the current correction reads

    .. math::
       \widehat{\boldsymbol{J}}^{\,n+1/2}_{\mathrm{correct}} = \widehat{\boldsymbol{J}}^{\,n+1/2}
       - \bigg(\boldsymbol{k}\cdot\widehat{\boldsymbol{J}}^{\,n+1/2}
       - i \frac{\widehat{\rho}^{n+1} - \widehat{\rho}^{n}}{\Delta{t}}\bigg) \frac{\boldsymbol{k}}{k^2}

    If ``psatd.v_galilean`` is non-zero, the spectral solver used is the Galilean PSATD scheme described in :cite:t:`param-LehePRE2016` and the current correction reads

    .. math::
       \widehat{\boldsymbol{J}}^{\,n+1/2}_{\mathrm{correct}} = \widehat{\boldsymbol{J}}^{\,n+1/2}
       - \bigg(\boldsymbol{k}\cdot\widehat{\boldsymbol{J}}^{\,n+1/2} - (\boldsymbol{k}\cdot\boldsymbol{v}_G)
       \,\frac{\widehat\rho^{n+1} - \widehat\rho^{n}\theta^2}{1 - \theta^2}\bigg) \frac{\boldsymbol{k}}{k^2}

    where :math:`\theta=\exp(i\,\boldsymbol{k}\cdot\boldsymbol{v}_G\,\Delta{t}/2)`.

    This option is currently implemented only for the standard PSATD, Galilean PSATD, and averaged Galilean PSATD schemes, while it is not yet available for the multi-J algorithm.

* ``psatd.update_with_rho`` (`0` or `1`)
    If true, the update equation for the electric field is expressed in terms of both the current density and the charge density, namely :math:`\widehat{\boldsymbol{J}}^{\,n+1/2}`, :math:`\widehat\rho^{n}`, and :math:`\widehat\rho^{n+1}`.
    If false, instead, the update equation for the electric field is expressed in terms of the current density :math:`\widehat{\boldsymbol{J}}^{\,n+1/2}` only.
    If charge is expected to be conserved (by setting, for example, ``psatd.current_correction=1``), then the two formulations are expected to be equivalent.

    If ``psatd.v_galilean`` is zero, the spectral solver used is the standard PSATD scheme described in :cite:t:`param-VayJCP2013`:

    1. if ``psatd.update_with_rho=0``, the update equation for the electric field reads

    .. math::
       \begin{split}
       \widehat{\boldsymbol{E}}^{\,n+1}= & \:
       C \widehat{\boldsymbol{E}}^{\,n} + i \, \frac{S c}{k} \boldsymbol{k}\times\widehat{\boldsymbol{B}}^{\,n}
       - \frac{S}{\epsilon_0 c \, k} \widehat{\boldsymbol{J}}^{\,n+1/2} \\[0.2cm]
       & +\frac{1-C}{k^2} (\boldsymbol{k}\cdot\widehat{\boldsymbol{E}}^{\,n}) \boldsymbol{k}
       + \frac{1}{\epsilon_0 k^2} \left(\frac{S}{c \, k}-\Delta{t}\right)
       (\boldsymbol{k}\cdot\widehat{\boldsymbol{J}}^{\,n+1/2}) \boldsymbol{k}
       \end{split}

    2. if ``psatd.update_with_rho=1``, the update equation for the electric field reads

    .. math::
       \begin{split}
       \widehat{\boldsymbol{E}}^{\,n+1}= & \:
       C\widehat{\boldsymbol{E}}^{\,n} + i \, \frac{S c}{k} \boldsymbol{k}\times\widehat{\boldsymbol{B}}^{\,n}
       - \frac{S}{\epsilon_0 c \, k} \widehat{\boldsymbol{J}}^{\,n+1/2} \\[0.2cm]
       & + \frac{i}{\epsilon_0 k^2} \left(C-\frac{S}{c\,k}\frac{1}{\Delta{t}}\right)
       \widehat{\rho}^{n} \boldsymbol{k} - \frac{i}{\epsilon_0 k^2} \left(1-\frac{S}{c \, k}
       \frac{1}{\Delta{t}}\right)\widehat{\rho}^{n+1} \boldsymbol{k}
       \end{split}

    The coefficients :math:`C` and :math:`S` are defined in :cite:t:`param-VayJCP2013`.

    If ``psatd.v_galilean`` is non-zero, the spectral solver used is the Galilean PSATD scheme described in :cite:t:`param-LehePRE2016`:

    1. if ``psatd.update_with_rho=0``, the update equation for the electric field reads

    .. math::
       \begin{split}
       \widehat{\boldsymbol{E}}^{\,n+1} = & \:
       \theta^{2} C \widehat{\boldsymbol{E}}^{\,n} + i \, \theta^{2} \frac{S c}{k}
       \boldsymbol{k}\times\widehat{\boldsymbol{B}}^{\,n}
       + \frac{i \, \nu \, \theta \, \chi_1 - \theta^{2} S}{\epsilon_0 c \, k}
       \widehat{\boldsymbol{J}}^{\,n+1/2} \\[0.2cm]
       & + \theta^{2} \frac{\chi_2-\chi_3}{k^{2}}
       (\boldsymbol{k}\cdot\widehat{\boldsymbol{E}}^{\,n}) \boldsymbol{k}
       + i \, \frac{\chi_2\left(\theta^{2}-1\right)}{\epsilon_0 c \, k^{3} \nu}
       (\boldsymbol{k}\cdot\widehat{\boldsymbol{J}}^{\,n+1/2}) \boldsymbol{k}
       \end{split}

    2. if ``psatd.update_with_rho=1``, the update equation for the electric field reads

    .. math::
       \begin{split}
       \widehat{\boldsymbol{E}}^{\,n+1} = & \:
       \theta^{2} C \widehat{\boldsymbol{E}}^{\,n} + i \, \theta^{2} \frac{S c}{k}
       \boldsymbol{k}\times\widehat{\boldsymbol{B}}^{\,n}
       + \frac{i \, \nu \, \theta \, \chi_1 - \theta^{2} S}{\epsilon_0 c \, k}
       \widehat{\boldsymbol{J}}^{\,n+1/2} \\[0.2cm]
       & + i \, \frac{\theta^{2} \chi_3}{\epsilon_0 k^{2}} \widehat{\rho}^{\,n} \boldsymbol{k}
       - i \, \frac{\chi_2}{\epsilon_0 k^{2}} \widehat{\rho}^{\,n+1} \boldsymbol{k}
       \end{split}

    The coefficients :math:`C`, :math:`S`, :math:`\theta`, :math:`\nu`, :math:`\chi_1`, :math:`\chi_2`, and :math:`\chi_3` are defined in :cite:t:`param-LehePRE2016`.

    The default value for ``psatd.update_with_rho`` is ``1`` if ``psatd.v_galilean`` is non-zero and ``0`` otherwise.
    The option ``psatd.update_with_rho=0`` is not implemented with the following algorithms:
    comoving PSATD (``psatd.v_comoving``), time averaging (``psatd.do_time_averaging=1``), div(E) cleaning (``warpx.do_dive_cleaning=1``), and multi-J (``warpx.do_multi_J=1``).

    Note that the update with and without rho is also supported in RZ geometry.

* ``psatd.J_in_time`` (``constant`` or ``linear``; default ``constant``)
    This determines whether the current density is assumed to be constant or linear in time, within the time step over which the electromagnetic fields are evolved.

* ``psatd.rho_in_time`` (``linear``; default ``linear``)
    This determines whether the charge density is assumed to be linear in time, within the time step over which the electromagnetic fields are evolved.

* ``psatd.v_galilean`` (`3 floats`, in units of the speed of light; default ``0. 0. 0.``)
    Defines the Galilean velocity.
    A non-zero velocity activates the Galilean algorithm, which suppresses numerical Cherenkov instabilities (NCI) in boosted-frame simulations (see the section :ref:`Numerical Stability and alternate formulation in a Galilean frame <theory-boostedframe-galilean>` for more information).
    This requires the code to be compiled with the spectral solver.
    It also requires the use of the direct current deposition algorithm (by setting ``algo.current_deposition = direct``).

* ``psatd.use_default_v_galilean`` (`0` or `1`; default: `0`)
    This can be used in boosted-frame simulations only and sets the Galilean velocity along the :math:`z` direction automatically as :math:`v_{G} = -\sqrt{1-1/\gamma^2}`, where :math:`\gamma` is the Lorentz factor of the boosted frame (set by ``warpx.gamma_boost``).
    See the section :ref:`Numerical Stability and alternate formulation in a Galilean frame <theory-boostedframe-galilean>` for more information on the Galilean algorithm for boosted-frame simulations.

* ``psatd.v_comoving`` (3 floating-point values, in units of the speed of light; default ``0. 0. 0.``)
    Defines the comoving velocity in the comoving PSATD scheme.
    A non-zero comoving velocity selects the comoving PSATD algorithm, which suppresses the numerical Cherenkov instability (NCI) in boosted-frame simulations, under certain assumptions. This option requires that WarpX is compiled with ``USE_FFT = TRUE``. It also requires the use of direct current deposition (``algo.current_deposition = direct``) and has not been neither implemented nor tested with other current deposition schemes.

* ``psatd.do_time_averaging`` (`0` or `1`; default: 0)
    Whether to use an averaged Galilean PSATD algorithm or standard Galilean PSATD.

* ``warpx.do_multi_J`` (`0` or `1`; default: `0`)
    Whether to use the multi-J algorithm, where current deposition and field update are performed multiple times within each time step. The number of sub-steps is determined by the input parameter ``warpx.do_multi_J_n_depositions``. Unlike sub-cycling, field gathering is performed only once per time step, as in regular PIC cycles. When ``warpx.do_multi_J = 1``, we perform linear interpolation of two distinct currents deposited at the beginning and the end of the time step, instead of using one single current deposited at half time. For simulations with strong numerical Cherenkov instability (NCI), it is recommended to use the multi-J algorithm in combination with ``psatd.do_time_averaging = 1``.

* ``warpx.do_multi_J_n_depositions`` (integer)
    Number of sub-steps to use with the multi-J algorithm, when ``warpx.do_multi_J = 1``.
    Note that this input parameter is not optional and must always be set in all input files where ``warpx.do_multi_J = 1``. No default value is provided automatically.

Maxwell solver: macroscopic media
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``algo.macroscopic_sigma_method`` (`string`, optional)
    The algorithm for updating electric field when ``algo.em_solver_medium`` is macroscopic. Available options are:

    - ``backwardeuler`` is a fully-implicit, first-order in time scheme for E-update (default).
    - ``laxwendroff`` is the semi-implicit, second order in time scheme for E-update.

    Comparing the two methods, Lax-Wendroff is more prone to developing oscillations and requires a smaller timestep for stability. On the other hand, Backward Euler is more robust but it is first-order accurate in time compared to the second-order Lax-Wendroff method.

* ``macroscopic.sigma_function(x,y,z)``, ``macroscopic.epsilon_function(x,y,z)``, ``macroscopic.mu_function(x,y,z)`` (`string`)
     To initialize spatially varying conductivity, permittivity, and permeability, respectively,
     using a mathematical function in the input. Constants required in the
     mathematical expression can be set using ``my_constants``. These parameters are parsed
     if ``algo.em_solver_medium=macroscopic``.

* ``macroscopic.sigma``, ``macroscopic.epsilon``, ``macroscopic.mu`` (`double`)
    To initialize a constant conductivity, permittivity, and permeability of the
    computational medium, respectively. The default values are the corresponding values
    in vacuum.

.. _running-cpp-parameters-hybrid-model:

Maxwell solver: kinetic-fluid hybrid
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``hybrid_pic_model.elec_temp`` (`float`)
    If ``algo.maxwell_solver`` is set to ``hybrid``, this sets the electron temperature, in eV, used to calculate
    the electron pressure (see :ref:`here <theory-hybrid-model-elec-temp>`).

* ``hybrid_pic_model.n0_ref`` (`float`)
    If ``algo.maxwell_solver`` is set to ``hybrid``, this sets the reference density, in :math:`m^{-3}`, used to calculate
    the electron pressure (see :ref:`here <theory-hybrid-model-elec-temp>`).

* ``hybrid_pic_model.gamma`` (`float`) optional (default ``5/3``)
    If ``algo.maxwell_solver`` is set to ``hybrid``, this sets the exponent used to calculate
    the electron pressure (see :ref:`here <theory-hybrid-model-elec-temp>`).

* ``hybrid_pic_model.plasma_resistivity(rho,J)`` (`float` or `str`) optional (default ``0``)
    If ``algo.maxwell_solver`` is set to ``hybrid``, this sets the plasma resistivity in :math:`\Omega m`.

* ``hybrid_pic_model.plasma_hyper_resistivity`` (`float` or `str`) optional (default ``0``)
    If ``algo.maxwell_solver`` is set to ``hybrid``, this sets the plasma hyper-resistivity in :math:`\Omega m^3`.

* ``hybrid_pic_model.J[x/y/z]_external_grid_function(x, y, z, t)`` (`float` or `str`) optional (default ``0``)
    If ``algo.maxwell_solver`` is set to ``hybrid``, this sets the external current (on the grid) in :math:`A/m^2`.

* ``hybrid_pic_model.n_floor`` (`float`) optional (default ``1``)
    If ``algo.maxwell_solver`` is set to ``hybrid``, this sets the plasma density floor, in :math:`m^{-3}`, which is useful since the generalized Ohm's law used to calculate the E-field includes a :math:`1/n` term.

* ``hybrid_pic_model.substeps`` (`int`) optional (default ``10``)
    If ``algo.maxwell_solver`` is set to ``hybrid``, this sets the number of sub-steps to take during the B-field update.

.. note::

    Based on results from :cite:t:`param-Stanier2020` it is recommended to use
    linear particles when using the hybrid-PIC model.

Grid types (collocated, staggered, hybrid)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``warpx.grid_type`` (`string`, ``collocated``, ``staggered`` or ``hybrid``)
    Whether to use a collocated grid (all fields defined at the cell nodes),
    a staggered grid (fields defined on a Yee grid), or a hybrid grid (fields
    and currents are interpolated back and forth between a staggered grid and a
    nodal grid, must be used with momentum-conserving field gathering algorithm,
    ``algo.field_gathering = momentum-conserving``).
    The option ``hybrid`` is currently not supported in RZ geometry.

    Default: ``warpx.grid_type = staggered``.

* ``interpolation.galerkin_scheme`` (`0` or `1`)
    Whether to use a Galerkin scheme when gathering fields to particles.
    When set to ``1``, the interpolation orders used for field-gathering are reduced for certain field components along certain directions.
    For example, :math:`E_z` is gathered using ``algo.particle_shape`` along :math:`(x,y)` and ``algo.particle_shape - 1`` along :math:`z`.
    See equations (21)-(23) of :cite:t:`param-Godfrey2013` and associated references for details.

    Default: ``interpolation.galerkin_scheme = 0`` with collocated grids, or momentum-conserving field gathering, or when ``algo.current_deposition = direct`` ; ``interpolation.galerkin_scheme = 1`` otherwise.

    .. warning::

        The default behavior should not normally be changed.
        At present, this parameter is intended mainly for testing and development purposes.

* ``warpx.field_centering_nox``, ``warpx.field_centering_noy``, ``warpx.field_centering_noz`` (`integer`, optional)
    The order of interpolation used with staggered or hybrid grids (``warpx.grid_type = staggered`` or ``warpx.grid_type = hybrid``) and momentum-conserving field gathering (``algo.field_gathering = momentum-conserving``) to interpolate the electric and magnetic fields from the cell centers to the cell nodes, before gathering the fields from the cell nodes to the particle positions.

    Default: ``warpx.field_centering_no<x,y,z> = 2`` with staggered grids, ``warpx.field_centering_no<x,y,z> = 8`` with hybrid grids (typically necessary to ensure stability in boosted-frame simulations of relativistic plasmas and beams).

* ``warpx.current_centering_nox``, ``warpx.current_centering_noy``, ``warpx.current_centering_noz`` (`integer`, optional)
    The order of interpolation used with hybrid grids (``warpx.grid_type = hybrid``) to interpolate the currents from the cell nodes to the cell centers when ``warpx.do_current_centering = 1``, before pushing the Maxwell fields on staggered grids.

    Default: ``warpx.current_centering_no<x,y,z> = 8`` with hybrid grids (typically necessary to ensure stability in boosted-frame simulations of relativistic plasmas and beams).

* ``warpx.do_current_centering`` (`bool`, `0` or `1`)
    If true, the current is deposited on a nodal grid and then centered to a staggered grid (Yee grid), using finite-order interpolation.

    Default: ``warpx.do_current_centering = 0`` with collocated or staggered grids, ``warpx.do_current_centering = 1`` with hybrid grids.

Additional parameters
^^^^^^^^^^^^^^^^^^^^^

* ``warpx.do_dive_cleaning`` (`0` or `1` ; default: 0)
    Whether to use modified Maxwell equations that progressively eliminate
    the error in :math:`div(E)-\rho`. This can be useful when using a current
    deposition algorithm which is not strictly charge-conserving, or when
    using mesh refinement. These modified Maxwell equation will cause the error
    to propagate (at the speed of light) to the boundaries of the simulation
    domain, where it can be absorbed.

* ``warpx.do_divb_cleaning_external`` (`0` or `1` ; default: 0)
    Whether to use projection method to scrub B field divergence in externally
    loaded fields. This is automatically turned on if external B fields are loaded.

* ``warpx.do_subcycling`` (`0` or `1`; default: 0)
    Whether or not to use sub-cycling. Different refinement levels have a
    different cell size, which results in different Courant–Friedrichs–Lewy
    (CFL) limits for the time step. By default, when using mesh refinement,
    the same time step is used for all levels. This time step is
    taken as the CFL limit of the finest level. Hence, for coarser
    levels, the timestep is only a fraction of the CFL limit for this
    level, which may lead to numerical artifacts. With sub-cycling, each level
    evolves with its own time step, set to its own CFL limit. In practice, it
    means that when level 0 performs one iteration, level 1 performs two
    iterations. Currently, this option is only supported when
    ``amr.max_level = 1``. More information can be found at
    https://ieeexplore.ieee.org/document/8659392.

* ``warpx.override_sync_intervals`` (`string`) optional (default `1`)
    Using the `Intervals parser`_ syntax, this string defines the timesteps at which
    synchronization of sources (`rho` and `J`) and fields (`E` and `B`) on grid nodes at box
    boundaries is performed. Since the grid nodes at the interface between two neighbor boxes are
    duplicated in both boxes, an instability can occur if they have too different values.
    This option makes sure that they are synchronized periodically.
    Note that if Perfectly Matched Layers (PML) are used, synchronization of the `E` and `B` fields
    is performed at every timestep regardless of this parameter.

* ``warpx.use_hybrid_QED`` (`bool`; default: 0)
    Will use the Hybrid QED Maxwell solver when pushing fields: a QED correction is added to the
    field solver to solve non-linear Maxwell's equations, according to :cite:t:`param-GrismayerNJP2021`.
    Note that this option can only be used with the PSATD build. Furthermore, one must set
    ``warpx.grid_type = collocated`` (which otherwise would be ``staggered`` by default).

* ``warpx.quantum_xi`` (`float`; default: 1.3050122.e-52)
     Overwrites the actual quantum parameter used in Maxwell's QED equations. Assigning a
     value here will make the simulation unphysical, but will allow QED effects to become more apparent.
     Note that this option will only have an effect if the ``warpx.use_Hybrid_QED`` flag is also triggered.

* ``warpx.do_device_synchronize`` (`bool`) optional (default `1`)
    When running in an accelerated platform, whether to call a ``amrex::Gpu::synchronize()`` around profiling regions.
    This allows the profiler to give meaningful timers, but (hardly) slows down the simulation.

* ``warpx.sort_intervals`` (`string`) optional (defaults: ``-1`` on CPU; ``4`` on GPU)
     Using the `Intervals parser`_ syntax, this string defines the timesteps at which particles are
     sorted.
     If ``<=0``, do not sort particles.
     It is turned on on GPUs for performance reasons (to improve memory locality).

* ``warpx.sort_particles_for_deposition`` (`bool`) optional (default: ``true`` for the CUDA backend, otherwise ``false``)
     This option controls the type of sorting used if particle sorting is turned on, i.e. if ``sort_intervals`` is not ``<=0``.
     If ``true``, particles will be sorted by cell to optimize deposition with many particles per cell, in the order x -> y -> z -> ppc.
     If ``false``, particles will be sorted by bin, using the ``sort_bin_size`` parameter below, in the order ppc -> x -> y -> z.
     ``true`` is recommend for best performance on NVIDIA GPUs, especially if there are many particles per cell.

* ``warpx.sort_idx_type`` (list of `int`) optional (default: ``0 0 0``)
    This controls the type of grid used to sort the particles when ``sort_particles_for_deposition`` is ``true``. Possible values are:
    ``idx_type = {0, 0, 0}``: Sort particles to a cell centered grid
    ``idx_type = {1, 1, 1}``: Sort particles to a node centered grid
    ``idx_type = {2, 2, 2}``: Compromise between a cell and node centered grid.
    In 2D (XZ and RZ), only the first two elements are read.
    In 1D, only the first element is read.

* ``warpx.sort_bin_size`` (list of `int`) optional (default ``1 1 1``)
     If ``sort_intervals`` is activated and ``sort_particles_for_deposition`` is ``false``, particles are sorted in bins of ``sort_bin_size`` cells.
     In 2D, only the first two elements are read.

* ``warpx.do_shared_mem_charge_deposition`` (`bool`) optional (default `false`)
     If activated, charge deposition will allocate and use small
     temporary buffers on which to accumulate deposited charge values
     from particles. On GPUs these buffers will reside in ``__shared__``
     memory, which is faster than the usual ``__global__``
     memory. Performance impact will depend on the relative overhead
     of assigning the particles to bins small enough to fit in the
     space available for the temporary buffers.

* ``warpx.do_shared_mem_current_deposition`` (`bool`) optional (default `false`)
     If activated, current deposition will allocate and use small
     temporary buffers on which to accumulate deposited current values
     from particles. On GPUs these buffers will reside in ``__shared__``
     memory, which is faster than the usual ``__global__``
     memory. Performance impact will depend on the relative overhead
     of assigning the particles to bins small enough to fit in the
     space available for the temporary buffers. Performance is mostly improved
     when there is lots of contention between particles writing to the same cell
     (e.g. for high particles per cell). This feature is only available for CUDA
     and HIP, and is only recommended for 3D or 2D.

* ``warpx.shared_tilesize`` (list of `int`) optional (default `6 6 8` in 3D; `14 14` in 2D; `1s` otherwise)
     Used to tune performance when ``do_shared_mem_current_deposition`` or
     ``do_shared_mem_charge_deposition`` is enabled. ``shared_tilesize`` is the
     size of the temporary buffer allocated in shared memory for a threadblock.
     A larger tilesize requires more shared memory, but gives more work to each
     threadblock, which can lead to higher occupancy, and allows for more
     buffered writes to ``__shared__`` instead of ``__global__``. The defaults
     in 2D and 3D
     are chosen from experimentation, but can be improved upon for specific
     problems. The other defaults are not optimized and should always be fine
     tuned for the problem.

* ``warpx.shared_mem_current_tpb`` (`int`) optional (default `128`)
     Used to tune performance when ``do_shared_mem_current_deposition`` is
     enabled. ``shared_mem_current_tpb`` controls the number of threads per
     block (tpb), i.e. the number of threads operating on a shared buffer.


.. _running-cpp-parameters-diagnostics:

Diagnostics and output
----------------------

.. _running-cpp-parameters-diagnostics-insitu:

In-situ visualization
^^^^^^^^^^^^^^^^^^^^^

WarpX has five types of diagnostics:
``Full`` diagnostics consist in dumps of fields and particles at given iterations,
``TimeAveraged`` diagnostics only allow field data, which they output after averaging over a period of time,
``BackTransformed`` diagnostics are used when running a simulation in a boosted frame, to reconstruct output data to the lab frame,
``BoundaryScraping`` diagnostics are used to collect the particles that are absorbed at the boundary, throughout the simulation, and
``ReducedDiags`` enable users to compute specific reduced quantities, such as particle temperature, energy histograms, or maximum field values, and efficiently save this in-situ analyzed data to files.
Similar to what is done for physical species, WarpX has a class Diagnostics that allows users to initialize different diagnostics, each of them with different fields, resolution and period.
This currently applies to standard diagnostics, but should be extended to back-transformed diagnostics and reduced diagnostics (and others) in a near future.

.. _running-cpp-parameters-diagnostics-full:

Full Diagnostics
^^^^^^^^^^^^^^^^

``FullDiagnostics`` consist in dumps of fields and particles at given iterations.
Similar to what is done for physical species, WarpX has a class Diagnostics that allows users to initialize different diagnostics, each of them with different fields, resolution and period.
The user specifies the number of diagnostics and the name of each of them, and then specifies options for each of them separately.
Note that some parameter (those that do not start with a ``<diag_name>.`` prefix) apply to all diagnostics.
This should be changed in the future.
In-situ capabilities can be used by turning on Sensei or Ascent (provided they are installed) through the output format, see below.

* ``diagnostics.enable`` (`0` or `1`, optional, default `1`)
    Whether to enable or disable diagnostics. This flag overwrites all other diagnostics input parameters.

* ``diagnostics.diags_names`` (list of `string` optional, default `empty`)
    Name of each diagnostics.
    example: ``diagnostics.diags_names = diag1 my_second_diag``.

* ``<diag_name>.intervals`` (`string`)
    Using the `Intervals parser`_ syntax, this string defines the timesteps at which data is dumped.
    Use a negative number or 0 to disable data dumping.
    example: ``diag1.intervals = 10,20:25:1``.
    Note that by default the last timestep is dumped regardless of this parameter. This can be
    changed using the parameter ``<diag_name>.dump_last_timestep`` described below.

* ``<diag_name>.dump_last_timestep`` (`bool` optional, default `1`)
    If this is `1`, the last timestep is dumped regardless of ``<diag_name>.intervals``.

* ``<diag_name>.diag_type`` (`string`)
    Type of diagnostics. ``Full``, ``BackTransformed``, and ``BoundaryScraping``
    example: ``diag1.diag_type = Full`` or ``diag1.diag_type = BackTransformed``

* ``<diag_name>.format`` (`string` optional, default ``plotfile``)
    Flush format. Possible values are:

    * ``plotfile`` for native AMReX format.

    * ``checkpoint`` for a checkpoint file, only works with ``<diag_name>.diag_type = Full``.

    * ``openpmd`` for OpenPMD format `openPMD <https://www.openPMD.org>`_.
      Requires to build WarpX with ``USE_OPENPMD=TRUE`` (see :ref:`instructions <building-openpmd>`).

    * ``ascent`` for in-situ visualization using Ascent.

    * ``sensei`` for in-situ visualization using Sensei.

    example: ``diag1.format = openpmd``.

* ``<diag_name>.sensei_config`` (`string`)
    Only read if ``<diag_name>.format = sensei``.
    Points to the SENSEI XML file which selects and configures the desired back end.

* ``<diag_name>.sensei_pin_mesh`` (`integer`; 0 by default)
    Only read if ``<diag_name>.format = sensei``.
    When 1 lower left corner of the mesh is pinned to 0.,0.,0.

* ``<diag_name>.openpmd_backend`` (``bp``, ``h5`` or ``json``) optional, only used if ``<diag_name>.format = openpmd``
    `I/O backend <https://openpmd-api.readthedocs.io/en/latest/backends/overview.html>`_ for `openPMD <https://www.openPMD.org>`_ data dumps.
    ``bp`` is the `ADIOS I/O library <https://csmd.ornl.gov/adios>`_, ``h5`` is the `HDF5 format <https://www.hdfgroup.org/solutions/hdf5/>`_, and ``json`` is a `simple text format <https://en.wikipedia.org/wiki/JSON>`_.
    ``json`` only works with serial/single-rank jobs.
    When WarpX is compiled with openPMD support, the first available backend in the order given above is taken.

* ``<diag_name>.openpmd_encoding`` (optional, ``v`` (variable based), ``f`` (file based) or ``g`` (group based) ) only read if ``<diag_name>.format = openpmd``.
     openPMD `file output encoding <https://openpmd-api.readthedocs.io/en/0.16.1/usage/concepts.html#iteration-and-series>`__.
     File based: one file per timestep (slower), group/variable based: one file for all steps (faster)).
     ``variable based`` is an `experimental feature with ADIOS2 <https://openpmd-api.readthedocs.io/en/0.16.1/backends/adios2.html#experimental-new-adios2-schema>`__ and not supported for back-transformed diagnostics.
     Default: ``f`` (full diagnostics)

* ``<diag_name>.adios2_operator.type`` (``zfp``, ``blosc``) optional,
    `ADIOS2 I/O operator type <https://openpmd-api.readthedocs.io/en/0.16.1/details/backendconfig.html#adios2>`__ for `openPMD <https://www.openPMD.org>`_ data dumps.

* ``<diag_name>.adios2_operator.parameters.*`` optional,
    `ADIOS2 I/O operator parameters <https://openpmd-api.readthedocs.io/en/0.16.1/details/backendconfig.html#adios2>`__ for `openPMD <https://www.openPMD.org>`_ data dumps.

    A typical example for `ADIOS2 output using lossless compression <https://openpmd-api.readthedocs.io/en/0.16.1/details/backendconfig.html#adios2>`__ with ``blosc`` using the ``zstd`` compressor and 6 CPU treads per MPI Rank (e.g. for a `GPU run with spare CPU resources <https://arxiv.org/abs/1706.00522>`__):

    .. code-block:: text

        <diag_name>.adios2_operator.type = blosc
        <diag_name>.adios2_operator.parameters.compressor = zstd
        <diag_name>.adios2_operator.parameters.clevel = 1
        <diag_name>.adios2_operator.parameters.doshuffle = BLOSC_BITSHUFFLE
        <diag_name>.adios2_operator.parameters.threshold = 2048
        <diag_name>.adios2_operator.parameters.nthreads = 6  # per MPI rank (and thus per GPU)

    or for the lossy ZFP compressor using very strong compression per scalar:

    .. code-block:: text

        <diag_name>.adios2_operator.type = zfp
        <diag_name>.adios2_operator.parameters.precision = 3

* ``<diag_name>.adios2_engine.type`` (``bp4``, ``sst``, ``ssc``, ``dataman``) optional,
    `ADIOS2 Engine type <https://openpmd-api.readthedocs.io/en/0.16.1/details/backendconfig.html#adios2>`__ for `openPMD <https://www.openPMD.org>`_ data dumps.
    See full list of engines at `ADIOS2 readthedocs <https://adios2.readthedocs.io/en/latest/engines/engines.html>`__

* ``<diag_name>.adios2_engine.parameters.*`` optional,
    `ADIOS2 Engine parameters <https://openpmd-api.readthedocs.io/en/0.16.1/details/backendconfig.html#adios2>`__ for `openPMD <https://www.openPMD.org>`_ data dumps.

    An example for parameters for the BP engine are setting the number of writers (``NumAggregators``), transparently redirecting data to burst buffers etc.
    A detailed list of engine-specific parameters are available at the official `ADIOS2 documentation <https://adios2.readthedocs.io/en/latest/engines/engines.html>`__

    .. code-block:: text

        <diag_name>.adios2_engine.parameters.NumAggregators = 2048
        <diag_name>.adios2_engine.parameters.BurstBufferPath="/mnt/bb/username"

* ``<diag_name>.fields_to_plot`` (list of `strings`, optional)
    Fields written to output.
    Possible scalar fields: ``part_per_cell`` ``rho`` ``phi`` ``F`` ``part_per_grid`` ``divE`` ``divB`` ``rho_<species_name>`` and ``T_<species_name>``, where ``<species_name>`` must match the name of one of the available particle species.
    ``T_<species_name>`` is the temperature in eV.
    Note that ``phi`` will only be written out when do_electrostatic==labframe. Also, note that for ``<diag_name>.diag_type = BackTransformed``, the only scalar field currently supported is ``rho``.
    Possible vector field components in Cartesian geometry: ``Ex`` ``Ey`` ``Ez`` ``Bx`` ``By`` ``Bz`` ``jx`` ``jy`` ``jz``.
    Possible vector field components in RZ geometry: ``Er`` ``Et`` ``Ez`` ``Br`` ``Bt`` ``Bz`` ``jr`` ``jt`` ``jz``.
    The default is ``<diag_name>.fields_to_plot = Ex Ey Ez Bx By Bz jx jy jz`` in Cartesian geometry and ``<diag_name>.fields_to_plot = Er Et Ez Br Bt Bz jr jt jz`` in RZ geometry.
    When the special value ``none`` is specified, no fields are written out.
    Note that the fields are averaged on the cell centers before they are written to file.
    Otherwise, we reconstruct a 2D Cartesian slice of the fields for output at :math:`\theta=0`.

* ``<diag_name>.dump_rz_modes`` (`0` or `1`) optional (default `0`)
    Whether to save all modes when in RZ.  When ``openpmd_backend = openpmd``, this parameter is ignored and all modes are saved.

* ``<diag_name>.particle_fields_to_plot`` (list of `strings`, optional)
   Names of per-cell diagnostics of particle properties to calculate and output as additional fields.
   Note that the deposition onto the grid does not respect the particle shape factor, but instead uses nearest-grid point interpolation.
   Default is none.
   Parser functions for these field names are specified by ``<diag_name>.particle_fields.<field_name>(x,y,z,ux,uy,uz)``.
   Also, note that this option is only available for ``<diag_name>.diag_type = Full``

* ``<diag_name>.particle_fields_species`` (list of `strings`, optional)
         Species for which to calculate ``particle_fields_to_plot``.
         Fields will be calculated separately for each specified species.
         The default is a list of all of the available particle species.

* ``<diag_name>.particle_fields.<field_name>.do_average`` (`0` or `1`) optional (default `1`)
   Whether the diagnostic is an average or a sum. With an average, the sum over the specified function is divided
   by the sum of the particle weights in each cell.

* ``<diag_name>.particle_fields.<field_name>(x,y,z,ux,uy,uz)`` (parser `string`)
   Parser function to be calculated for each particle per cell. The averaged field written is

   .. math::

      \texttt{<field_name>_<species>} = \frac{\sum_{i=1}^N w_i \, f(x_i,y_i,z_i,u_{x,i},u_{y,i},u_{z,i})}{\sum_{i=1}^N w_i}

   where :math:`w_i` is the particle weight, :math:`f()` is the parser function, and :math:`(x_i,y_i,z_i)` are particle positions in units of a meter. The sums are over all particles of type ``<species>`` in a cell (ignoring the particle shape factor) that satisfy ``<diag_name>.particle_fields.<field_name>.filter(x,y,z,ux,uy,uz)``.
   When ``<diag_name>.particle_fields.<field_name>.do_average`` is `0`, the division by the sum over particle weights is not done.
   In 1D or 2D, the particle coordinates will follow the WarpX convention. :math:`(u_{x,i},u_{y,i},u_{z,i})` are components of the particle four-momentum. :math:`u = \gamma v/c`, :math:`\gamma` is the Lorentz factor, :math:`v` is the particle velocity and :math:`c` is the speed of light.
   For photons, we use the standardized momentum :math:`u = p/(m_{e}c)`, where :math:`p` is the momentum of the photon and :math:`m_{e}` the mass of an electron.

* ``<diag_name>.particle_fields.<field_name>.filter(x,y,z,ux,uy,uz)`` (parser `string`, optional)
    Parser function returning a boolean for whether to include a particle in the diagnostic.
    If not specified, all particles will be included (see above).
    The function arguments are the same as above.

* ``<diag_name>.plot_raw_fields`` (`0` or `1`) optional (default `0`)
    By default, the fields written in the plot files are averaged on the cell centers.
    When ``<diag_name>.plot_raw_fields = 1``, then the raw (i.e. non-averaged)
    fields are also saved in the output files.
    Only works with ``<diag_name>.format = plotfile``.
    See `this section <https://yt-project.org/doc/examining/loading_data.html#viewing-raw-fields-in-warpx>`__
    in the yt documentation for more details on how to view raw fields.

* ``<diag_name>.plot_raw_fields_guards`` (`0` or `1`) optional (default `0`)
    Only used when ``<diag_name>.plot_raw_fields = 1``.
    Whether to include the guard cells in the output of the raw fields.
    Only works with ``<diag_name>.format = plotfile``.

* ``<diag_name>.coarsening_ratio`` (list of `int`) optional (default `1 1 1`)
    Reduce size of the selected diagnostic fields output by this ratio in each dimension.
    (For a ratio of N, this is done by averaging the fields over N or (N+1) points depending on the staggering).
    If ``blocking_factor`` and ``max_grid_size`` are used for the domain decomposition, as detailed in
    the :ref:`domain decomposition <usage_domain_decomposition>` section, ``coarsening_ratio`` should be an integer
    divisor of ``blocking_factor``. If ``warpx.numprocs`` is used instead, the total number of cells in a given
    dimension must be a multiple of the ``coarsening_ratio`` multiplied by ``numprocs`` in that dimension.

* ``<diag_name>.file_prefix`` (`string`) optional (default `diags/<diag_name>`)
    Root for output file names. Supports sub-directories.

* ``<diag_name>.file_min_digits`` (`int`) optional (default `6`)
    The minimum number of digits used for the iteration number appended to the diagnostic file names.

* ``<diag_name>.diag_lo`` (list `float`, 1 per dimension) optional (default `-infinity -infinity -infinity`)
    Lower corner of the output fields (if smaller than ``warpx.dom_lo``, then set to ``warpx.dom_lo``). Currently, when the ``diag_lo`` is different from ``warpx.dom_lo``, particle output is disabled.

* ``<diag_name>.diag_hi`` (list `float`, 1 per dimension) optional (default `+infinity +infinity +infinity`)
    Higher corner of the output fields (if larger than ``warpx.dom_hi``, then set to ``warpx.dom_hi``). Currently, when the ``diag_hi`` is different from ``warpx.dom_hi``, particle output is disabled.

* ``<diag_name>.write_species`` (`0` or `1`) optional (default `1`)
    Whether to write species output or not. For checkpoint format, always set this parameter to 1.

* ``<diag_name>.species`` (list of `string`, default all physical species in the simulation)
    Which species dumped in this diagnostics.

* ``<diag_name>.<species_name>.variables`` (list of `strings` separated by spaces, optional)
    List of particle quantities to write to output.
    Choices are ``x``, ``y``, ``z`` for the particle positions (3D and RZ), ``x`` & ``z`` in 2D, ``z`` in 1D,
    ``w`` for the particle weight and ``ux``, ``uy``, ``uz`` for the particle momenta.
    When using the lab-frame electrostatic solver, ``phi`` (electrostatic potential, on the macroparticles) is also available.
    By default, all particle quantities (except ``phi``) are written.
    If ``<diag_name>.<species_name>.variables = none``, no particle data are written.

* ``<diag_name>.<species_name>.random_fraction`` (`float`) optional
    If provided ``<diag_name>.<species_name>.random_fraction = a``, only `a` fraction of the particle data of this species will be dumped randomly in diag ``<diag_name>``, i.e. if `rand() < a`, this particle will be dumped, where `rand()` denotes a random number generator.
    The value `a` provided should be between 0 and 1.

* ``<diag_name>.<species_name>.uniform_stride`` (`int`) optional
    If provided ``<diag_name>.<species_name>.uniform_stride = n``,
    every `n` particle of this species will be dumped, selected uniformly.
    The value provided should be an integer greater than or equal to 0.

* ``<diag_name>.<species_name>.plot_filter_function(t,x,y,z,ux,uy,uz)`` (`string`) optional
    Users can provide an expression returning a boolean for whether a particle is dumped.
    `t` represents the physical time in seconds during the simulation.
    `x, y, z` represent particle positions in the unit of meter.
    `ux, uy, uz` represent particle momenta in the unit of
    :math:`\gamma v/c`, where
    :math:`\gamma` is the Lorentz factor,
    :math:`v/c` is the particle velocity normalized by the speed of light.
    E.g. If provided `(x>0.0)*(uz<10.0)` only those particles located at
    positions `x` greater than `0`, and those having momentum `uz` less than 10,
    will be dumped.

* ``amrex.async_out`` (`0` or `1`) optional (default `0`)
    Whether to use asynchronous IO when writing plotfiles. This only has an effect
    when using the AMReX plotfile format.
    Please see the :ref:`data analysis section <dataanalysis-formats>` for more information.

* ``amrex.async_out_nfiles`` (`int`) optional (default `64`)
    The maximum number of files to write to when using asynchronous IO.
    To use asynchronous IO with more than ``amrex.async_out_nfiles`` MPI ranks,
    WarpX must be configured with ``-DWarpX_MPI_THREAD_MULTIPLE=ON``.
    Please see the :ref:`data analysis section <dataanalysis-formats>` for more information.

* ``warpx.field_io_nfiles`` and ``warpx.particle_io_nfiles`` (`int`) optional (default `1024`)
    The maximum number of files to use when writing field and particle data to plotfile directories.

* ``warpx.mffile_nstreams`` (`int`) optional (default `4`)
    Limit the number of concurrent readers per file.


.. _running-cpp-parameters-diagnostics-timeavg:

Time-Averaged Diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^^

``TimeAveraged`` diagnostics are a special type of ``Full`` diagnostics that allows for the output of time-averaged field data.
This type of diagnostics can be created using ``<diag_name>.diag_type = TimeAveraged``.
We support only field data and related options from the list at `Full Diagnostics`_.

.. note::

    As with ``Full`` diagnostics, ``TimeAveraged`` diagnostics output the initial **instantaneous** conditions of the selected fields on step 0 (unless more specific output intervals exclude output for step 0).

In addition, ``TimeAveraged`` diagnostic options include:

* ``<diag_name>.time_average_mode`` (`string`, default `none`)
    Describes the operating mode for time averaged field output.

    * ``none`` for no averaging (instantaneous fields)

    * ``fixed_start`` for a diagnostic that averages all fields between the current output step and a fixed point in time

    * ``dynamic_start`` for a constant averaging period and output at different points in time (non-overlapping)

    .. note::

        To enable time-averaged field output with intervals tightly spaced enough for overlapping averaging periods,
        please create additional instances of ``TimeAveraged`` diagnostics.

* ``<diag_name>.average_period_steps`` (`int`)
    Configures the number of time steps in an averaging period.
    Set this only in the ``dynamic_start`` mode and only if ``average_period_time`` has not already been set.
    Will be ignored in the ``fixed_start`` mode (with warning).

* ``<diag_name>.average_period_time`` (`float`, in seconds)
    Configures the time (SI units) in an averaging period.
    Set this only in the ``dynamic_start`` mode and only if ``average_period_steps`` has not already been set.
    Will be ignored in the ``fixed_start`` mode (with warning).

* ``<diag_name>.average_start_step`` (`int`)
    Configures the time step at which time-averaging begins.
    Set this only in the ``fixed_start`` mode.
    Will be ignored in the ``dynamic_start`` mode (with warning).

.. _running-cpp-parameters-diagnostics-btd:

BackTransformed Diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``BackTransformed`` diag type are used when running a simulation in a boosted frame, to reconstruct output data to the lab frame. This option can be set using ``<diag_name>.diag_type = BackTransformed``. We support the following list of options from `Full Diagnostics`_

    ``<diag_name>.format``, ``<diag_name>.openpmd_backend``, ``<diag_name>.dump_rz_modes``, ``<diag_name>.file_prefix``, ``<diag_name>.diag_lo``, ``<diag_name>.diag_hi``, ``<diag_name>.write_species``, ``<diag_name>.species``.

    Additional options for this diagnostic include:

* ``<diag_name>.num_snapshots_lab`` (`integer`)
    Only used when ``<diag_name>.diag_type`` is ``BackTransformed``.
    The number of lab-frame snapshots that will be written.
    Only this option or ``intervals`` should be specified;
    a run-time error occurs if the user attempts to set both ``num_snapshots_lab`` and ``intervals``.

* ``<diag_name>.intervals`` (`string`)
    Only used when ``<diag_name>.diag_type`` is ``BackTransformed``.
    Using the `Intervals parser`_ syntax, this string defines the lab frame times at which data is dumped,
    given as multiples of the step size ``dt_snapshots_lab`` or ``dz_snapshots_lab`` described below.
    Example: ``btdiag1.intervals = 10:11,20:24:2`` and ``btdiag1.dt_snapshots_lab = 1.e-12``
    indicate to dump at lab times ``1e-11``, ``1.1e-11``, ``2e-11``, ``2.2e-11``, and ``2.4e-11`` seconds.
    Note that the stop interval, the second number in the slice, must always be specified.
    Only this option or ``num_snapshots_lab`` should be specified;
    a run-time error occurs if the user attempts to set both ``num_snapshots_lab`` and ``intervals``.

* ``<diag_name>.dt_snapshots_lab`` (`float`, in seconds)
    Only used when ``<diag_name>.diag_type`` is ``BackTransformed``.
    The time interval in between the lab-frame snapshots (where this
    time interval is expressed in the laboratory frame).

* ``<diag_name>.dz_snapshots_lab`` (`float`, in meters)
    Only used when ``<diag_name>.diag_type`` is ``BackTransformed``.
    Distance between the lab-frame snapshots (expressed in the laboratory
    frame). ``dt_snapshots_lab`` is then computed by
    ``dt_snapshots_lab = dz_snapshots_lab/c``. Either `dt_snapshots_lab`
    or `dz_snapshot_lab` is required.

* ``<diag_name>.buffer_size`` (`integer`)
    Only used when ``<diag_name>.diag_type`` is ``BackTransformed``.
    The default size of the back transformed diagnostic buffers used to generate lab-frame
    data is 256. That is, when the multifab with lab-frame data has 256 z-slices,
    the data will be flushed out. However, if many lab-frame snapshots are required for
    diagnostics and visualization, the GPU may run out of memory with many large boxes with
    a size of 256 in the z-direction. This input parameter can then be used to set a
    smaller buffer-size, preferably multiples of 8, such that, a large number of
    lab-frame snapshot data can be generated without running out of gpu memory.
    The downside to using a small buffer size, is that the I/O time may increase due
    to frequent flushes of the lab-frame data. The other option is to keep the default
    value for buffer size and use slices to reduce the memory footprint and maintain
    optimum I/O performance.

* ``<diag_name>.do_back_transformed_fields`` (`0` or `1`) optional (default `1`)
    Only used when ``<diag_name>.diag_type`` is ``BackTransformed``
    Whether to back transform the fields or not.
    Note that for ``BackTransformed`` diagnostics, at least one of the options
    ``<diag_name>.do_back_transformed_fields`` or ``<diag_name>.do_back_transformed_particles`` must be 1.

* ``<diag_name>.do_back_transformed_particles`` (`0` or `1`) optional (default `1`)
    Only used when ``<diag_name>.diag_type`` is ``BackTransformed``
    Whether to back transform the particle data or not.
    Note that for ``BackTransformed`` diagnostics, at least one of the options
    ``<diag_name>.do_back_transformed_fields`` or ``<diag_name>.do_back_transformed_particles`` must be 1.
    If ``diag_name.write_species = 0``, then ``<diag_name>.do_back_transformed_particles`` will be set
    to 0 in the simulation and particles will not be backtransformed.

Boundary Scraping Diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``BoundaryScrapingDiagnostics`` are used to collect the particles that are absorbed at the boundaries, throughout the simulation.
This diagnostic type is specified by setting ``<diag_name>.diag_type`` = ``BoundaryScraping``.
Currently, the only supported output format is openPMD, so the user also needs to set ``<diag>.format=openpmd`` and WarpX must be compiled with openPMD turned on.
The data that is to be collected and recorded is controlled per species and per boundary by setting one or more of the flags to ``1``,
``<species>.save_particles_at_xlo/ylo/zlo``, ``<species>.save_particles_at_xhi/yhi/zhi``, and ``<species>.save_particles_at_eb``.
(Note that this diagnostics does not save any field ; it only saves particles.)

The data collected at each boundary is written out to a subdirectory of the diagnostics directory with the name of the boundary, for example, ``particles_at_xlo``, ``particles_at_zhi``, or ``particles_at_eb``.
By default, all of the collected particle data is written out at the end of the simulation. Optionally, the ``<diag_name>.intervals`` parameter can be given to specify writing out the data more often.
This can be important if a large number of particles are lost, avoiding filling up memory with the accumulated lost particle data.

In addition to their usual attributes, the saved particles have
   an integer attribute ``stepScraped``, which indicates the PIC iteration at which each particle was absorbed at the boundary,
   a real attribute ``deltaTimeScraped``, which indicates the time between the time associated to `stepScraped`
   and the exact time when each particle hits the boundary.
   3 real attributes ``nx``, ``ny``, ``nz``, which represents the three components of the normal to the boundary on the point of contact of the particles (not saved if they reach non-EB boundaries)

``BoundaryScrapingDiagnostics`` can be used with ``<diag_name>.<species>.random_fraction``, ``<diag_name>.<species>.uniform_stride``, and ``<diag_name>.<species>.plot_filter_function``, which have the same behavior as for ``FullDiagnostics``. For ``BoundaryScrapingDiagnostics``, these filters are applied at the time the data is written to file. An implication of this is that more particles may initially be accumulated in memory than are ultimately written. ``t`` in ``plot_filter_function`` refers to the time the diagnostic is written rather than the time the particle crossed the boundary.

.. _running-cpp-parameters-diagnostics-reduced:

Reduced Diagnostics
^^^^^^^^^^^^^^^^^^^

``ReducedDiags`` enable users to compute specific reduced quantities, such as particle temperature, energy histograms, or maximum field values, and efficiently save this in-situ analyzed data to files.
This shifts analysis from post-processing to runtime calculation of reduction operations (average, maximum, ...) and can greatly save disk space when "raw" particle and field outputs from `FullDiagnostics` can be avoided in favor of single values, 1D or 2D data at possibly even higher time resolution.

* ``warpx.reduced_diags_names`` (`strings`, separated by spaces)
    A list of user-given names for reduced diagnostics.
    By default, these names are also prefixing the names of output files.
    If ``warpx.reduced_diags_names`` is not provided in the input file,
    no reduced diagnostics will be activated during the run.
    This is then used in the rest of the input deck;
    in this documentation we use ``<reduced_diags_name>`` as a placeholder.

* ``<reduced_diags_name>.type`` (`string`)
    The type of reduced diagnostics associated with this ``<reduced_diags_name>``.
    For example, ``ParticleEnergy``, ``FieldEnergy``, etc.
    All available types are described below in detail.
    For all reduced diagnostics that are writing tabular data into text files,
    the first and the second columns in the output file are
    the time step and the corresponding physical time in seconds, respectively.

    * ``ParticleEnergy``
        This type computes the total and mean relativistic particle kinetic energy among all species:

        .. math::

            E_p = \sum_{i=1}^N w_i \, \left( \sqrt{|\boldsymbol{p}_i|^2 c^2 + m_0^2 c^4} - m_0 c^2 \right)

        where :math:`\boldsymbol{p}_i` is the relativistic momentum of the :math:`i`-th particle, :math:`c` is the speed of light, :math:`m_0` is the rest mass, :math:`N` is the number of particles, and :math:`w_i` is the weight of the :math:`i`-th particle.

        The output columns are the total energy of all species, the total energy per species, the total mean energy :math:`E_p / \sum_i w_i` of all species, and the total mean energy per species.

    * ``ParticleMomentum``
        This type computes the total and mean relativistic particle momentum among all species:

        .. math::

            \boldsymbol{P}_p = \sum_{i=1}^N w_i \, \boldsymbol{p}_i

        where :math:`\boldsymbol{p}_i` is the relativistic momentum of the :math:`i`-th particle, :math:`N` is the number of particles, and :math:`w_i` is the weight of the :math:`i`-th particle.

        The output columns are the components of the total momentum of all species, the total momentum per species, the total mean momentum :math:`\boldsymbol{P}_p / \sum_i w_i` of all species, and the total mean momentum per species.

    * ``FieldEnergy``
        This type computes the electromagnetic field energy

        .. math::

            E_f = \frac{1}{2} \sum_{\text{cells}} \left( \varepsilon_0 |\boldsymbol{E}|^2 + \frac{|\boldsymbol{B}|^2}{\mu_0} \right) \Delta V

        where :math:`\boldsymbol{E}` is the electric field, :math:`\boldsymbol{B}` is the magnetic field, :math:`\varepsilon_0` is the vacuum permittivity, :math:`\mu_0` is the vacuum permeability, :math:`\Delta V` is the cell volume (or cell area in 2D), and the sum is over all cells.

        The output columns are the total field energy :math:`E_f`, the :math:`\boldsymbol{E}` field energy, and the :math:`\boldsymbol{B}` field energy, at each mesh refinement level.

    * ``FieldMomentum``
        This type computes the electromagnetic field momentum

        .. math::

            \boldsymbol{P}_f = \varepsilon_0 \sum_{\text{cells}} \left( \boldsymbol{E} \times \boldsymbol{B} \right) \Delta V

        where :math:`\boldsymbol{E}` is the electric field, :math:`\boldsymbol{B}` is the magnetic field, :math:`\varepsilon_0` is the vacuum permittivity, :math:`\Delta V` is the cell volume (or cell area in 2D), and the sum is over all cells.

        The output columns are the components of the total field momentum :math:`\boldsymbol{P}_f` at each mesh refinement level.

        Note that the fields are *not* averaged on the cell centers before their energy is
        computed.

    * ``FieldMaximum``
        This type computes the maximum value of each component of the electric and magnetic fields
        and of the norm of the electric and magnetic field vectors.
        Measuring maximum fields in a plasma might be very noisy in PIC, use this instead
        for analysis of scenarios such as an electromagnetic wave propagating in vacuum.

        The output columns are
        the maximum value of the :math:`E_x` field,
        the maximum value of the :math:`E_y` field,
        the maximum value of the :math:`E_z` field,
        the maximum value of the norm :math:`|E|` of the electric field,
        the maximum value of the :math:`B_x` field,
        the maximum value of the :math:`B_y` field,
        the maximum value of the :math:`B_z` field and
        the maximum value of the norm :math:`|B|` of the magnetic field,
        at mesh refinement levels from  0 to :math:`n`.

        Note that the fields are averaged on the cell centers before their maximum values are
        computed.

    * ``FieldProbe``
        This type computes the value of each component of the electric and magnetic fields
        and of the Poynting vector (a measure of electromagnetic flux) at points in the domain.

        Multiple geometries for point probes can be specified via ``<reduced_diags_name>.probe_geometry = ...``:

        * ``Point`` (default): a single point
        * ``Line``: a line of points with equal spacing
        * ``Plane``: a plane of points with equal spacing

        **Point**: The point where the fields are measured is specified through the input parameters ``<reduced_diags_name>.x_probe``, ``<reduced_diags_name>.y_probe`` and ``<reduced_diags_name>.z_probe``.

        **Line**: probe a 1 dimensional line of points to create a line detector.
        Initial input parameters ``x_probe``, ``y_probe``, and ``z_probe`` designate one end of the line detector, while the far end is specified via ``<reduced_diags_name>.x1_probe``, ``<reduced_diags_name>.y1_probe``, ``<reduced_diags_name>.z1_probe``.
        Additionally, ``<reduced_diags_name>.resolution`` must be defined to give the number of detector points along the line (equally spaced) to probe.

        **Plane**: probe a 2 dimensional plane of points to create a square plane detector.
        Initial input parameters ``x_probe``, ``y_probe``, and ``z_probe`` designate the center of the detector.
        The detector plane is normal to a vector specified by ``<reduced_diags_name>.target_normal_x``, ``<reduced_diags_name>.target_normal_y``, and ``<reduced_diags_name>.target_normal_z``.
        Note that it is not necessary to specify the ``target_normal`` vector in a 2D simulation (the only supported normal is in ``y``).
        The top of the plane is perpendicular to an "up" vector denoted by ``<reduced_diags_name>.target_up_x``, ``<reduced_diags_name>.target_up_y``, and ``<reduced_diags_name>.target_up_z``.
        The detector has a square radius to be determined by ``<reduced_diags_name>.detector_radius``.
        Similarly to the line detector, the plane detector requires a resolution ``<reduced_diags_name>.resolution``, which denotes the number of detector particles along each side of the square detector.

        The output columns are
        the value of the :math:`E_x` field,
        the value of the :math:`E_y` field,
        the value of the :math:`E_z` field,
        the value of the :math:`B_x` field,
        the value of the :math:`B_y` field,
        the value of the :math:`B_z` field and
        the value of the Poynting Vector :math:`|S|` of the electromagnetic fields,
        at mesh refinement levels from  0 to :math:`n`, at point (:math:`x`, :math:`y`, :math:`z`).

        The fields are always interpolated to the measurement point.
        The interpolation order can be set by specifying ``<reduced_diags_name>.interp_order``,
        defaulting to ``1``.
        In RZ geometry, this only saves the
        0'th azimuthal mode component of the fields.
        Time integrated electric and magnetic field components can instead be obtained by specifying
        ``<reduced_diags_name>.integrate = true``.
        The integration is done every time step even when the data is written out less often.
        In a *moving window* simulation, the FieldProbe can be set to follow the moving frame by specifying ``<reduced_diags_name>.do_moving_window_FP = 1`` (default 0).

        .. warning::

           The FieldProbe reduced diagnostic does not yet add a Lorentz back transformation for boosted frame simulations.
           Thus, it records field data in the boosted frame, not (yet) in the lab frame.

    * ``RhoMaximum``
        This type computes the maximum and minimum values of the total charge density as well as
        the maximum absolute value of the charge density of each charged species.
        Please be aware that measuring maximum charge densities might be very noisy in PIC simulations.

        The output columns are
        the maximum value of the :math:`rho` field,
        the minimum value of the :math:`rho` field,
        the maximum value of the absolute :math:`|rho|` field of each charged species.

        Note that the charge densities are averaged on the cell centers before their maximum values
        are computed.

    * ``FieldReduction``
        This type computes an arbitrary reduction of the positions, the current density, and the electromagnetic fields.

        * ``<reduced_diags_name>.reduced_function(x,y,z,Ex,Ey,Ez,Bx,By,Bz,jx,jy,jz)`` (`string`)
            An analytic function to be reduced must be provided, using the math parser.

        * ``<reduced_diags_name>.reduction_type`` (`string`)
            The type of reduction to be performed. It must be either ``Maximum``, ``Minimum`` or
            ``Integral``.
            ``Integral`` computes the spatial integral of the function defined in the parser by
            summing its value on all grid points and multiplying the result by the volume of a
            cell.
            Please be also aware that measuring maximum quantities might be very noisy in PIC
            simulations.

        The only output column is the reduced value.

        Note that the fields are averaged on the cell centers before the reduction is performed.

    * ``ParticleNumber``
        This type computes the total number of macroparticles and of physical particles (i.e. the
        sum of their weights) in the whole simulation domain (for each species and summed over all
        species). It can be useful in particular for simulations with creation (ionization, QED
        processes) or removal (resampling) of particles.

        The output columns are
        total number of macroparticles summed over all species,
        total number of macroparticles of each species,
        sum of the particles' weight summed over all species,
        sum of the particles' weight of each species.

    * ``BeamRelevant``
        This type computes properties of a particle beam relevant for particle accelerators, like position, momentum, emittance, etc.

        ``<reduced_diags_name>.species`` must be provided, such that the diagnostics are done for this (beam-like) species only.

        The output columns (for 3D-XYZ) are the following, where the average is done over the whole species (typical usage: the particle beam is in a separate species):

        [0]: simulation step (iteration).

        [1]: time (s).

        [2], [3], [4]: The mean values of beam positions (m)
        :math:`\langle x \rangle`,
        :math:`\langle y \rangle`,
        :math:`\langle z \rangle`.

        [5], [6], [7]: The mean values of beam relativistic momenta (kg m/s)
        :math:`\langle p_x \rangle`,
        :math:`\langle p_y \rangle`,
        :math:`\langle p_z \rangle`.

        [8]: The mean Lorentz factor :math:`\langle \gamma \rangle`.

        [9], [10], [11]: The RMS values of beam positions (m)
        :math:`\delta_x = \sqrt{ \langle (x - \langle x \rangle)^2 \rangle }`,
        :math:`\delta_y = \sqrt{ \langle (y - \langle y \rangle)^2 \rangle }`,
        :math:`\delta_z = \sqrt{ \langle (z - \langle z \rangle)^2 \rangle }`.

        [12], [13], [14]: The RMS values of beam relativistic momenta (kg m/s)
        :math:`\delta_{px} = \sqrt{ \langle (p_x - \langle p_x \rangle)^2 \rangle }`,
        :math:`\delta_{py} = \sqrt{ \langle (p_y - \langle p_y \rangle)^2 \rangle }`,
        :math:`\delta_{pz} = \sqrt{ \langle (p_z - \langle p_z \rangle)^2 \rangle }`.

        [15]: The RMS value of the Lorentz factor
        :math:`\sqrt{ \langle (\gamma - \langle \gamma \rangle)^2 \rangle }`.

        [16], [17], [18]: beam projected transverse RMS normalized emittance (m)
        :math:`\epsilon_x = \dfrac{1}{mc} \sqrt{\delta_x^2 \delta_{px}^2 -
        \Big\langle (x-\langle x \rangle) (p_x-\langle p_x \rangle) \Big\rangle^2}`,
        :math:`\epsilon_y = \dfrac{1}{mc} \sqrt{\delta_y^2 \delta_{py}^2 -
        \Big\langle (y-\langle y \rangle) (p_y-\langle p_y \rangle) \Big\rangle^2}`,
        :math:`\epsilon_z = \dfrac{1}{mc} \sqrt{\delta_z^2 \delta_{pz}^2 -
        \Big\langle (z-\langle z \rangle) (p_z-\langle p_z \rangle) \Big\rangle^2}`.

        [19], [20]: Twiss alpha for the transverse directions
        :math:`\alpha_x = - \Big\langle (x-\langle x \rangle) (p_x-\langle p_x \rangle) \Big\rangle \Big/ \epsilon_x`,
        :math:`\alpha_y = - \Big\langle (y-\langle y \rangle) (p_y-\langle p_y \rangle) \Big\rangle \Big/ \epsilon_y`.

        [21], [22]: beta function for the transverse directions (m)
        :math:`\beta_x = \dfrac{{\delta_x}^2}{\epsilon_x}`,
        :math:`\beta_y = \dfrac{{\delta_y}^2}{\epsilon_y}`.

        [23]: The charge of the beam (C).

        For 2D-XZ,
        :math:`\langle y \rangle`,
        :math:`\delta_y`, and
        :math:`\epsilon_y` will not be outputted.

    * ``LoadBalanceCosts``
        This type computes the cost, used in load balancing, for each box on the domain.
        The cost :math:`c` is computed as

        .. math::

            c = n_{\text{particle}} \cdot w_{\text{particle}} + n_{\text{cell}} \cdot w_{\text{cell}},

        where
        :math:`n_{\text{particle}}` is the number of particles on the box,
        :math:`w_{\text{particle}}` is the particle cost weight factor (controlled by ``algo.costs_heuristic_particles_wt``),
        :math:`n_{\text{cell}}` is the number of cells on the box, and
        :math:`w_{\text{cell}}` is the cell cost weight factor (controlled by ``algo.costs_heuristic_cells_wt``).

    * ``LoadBalanceEfficiency``
        This type computes the load balance efficiency, given the present costs
        and distribution mapping. Load balance efficiency is computed as the
        mean cost over all ranks, divided by the maximum cost over all ranks.
        Until costs are recorded, load balance efficiency is output as `-1`;
        at earliest, the load balance efficiency can be output starting at step
        `2`, since costs are not recorded until step `1`.

    * ``ParticleHistogram``
        This type computes a user defined particle histogram.

        * ``<reduced_diags_name>.species`` (`string`)
            A species name must be provided,
            such that the diagnostics are done for this species.

        * ``<reduced_diags_name>.histogram_function(t,x,y,z,ux,uy,uz)`` (`string`)
            A histogram function must be provided.
            `t` represents the physical time in seconds during the simulation.
            `x, y, z` represent particle positions in the unit of meter.
            `ux, uy, uz` represent the particle momenta in the unit of
            :math:`\gamma v/c`, where
            :math:`\gamma` is the Lorentz factor,
            :math:`v/c` is the particle velocity normalized by the speed of light.
            E.g.
            ``x`` produces the position (density) distribution in `x`.
            ``ux`` produces the momentum distribution in `x`,
            ``sqrt(ux*ux+uy*uy+uz*uz)`` produces the speed distribution.
            The default value of the histogram without normalization is
            :math:`f = \sum\limits_{i=1}^N w_i`, where
            :math:`\sum\limits_{i=1}^N` is the sum over :math:`N` particles
            in that bin,
            :math:`w_i` denotes the weight of the ith particle.

        * ``<reduced_diags_name>.bin_number`` (`int` > 0)
            This is the number of bins used for the histogram.

        * ``<reduced_diags_name>.bin_max`` (`float`)
            This is the maximum value of the bins.

        * ``<reduced_diags_name>.bin_min`` (`float`)
            This is the minimum value of the bins.

        * ``<reduced_diags_name>.normalization`` (optional)
            This provides options to normalize the histogram:

            ``unity_particle_weight``
            uses unity particle weight to compute the histogram,
            such that the values of the histogram are
            the number of counted macroparticles in that bin,
            i.e.  :math:`f = \sum\limits_{i=1}^N 1`,
            :math:`N` is the number of particles in that bin.

            ``max_to_unity`` will normalize the histogram such that
            its maximum value is one.

            ``area_to_unity`` will normalize the histogram such that
            the area under the histogram is one,
            so the histogram is also the probability density function.

            If nothing is provided,
            the macroparticle weight will be used to compute
            the histogram, and no normalization will be done.

        * ``<reduced_diags_name>.filter_function(t,x,y,z,ux,uy,uz)`` (`string`) optional
            Users can provide an expression returning a boolean for whether a particle is taken
            into account when calculating the histogram.
            `t` represents the physical time in seconds during the simulation.
            `x, y, z` represent particle positions in the unit of meter.
            `ux, uy, uz` represent particle momenta in the unit of
            :math:`\gamma v/c`, where
            :math:`\gamma` is the Lorentz factor,
            :math:`v/c` is the particle velocity normalized by the speed of light.
            E.g. If provided `(x>0.0)*(uz<10.0)` only those particles located at
            positions `x` greater than `0`, and those having momentum `uz` less than 10,
            will be taken into account when calculating the histogram.

        The output columns are
        values of the 1st bin, the 2nd bin, ..., the nth bin.
        An example input file and a loading python script of
        using the histogram reduced diagnostics
        are given in ``Examples/Tests/initial_distribution/``.

    * ``ParticleHistogram2D``
        This type computes a user defined, 2D particle histogram.

        * ``<reduced_diags_name>.species`` (`string`)
            A species name must be provided,
            such that the diagnostics are done for this species.

        * ``<reduced_diags_name>.file_min_digits`` (`int`) optional (default `6`)
            The minimum number of digits used for the iteration number appended to the diagnostic file names.

        * ``<reduced_diags_name>.histogram_function_abs(t,x,y,z,ux,uy,uz,w)`` (`string`)
            A histogram function must be provided for the abscissa axis.
            `t` represents the physical time in seconds during the simulation.
            `x, y, z` represent particle positions in the unit of meter.
            `ux, uy, uz` represent the particle velocities in the unit of
            :math:`\gamma v/c`, where
            :math:`\gamma` is the Lorentz factor,
            :math:`v/c` is the particle velocity normalized by the speed of light.
            `w` represents the weight.

        * ``<reduced_diags_name>.histogram_function_ord(t,x,y,z,ux,uy,uz,w)`` (`string`)
            A histogram function must be provided for the ordinate axis.

        * ``<reduced_diags_name>.bin_number_abs`` (`int` > 0) and ``<reduced_diags_name>.bin_number_ord`` (`int` > 0)
            These are the number of bins used for the histogram for the abscissa and ordinate axis respectively.

        * ``<reduced_diags_name>.bin_max_abs`` (`float`) and ``<reduced_diags_name>.bin_max_ord`` (`float`)
            These are the maximum value of the bins for the abscissa and ordinate axis respectively.
            Particles with values outside of these ranges are discarded.

        * ``<reduced_diags_name>.bin_min_abs`` (`float`) and ``<reduced_diags_name>.bin_min_ord`` (`float`)
            These are the minimum value of the bins for the abscissa and ordinate axis respectively.
            Particles with values outside of these ranges are discarded.

        * ``<reduced_diags_name>.filter_function(t,x,y,z,ux,uy,uz,w)`` (`string`) optional
            Users can provide an expression returning a boolean for whether a particle is taken
            into account when calculating the histogram.
            `t` represents the physical time in seconds during the simulation.
            `x, y, z` represent particle positions in the unit of meter.
            `ux, uy, uz` represent particle velocities in the unit of
            :math:`\gamma v/c`, where
            :math:`\gamma` is the Lorentz factor,
            :math:`v/c` is the particle velocity normalized by the speed of light.
            `w` represents the weight.

        * ``<reduced_diags_name>.value_function(t,x,y,z,ux,uy,uz,w)`` (`string`) optional
            Users can provide an expression for the weight used to calculate the number of particles
            per cell associated with the selected abscissa and ordinate functions and/or the filter function.
            `t` represents the physical time in seconds during the simulation.
            `x, y, z` represent particle positions in the unit of meter.
            `ux, uy, uz` represent particle velocities in the unit of
            :math:`\gamma v/c`, where
            :math:`\gamma` is the Lorentz factor,
            :math:`v/c` is the particle velocity normalized by the speed of light.
            `w` represents the weight.

        The output is a ``<reduced_diags_name>`` folder containing a set of openPMD files.
        An example input file and a loading python script of
        using the histogram2D reduced diagnostics
        are given in ``Examples/Tests/histogram2D/``.

    * ``ParticleExtrema``
        This type computes the minimum and maximum values of
        particle position, momentum, gamma, weight,
        and the :math:`\chi` parameter for QED species.

        ``<reduced_diags_name>.species`` must be provided,
        such that the diagnostics are done for this species only.

        The output columns are
        minimum and maximum position :math:`x`, :math:`y`, :math:`z`;
        minimum and maximum momentum :math:`p_x`, :math:`p_y`, :math:`p_z`;
        minimum and maximum gamma :math:`\gamma`;
        minimum and maximum weight :math:`w`;
        minimum and maximum :math:`\chi`.

        Note that when the QED parameter :math:`\chi` is computed,
        field gather is carried out at every output,
        so the time of the diagnostic may be long
        depending on the simulation size.

    * ``ChargeOnEB``
        This type computes the total surface charge on the embedded boundary
        (in Coulombs), by using the formula

        .. math::

            Q_{tot} = \epsilon_0 \iint dS \cdot E

        where the integral is performed over the surface of the embedded boundary.

        When providing ``<reduced_diags_name>.weighting_function(x,y,z)``, the
        computed integral is weighted:

        .. math::

            Q = \epsilon_0 \iint dS \cdot E \times weighting(x, y, z)

        In particular, by choosing a weighting function which returns either
        1 or 0, it is possible to compute the charge on only some part of the
        embedded boundary.

    * ``ColliderRelevant``
        This diagnostics computes properties of two colliding beams that are relevant for particle colliders.
        Two species must be specified. Photon species are not supported yet.
        It is assumed that the two species propagate and collide along the ``z`` direction.
        The output columns (for 3D-XYZ) are the following, where the minimum, average and maximum
        are done over the whole species:

        [0]: simulation step (iteration).

        [1]: time (s).

        [2]: time derivative of the luminosity (:math:`m^{-2}s^{-1}`) defined as:

        .. math::

            \frac{dL}{dt} = 2 c \iiint  n_1(x,y,z) n_2(x,y,z) dx dy dz

        where :math:`n_1`, :math:`n_2` are the number densities of the two colliding species.

        [3], [4], [5]: If, QED is enabled, the minimum, average and maximum values of the quantum parameter :math:`\chi` of species 1:
        :math:`\chi_{min}`,
        :math:`\langle \chi \rangle`,
        :math:`\chi_{max}`.
        If QED is not enabled, these numbers are not computed.

        [6], [7]: The average and standard deviation of the values of the transverse coordinate :math:`x` (m) of species 1:
        :math:`\langle x \rangle`,
        :math:`\sqrt{\langle x- \langle x \rangle \rangle^2}`.

        [8], [9]: The average and standard deviation of the values of the transverse coordinate :math:`y` (m) of species 1:
        :math:`\langle y \rangle`,
        :math:`\sqrt{\langle y- \langle y \rangle \rangle^2}`.

        [10], [11], [12], [13]: The minimum, average, maximum and standard deviation of the angle :math:`\theta_x = \angle (u_x, u_z)` (rad) of species 1:
        :math:`{\theta_x}_{min}`,
        :math:`\langle \theta_x \rangle`,
        :math:`{\theta_x}_{max}`,
        :math:`\sqrt{\langle \theta_x- \langle \theta_x \rangle \rangle^2}`.

        [14], [15], [16], [17]:  The minimum, average, maximum and standard deviation of the angle :math:`\theta_y = \angle (u_y, u_z)` (rad) of species 1:
        :math:`{\theta_y}_{min}`,
        :math:`\langle \theta_y \rangle`,
        :math:`{\theta_y}_{max}`,
        :math:`\sqrt{\langle \theta_y- \langle \theta_y \rangle \rangle^2}`.

        [18], ..., [32]: Analogous quantities for species 2.

        For 2D-XZ, :math:`y`-related quantities are not outputted.
        For 1D-Z, :math:`x`-related and :math:`y`-related quantities are not outputted.
        RZ geometry is not supported yet.

    * ``DifferentialLuminosity``
        This type computes the differential luminosity between two species, defined as:

        .. math::

            \frac{d\mathcal{L}}{d\mathcal{E}^*}(\mathcal{E}^*, t) = \int_0^t dt'\int d\boldsymbol{x}\,d\boldsymbol{p}_1 d\boldsymbol{p}_2\;
             \sqrt{ |\boldsymbol{v}_1 - \boldsymbol{v}_2|^2 - |\boldsymbol{v}_1\times\boldsymbol{v}_2|^2/c^2} \\ f_1(\boldsymbol{x}, \boldsymbol{p}_1, t')f_2(\boldsymbol{x}, \boldsymbol{p}_2, t') \delta(\mathcal{E}^* - \mathcal{E}^*(\boldsymbol{p}_1, \boldsymbol{p}_2))

        where :math:`f_i` is the distribution function of species :math:`i` and
        :math:`\mathcal{E}^*(\boldsymbol{p}_1, \boldsymbol{p}_2) = \sqrt{m_1^2c^4 + m_2^2c^4 + 2 c^2{p_1}^\mu {p_2}_\mu}`
        is the energy in the center-of-mass frame, where :math:`p^\mu = (\sqrt{m^2 c^2 + \boldsymbol{p}^2}, \boldsymbol{p})`
        represents the 4-momentum. Note that, if :math:`\sigma^*(\mathcal{E}^*)`
        is the center-of-mass cross-section of a given collision process, then
        :math:`\int d\mathcal{E}^* \frac{d\mathcal{L}}{d\mathcal{E}^*} (\mathcal{E}^*, t)\sigma^*(\mathcal{E}^*)`
        gives the total number of collisions of that process (from the beginning of the simulation up until time :math:`t`).

        The differential luminosity is given in units of :math:`\text{m}^{-2}.\text{eV}^{-1}`. For collider-relevant WarpX simulations
        involving two crossing, high-energy beams of particles, the differential luminosity in :math:`\text{s}^{-1}.\text{m}^{-2}.\text{eV}^{-1}`
        can be obtained by multiplying the above differential luminosity by the expected repetition rate of the beams.

        In practice, the above expression of the differential luminosity is evaluated over discrete bins in energy :math:`\mathcal{E}^*`,
        and by summing over macroparticles.

        * ``<reduced_diags_name>.species`` (`list of two strings`)
            The names of the two species for which the differential luminosity is computed.

        * ``<reduced_diags_name>.bin_number`` (`int` > 0)
            The number of bins in energy :math:`\mathcal{E}^*`

        * ``<reduced_diags_name>.bin_max`` (`float`, in eV)
            The minimum value of :math:`\mathcal{E}^*` for which the differential luminosity is computed.

        * ``<reduced_diags_name>.bin_min`` (`float`, in eV)
            The maximum value of :math:`\mathcal{E}^*` for which the differential luminosity is computed.

    * ``Timestep``
        This type outputs the simulation's physical timestep (in seconds) at each mesh refinement level.

* ``reduced_diags.intervals`` (`string`)
    Using the `Intervals Parser`_ syntax, this string defines the timesteps at which reduced
    diagnostics are written to the file.
    This can also be specified for the specific diagnostic by setting ``<reduced_diags_name>.intervals``.

* ``reduced_diags.path`` (`string`) optional (default `./diags/reducedfiles/`)
    The path where the output file will be stored.
    This can also be specified for the specific diagnostic by setting ``<reduced_diags_name>.path``.

* ``reduced_diags.extension`` (`string`) optional (default `txt`)
    The extension of the output file (the suffix).
    This can also be specified for the specific diagnostic by setting ``<reduced_diags_name>.extension``.

* ``reduced_diags.separator`` (`string`) optional (default a `whitespace`)
    The separator between row values in the output file.
    The default separator is a whitespace.
    This can also be specified for the specific diagnostic by setting ``<reduced_diags_name>.separator``.

* ``reduced_diags.precision`` (`integer`) optional (default `14`)
    The precision used when writing out the data to the text files.
    This can also be specified for the specific diagnostic by setting ``<reduced_diags_name>.precision``.

Lookup tables and other settings for QED modules
------------------------------------------------

Lookup tables store pre-computed values for functions used by the QED modules.
**This feature requires to compile with QED=TRUE (and also with QED_TABLE_GEN=TRUE for table generation)**

* ``qed_bw.lookup_table_mode`` (`string`)
    There are three options to prepare the lookup table required by the Breit-Wheeler module:

    * ``builtin``:  a built-in table is used (Warning: the table gives reasonable results but its resolution is quite low).

    * ``generate``: a new table is generated. This option requires Boost math library
      (version >= 1.66) and to compile with ``QED_TABLE_GEN=TRUE``. All
      the following parameters must be specified (table 1 is used to evolve the optical depth
      of the photons, while table 2 is used for pair generation):

        * ``qed_bw.tab_dndt_chi_min`` (`float`): minimum chi parameter for lookup table 1 (
          used for the evolution of the optical depth of the photons)

        * ``qed_bw.tab_dndt_chi_max`` (`float`): maximum chi parameter for lookup table 1

        * ``qed_bw.tab_dndt_how_many`` (`int`): number of points to be used for lookup table 1

        * ``qed_bw.tab_pair_chi_min`` (`float`): minimum chi parameter for lookup table 2 (
          used for pair generation)

        * ``qed_bw.tab_pair_chi_max`` (`float`): maximum chi parameter for lookup table 2

        * ``qed_bw.tab_pair_chi_how_many`` (`int`): number of points to be used for chi axis in lookup table 2

        * ``qed_bw.tab_pair_frac_how_many`` (`int`): number of points to be used for the second axis in lookup table 2
          (the second axis is the ratio between the quantum parameter of the less energetic particle of the pair and the
          quantum parameter of the photon).

        * ``qed_bw.save_table_in`` (`string`): where to save the lookup table

      Alternatively, the lookup table can be generated using a standalone tool (see :ref:`qed tools section <generate-lookup-tables-with-tools>`).

    * ``load``: a lookup table is loaded from a pre-generated binary file. The following parameter
      must be specified:

        * ``qed_bw.load_table_from`` (`string`): name of the lookup table file to read from.

* ``qed_qs.lookup_table_mode`` (`string`)
    There are three options to prepare the lookup table required by the Quantum Synchrotron module:

    * ``builtin``: a built-in table is used (Warning: the table gives reasonable results but its resolution is quite low).

    * ``generate``: a new table is generated. This option requires Boost math library
      (version >= 1.66) and to compile with ``QED_TABLE_GEN=TRUE``. All
      the following parameters must be specified (table 1 is used to evolve the optical depth
      of the particles, while table 2 is used for photon emission):

        * ``qed_qs.tab_dndt_chi_min`` (`float`): minimum chi parameter for lookup table 1 (
          used for the evolution of the optical depth of electrons and positrons)

        * ``qed_qs.tab_dndt_chi_max`` (`float`): maximum chi parameter for lookup table 1

        * ``qed_qs.tab_dndt_how_many`` (`int`): number of points to be used for lookup table 1

        * ``qed_qs.tab_em_chi_min`` (`float`): minimum chi parameter for lookup table 2 (
          used for photon emission)

        * ``qed_qs.tab_em_chi_max`` (`float`): maximum chi parameter for lookup table 2

        * ``qed_qs.tab_em_chi_how_many`` (`int`): number of points to be used for chi axis in lookup table 2

        * ``qed_qs.tab_em_frac_how_many`` (`int`): number of points to be used for the second axis in lookup table 2
          (the second axis is the ratio between the quantum parameter of the photon and the
          quantum parameter of the charged particle).

        * ``qed_qs.tab_em_frac_min`` (`float`): minimum value to be considered for the second axis of lookup table 2

        * ``qed_qs.save_table_in`` (`string`): where to save the lookup table

      Alternatively, the lookup table can be generated using a standalone tool (see :ref:`qed tools section <generate-lookup-tables-with-tools>`).

    * ``load``: a lookup table is loaded from a pre-generated binary file. The following parameter
      must be specified:

        * ``qed_qs.load_table_from`` (`string`): name of the lookup table file to read from.

* ``qed_bw.chi_min`` (`float`): minimum chi parameter to be considered by the Breit-Wheeler engine
    (suggested value : 0.01)

* ``qed_qs.chi_min`` (`float`): minimum chi parameter to be considered by the Quantum Synchrotron engine
    (suggested value : 0.001)

* ``qed_qs.photon_creation_energy_threshold`` (`float`) optional (default `2`)
    Energy threshold for photon particle creation in `*me*c^2` units.

* ``warpx.do_qed_schwinger`` (`bool`) optional (default `0`)
    If this is 1, Schwinger electron-positron pairs can be generated in vacuum in the cells where the EM field is high enough.
    Activating the Schwinger process requires the code to be compiled with ``QED=TRUE`` and ``PICSAR``.
    If ``warpx.do_qed_schwinger = 1``, Schwinger product species must be specified with
    ``qed_schwinger.ele_product_species`` and ``qed_schwinger.pos_product_species``.
    Schwinger process requires either ``warpx.grid_type = collocated`` or
    ``algo.field_gathering=momentum-conserving`` (so that different field components are computed
    at the same location in the grid) and does not currently support mesh refinement, cylindrical
    coordinates or single precision.

* ``qed_schwinger.ele_product_species`` (`string`)
    If Schwinger process is activated, an electron product species must be specified
    (the name of an existing electron species must be provided).

* ``qed_schwinger.pos_product_species`` (`string`)
    If Schwinger process is activated, a positron product species must be specified
    (the name of an existing positron species must be provided).

* ``qed_schwinger.y_size`` (`float`; in meters)
    If Schwinger process is activated with ``DIM=2D``, a transverse size must be specified.
    It is used to convert the pair production rate per unit volume into an actual number of created particles.
    This value should correspond to the typical transverse extent for which the EM field has a very high value
    (e.g. the beam waist for a focused laser beam).

* ``qed_schwinger.xmin,ymin,zmin`` and ``qed_schwinger.xmax,ymax,zmax`` (`float`) optional (default unlimited)
    When ``qed_schwinger.xmin`` and ``qed_schwinger.xmax`` are set, they delimit the region within
    which Schwinger pairs can be created.
    The same is applicable in the other directions.

* ``qed_schwinger.threshold_poisson_gaussian`` (`integer`) optional (default `25`)
    If the expected number of physical pairs created in a cell at a given timestep is smaller than this threshold,
    a Poisson distribution is used to draw the actual number of physical pairs created.
    Otherwise a Gaussian distribution is used.
    Note that, regardless of this parameter, the number of macroparticles created is at most one per cell
    per timestep per species (with a weight corresponding to the number of physical pairs created).

Checkpoints and restart
-----------------------
WarpX supports checkpoints/restart via AMReX.
The checkpoint capability can be turned with regular diagnostics: ``<diag_name>.format = checkpoint``.

* ``amr.restart`` (`string`)
    Name of the checkpoint file to restart from. Returns an error if the folder does not exist
    or if it is not properly formatted.

* ``warpx.write_diagnostics_on_restart`` (`bool`) optional (default `false`)
    When `true`, write the diagnostics after restart at the time of the restart.

Intervals parser
----------------

WarpX can parse time step interval expressions of the form ``start:stop:period``, e.g.
``1:2:3, 4::, 5:6, :, ::10``.
A comma is used as a separator between groups of intervals, which we call slices.
The resulting time steps are the `union set <https://en.wikipedia.org/wiki/Union_(set_theory)>`_ of all given slices.
White spaces are ignored.
A single slice can have 0, 1 or 2 colons ``:``, just as `numpy slices <https://numpy.org/doc/stable/reference/generated/numpy.s_.html>`_, but with inclusive upper bound for ``stop``.

* For 0 colon the given value is the period

* For 1 colon the given string is of the type ``start:stop``

* For 2 colons the given string is of the type ``start:stop:period``

Any value that is not given is set to default.
Default is ``0`` for the start, ``std::numeric_limits<int>::max()`` for the stop and ``1`` for the
period.
For the 1 and 2 colon syntax, actually having values in the string is optional
(this means that ``::5``, ``100 ::10`` and ``100 :`` are all valid syntaxes).

All values can be expressions that will be parsed in the same way as other integer input parameters.

**Examples**

* ``something_intervals = 50`` -> do something at timesteps 0, 50, 100, 150, etc.
  (equivalent to ``something_intervals = ::50``)

* ``something_intervals = 300:600:100`` -> do something at timesteps 300, 400, 500 and 600.

* ``something_intervals = 300::50`` -> do something at timesteps 300, 350, 400, 450, etc.

* ``something_intervals = 105:108,205:208`` -> do something at timesteps 105, 106, 107, 108,
  205, 206, 207 and 208. (equivalent to ``something_intervals = 105 : 108 : , 205 : 208 :``)

* ``something_intervals = :`` or  ``something_intervals = ::`` -> do something at every timestep.

* ``something_intervals = 167:167,253:253,275:425:50`` do something at timesteps 167, 253, 275,
  325, 375 and 425.

This is essentially the python slicing syntax except that the stop is inclusive
(``0:100`` contains 100) and that no colon means that the given value is the period.

Note that if a given period is zero or negative, the corresponding slice is disregarded.
For example, ``something_intervals = -1`` deactivates ``something`` and
``something_intervals = ::-1,100:1000:25`` is equivalent to ``something_intervals = 100:1000:25``.


.. _running-cpp-parameters-test-debug:

Testing and Debugging
---------------------

When developing, testing and :ref:`debugging WarpX <debugging_warpx>`, the following options can be considered.

* ``warpx.verbose`` (``0`` or ``1``; default is ``1`` for true)
    Controls how much information is printed to the terminal, when running WarpX.

* ``warpx.always_warn_immediately`` (``0`` or ``1``; default is ``0`` for false)
    If set to ``1``, WarpX immediately prints every warning message as soon as
    it is generated. It is mainly intended for debug purposes, in case a simulation
    crashes before a global warning report can be printed.

* ``warpx.abort_on_warning_threshold`` (string: ``low``, ``medium`` or ``high``) optional
    Optional threshold to abort as soon as a warning is raised.
    If the threshold is set, warning messages with priority greater than or
    equal to the threshold trigger an immediate abort.
    It is mainly intended for debug purposes, and is best used with
    ``warpx.always_warn_immediately=1``.

* ``amrex.abort_on_unused_inputs`` (``0`` or ``1``; default is ``0`` for false)
    When set to ``1``, this option causes simulation to fail *after* its completion if there were unused parameters.
    It is mainly intended for continuous integration and automated testing to check that all tests and inputs are adapted to API changes.

* ``amrex.use_profiler_syncs`` (``0`` or ``1``; default is ``0`` for false)
    Adds a synchronization at the start of communication, so any load balance will be caught there (the timer is called ``SyncBeforeComms``), then the comm operation will run.
    This will slow down the run.

* ``warpx.serialize_initial_conditions`` (`0` or `1`) optional (default `0`)
    Serialize the initial conditions for reproducible testing, e.g, in our continuous integration tests.
    Mainly whether or not to use OpenMP threading for particle initialization.

* ``warpx.safe_guard_cells`` (`0` or `1`) optional (default `0`)
    Run in safe mode, exchanging more guard cells, and more often in the PIC loop (for debugging).

* ``ablastr.fillboundary_always_sync`` (`0` or `1`) optional (default `0`)
    Run all ``FillBoundary`` operations on ``MultiFab`` to force-synchronize shared nodal points.
    This slightly increases communication cost and can help to spot missing ``nodal_sync`` flags in these operations.

.. bibliography::
    :keyprefix: param-
