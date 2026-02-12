Estimate memory cost for a simulation
=====================================

Aside from the detailed compute load and memory cost information that WarpX provides
via the :ref:`ReducedDiagnostic <dataanalysis-reduced-diagnostics>` option :ref:`LoadBalanceCosts <running-cpp-parameters-diagnostics>`
 it is useful to make
good initial estimates on how much memory a simulation will take up before actually running it.
Especially when running on multiple hardware accelerators (e.g. GPUs), the available memory is limited
and running out of memory on only a single device can cause a crash.

In ``Tools/RunPlanning`` there is a tool named ``memory_calculator.py`` that contains
the class ``MemoryCalculator`` which calculates the memory requirement of a simulation
(sub-)domain for fields, particles, random number generation and diagnostic overhead.

It is reasonable to start the estimation by dividing the whole simulation
domain into rectangular boxes with one box per device. To help with the initial setup
for the number of cells given a desired resolution, WarpX provides a python helper script
in the :ref:`parallelization workflow <parallelization_warpx>` section. While WarpX
supports dynamic load balancing through `AMReX <https://amrex-codes.github.io/amrex/docs_html/ManagingGridHierarchy_Chapter.html?highlight=load+balancing#gridding-and-load-balancing>`_
the ensuing distribution mapping is difficult to predict and the provided script does
not cover this complexity. However, the actual load per MPI rank can later be visualized
later by following the :ref:`Distribution Mapping Visualization workflow <plot_distribution_mapping>`.

Below is an example for calculating the runtime memory footprint of the

.. literalinclude:: ./memoryPerDevice.py
    :language: python3
    :lines: 12,14,28-

This will give the following output:

.. program-output:: bash -c "PYTHONPATH=$(pwd)/../../../../:$PYTHONPATH ./memory_per_device.py"

Notes:

Limitations
-----------

The ``MemoryCalculator`` provides **estimates** of simulation memory requirements.
Actual memory usage may differ due to:

* **AMReX overhead**: Memory management, metadata, and communication buffers
* **Dynamic load balancing**: Box sizes and distribution may vary at runtime
* **Particle dynamics**: Particle counts can change due to ionization, pair creation, etc.
* **Diagnostics**: Output buffers and in-situ visualization can add significant overhead
* **Operating system**: Memory fragmentation and allocation overhead

Typically, the calculator provides a lower bound estimate that is 70-90% of actual memory usage.

For accurate runtime memory measurements, use the :ref:`LoadBalanceCosts diagnostic <running-cpp-parameters-diagnostics>`.

New Features
------------

The memory calculator now supports:

**Solver Types** (specify via ``solver_type`` parameter):

* **electromagnetic** (default): Full EM-PIC with E, B, J fields (9 components)
* **electrostatic**: Electrostatic solver with E, phi, rho only (5 components, ~45% less memory)
* **magnetostatic**: Magnetostatic with B from currents (8 components)
* **hybrid**: Hybrid-PIC with kinetic ions and fluid electrons (~55% more memory)

**Transparency**: Call ``get_field_breakdown()`` after ``mem_req_by_fields()`` to see exactly
which field components are allocated and their memory usage.

**Other Features**:

* **Mesh refinement**: Account for auxiliary grids with ``num_mr_levels``
* **QED physics**: Include optical depth attributes with ``enable_qed=True``
* **Ionization**: Include ionization level with ``enable_ionization=True``
* **PSATD solver**: Account for FFT buffers with ``use_psatd=True``
* **Multiple GPU models**: A100, H100, V100, MI250X, or custom specifications
* **Divergence cleaning**: F and G field memory automatically included

Example with solver type::

    from Tools.RunPlanning.memory_calculator import MemoryCalculator as MC

    # Electrostatic simulation
    mc_es = MC(256, 256, 256, build_dim=3, solver_type="electrostatic")
    field_mem_es = mc_es.mem_req_by_fields(256, 256, 256, pml_ncell=0)
    breakdown_es = mc_es.get_field_breakdown()
    # Shows: {'E_field': 3, 'phi_potential': 1, 'rho_charge': 1}

    # Electromagnetic simulation
    mc_em = MC(256, 256, 256, build_dim=3, solver_type="electromagnetic")
    field_mem_em = mc_em.mem_req_by_fields(256, 256, 256, pml_ncell=0)
    breakdown_em = mc_em.get_field_breakdown()
    # Shows: {'E_field': 3, 'B_field': 3, 'J_current': 3}

See Also
--------

* :ref:`Distribution mapping visualization <plot_distribution_mapping>`
* :ref:`LoadBalanceCosts diagnostic <running-cpp-parameters-diagnostics>`
* :ref:`Parallelization workflow <parallelization_warpx>`
