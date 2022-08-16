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
