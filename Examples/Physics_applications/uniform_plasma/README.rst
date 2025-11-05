.. _examples-uniform-plasma:

Uniform Plasma
==============

This example evolves a uniformly distributed, hot plasma over time.


Run
---

For `MPI-parallel <https://www.mpi-forum.org>`__ runs, prefix these lines with ``mpiexec -n 4 ...`` or ``srun -n 4 ...``, depending on the system.

.. tab-set::

   .. tab-item:: 3D

      .. tab-set::

         .. tab-item:: Inputs: Parameter List

            This example can be run as WarpX **executable** using an input file: ``warpx.3d inputs_base_3d``

             .. literalinclude:: inputs_base_3d
                :language: none
                :caption: You can copy this file from ``Examples/Physics_applications/uniform_plasma/inputs_base_3d``.

   .. tab-item:: 2D

      .. tab-set::

         .. tab-item:: Inputs: Parameter List

            This example can be run as WarpX **executable** using an input file: ``warpx.2d inputs_test_2d_uniform_plasma``

             .. literalinclude:: inputs_test_2d_uniform_plasma
                :language: none
                :caption: You can copy this file from ``Examples/Physics_applications/uniform_plasma/inputs_test_2d_uniform_plasma``.
