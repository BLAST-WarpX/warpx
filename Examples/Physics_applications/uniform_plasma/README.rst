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

         .. tab-item:: Python: Script

            .. note::

               TODO: This input file should be created following the ``inputs_test_3d_uniform_plasma`` file.

         .. tab-item:: Executable: Input File

            This example can be run **either** as WarpX **executable** using an input file: ``warpx.3d inputs_test_3d_uniform_plasma``

             .. literalinclude:: inputs_test_3d_uniform_plasma
                :language: ini
                :caption: You can copy this file from ``usage/examples/lwfa/inputs_test_3d_uniform_plasma``.

   .. tab-item:: 2D

      .. tab-set::

         .. tab-item:: Python: Script

            .. note::

               TODO: This input file should be created following the ``inputs_test_2d_uniform_plasma`` file.

         .. tab-item:: Executable: Input File

            This example can be run **either** as WarpX **executable** using an input file: ``warpx.2d inputs_test_2d_uniform_plasma``

             .. literalinclude:: inputs_test_2d_uniform_plasma
                :language: ini
                :caption: You can copy this file from ``usage/examples/lwfa/inputs_test_2d_uniform_plasma``.

Analyze
-------

.. note::

   This section is TODO.


Visualize
---------

.. note::

   This section is TODO.
