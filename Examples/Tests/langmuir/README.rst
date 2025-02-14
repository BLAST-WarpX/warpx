.. _examples-langmuir:

Langmuir Waves
==============

These are examples of Plasma oscillations (`Langmuir waves <https://en.wikipedia.org/wiki/Plasma_oscillation>`__) in a uniform plasma in 1D, 2D, 3D, and RZ.

In each case, a uniform plasma is setup with a sinusoidal perturbation in the electron momentum along each axis.
The plasma is followed for a short period of time, long enough so that E fields develop.
The resulting fields can be compared to the analytic solutions.


Run
---

For `MPI-parallel <https://www.mpi-forum.org>`__ runs, prefix these lines with ``mpiexec -n 4 ...`` or ``srun -n 4 ...``, depending on the system.

.. tab-set::

   .. tab-item:: 3D

      .. tab-set::

         .. tab-item:: Python: Script

            This example can be run as a **Python** script: ``python3 inputs_test_3d_langmuir_multi_picmi.py``.

            .. literalinclude:: inputs_test_3d_langmuir_multi_picmi.py
               :language: python3
               :caption: You can copy this file from ``Examples/Tests/langmuir/inputs_test_3d_langmuir_multi_picmi.py``.

         .. tab-item:: Executable: Input File

            This example can be run as WarpX **executable** using an input file: ``warpx.3d inputs_test_3d_langmuir_multi``

            .. literalinclude:: inputs_test_3d_langmuir_multi
               :language: ini
               :caption: You can copy this file from ``Examples/Tests/langmuir/inputs_test_3d_langmuir_multi``.

   .. tab-item:: 2D

      .. tab-set::

         .. tab-item:: Python: Script

            This example can be run as a **Python** script: ``python3 inputs_test_2d_langmuir_multi_picmi.py``.

            .. literalinclude:: inputs_test_2d_langmuir_multi_picmi.py
               :language: python3
               :caption: You can copy this file from ``Examples/Tests/langmuir/inputs_test_2d_langmuir_multi_picmi.py``.

         .. tab-item:: Executable: Input File

            This example can be run as WarpX **executable** using an input file: ``warpx.2d inputs_test_2d_langmuir_multi``

            .. literalinclude:: inputs_test_2d_langmuir_multi
               :language: ini
               :caption: You can copy this file from ``Examples/Tests/langmuir/inputs_test_2d_langmuir_multi``.


   .. tab-item:: RZ

      .. tab-set::

         .. tab-item:: Python: Script

            This example can be run as a **Python** script: ``python3 inputs_test_rz_langmuir_multi_picmi.py``.

            .. literalinclude:: inputs_test_rz_langmuir_multi_picmi.py
               :language: python3
               :caption: You can copy this file from ``Examples/Tests/langmuir/inputs_test_rz_langmuir_multi_picmi.py``.

         .. tab-item:: Executable: Input File

            This example can be run as WarpX **executable** using an input file: ``warpx.rz inputs_test_rz_langmuir_multi``

            .. literalinclude:: inputs_test_rz_langmuir_multi
               :language: ini
               :caption: You can copy this file from ``Examples/Tests/langmuir/inputs_test_rz_langmuir_multi``.


   .. tab-item:: 1D

      .. tab-set::

         .. tab-item:: Python: Script

            .. note::

               TODO: This input file should be created, like the ``inputs_test_1d_langmuir_multi`` file.

         .. tab-item:: Executable: Input File

            This example can be run as WarpX **executable** using an input file: ``warpx.1d inputs_test_1d_langmuir_multi``

            .. literalinclude:: inputs_test_1d_langmuir_multi
               :language: ini
               :caption: You can copy this file from ``Examples/Tests/langmuir/inputs_test_1d_langmuir_multi``.


Analyze
-------

We run the following script to analyze correctness:

.. tab-set::

   .. tab-item:: 3D

      .. dropdown:: Script ``analysis_3d.py``

         .. literalinclude:: analysis_3d.py
            :language: python3
            :caption: You can copy this file from ``Examples/Tests/langmuir/analysis_3d.py``.

   .. tab-item:: 2D

      .. dropdown:: Script ``analysis_2d.py``

         .. literalinclude:: analysis_2d.py
            :language: python3
            :caption: You can copy this file from ``Examples/Tests/langmuir/analysis_2d.py``.

   .. tab-item:: RZ

      .. dropdown:: Script ``analysis_rz.py``

         .. literalinclude:: analysis_rz.py
            :language: python3
            :caption: You can copy this file from ``Examples/Tests/langmuir/analysis_rz.py``.

   .. tab-item:: 1D

      .. dropdown:: Script ``analysis_1d.py``

         .. literalinclude:: analysis_1d.py
            :language: python3
            :caption: You can copy this file from ``Examples/Tests/langmuir/analysis_1d.py``.


Visualize
---------

.. note::

   This section is TODO.
