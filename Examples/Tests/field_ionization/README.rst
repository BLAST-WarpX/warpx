.. _examples-tests-field_ionization:

Field Ionization
================

Run Test
--------

For `MPI-parallel <https://www.mpi-forum.org>`__ runs, prefix these lines with ``mpiexec -n 4 ...`` or ``srun -n 4 ...``, depending on the system.

.. tab-set::

   .. tab-item:: lab frame

        This example can be run **either** as:

        * **Python** script: ``python3 inputs_test_2d_ionization_picmi.py`` or
        * WarpX **executable** using an input file: ``warpx.2d inputs_test_2d_ionization_lab max_step=1600``

        .. tab-set::

            .. tab-item:: Python: Script

                .. literalinclude:: inputs_test_2d_ionization_picmi.py
                    :language: python3
                    :caption: You can copy this file from ``Examples/Tests/field_ionization/inputs_test_2d_ionization_picmi.py``.

            .. tab-item:: Executable: Input File

                .. literalinclude:: inputs_test_2d_ionization_lab
                    :language: ini
                    :caption: You can copy this file from ``Examples/Tests/field_ionization/inputs_test_2d_ionization_lab``.

   .. tab-item:: boosted frame

        This example can be run as:

        * WarpX **executable** using an input file: ``warpx.2d inputs_test_2d_ionization_boost max_step=420``

        .. literalinclude:: inputs_test_2d_ionization_boost
            :language: ini
            :caption: You can copy this file from ``Examples/Tests/field_ionization/inputs_test_2d_ionization_boost``.

Analyze
-------

.. dropdown:: Script ``analysis.py``

   .. literalinclude:: analysis.py
      :language: python3
      :caption: You can copy this file from ``Examples/Tests/field_ionization/analysis.py``.

Visualize
---------

.. figure:: https://gist.githubusercontent.com/johvandewetering/48d092c003915f1d1689b507caa2865b/raw/29f5d12ed77831047ca12f456a07dbf3b99770d5/image_ionization.png
   :alt: Electric field of the laser pulse with (top) ions with ionization levels and (bottom) ionized electrons.

   Electric field of the laser pulse with (top) ions with ionization levels and (bottom) ionized electrons.
