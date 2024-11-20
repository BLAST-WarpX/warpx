.. _examples-lwfa:

Laser-Wakefield Acceleration of Electrons
=========================================

This example shows how to model a laser-wakefield accelerator (LWFA) :cite:p:`ex-TajimaDawson1982,ex-Esarey1996`.

Laser-wakefield acceleration is best performed in 3D or quasi-cylindrical (RZ) geometry, in order to correctly capture some of the key physics (laser diffraction, beamloading, shape of the accelerating bubble in the blowout regime, etc.).
For physical situations that have close-to-cylindrical symmetry, simulations in RZ geometry capture the relevant physics at a fraction of the computational cost of a 3D simulation.
On the other hand, for physical situation with strong asymmetries (e.g., non-round laser driver, strong hosing of the accelerated beam, etc.), only 3D simulations are suitable.

For LWFA scenarios with long propagation lengths, use the :ref:`boosted frame method <theory-boostedframe>`.
An example can be seen in the :ref:`PWFA example <examples-pwfa>`.


Run
---

For `MPI-parallel <https://www.mpi-forum.org>`__ runs, prefix these lines with ``mpiexec -n 4 ...`` or ``srun -n 4 ...``, depending on the system.

.. tab-set::

   .. tab-item:: 3D

      This example can be run **either** as:

      * **Python** script: ``python3 inputs_test_3d_laser_acceleration_picmi.py`` or
      * WarpX **executable** using an input file: ``warpx.3d inputs_test_3d_laser_acceleration max_step=400``

      .. tab-set::

         .. tab-item:: Python: Script

             .. literalinclude:: inputs_test_3d_laser_acceleration_picmi.py
                :language: python3
                :caption: You can copy this file from ``Examples/Physics_applications/laser_acceleration/inputs_test_3d_laser_acceleration_picmi.py``.

         .. tab-item:: Executable: Input File

             .. literalinclude:: inputs_test_3d_laser_acceleration
                :language: ini
                :caption: You can copy this file from ``Examples/Physics_applications/laser_acceleration/inputs_test_3d_laser_acceleration``.

   .. tab-item:: RZ

      This example can be run **either** as:

      * **Python** script: ``python3 inputs_test_rz_laser_acceleration_picmi.py`` or
      * WarpX **executable** using an input file: ``warpx.rz inputs_test_rz_laser_acceleration max_step=400``

      .. tab-set::

         .. tab-item:: Python: Script

             .. literalinclude:: inputs_test_rz_laser_acceleration_picmi.py
                :language: python3
                :caption: You can copy this file from ``Examples/Physics_applications/laser_acceleration/inputs_test_rz_laser_acceleration_picmi.py``.

         .. tab-item:: Executable: Input File

             .. literalinclude:: inputs_test_rz_laser_acceleration
                :language: ini
                :caption: You can copy this file from ``Examples/Physics_applications/laser_acceleration/inputs_test_rz_laser_acceleration``.

Analyze
-------

.. note::

   This section is TODO.


Visualize
---------

You can run the following script to visualize the beam evolution over time:

.. dropdown:: Script ``plot_3d.py``

   .. literalinclude:: plot_3d.py
      :language: python3
      :caption: You can copy this file from ``Examples/Physics_applications/laser_acceleration/plot_3d.py diags/diag1000400/``.

.. figure:: https://user-images.githubusercontent.com/1353258/287800852-f994a020-4ecc-4987-bffc-2cb7df6144a9.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTEiLCJleHAiOjE3MDE3MTYyNjksIm5iZiI6MTcwMTcxNTk2OSwicGF0aCI6Ii8xMzUzMjU4LzI4NzgwMDg1Mi1mOTk0YTAyMC00ZWNjLTQ5ODctYmZmYy0yY2I3ZGY2MTQ0YTkucG5nP1gtQW16LUFsZ29yaXRobT1BV1M0LUhNQUMtU0hBMjU2JlgtQW16LUNyZWRlbnRpYWw9QUtJQUlXTkpZQVg0Q1NWRUg1M0ElMkYyMDIzMTIwNCUyRnVzLWVhc3QtMSUyRnMzJTJGYXdzNF9yZXF1ZXN0JlgtQW16LURhdGU9MjAyMzEyMDRUMTg1MjQ5WiZYLUFtei1FeHBpcmVzPTMwMCZYLUFtei1TaWduYXR1cmU9NDkyNWJkMTg2NWM3ZjcwZjVkMjlmNDE1NmRjNWEyZWM5MzgxMWJhZTVjMGMxNjdkZDg1Zjk0NmQ1NGEwMjNiMiZYLUFtei1TaWduZWRIZWFkZXJzPWhvc3QmYWN0b3JfaWQ9MCZrZXlfaWQ9MCZyZXBvX2lkPTAifQ.C_NQceQcqiCDzBoSzIjm3c8QdTLNDdtjJmkQjkhW4c8
   :alt: (top) Electric field of the laser pulse and (bottom) absolute density.

   (top) Electric field of the laser pulse and (bottom) absolute density.
