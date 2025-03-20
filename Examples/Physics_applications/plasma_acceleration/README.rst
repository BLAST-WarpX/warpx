.. _examples-pwfa:

Beam-Driven Wakefield Acceleration of Electrons
===============================================

This example shows how to model a beam-driven plasma-wakefield accelerator (PWFA) :cite:p:`ex-TajimaDawson1982,ex-Esarey1996`.

PWFA is best performed in 3D or quasi-cylindrical (RZ) geometry, in order to correctly capture some of the key physics (structure of the space-charge fields, beamloading, shape of the accelerating bubble in the blowout regime, etc.).
For physical situations that have close-to-cylindrical symmetry, simulations in RZ geometry capture the relevant physics at a fraction of the computational cost of a 3D simulation.
On the other hand, for physical situation with strong asymmetries (e.g., non-round driver, strong hosing of the accelerated beam, etc.), only 3D simulations are suitable.

Additionally, to speed up computation, this example uses the :ref:`boosted frame method <theory-boostedframe>` to effectively model long acceleration lengths.

Alternatively, an other common approximation for PWFAs is quasi-static modeling, e.g., if effects such as self-injection can be ignored.
In the Beam, Plasma & Accelerator Simulation Toolkit (BLAST), `HiPACE++ <https://hipace.readthedocs.io>`__ provides such methods.

.. note::

   TODO: The Python (PICMI) input file should use the boosted frame method, like the ``inputs_test_3d_plasma_acceleration_boosted`` file.


Run
---

This example can be run **either** as:

* **Python** script: ``python3 inputs_test_3d_plasma_acceleration_picmi.py`` or
* WarpX **executable** using an input file: ``warpx.3d inputs_test_3d_plasma_acceleration_boosted``

For `MPI-parallel <https://www.mpi-forum.org>`__ runs, prefix these lines with ``mpiexec -n 4 ...`` or ``srun -n 4 ...``, depending on the system.

.. tab-set::

   .. tab-item:: Python: Script

      .. note::

         TODO: This input file should use the boosted frame method, like the ``inputs_test_3d_plasma_acceleration_boosted`` file.

      .. literalinclude:: inputs_test_3d_plasma_acceleration_picmi.py
         :language: python3
         :caption: You can copy this file from ``Examples/Physics_applications/plasma_acceleration/inputs_test_3d_plasma_acceleration_picmi.py``.

   .. tab-item:: Executable: Input File

      .. literalinclude:: inputs_test_3d_plasma_acceleration_boosted
         :language: ini
         :caption: You can copy this file from ``Examples/Physics_applications/plasma_acceleration/inputs_test_3d_plasma_acceleration_boosted``.

Analyze
-------

.. note::

   This section is TODO.


Visualize
---------

.. note::

   This section is TODO.
