.. _examples-plasma-mirror:

Plasma-Mirror
=============

This example shows how to model a plasma mirror, using a planar target of solid density :cite:p:`ex-Dromey2004,ex-Roedel2010`.

Although laser-solid interaction modeling requires full 3D modeling for adequate description of the dynamics at play, this example models a 2D example.
2D modeling provide a qualitative overview of the dynamics, but mostly saves computational costs since the plasma frequency (and Debye length) of the surface plasma determines the resolution need in laser-solid interaction modeling.

.. note::

   TODO: The Python (PICMI) input file needs to be created.


Run
---

This example can be run **either** as:

* **Python** script: (*TODO*) or
* WarpX **executable** using an input file: ``warpx.2d inputs_2d``

For `MPI-parallel <https://www.mpi-forum.org>`__ runs, prefix these lines with ``mpiexec -n 4 ...`` or ``srun -n 4 ...``, depending on the system.

.. tab-set::

   .. tab-item:: Python: Script

      .. note::

         TODO: This input file should be created following the ``inputs_2d`` file.

   .. tab-item:: Executable: Input File

      .. literalinclude:: inputs_test_2d_plasma_mirror
         :language: ini
         :caption: You can copy this file from ``Examples/Physics_applications/plasma_mirror/inputs_test_2d_plasma_mirror``.


Analyze
-------

.. note::

   This section is TODO.


Visualize
---------

.. note::

   This section is TODO.
