.. _examples-laser-ion:

Laser-Ion Acceleration with a Planar Target
===========================================

This example shows how to model laser-ion acceleration with planar targets of solid density :cite:p:`ex-Wilks2001,ex-Bulanov2008,ex-Macchi2013`.
The acceleration mechanism in this scenario depends on target parameters.

Although laser-ion acceleration requires full 3D modeling for adequate description of the acceleration dynamics, especially the acceleration field lengths and decay times, this example models a 2D example.
2D modeling can often hint at a qualitative overview of the dynamics, but mostly saves computational costs since the plasma frequency (and Debye length) of the plasma determines the resolution need in laser-solid interaction modeling.

.. note::

   The resolution of this 2D case is extremely low by default.
   This includes spatial and temporal resolution, but also the number of macro-particles per cell representing the target density for proper phase space sampling.
   You will need a computing cluster for adequate resolution of the target density, see comments in the input file.


Run
---

This example can be run **either** as:

* **Python** script: ``mpiexec -n 2 python3 inputs_test_2d_laser_ion_acc_picmi.py`` or
* WarpX **executable** using an input file: ``mpiexec -n 2 warpx.2d inputs_test_2d_laser_ion_acc``

.. tip::

   For `MPI-parallel <https://www.mpi-forum.org>`__ runs on computing clusters, change the prefix to ``mpiexec -n <no. of MPI ranks> ...`` or ``srun -n <no. of MPI ranks> ...``, depending on the system and number of MPI ranks you want to allocate.

   The input option ``warpx_numprocs`` / ``warpx.numprocs`` needs to be adjusted for parallel :ref:`domain decomposition <usage_domain_decomposition>`, to match the number of MPI ranks used.
   In order to use dynamic load balancing, use the :ref:`more general method <usage_domain_decomposition-general>` of setting blocks.

.. tab-set::

   .. tab-item:: Python: Script

      .. literalinclude:: inputs_test_2d_laser_ion_acc_picmi.py
         :language: python3
         :caption: You can copy this file from ``Examples/Physics_applications/laser_ion/inputs_test_2d_laser_ion_acc_picmi.py``.


   .. tab-item:: Executable: Input File

      .. literalinclude:: inputs_test_2d_laser_ion_acc
         :language: ini
         :caption: You can copy this file from ``Examples/Physics_applications/laser_ion/inputs_test_2d_laser_ion_acc``.

Analyze
-------

.. _fig-tnsa-ps-electrons-pinhole:

.. figure:: https://user-images.githubusercontent.com/5416860/295003882-c755fd47-4bb3-4439-9319-c48214cbaafd.png
   :alt: Longitudinal phase space of forward-moving electrons in a 2 degree opening angle.
   :width: 100%

   Longitudinal phase space of forward-moving electrons in a 2 degree opening angle.

.. _fig-tnsa-ps-protons-pinhole:

.. figure:: https://user-images.githubusercontent.com/5416860/295003988-dea3dfb7-0d55-4616-b32d-061fb429f9ac.png
   :alt: Longitudinal phase space of forward-moving protons in a 2 degree opening angle.
   :width: 100%

   Longitudinal phase space of forward-moving protons in a 2 degree opening angle.

Time-resolved phase electron space analysis as in :numref:`fig-tnsa-ps-electrons-pinhole` gives information about, e.g., how laser energy is locally converted into electron kinetic energy.
Later in time, ion phase spaces like :numref:`fig-tnsa-ps-protons-pinhole` can reveal where accelerated ion populations originate.

.. dropdown:: Script ``analysis_histogram_2D.py``

   .. literalinclude:: analysis_histogram_2D.py
      :language: python3
      :caption: You can copy this file from ``Examples/Physics_applications/laser_ion/analysis_histogram_2D.py``.

Visualize
---------

.. note::

   The following images for densities and electromagnetic fields were created with a run on 64 NVidia A100 GPUs featuring a total number of cells of ``nx = 8192`` and ``nz = 16384``, as well as 64 particles per cell per species.

.. _fig-tnsa-densities:

.. figure:: https://user-images.githubusercontent.com/5416860/296338802-8059c39c-0be8-4e4d-b41b-f976b626bd7f.png
   :alt: Particle densities for electrons (top), protons (middle), and electrons again in logarithmic scale (bottom).
   :width: 80%

   Particle densities for electrons (top), protons (middle), and electrons again in logarithmic scale (bottom).

Particle density output illustrates the evolution of the target in time and space.
Logarithmic scales can help to identify where the target becomes transparent for the laser pulse (bottom panel in :numref:`fig-tnsa-densities` ).

.. _fig-tnsa-fields:

.. figure:: https://user-images.githubusercontent.com/5416860/296338609-a49eee7f-6793-4b55-92f1-0b887e6437ab.png
   :alt: Electromagnetic field visualization for E_x (top), B_y (middle), and E_z (bottom).
   :width: 80%

   Electromagnetic field visualization for :math:`E_x` (top), :math:`B_y` (middle), and :math:`E_z` (bottom).

Electromagnetic field output shows where the laser field is strongest at a given point in time, and where accelerating fields build up :numref:`fig-tnsa-fields`.

.. dropdown:: Script ``plot_2d.py``

   .. literalinclude:: plot_2d.py
      :language: python3
      :caption: You can copy this file from ``Examples/Physics_applications/laser_ion/plot_2d.py``.
