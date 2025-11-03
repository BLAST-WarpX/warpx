.. _examples-capacitive-discharge:

Capacitive Discharge
====================

The examples in this directory are based on the benchmark cases from Turner et al. (Phys. Plasmas 20, 013507, 2013) :cite:p:`ex-Turner2013`.

The Monte-Carlo collision (MCC) model can be used to simulate electron and ion collisions with a neutral background gas.
In particular this can be used to study capacitive discharges between parallel plates.
The implementation has been tested against the benchmark results from :cite:t:`ex-Turner2013`.

.. note::

   This example needs `additional calibration data for cross sections <https://github.com/BLAST-WarpX/warpx-data>`__.
   Download this data alongside your inputs file and update the paths in the inputs file:

   .. code-block:: bash

      git clone https://github.com/BLAST-WarpX/warpx-data.git


Run
---

The 1D PICMI input file can be used to reproduce the results from Turner et al. for a given case, ``N`` from 1 to 4, by executing ``python3 inputs_base_1d_picmi.py -n N``, e.g.,

.. code-block:: bash

   python3 inputs_base_1d_picmi.py -n 1

For `MPI-parallel <https://www.mpi-forum.org>`__ runs, prefix these lines with ``mpiexec -n 4 ...`` or ``srun -n 4 ...``, depending on the system.

.. literalinclude:: inputs_base_1d_picmi.py
   :language: python3
   :caption: You can copy this file from ``Examples/Physics_applications/capacitive_discharge/inputs_base_1d_picmi.py``.


Analyze
-------

Once the simulation completes an output file ``avg_ion_density.npy`` will be created which can be compared to the literature results as in the plot below.
Running case ``1`` on four CPU processors takes roughly 20 minutes to complete.


Visualize
---------

The figure below shows a comparison of the ion density as calculated in WarpX (in June 2022 with `PR #3118 <https://github.com/BLAST-WarpX/warpx/pull/3118>`_) compared to the literature results (which can be found `in the supplementary materials of Turner et al. <https://aip.scitation.org/doi/suppl/10.1063/1.4775084>`__).

.. figure:: https://user-images.githubusercontent.com/40245517/171573007-f7d733c7-c0de-490c-9ed6-ff4c02154358.png
   :alt: MCC benchmark against :cite:t:`ex-Turner2013`.
   :width: 80%

   MCC benchmark against :cite:t:`ex-Turner2013`.
