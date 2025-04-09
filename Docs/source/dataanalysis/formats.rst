.. _dataanalysis-formats:

Output formats
==============

WarpX can :ref:`write diagnostics data <running-cpp-parameters>` either in

* `plotfile format <https://amrex-codes.github.io/amrex/docs_html/IO.html>`__ or in
* `openPMD format(s) <https://www.openpmd.org/>`__.

Plotfiles are AMReX' native data format, while openPMD is implemented in popular community formats such as `ADIOS <https://csmd.ornl.gov/adios>`__ and `HDF5 <https://www.hdfgroup.org/>`__.

This section describes some of the tools available to visualize the data.

Asynchronous IO
---------------

When using the AMReX `plotfile` format, users can set the ``amrex.async_out=1``
option to perform the IO in a non-blocking fashion, meaning that the simulation
will continue to run while an IO thread controls writing the data to disk.
This can significantly reduce the overall time spent in IO. This is primarily intended for
large runs on supercomputers (e.g. at OLCF or NERSC); depending on the MPI
implementation you are using, you may not see a benefit on your workstation.

When writing plotfiles, each rank will write to a separate file, up to some maximum number
(by default, 64). This maximum can be adjusted using the ``amrex.async_out_nfiles`` inputs
parameter. To use asynchronous IO with than ``amrex.async_out_nfiles`` MPI ranks, WarpX
WarpX must be configured with ``-DWarpX_MPI_THREAD_MULTIPLE=ON``.
Please see :ref:`the building instructions <install-developers>` for details.

In Situ Capabilities
--------------------

WarpX includes so-called :ref:`reduced diagnostics <running-cpp-parameters-diagnostics-reduced>`.
Reduced diagnostics create observables on-the-fly, such as energy histograms or particle beam statistics and are :ref:`easily visualized in post-processing <dataanalysis-reduced-diagnostics>`.

In addition, WarpX also has vn-situ visualization capabilities (i.e.
visualizing the data directly from the simulation, without dumping data
files to disk).

.. toctree::
   :maxdepth: 1

   ascent
   catalyst
   sensei

If you like the 3D rendering of laser wakefield acceleration
on the WarpX documentation front page (which is
also the avatar of the ECP-WarpX organization), you can find the serial
analysis script :download:`video_yt.py <../../../Tools/PostProcessing/video_yt.py>` as well
as a parallel analysis script
:download:`video_yt.py <../../../Tools/PostProcessing/yt3d_mpi.py>` used to make a similar
rendering for a beam-driven wakefield simulation, running parallel.

Staggering in Data Output
-------------------------

Warning: currently, quantities in the output file for iteration ``n`` are not all defined at the same physical time due to the staggering in time in WarpX.
The table below provides the physical time at which each quantity in the output file is written, in units of time step, for time step ``n``.

======== ===== =====
quantity staggering
-------- -----------
         FDTD  PSATD
======== ===== =====
E        n     n
B        n     n
j        n-1/2 n-1/2
rho      n     n
position n     n
momentum n-1/2 n-1/2
======== ===== =====
