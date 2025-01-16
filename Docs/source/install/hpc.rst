.. _install-hpc:

HPC
===

On selected high-performance computing (HPC) systems, WarpX has documented or even pre-build installation routines.
Follow the guide here instead of the generic installation routines for optimal stability and best performance.


.. _install-hpc-profile:

warpx.profile
-------------

Use a ``warpx.profile`` file to set up your software environment without colliding with other software.
Ideally, store that file directly in your ``$HOME/`` and source it after connecting to the machine:

.. code-block:: bash

   source $HOME/warpx.profile

We list example ``warpx.profile`` files below, which can be used to set up WarpX on various HPC systems.


.. _install-hpc-machines:

HPC Machines
------------

This section documents quick-start guides for a selection of supercomputers that WarpX users are active on.

.. toctree::
   :maxdepth: 1

   hpc/adastra
   hpc/crusher
   hpc/frontier
   hpc/fugaku
   hpc/greatlakes
   hpc/hpc3
   hpc/juwels
   hpc/karolina
   hpc/lassen
   hpc/lawrencium
   hpc/leonardo
   hpc/lonestar6
   hpc/lumi
   hpc/lxplus
   hpc/ookami
   hpc/perlmutter
   hpc/pitzer
   hpc/polaris
   hpc/dane
   hpc/summit
   hpc/taurus
   hpc/tioga

.. tip::

   Your HPC system is not in the list?
   `Open an issue <https://github.com/ECP-WarpX/WarpX/issues>`__ and together we can document it!


.. _install-hpc-batch:

Batch Systems
-------------

HPC systems use a scheduling ("batch") system for time sharing of computing resources.
The batch system is used to request, queue, schedule and execute compute jobs asynchronously.
The individual HPC machines above document job submission example scripts, as templates for your modifications.

In this section, we document a quick reference guide (or cheat sheet) to interact in more detail with the various batch systems that you might encounter on different systems.

Slurm
"""""

Slurm is a modern and very popular batch system.
Slurm is used at NERSC, OLCF Frontier, among others.

.. include:: batch/slurm.rst


LSF
"""

LSF (for *Load Sharing Facility*) is an IBM batch system.
It is used at OLCF Summit, LLNL Lassen, and other IBM systems.

.. include:: batch/lsf.rst


PBS
"""

PBS (for *Portable Batch System*) is a popular HPC batch system.
The OpenPBS project is related to `PBS, PBS Pro and TORQUE <https://en.wikipedia.org/wiki/Portable_Batch_System>`__.

.. include:: batch/pbs.rst


PJM
"""

PJM (probably for *Parallel Job Manager*?) is a Fujitsu batch system
It is used at RIKEN Fugaku and on other Fujitsu systems.

.. include:: batch/pjm.rst
