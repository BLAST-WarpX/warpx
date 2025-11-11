Using AMReX functionalities through pyAMReX
-------------------------------------------

The Python interface to WarpX is provided through `pyAMReX <https://github.com/AMReX-Codes/pyamrex>`__.
After the simulation is initialized, the pyAMReX module can be accessed (if needed) via

.. code-block:: python

   from pywarpx import picmi, libwarpx

   # ... simulation definition ...

   # equivalent to
   #   import amrex.space3d as amr
   # for a 3D simulation
   amr = libwarpx.amr  # picks the right 1d, 2d or 3d variant


Full details for pyAMReX APIs are `documented here <https://pyamrex.readthedocs.io/en/latest/usage/api.html>`__.
The major objects used in the WarpX interface will be of types defined by pyAMReX.
Important APIs include:

* `amr.ParallelDescriptor <https://pyamrex.readthedocs.io/en/latest/usage/api.html#amrex.space3d.ParallelDescriptor.IOProcessor>`__: MPI-parallel rank information
* `amr.MultiFab <https://pyamrex.readthedocs.io/en/latest/usage/api.html#amrex.space3d.MultiFab>`__: MPI-parallel field data
* `amr.ParticleContainer_* <https://pyamrex.readthedocs.io/en/latest/usage/api.html#amrex.space3d.ParticleContainer_1_1_2_1_default>`__: MPI-parallel particle data for a particle species
