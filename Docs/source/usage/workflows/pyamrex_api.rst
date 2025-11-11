Using AMReX functionalities through pyAMReX
-------------------------------------------

WarpX uses the AMReX library for its grid and particle data structures.
The Python interface to AMReX is provided through `pyAMReX <https://github.com/AMReX-Codes/pyamrex>`__.

After the simulation is initialized, the pyAMReX module can be accessed (if needed) via

.. code-block:: python

   from pywarpx import picmi, libwarpx

   # ... simulation definition ...