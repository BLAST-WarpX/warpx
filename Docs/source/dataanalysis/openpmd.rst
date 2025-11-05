.. _dataanalysis-openpmd:

Read openPMD Data
=================

openPMD is implemented in popular community formats such as `ADIOS <https://csmd.ornl.gov/adios>`__ and `HDF5 <https://www.hdfgroup.org/>`__.

To start, there are two popular Python libraries to interact with openPMD data:

.. toctree::
   :maxdepth: 1

   openpmdviewer
   openpmdapi

The viewer provides a simple python interface to open particle and field data, and perform other more advanced analysis, such as reconstructing the trajectories of selected particles.
However, not all the original openPMD metadata is provided. This is not a limitation in many cases, e.g. when analyzing simulation results.
The implementation is serial, for now.

The api provides a more sophisticated C++ and python interface to open particle and field diagnostics, and allows full control of the original data.
It's particularly useful for debugging purposes or to write a new dataset.
The data can be handled in parallel in chunks.

openPMD-api also enables seamless coupling to `Pandas <https://openpmd-api.readthedocs.io/en/latest/analysis/pandas.html>`__, `DASK <https://openpmd-api.readthedocs.io/en/latest/analysis/dask.html>`__, `RAPIDS <https://openpmd-api.readthedocs.io/en/latest/analysis/rapids.html>`__ and `other frameworks <https://openpmd-api.readthedocs.io/en/latest/analysis/contrib.html>`__.

Furthermore, consider with our :ref:`3D visualization <dataanalysis-3dvisualizations>` section for openPMD support in those.
