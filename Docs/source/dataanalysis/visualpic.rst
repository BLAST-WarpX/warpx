.. _dataanalysis-visualpic:

VisualPIC
=========

VisualPIC is an open-source Python GUI for visual data analysis, especially for advanced accelerator simulations.
It supports WarpX' data through openPMD files.

Installation
""""""""""""

.. code-block:: bash

   mamba install -c conda-forge python vtk pyvista pyqt
   python3 -m pip install git+https://github.com/AngelFP/VisualPIC.git@dev


Usage
"""""

VisualPIC provides a Python data reader API and plotting capabilties.
It is designed for small to medium-size data sets that fit in the RAM of a single computer.

Plotting can be performed via a command line tools or scripted with Python.
**Command line tools** are:

* ``vpic [options] <path/to/diagnostics/>``: 2D matplotlib plotter, e.g., for particle phase space
* ``vpic3d [options] <path/to/diagnostics/>``: 3D VTK renderer

Example: ``vpic3d -s beam -rho -Ez diags/diag1/`` could be used to visualize the witness beam, plasma density, and accelerating field of an LWFA.

Example: ``vpic3d -Ex diags/diag1/`` could be used to visualize the transverse focusing field :math:`E_x` in a plasma wake behind a laser pulse (linearly polarized in :math:`E_y`), see below:

.. figure:: https://user-images.githubusercontent.com/1353258/233236692-4d75b12f-de44-43dc-97bd-c96b04ee68ac.png
   :alt: Example view of a 3D rendering with VisualPIC.
   :width: 100%

The **Python script** controlled rendering allows more flexible options, such as selecting and cutting views, rendering directly into an image file, looping for animations, etc.
As with matplotlib scripts, Python script scenes can also be used to open a GUI and then browse time series interactively.
The `VisualPIC examples <https://github.com/AngelFP/VisualPIC/tree/dev/examples>`__ provide showcases for scripting.


Repository
""""""""""

The source code can be found under:
  https://github.com/AngelFP/VisualPIC
