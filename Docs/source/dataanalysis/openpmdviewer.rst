.. _dataanalysis-openpmd-viewer:

openPMD-viewer
==============

`openPMD-viewer <https://github.com/openPMD/openPMD-viewer>`__ is an open-source Python package to access openPMD data.

It allows to:

* Quickly browse through the data, with a GUI-type interface in the Jupyter notebook
* Have access to the data numpy array, for more detailed analysis

Installation
------------

openPMD-viewer can be installed via ``conda`` or ``pip``:

.. code-block:: bash

    conda install -c conda-forge openpmd-viewer openpmd-api

.. code-block:: bash

    python3 -m pip install openPMD-viewer openPMD-api

Usage
-----

openPMD-viewer can be used either in simple Python scripts or in `Jupyter <https://jupyter.org>`__.
For interactive plots in Jupyter notebook or Jupyter Lab, add this `"cell magic" <https://ipython.readthedocs.io/en/stable/interactive/magics.html>`__ to the first line of your notebook:

.. code-block:: python

   %matplotlib widget

If none of those work, e.g. because `ipympl <https://github.com/matplotlib/ipympl#installation>`__ is not properly installed, you can as a last resort always try ``%matplotlib inline`` for non-interactive plots.

In both interactive and scripted usage, you can import openPMD-viewer, and load the data with the following commands:

.. code-block:: python

    from openpmd_viewer import OpenPMDTimeSeries
    ts = OpenPMDTimeSeries('./diags/diag1/')

.. note::

    If you are using the Jupyter notebook, then you can start a pre-filled
    notebook, which already contains the above lines, by typing in a terminal:

    ::

        openPMD_notebook

When using the Jupyter notebook, you can quickly browse through the data
by using the command:

::

    ts.slider()

You can also access the particle and field data as numpy arrays with the methods ``ts.get_field`` and ``ts.get_particle``.
See the openPMD-viewer tutorials `here <https://github.com/openPMD/openPMD-viewer/tree/dev/docs/source/tutorials>`_ for more info.
