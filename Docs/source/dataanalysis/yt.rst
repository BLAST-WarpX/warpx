.. _dataanalysis-amrex-plotfiles:
.. _dataanalysis-yt:

Read AMReX Plotfiles with yt
============================

`yt <http://yt-project.org/>`__ is a Python package that can help in analyzing and visualizing WarpX data (among other data formats).
It is convenient to use yt within a `Jupyter notebook <http://jupyter.org/>`__.

Data Support
------------

yt primarily supports WarpX through plotfiles.
There is also support for openPMD HDF5 files in yt (w/o mesh refinement).

Installation
------------

From the terminal, install the latest version of yt:

.. code-block:: bash

    python3 -m pip install cython
    python3 -m pip install --upgrade yt

Alternatively, yt can be installed via their installation script, see `yt installation web page <https://yt-project.org/doc/installing.html>`__.

Visualizing the data
--------------------

Once data ("plotfiles") has been created by the simulation, open a Jupyter notebook from
the terminal:

.. code-block:: bash

    jupyter notebook

Then use the following commands in the first cell of the notebook to import yt
and load the first plot file:

.. code-block:: python

    import yt
    ds = yt.load('./diags/plotfiles/plt00000/')

The list of field data and particle data stored can be seen with:

.. code-block:: python

    ds.field_list

For a quick start-up, the most useful commands for post-processing can be found
in our Jupyter notebook
:download:`Visualization.ipynb<../../../Tools/PostProcessing/Visualization.ipynb>`

Field data
~~~~~~~~~~

Field data can be visualized using ``yt.SlicePlot`` (see the docstring of
this function `here <http://yt-project.org/doc/reference/api/yt.visualization.plot_window.html#yt.visualization.plot_window.SlicePlot>`__)

For instance, in order to plot the field ``Ex`` in a slice orthogonal to ``y`` (``1``):

.. code-block:: python

    yt.SlicePlot( ds, 1, 'Ex', origin='native' )

.. note::

    `yt.SlicePlot` creates a 2D plot with the same aspect ratio as the physical
    size of the simulation box. Sometimes this can lead to very elongated plots
    that are difficult to read. You can modify the aspect ratio with the
    `aspect` argument ; for instance:

    .. code-block:: python

        yt.SlicePlot( ds, 1, 'Ex', aspect=1./10 )


Alternatively, the data can be obtained as a `numpy <http://www.numpy.org/>`__ array.

For instance, in order to obtain the field `jz` (on level 0) as a numpy array:

.. code-block:: python

    ad0 = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions)
    jz_array = ad0['jz'].to_ndarray()


Particle data
~~~~~~~~~~~~~

Particle data can be visualized using ``yt.ParticlePhasePlot`` (see the docstring
`here <http://yt-project.org/doc/reference/api/yt.visualization.particle_plots.html?highlight=particlephaseplot#yt.visualization.particle_plots.ParticlePhasePlot>`__).

For instance, in order to plot the particles' ``x`` and ``y`` positions:

.. code-block:: python

    yt.ParticlePhasePlot( ds.all_data(), 'particle_position_x', 'particle_position_y', 'particle_weight')

Alternatively, the data can be obtained as a `numpy <http://www.numpy.org/>`__ array.

For instance, in order to obtain the array of position `x` as a numpy array:

.. code-block:: python

    ad = ds.all_data()
    x = ad['particle_position_x'].to_ndarray()

Further information
-------------------

A lot more information can be obtained from the yt documentation, and the
corresponding notebook tutorials `here <http://yt-project.org/doc/>`__.

.. toctree::
   :maxdepth: 2

   plot_parallel
   advanced

In the WarpX repository you can find other examples of visualization scripts, e.g., the serial script :download:`video_yt.py <../../../Tools/PostProcessing/video_yt.py>` and the parallel script :download:`yt3d_mpi.py <../../../Tools/PostProcessing/yt3d_mpi.py>`.
