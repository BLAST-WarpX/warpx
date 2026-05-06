Advanced Visualization of Plotfiles With yt (for developers)
============================================================

This sections contains yt commands for advanced users. The Particle-In-Cell methods uses a
staggered grid (see :ref:`particle-in-cell theory <theory-pic>`), so that the x, y, and z components of the
electric and magnetic fields are all defined at different locations in space. Regular output
(see the :doc:`yt` page, or the notebook at ``WarpX/Tools/PostProcessing/Visualization.ipynb`` for an example)
returns cell-centered data for convenience, which involves an additional operation. It is sometimes
useful to access the raw data directly. Furthermore,
the WarpX implementation for mesh refinement contains a number of grids for each level (coarse,
fine and auxiliary, see :ref:`the theory <theory>` for more details), and it is sometimes useful to access each of
them (regular output return the auxiliary grid only). This page provides information to read
raw data of all grids.

Write Raw Data
--------------

For a given diagnostic the user has the option to write the raw data by setting :pp:param:`<diag_name>.plot_raw_fields = 1`.
Moreover, the user has the option to write also the values of the fields in the guard cells by setting :pp:param:`<diag_name>.plot_raw_fields_guards = 1`.
Please refer to :ref:`Input Parameters <running-cpp-parameters>` for more information.

Read Raw Data
-------------

Meta-data relevant to this topic (for example, number and locations of grids in the simulation) are accessed with

.. code-block:: python

    import yt
    # get yt dataset
    ds = yt.load( './plotfiles/plt00004' )
    # Index of data in the plotfile
    ds_index = ds.index
    # Print the number of grids in the simulation
    ds_index.grids.shape
    # Left and right physical boundary of each grid
    ds_index.grid_left_edge
    ds_index.grid_right_edge
    # List available fields
    ds.field_list

When :pp:param:`<diag_name>.plot_raw_fields = 1`, here are some useful commands to access properties of a grid and the Ex field on the fine patch:

.. code-block:: python

    # store grid number 2 into my_grid
    my_grid = ds.index.grids[2]
    # Get left and right edges of my_grid
    my_grid.LeftEdge
    my_grid.RightEdge
    # Get Level of my_grid
    my_grid.Level
    # left edge of the grid, in number of points
    my_grid.start_index

Return the ``Ex`` field on the fine patch of grid ``my_grid``:

.. code-block:: python

    my_field = my_grid['raw', 'Ex_fp'].squeeze().v

For a 2D plotfile, ``my_field`` has shape ``(nx,nz,2)``. The last component stands for the
two values on the edges of each cell for the electric field, due to field staggering. Numpy
function ``squeeze`` removes empty components. While ``yt`` arrays are unit-aware, it is
sometimes useful to extract the data into unitless numpy arrays. This is achieved with ``.v``.
In the case of ``Ex_fp``, the staggering is on direction ``x``, so that
``my_field[:,:-1,1] == my_field[:,1:,0]``.

All combinations of the fields (``E`` or ``B``), the component (``x``, ``y`` or ``z``) and the
grid (``_fp`` for fine, ``_cp`` for coarse and ``_aux`` for auxiliary) can be accessed in this
way, i.e., ``my_grid['raw', 'Ey_aux']`` or ``my_grid['raw', 'Bz_cp']`` are valid queries.

Read Raw Data With Guard Cells
------------------------------

When the output includes the data in the guard cells, the user can read such data using the post-processing tool ``read_raw_data.py``, available in ``Tools/PostProcessing/``, as illustrated in the following example:

.. code-block:: python

    from read_raw_data import read_data

    # Load all data saved in a given path
    path = './diags/diag00200/'
    data = read_data(path)

    # Load Ex_fp on mesh refinement level 0
    level = 0
    field = 'Ex_fp'
    # data[level] is a dictionary, data[level][field] is a numpy array
    my_field = data[level][field]

Note that a list of all available raw fields written to output, that is, a list of all valid strings that the variable ``field`` in the example above can be assigned to, can be obtained by calling ``data[level].keys()``.

In order to plot a 2D slice of the data with methods like ``matplotlib.axes.Axes.imshow``, one might want to pass the correct ``extent`` (the bounding box in data coordinates that the image will fill), including the guard cells. One way to set the correct ``extent`` is illustrated in the following example (case of a 2D slice in the ``(x,z)`` plane):

.. code-block:: python

    import yt
    import numpy as np

    from read_raw_data import read_data

    # Load all data saved in a given path
    path = './diags/diag00200/'
    data = read_data(path)

    # Load Ex_fp on mesh refinement level 0
    level = 0
    field = 'Ex_fp'
    # data[level] is a dictionary, data[level][field] is a numpy array
    my_field = data[level][field]

    # Set the number of cells in the valid domain
    # by loading the standard output data with yt
    ncells = yt.load(path).domain_dimensions

    # Set the number of dimensions automatically (2D or 3D)
    dim = 2 if (ncells[2] == 1) else 3

    xdir = 0
    zdir = 1 if (dim == 2) else 2

    # Set the extent (bounding box in data coordinates, including guard cells)
    # to be passed to matplotlib.axes.Axes.imshow
    left_edge_x  = 0            - (my_field.shape[xdir] - ncells[xdir]) // 2
    right_edge_x = ncells[xdir] + (my_field.shape[xdir] - ncells[xdir]) // 2
    left_edge_z  = 0            - (my_field.shape[zdir] - ncells[zdir]) // 2
    right_edge_z = ncells[zdir] + (my_field.shape[zdir] - ncells[zdir]) // 2
    extent = np.array([left_edge_z, right_edge_z, left_edge_x, right_edge_x])
