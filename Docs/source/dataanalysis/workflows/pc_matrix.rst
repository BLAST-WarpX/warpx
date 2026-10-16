.. _dataanalysis-pc-matrix:

Preconditioner Matrices
=======================

The implicit solvers can write the matrix of a preconditioner to a file, for debugging and analysis.
This is available for ``jacobian.pc_type = pc_jacobi``, ``pc_petsc``, and ``pc_algmg``.
It is enabled with the parameters below, which use the name of the preconditioner as prefix (see :ref:`running-cpp-parameters`):

* ``<prefix>.print_matrix`` (``bool``, default: false)
* ``<prefix>.print_matrix_steps`` (list of ``int``) Zero-based time steps at which the matrix is written, at every update of the preconditioner in these steps.
  By default, only the first update of the run is written.
* ``<prefix>.print_matrix_dir`` (``string``, default: ``diags/pc_matrices``)

File format
-----------

Each matrix is written to ``<dir>/<prefix>_step<NNNNN>_update<k>.mtx``, where ``k`` counts the preconditioner updates within the time step.
The file is in `MatrixMarket <https://math.nist.gov/MatrixMarket/formats.html>`__ coordinate format.
Rows and columns are the global indices (1-based) of the degrees of freedom of the electric field.
Each rank writes its own rows; the matrix is never gathered on one rank.
For ``pc_jacobi``, every entry of the mass-matrix stencil is written, including entries that are zero.

The header holds metadata as ``% key: value`` comment lines, for example:

.. code-block:: text

   %%MatrixMarket matrix coordinate real general
   % name: pc_jacobi
   % step: 0
   % update_in_step: 0
   % omega: 0.830000
   % matrix_ops: MultiFabMatrixSolverOps (stencil)
   % operator: A = I + M, M = PC mass matrices
   % stencil_size_comp0: 5 5 1
   % diagonal_min: 1.0213
   % max_offdiag_ratio: 0.4096
   % gershgorin_bounds_DinvA: 0.5904 1.4096
   % gershgorin_omega: 0.83
   % global_size: 21504 x 21504
   % nnz: 523776
   % ranks: 2
   % row_offsets: 0 10752 21504
   % local_nnz: 261888 261888
   % dof_map: dofmap.txt
   21504 21504 523776
   1 1 1.0213
   1 2 -0.0042

The row statistics are computed from the written entries:

* ``max_offdiag_ratio`` is :math:`r = \max_i \sum_{j \neq i} |a_{ij}| / |a_{ii}|`.
* ``gershgorin_bounds_DinvA`` bound the eigenvalues of :math:`D^{-1} A`, with :math:`D` the diagonal of :math:`A`, by :math:`[1 - r, 1 + r]`.
* ``gershgorin_omega`` is :math:`2 / (2 + r)`, the relaxation factor that ``pc_jacobi`` starts from.

The file ``dofmap.txt`` in the same directory maps each row to its AMR level, field component (0: x, 1: y, 2: z), and cell indices ``i j k``.
Its header gives the index type (cell-centered or nodal) of each field component.

Reading the files
-----------------

``Tools/PostProcessing/read_pc_matrix.py`` reads a matrix, its metadata, and the DOF map, and prints a summary (``--plot`` shows the sparsity pattern):

.. code-block:: python

   import sys

   sys.path.append("<WarpX>/Tools/PostProcessing")
   from read_pc_matrix import read_dofmap, read_pc_matrix

   A, info = read_pc_matrix("diags/pc_matrices/pc_jacobi_step00000_update0.mtx")
   dofmap = read_dofmap("diags/pc_matrices")
   rows_x = dofmap["comp"] == 0  # rows of Ex

The matrix can also be read directly with ``scipy.io.mmread``.

``Tools/PostProcessing/read_pc_matrix.m`` does the same in MATLAB and Octave:

.. code-block:: matlab

   [A, info, dofmap] = read_pc_matrix('diags/pc_matrices/pc_jacobi_step00000_update0.mtx');
   spy(A)

In Julia, use `MatrixMarket.jl <https://github.com/JuliaSparse/MatrixMarket.jl>`__:

.. code-block:: julia

   using MatrixMarket
   A = MatrixMarket.mmread("diags/pc_matrices/pc_jacobi_step00000_update0.mtx")
