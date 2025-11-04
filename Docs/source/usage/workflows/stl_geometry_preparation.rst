.. _workflows-stl-geometry-preparation:

STL Geometry Preparation for WarpX
===================================

This workflow describes how to prepare multiple STL (STereoLithography) files of a device for input into WarpX.
STL files are a common format for representing 3D geometries and can be used to define complex embedded boundaries in WarpX simulations.

Overview
--------

When working with STL files in WarpX, you need to ensure that:

1. Multiple STL files representing different parts of a device are combined into a single file
2. The resulting geometry is "watertight" (no gaps or holes in the mesh)
3. The STL file is properly formatted for WarpX input

Step 1: Combining Multiple STL Files
-------------------------------------

If your device consists of multiple parts stored in separate STL files, you'll need to combine them into a single STL file.

Using MeshLab
^^^^^^^^^^^^^

`MeshLab <https://www.meshlab.net/>`__ is a free, open-source tool for processing 3D meshes:

1. Open MeshLab
2. Import the first STL file: ``File -> Import Mesh``
3. Import additional STL files: ``File -> Import Mesh`` (repeat for each file)
4. All parts should now be visible in the same scene
5. Export the combined mesh: ``File -> Export Mesh As``
6. Choose STL format and save

Using Python with trimesh
^^^^^^^^^^^^^^^^^^^^^^^^^^

You can also use Python with the ``trimesh`` library:

.. code-block:: python

   import trimesh

   # Load individual STL files
   mesh1 = trimesh.load('part1.stl')
   mesh2 = trimesh.load('part2.stl')
   mesh3 = trimesh.load('part3.stl')

   # Combine meshes
   combined = trimesh.util.concatenate([mesh1, mesh2, mesh3])

   # Export combined mesh
   combined.export('device_combined.stl')

Step 2: Making the Geometry Watertight
---------------------------------------

WarpX requires that STL geometries be "watertight" (manifold), meaning the mesh has no holes, gaps, or open edges.
This ensures that WarpX can properly determine which regions are inside or outside the geometry.

Using MeshLab
^^^^^^^^^^^^^

To check and repair a mesh in MeshLab:

1. **Check for issues:**
   
   * ``Filters -> Quality Measure and Computations -> Compute Topological Measures``
   * Look for non-manifold edges and vertices in the output

2. **Repair the mesh:**
   
   * ``Filters -> Cleaning and Repairing -> Remove Duplicate Faces``
   * ``Filters -> Cleaning and Repairing -> Remove Duplicate Vertices``
   * ``Filters -> Cleaning and Repairing -> Remove Zero Area Faces``
   * ``Filters -> Remeshing, Simplification and Reconstruction -> Close Holes``

3. **Verify the mesh is watertight:**
   
   * Run ``Filters -> Quality Measure and Computations -> Compute Topological Measures`` again
   * Check that the mesh is manifold (no warnings about non-manifold edges)

Using Python with trimesh
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import trimesh

   # Load the mesh
   mesh = trimesh.load('device_combined.stl')

   # Check if watertight
   print(f"Is watertight: {mesh.is_watertight}")
   print(f"Is manifold: {mesh.is_winding_consistent}")

   # Attempt to repair
   trimesh.repair.fix_normals(mesh)
   trimesh.repair.fill_holes(mesh)
   trimesh.repair.fix_inversion(mesh)

   # Check again
   print(f"After repair - Is watertight: {mesh.is_watertight}")

   # Export repaired mesh
   mesh.export('device_watertight.stl')

Using netfabb or Meshmixer
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Other popular tools for repairing STL files include:

* `Autodesk Meshmixer <https://www.meshmixer.com/>`__: Free tool with analysis and repair features
* `netfabb <https://www.autodesk.com/products/netfabb/overview>`__: Professional tool with automatic repair

Step 3: Using STL Files in WarpX
---------------------------------

Once you have a single, watertight STL file, you can use it in WarpX by setting the following parameters in your input file:

.. code-block:: text

   # Specify that the embedded boundary comes from an STL file
   eb2.geom_type = stl
   
   # Path to your STL file
   eb2.stl_file = path/to/device_watertight.stl

You can also optionally set an electric potential on the embedded boundary:

.. code-block:: text

   # Define electric potential at the embedded boundary (optional)
   warpx.eb_potential(x,y,z,t) = "voltage_function"

For more details on embedded boundary parameters, see :ref:`running-cpp-parameters-eb`.

Tips and Best Practices
------------------------

* *Units:* Ensure your STL file uses the same unit system as your WarpX simulation
* *Scale:* If needed, scale your geometry in your CAD software or mesh editor before exporting
* *Orientation:* Check that your geometry is properly oriented relative to WarpX's coordinate system
* *Resolution:* The STL mesh resolution should be appropriate for your simulation - too coarse may miss important features, too fine may slow down initialization
* *Binary vs ASCII:* WarpX can read both binary and ASCII STL files, but binary files are typically smaller and faster to load

Further Reading
---------------

* :ref:`running-cpp-parameters-eb` - Full documentation on embedded boundary parameters
* `STL file format <https://en.wikipedia.org/wiki/STL_(file_format)>`__ - Wikipedia article on STL format
* `AMReX Embedded Boundary documentation <https://amrex-codes.github.io/amrex/docs_html/EB.html>`__ - Details on how AMReX handles embedded boundaries

Examples
--------

.. note::

   **TODO**: WarpX does not yet have examples that use STL files for embedded boundaries.
   
   Current embedded boundary examples in ``Examples/Tests/embedded_boundary_*`` use analytical
   functions to define geometries. A complete example demonstrating STL file usage will be
   added in the future.
