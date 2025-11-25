.. _workflows-stl-geometry-preparation:

STL Geometry Preparation for WarpX
==================================

WarpX has the ability to define the embedded boundary (EB) in a simulation using externally provided STL (STereoLithography) files.
STL files are a common format for representing 3D geometries and can be used to define complex embedded boundaries in WarpX simulations, without having to define extensive EB implicit functions.
However, there are some limitations to using STL files which must be resolved, including:

1. Multiple STL files representing different parts of a device must be combined into a single file
2. The STL geometry is "watertight" (The STL file shouldn't have duplicate faces or vertices, overlapping vertices should be merged, and there shouldn't be any gaps or holes in the mesh)
3. The STL file is properly formatted for WarpX input

.. 2. The STL file shouldn't have duplicate faces or vertices, and overlapping vertices should be merged. (issues with vertices can cause the MLMG solver to crash)

Step 1: Combining Multiple STL Files
------------------------------------

If your device consists of multiple parts stored in separate STL files, you'll need to combine them into a single STL file.

Using MeshLab
^^^^^^^^^^^^^

`MeshLab <https://www.meshlab.net/>`__ is a free, open-source tool for processing 3D meshes:

1. Open MeshLab
2. Import the first STL file: ``File -> Import Mesh``
3. Import additional STL files: ``File -> Import Mesh`` (repeat for each file)
4. All parts should now be visible in the same scene
5. Create a union of the two parts, either by ``Filters -> Remeshing, Simplification and Reconstruction -> Mesh Boolean: Union``
   or by right-clicking on one of the parts in the Layer Dialog panel on the right and clicking ``Mesh Boolean: Union``
6. Select the union and export the combined mesh: ``File -> Export Mesh As``
7. Choose STL format and save

Using Python with trimesh
^^^^^^^^^^^^^^^^^^^^^^^^^

You can also use Python with the `trimesh <https://github.com/mikedh/trimesh>`__ library:

.. code-block:: python

   import trimesh

   # Load individual STL files
   mesh1 = trimesh.load('part1.stl')
   mesh2 = trimesh.load('part2.stl')

   # Combine meshes
   combined = trimesh.util.concatenate([mesh1, mesh2])

   # Export combined mesh
   combined.export('device_combined.stl')

Step 2: Making the Geometry Watertight and Removing and Merging Duplicate Faces and Vertices
---------------------------------------------------------------------------------------------------


WarpX requires that STL geometries be "watertight" (manifold), meaning the mesh has no holes, gaps, or open edges.
This ensures that WarpX can properly determine which regions are inside or outside the geometry.
STL files that have overlapping or duplicate faces and vertices can cause the MLMG solver to crash, while a non-manifold file can be inherently leaky for Poynting flux in an EM simulation.
These issues with the mesh might happen while exporting the STL file from a CAD program (such as SolidWorks)

Using MeshLab
^^^^^^^^^^^^^

To check and repair a mesh in MeshLab:

1. **Check for issues:**

   * ``Filters -> Quality Measure and Computations -> Compute Topological Measures``
   * Look for holes in the output
   * Look for non-manifold edges and vertices in the output

2. **Repair the mesh:**

   * When loading in the mesh, there is an option to ``Unify Duplicated Vertices in STL files``, this will perform the next steps automatically, but gives the user less control.
   * ``Filters -> Cleaning and Repairing -> Remove Duplicate Faces``
   * ``Filters -> Cleaning and Repairing -> Remove Duplicate Vertices``
   * ``Filters -> Cleaning and Repairing -> Merge Close Vertices``

      * This filter gives the user an option to set an absolute or relative tolerance on the distance between the vertices to merge.

   .. * ``Filters -> Cleaning and Repairing -> Remove Zero Area Faces``
   .. * ``Filters -> Remeshing, Simplification and Reconstruction -> Close Holes``

3. **Verify the mesh is watertight:**

   * Run ``Filters -> Quality Measure and Computations -> Compute Topological Measures`` again
   * Check that the mesh has no holes
   * Check that the mesh is manifold (no warnings about non-manifold edges)

      * An STL file with non-manifold edges won't necessarily crash a simulation, but may present other numerical issues.




Step 3: Using STL Files in WarpX
--------------------------------

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

For more details on embedded boundary parameters, see:

* `Embedded Boundary Input Parameters <https://warpx.readthedocs.io/en/latest/usage/parameters.html#embedded-boundary-conditions:~:text=in%20this%20case.-,Embedded%20Boundary%20Conditions,-%EF%83%81>`__
* `Embedded Boundary PICMI Input Parameters <https://warpx.readthedocs.io/en/latest/usage/python.html#pywarpx.picmi.EmbeddedBoundary:~:text=Custom%20class%20to,translated%20and%20inverted.>`__

Tips and Best Practices
-----------------------

* *Units:* Ensure your STL file uses the same unit system as your WarpX simulation
* *Scale:* If needed, scale your geometry in your CAD software or mesh editor before exporting
* *Orientation:* Check that your geometry is properly oriented relative to WarpX's coordinate system
* *Resolution:* The STL mesh resolution should be appropriate for your simulation - too coarse may miss important features, too fine may slow down initialization

      * It's generally a good idea to simplify your geometery and remove features that are significantly smaller than the WarpX simulation grid

* *Binary vs ASCII:* WarpX can read both binary and ASCII STL files, but binary files are typically smaller and faster to load


Examples
--------

.. note::

   **TODO**: WarpX does not yet have examples that use STL files for embedded boundaries.

   Current embedded boundary examples in ``Examples/Tests/embedded_boundary_*`` use analytical
   functions to define geometries. A complete example demonstrating STL file usage will be
   added in the future.
