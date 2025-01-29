.. _visualization-ascent:

In situ Visualization with Ascent
=================================
Ascent is a system designed to meet the in-situ visualization and analysis needs of simulation code teams running multi-physics calculations on many-core HPC architectures.
It provides rendering runtimes that can leverage many-core CPUs and GPUs to render images of simulation meshes.

Compiling with GNU Make
-----------------------
After building and installing Ascent according to the instructions at `Building Ascent <https://ascent.readthedocs.io/en/latest/BuildingAscent.html>`_, you can enable support for it in WarpX by changing the line

.. code-block:: bash

   USE_ASCENT_INSITU=FALSE

in ``GNUmakefile`` to

.. code-block:: bash

   USE_ASCENT_INSITU=TRUE

Furthermore, you must ensure that either the :code:`ASCENT_DIR` shell environment variable contains the directory where Ascent is installed or you must specify this location when invoking make, i.e.,

.. code-block:: bash

   make -j 8 USE_ASCENT_INSITU=TRUE ASCENT_DIR=/path/to/ascent/install


Inputs File Configuration
-------------------------

Once WarpX has been compiled with Ascent support, it will need to be enabled and configured at runtime.
This is done using our usual inputs file (read with ``amrex::ParmParse``).
The supported parameters are part of the :ref:`FullDiagnostics <running-cpp-parameters-diagnostics>` with ``<diag_name>.format`` parameter set to ``ascent``.


Visualization/Analysis Pipeline Configuration
---------------------------------------------
Ascent uses the file :code:`ascent_actions.yaml` to configure analysis and visualization pipelines.
Ascent looks for the :code:`ascent_actions.yaml` file in the current working directory.

For example, the following :code:`ascent_actions.yaml` file extracts an isosurface of the field ``Ex`` for ``15`` levels and saves the resulting images to :code:`levels_<nnnn>.png`.
`Ascent Actions <https://ascent.readthedocs.io/en/latest/Actions/index.html>`_ provides an overview over all available analysis and visualization actions.

.. code-block:: yaml

    -
      action: "add_pipelines"
      pipelines:
        p1:
          f1:
            type: "contour"
            params:
               field: "Ex"
               levels: 15
    -
      action: "add_scenes"
      scenes:
        scene1:
          image_prefix: "levels_%04d"
          plots:
            plot1:
              type: "pseudocolor"
              pipeline: "p1"
              field: "Ex"

Here is another :code:`ascent_actions.yaml` example that renders isosurfaces and particles:

.. code-block:: yaml

    -
      action: "add_pipelines"
      pipelines:
        p1:
          f1:
            type: "contour"
            params:
               field: "Bx"
               levels: 3
    -
      action: "add_scenes"
      scenes:
        scene1:
          plots:
            plot1:
              type: "pseudocolor"
              pipeline: "p1"
              field: "Bx"
            plot2:
              type: "pseudocolor"
              field: "particle_electrons_Bx"
              points:
                radius: 0.0000005
          renders:
            r1:
              camera:
                azimuth: 100
                elevation: 10
              image_prefix: "out_render_3d_%06d"


Finally, here is a more complex :code:`ascent_actions.yaml` example that creates the same images as the prior example, but adds a trigger that creates a Cinema Database at cycle ``300``:

.. code-block:: yaml

    -
      action: "add_triggers"
      triggers:
        t1:
          params:
            condition: "cycle() == 300"
            actions_file: "trigger.yaml"
    -
      action: "add_pipelines"
      pipelines:
        p1:
          f1:
            type: "contour"
            params:
               field: "jy"
               iso_values: [ 1000000000000.0, -1000000000000.0]
    -
      action: "add_scenes"
      scenes:
        scene1:
          plots:
            plot1:
              type: "pseudocolor"
              pipeline: "p1"
              field: "jy"
            plot2:
              type: "pseudocolor"
              field: "particle_electrons_w"
              points:
                radius: 0.0000002
          renders:
            r1:
              camera:
                azimuth: 100
                elevation: 10
              image_prefix: "out_render_jy_part_w_3d_%06d"


When the trigger condition is meet, ``cycle() == 300``, the actions in :code:`trigger.yaml` are also executed:

.. code-block:: yaml

    -
      action: "add_pipelines"
      pipelines:
        p1:
          f1:
            type: "contour"
            params:
               field: "jy"
               iso_values: [ 1000000000000.0, -1000000000000.0]
    -
      action: "add_scenes"
      scenes:
        scene1:
          plots:
            plot1:
              type: "pseudocolor"
              pipeline: "p1"
              field: "jy"
            plot2:
              type: "pseudocolor"
              field: "particle_electrons_w"
              points:
                radius: 0.0000001
          renders:
            r1:
              type: "cinema"
              phi: 10
              theta: 10
              db_name: "cinema_out"

You can view the Cinema Database result by opening :code:`cinema_databases/cinema_out/index.html`.


Replay
------

With Ascent/Conduit, one can store the intermediate data files before the rendering step is applied to custom files.
These so-called *Conduit Blueprint HDF5* files can be "replayed", i.e. rendered without running the simulation again.
VisIt 3.0+ also supports those files.

`Replay <https://ascent.readthedocs.io/en/latest/Utilities.html#getting-data-for-replay>`_ is a utility that allows the user to replay a simulation from aforementioned files and rendering them with Ascent.
Replay enables the user or developer to pick specific time steps and load them for Ascent visualization, without running the simulation again.

We will guide you through the replay procedure.


Get Blueprint Files
^^^^^^^^^^^^^^^^^^^
To use replay, you first need *Conduit Blueprint HDF5* files.
The following block can be used in an ascent action to extract *Conduit Blueprint HDF5* files from a simulation run.

.. code-block:: yaml

    -
      action: "add_extracts"
      extracts:
        e1:
          type: "relay"
          params:
            path: "conduit_blueprint"
            protocol: "blueprint/mesh/hdf5"

The output in the WarpX run directory will look as in the following listing.
The ``.root`` file is a metadata file and the corresponding directory contains the conduit blueprint data in an internal format that is based on HDF5.

.. code-block:: bash

   conduit_blueprint.cycle_000000/
   conduit_blueprint.cycle_000000.root
   conduit_blueprint.cycle_000050/
   conduit_blueprint.cycle_000050.root
   conduit_blueprint.cycle_000100/
   conduit_blueprint.cycle_000100.root

In order to select a few time steps after the fact, a so-called *cycles file* can be created.
A cycles file is a simple text file that lists one root file per line, e.g.:

.. code-block:: bash

  conduit_blueprint.cycle_000100.root
  conduit_blueprint.cycle_000050.root


Run Replay
^^^^^^^^^^

For Ascent Replay, two command line tools are provided in the utilities/replay directory of the Ascent installation.
There are two version of replay: the MPI-parallel version ``replay_mpi`` and a serial version, ``replay_ser``.
Use an MPI-parallel replay with data sets created with MPI-parallel builds of WarpX.
Here we use ``replay_mpi`` as an example.

The options for replay are:

* ``--root``: specifies Blueprint root file to load
* ``--cycles``: specifies a text file containing a list of Blueprint root files to load
* ``--actions``: specifies the name of the actions file to use (default: ``ascent_actions.yaml``)

Instead of starting a simulation that generates data for Ascent, we now execute ``replay_ser``/``replay_mpi``.
Replay will loop over the files listed in ``cycles`` in the order in which they appear in the cycles file.

For example, for a small data example that fits on a single computer:

.. code-block:: bash

  ./replay_ser --root=conduit_blueprint.cycle_000400.root --actions=ascent_actions.yaml

Will replay the data of WarpX step 400 ("cycle" 400).
A whole set of steps can be replayed with the above mentioned *cycles file*:

.. code-block:: bash

  ./replay_ser --cycles=warpx_list.txt --actions=ascent_actions.yaml

For larger examples, e.g. on a cluster with *Slurm* batch system, a parallel launch could look like this:

.. code-block:: bash

  # one step
  srun -n 8 ./replay_mpi --root=conduit_blueprint.cycle_000400.root --actions=ascent_actions.yaml
  # multiple steps
  srun -n 8 ./replay_mpi --cycles=warpx_list.txt --actions=ascent_actions.yaml


Example Actions
^^^^^^^^^^^^^^^

A visualization of the electric field component :math:`E_x` (variable: ``Ex``) with a contour plot and with added particles can be obtained with the following Ascent Action.
This action can be used both in replay as well as in situ runs.

.. literalinclude:: ../../../Examples/Physics_applications/laser_acceleration/ascent_actions.yaml
   :language: yaml

There are more `Ascent Actions examples available <https://ascent.readthedocs.io/en/latest/Actions/Examples.html#yaml-examples>`_ for you to play.


Workflow
^^^^^^^^

.. note::

   This section is in-progress.
   TODOs: finalize acceptance testing; update 3D LWFA example

In the preparation of simulations, it is generally useful to run small, under-resolved versions of the planned simulation layout first.
Ascent replay is helpful in the setup of an in situ visualization pipeline during this process.
In the following, a Jupyter-based workflow is shown that can be used to quickly iterate on the design of a ``ascent_actions.yaml`` file, repeatedly rendering the same (small) data.

First, run a small simulation, e.g. on a local computer, and create conduit blueprint files (see above).
Second, copy the Jupyter Notebook file :download:`ascent_replay_warpx.ipynb <../../../Tools/Ascent/ascent_replay_warpx.ipynb>` into the simulation output directory.
Third, download and start a Docker container with a prepared Jupyter installation and Ascent Python bindings from the simulation output directory:

.. code-block:: bash

   docker pull alpinedav/ascent-jupyter:latest
   docker run -v$PWD:/home/user/ascent/install-debug/examples/ascent/tutorial/ascent_intro/notebooks/replay -p 8000:8000 -p 8888:8888 -p 9000:9000 -p 10000:10000 -t -i alpinedav/ascent-jupyter:latest

Now, access Jupyter Lab via: http://localhost:8888/lab (password: ``learn``).

Inside the Jupyter Lab is a ``replay/`` directory, which mounts the outer working directory.
You can now open ``ascent_replay_warpx.ipynb`` and execute all cells.
The last two cells are the replay action that can be quickly iterated: change ``replay_actions.yaml`` cell and execute both.

.. note::

   * Keep an eye on the terminal, if a replay action is erroneous it will show up on the terminal that started the docker container.
     (TODO: We might want to catch that inside python and print it in Jupyter instead.)
   * If you remove a "key" from the replay action, you might see an error in the ``AscentViewer``.
     Restart and execute all cells in that case.
