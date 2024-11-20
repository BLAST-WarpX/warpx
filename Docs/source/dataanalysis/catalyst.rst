.. _visualization-catalyst:

In situ Visualization with Catalyst 2
=====================================
Catalyst 2 (further referred to as just Catalyst) is a lightweight in-situ visualization and analysis framework API developed for simulations and other scientific data producers. It has a lightweight implementation
(or **stub**) and an SDK to develop custom implementations of Catalyst. ParaView comes with its own implementation (known as **ParaView Catalyst**) for leveraging ParaView's
visualization and analysis capabilities, which is what this document will focus on.


Enabling Catalyst
-----------------
In order to use Catalyst with WarpX, we need to ensure that we will be using the same version of
conduit across all libraries i.e Catalyst, AMReX and ParaView. One way to achieve this is to
build conduit externally and use it for compiling all the above packages.
This ensures compatibility when passing conduit nodes between WarpX and ParaView.

First, we build
`Conduit <https://llnl-conduit.readthedocs.io/en/latest/building.html>`_ and then
build `Catalyst 2 <https://catalyst-in-situ.readthedocs.io/en/latest/build_and_install.html>`_
using the conduit library created in the previous step.
The latter can be achieved by adding the installation path of conduit to the environmental
variable ``CMAKE_PREFIX_PATH`` and setting ``CATALYST_WITH_EXTERNAL_CONDUIT=ON`` during the configuration step of Catalyst.

Then we build ParaView master (on a commit after 2024.07.01, tested on ``4ef351a54ff747ef7169e2e52e77d9703a9dfa77``) following the developer instructions provided
`here <https://github.com/Kitware/ParaView/blob/master/Documentation/dev/build.md>`__ .
A representative set of options for a headless ParaView installation is provided
`here <https://gitlab.kitware.com/christos.tsolakis/catalyst-amrex-docker-images/-/blob/ci-catalyst-amrex-warpx-20240701/docker/ubuntu22_04/install_paraview.sh>`__
Afterward, WarpX must be built with ``WarpX_CATALYST=ON``.
Also, make sure to provide the installed paths of Conduit and Catalyst via
``CMAKE_PREFIX_PATH`` before configuring WarpX.

Inputs File Configuration
-------------------------
Once WarpX has been compiled with Catalyst support, it will need to be enabled and configured at runtime.
This is done using our usual inputs file (read with ``amrex::ParmParse``).
The supported parameters are part of the :ref:`FullDiagnostics <running-cpp-parameters-diagnostics>` with ``<diag_name>.format`` parameter set to ``catalyst``.

In addition to configuring the diagnostics, the following parameters must be included:
    * ``catalyst.script_paths``: The locations of the pipeline scripts, separated by either a colon or semicolon (e.g. ``/path/to/script1.py;/path/to/script2.py``).
    * ``catalyst.implementation`` (default ``paraview``): The name of the implementation being used (case sensitive).
    * ``catalyst.implementation_search_paths``: The locations to search for the given implementation. The specific file being searched for will be ``catalyst_{implementation}.so``.

The latter two can also be given via the environmental variables
``CATALYST_IMPLEMENTATION_NAME`` and ``CATALYST_IMPLEMENTATION_PATHS``
respectively.

Because the scripts and implementations are global, Catalyst does not benefit from nor differentiate between multiple diagnostics.


Visualization/Analysis Pipeline Configuration
---------------------------------------------
Catalyst uses the files specified in ``catalyst.script_paths`` to run all analysis.

The following script, :code:`simple_catalyst_pipeline.py`, automatically detects the type of data for both the mesh and particles, then creates an extractor for them. In most
cases, these will be saved as ``.VTPC`` files which can be read with the ``XML Partitioned Dataset Collection Reader``.

.. literalinclude:: catalyst/catalyst_simple_pipeline.py
   :language: python
   :caption:  You can copy this file from ``Docs/source/dataanalysis/catalyst/catalyst_simple_pipeline.py``.



For the case of ParaView Catalyst, pipelines are run with ParaView's included ``pvbatch`` executable and use the ``paraview`` library to modify the data. While pipeline scripts
could be written manually, this is not advised for anything beyond the script above. It is much more practical to use ParaView's built in ``Save Catalyst State`` button.

The process for creating a pipeline is as follows:
    1. Run at least one step of simulation and save the data in a ParaView compatible format, then open it in ParaView.
    2. Set up the desired scene, including filters, camera and views, and extractors.
    3. Press ``Save Catalyst State``, or the multicolored flask icon in the top left corner, and save it to a desired location.
    4. Open the script and replace the used producer with ``PVTrivialProducer``, setting the ``registrationName`` to either ``mesh`` or ``particles`` based on what data is used.

As an example for step four, here are a few lines from a script directly exported from ParaView:

.. code-block:: python

    # create a new 'XML Image Data Reader'
    meshdatavti = XMLImageDataReader(registrationName='meshdata.vti', FileName=['/path/to/meshdata.vti'])
    meshdatavti.CellArrayStatus = ['Bx', 'By', 'Bz', 'Ex', 'Ey', 'Ez']
    meshdatavti.TimeArray = 'None'

    # Calculator sample filter
    calculator1 = Calculator(registrationName='Calculator1', Input=meshdatavti)
    calculator1.AttributeType = 'Cell Data'
    calculator1.ResultArrayName = 'BField'
    calculator1.Function = 'sqrt(Bx^2 + By^2 + Bz^2)'

In order to use it with the mesh data coming from the simulation, the above code would be changed to:

.. code-block:: python

    # create the producer
    meshdata = PVTrivialProducer(registrationName='mesh')
    meshdata.CellArrayStatus = ['Bx', 'By', 'Bz', 'Ex', 'Ey', 'Ez']
    meshdata.TimeArray = 'None'

    # Calculator sample filter
    calculator1 = Calculator(registrationName='Calculator1', Input=meshdata)
    calculator1.AttributeType = 'Cell Data'
    calculator1.ResultArrayName = 'BField'
    calculator1.Function = 'sqrt(Bx^2 + By^2 + Bz^2)'

Steps one is advised so that proper scaling and framing can be done, however in certain cases it may not be possible. If this is the case, a dummy object can be used instead
(such as a wavelet or geometric shape scaled appropriately) and the rest of the steps can be followed as usual.

Replay
------
Catalyst 2.0 supports generating binary data dumps for the conduit nodes passed to each ``catalyst_`` call at each iteration. This allows to debug/adapt catalyst scripts without having to rerun the simulation each time.

To generate the data dumps one must first set the environmental variable ``CATALYST_DATA_DUMP_DIRECTORY`` to the path where the dumps should be saved. Then, run the simulation as normal but replace ``catalyst.implementation=stub`` either in the calling script of WarpX or as an additional argument.

This will run the simulation and write the conduit nodes under ``CATALYST_DATA_DUMP_DIRECTORY``.

Afterward, one can replay the generated nodes by setting up the ``CATALYST_IMPLEMENTATION_*`` variables for the ``catalyst_replay`` executable (which can be found in the catalyst build directory) appropriately. For example:

.. code-block:: bash

    # dump conduit nodes
    export CATALYST_DATA_DUMP_DIRECTORY=./raw_data
    mpiexec -n N <WarpX build directory>/bin/warpx.2d ./inputs_2d catalyst.script_paths=catalyst_pipeline.py catalyst.implementation="stub"
    # validate that files have been written
    ls ./raw_data/
    ... many files of the format XXXX.conduit_bin.Y.Z

    # replay them
    export CATALYST_IMPLEMENTATION_NAME=paraview
    export CATALYST_IMPLEMENTATION_PATHS=<paraview install path>/lib/catalyst
    export CATALYST_IMPLEMENTATION_PREFER_ENV=YES
    export CATALYST_DEBUG=1 # optional but helps to make sure the right paths are used
    export PYTHONPATH=${PYTHONPATH}/$(pwd) # or the path containing catalyst_pipeline.py in general
    # N needs to be the same as when we generated the dump
    mpiexec -n N <catalyst install path>/bin/catalyst_replay ./raw_data
    # check extractor output  e.g
    ls ./datasets/

For more information see the documentation for catalyst replay `here <https://catalyst-in-situ.readthedocs.io/en/latest/catalyst_replay.html>`__ .
