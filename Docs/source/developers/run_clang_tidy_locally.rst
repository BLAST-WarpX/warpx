.. _developers-run_clang_tidy_locally:

The clang-tidy linter
=====================

Clang-tidy CI test
------------------

WarpX's CI tests include several checks performed with the
`clang-tidy <https://releases.llvm.org/15.0.0/tools/clang/tools/extra/docs/clang-tidy/index.html>`__ linter
(currently the version 15 of this tool). The complete list of checks
enforced in CI tests can be found in the ``.clang-tidy`` configuration file.

.. dropdown:: clang-tidy configuration file
   :color: light
   :icon: info
   :animate: fade-in-slide-down

   .. literalinclude:: ../../../.clang-tidy
      :language: yaml

Run clang-tidy linter locally
-----------------------------

We provide a script to run clang-tidy locally. The script can be run as follows,
provided that all the requirements to compile WarpX are met (see `building from source <install-developers>`).
The script generates a simple wrapper to ensure that `clang-tidy` is only applied to WarpX source files
and compiles WarpX in 1D,2D,3D, and RZ using such wrapper. By default WarpX is compiled in single precision
with PSATD solver, QED module, QED table generator and Embedded boundary in order to find more
potential issues with the `clang-tidy` tool.

Few optional environment variables can be set to tune the behavior of the script:

* ``WARPX_TOOLS_LINTER_PARALLEL``: sets the number of cores to be used for the compilation
* ``CLANG``, ``CLANGXX``, and ``CLANGTIDY`` : set the version of the compiler and of the linter

Note: clang v15 is currently used in CI tests. It is therefore recommended to use this version.
Otherwise, a newer version may find issues not currently covered by CI tests (checks are opt-in)
while older versions may not find all the issues.

.. code-block:: bash

   export WARPX_TOOLS_LINTER_PARALLEL=12
   export CLANG=clang-15
   export CLANGXX=clang++-15
   export CLANGTIDY=clang-tidy-15
   ./Tools/Linter/runClangTidy.sh

.. dropdown:: Script Details
   :color: light
   :icon: info
   :animate: fade-in-slide-down

   .. literalinclude:: ../../../Tools/Linter/runClangTidy.sh
      :language: bash
