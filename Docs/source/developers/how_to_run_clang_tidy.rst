.. _developers-run_clang_tidy_locally:

How to run the clang-tidy linter
================================

WarpX's CI tests include several checks performed with the `clang-tidy <https://clang.llvm.org/extra/clang-tidy/>`__ linter.
The complete list of checks performed is defined in the ``.clang-tidy`` configuration file.

.. dropdown:: clang-tidy configuration file
   :color: light
   :icon: info
   :animate: fade-in-slide-down

   .. literalinclude:: ../../../.clang-tidy
      :language: yaml

Under `Tools/Linter <https://github.com/ECP-WarpX/WarpX/blob/development/Tools/Linter>`__, the script ``runClangTidy.sh`` can be used to run the clang-tidy linter locally.

.. dropdown:: clang-tidy local run script
   :color: light
   :icon: info
   :animate: fade-in-slide-down

   .. literalinclude:: ../../../Tools/Linter/runClangTidy.sh
      :language: bash

It is a prerequisite that WarpX is compiled following the instructions that you find in our :ref:`Users <install-cmake>` or :ref:`Developers <building-cmake>` sections.

The script generates a wrapper to ensure that clang-tidy is only applied to WarpX source files and compiles WarpX in 1D, 2D, 3D, and RZ geometry, using such wrapper.

By default WarpX is compiled in single precision with PSATD solver, QED module, QED table generator and embedded boundary in order to ensure broader coverage with the clang-tidy tool.

Few optional environment variables can be set to tune the behavior of the script:

* ``WARPX_TOOLS_LINTER_PARALLEL``: set the number of cores used for compilation;

* ``CLANG``, ``CLANGXX``, and ``CLANGTIDY``: set the version of the compiler and the linter.

For continuous integration we currently use clang version 15.0.0 and it is recommended to use this version locally as well.
A newer version may find issues not currently covered by CI tests (checks are opt-in), while older versions may not find all the issues.

Here's an example of how to run the script after setting the appropriate environment variables:

.. code-block:: bash

   export WARPX_TOOLS_LINTER_PARALLEL=12
   export CLANG=clang-15
   export CLANGXX=clang++-15
   export CLANGTIDY=clang-tidy-15

   ./Tools/Linter/runClangTidy.sh
