.. _developers-testing:

Testing the Code
================

When proposing a code change, you want to make sure that

* the code change does not break the existing code;
* the code change gives correct results (numerics, physics, etc.).

WarpX follows the continuous integration (CI) software development practice, where automated builds and tests are run after merging code changes into the main branch.

While the code is tested regularly remotely (on the cloud when commits are pushed to an open PR, and every night on local clusters), it can also be useful to run tests on your custom input file.

How to run pre-commit tests locally
-----------------------------------

First, when proposing a code change, we perform a couple of automated style and correctness checks.

If you install the ``pre-commit`` tool on your local machine via

.. code-block:: sh

   python -m pip install -U pre-commit
   pre-commit install

the style and correctness checks will run automatically on your local machine, after you commit the change and before you push.

If you do not install the ``pre-commit`` tool on your local machine, these checks will run automatically as part of our CI workflows and a commit containing style and correctness changes might be added automatically to your branch.
In that case, you will need to pull that automated commit before pushing further changes.

See `pre-commit.com <https://pre-commit.com>`__ and our ``.pre-commit-config.yaml`` file in the repository for more details.

How to configure the automated tests
------------------------------------

Our regression tests are run with `CTest <https://cmake.org/cmake/help/book/mastering-cmake/chapter/Testing%20With%20CMake%20and%20CTest.html#>`__, an executable that comes with CMake.

The test suite is ready to run once you have configured and built WarpX with CMake, following the instructions that you find in our :ref:`Users <install-cmake>` or :ref:`Developers <building-cmake>` sections.

A test that requires a build option that was not configured and built will be skipped automatically. For example, if you configure and build WarpX in 1D only, any test of dimensionality other than 1D, which would require WarpX to be configured and built in the corresponding dimensionality, will be skipped automatically.

How to run automated tests locally
----------------------------------

Once your new feature is ready, there are ways to check that you did not break anything.
WarpX has automated tests running every time a commit is pushed to an open pull request.
The input files and scripts used by the automated tests can be found in the `Examples <https://github.com/ECP-WarpX/WarpX/tree/development/Examples>`__ directory, either under `Physics_applications <https://github.com/ECP-WarpX/WarpX/tree/development/Examples/Physics_applications>`__ or `Tests <https://github.com/ECP-WarpX/WarpX/tree/development/Examples/Tests>`__.

For easier debugging, it can be convenient to run the tests on your local machine by executing CTest as illustrated in the examples below (where we assume that WarpX was configured and built in the directory ``build``):

* List tests available for the current build options:

  .. code-block:: sh

       ctest --test-dir build -N

* Run tests available for the current build options:

  .. code-block:: sh

       ctest --test-dir build

* Run tests available for the current build options in parallel (while preserving existing dependencies between tests):

  .. code-block:: sh

       ctest --test-dir build -j 2

* Run tests available for the current build options and output anything outputted by the test program if the test should fail:

  .. code-block:: sh

       ctest --test-dir build --output-on-failure

* Run tests available for the current build options with verbose output:

  .. code-block:: sh

       ctest --test-dir build --verbose

* Run tests matching the regular expression ``laser_acceleration``:

  .. code-block:: sh

       ctest --test-dir build -R laser_acceleration

* Run tests except those matching the regular expression ``laser_acceleration``:

  .. code-block:: sh

       ctest --test-dir build -E laser_acceleration

* Sometimes two or more tests share a large number of input parameters and differ by a small set of options.
  Such tests typically also share a base string in their names.
  For example, you can find three different tests named ``test_3d_langmuir_multi``, ``test_3d_langmuir_multi_nodal`` and ``test_3d_langmuir_multi_picmi``.
  In such a case, if you wish to run the test ``test_3d_langmuir_multi`` only, this can be done again with the ``-R`` regular `expression filter <https://regex101.com>`__ via

  .. code-block:: sh

       ctest --test-dir build -R "test_3d_langmuir_multi\..*"

  Note that filtering with ``-R "test_3d_langmuir_multi"`` would include the additional tests that have the same substring in their name and would not be sufficient to isolate a single test.
  Note also that the escaping ``\.`` in the regular expression is necessary in order to take into account the fact that each test is automatically appended with the strings ``.run``, ``.analysis`` and possibly ``.cleanup``.

* Run only tests not labeled with the ``slow`` label:

  .. code-block:: sh

       ctest --test-dir build -LE slow

Once the execution of CTest is completed, you can find all files associated with each test in its corresponding directory under ``build/bin/``.
For example, if you run the single test ``test_3d_laser_acceleration``, you can find all files associated with this test in the directory ``build/bin/test_3d_laser_acceleration/``.

If you modify the code base locally and want to assess the effects of your code changes on the automated tests, you need to first rebuild WarpX including your code changes and then rerun CTest.

How to add automated tests
--------------------------

An automated test typically consists of the following components:

* input file or PICMI input script;
* analysis script;
* checksum file.

To learn more about how to use checksums in automated tests, please see the corresponding section :ref:`Checksums on Tests <developers-checksum>`.

As mentioned above, the input files and scripts used by the automated tests can be found in the `Examples <https://github.com/ECP-WarpX/WarpX/tree/development/Examples>`__ directory, under either `Physics_applications <https://github.com/ECP-WarpX/WarpX/tree/development/Examples/Physics_applications>`__ or `Tests <https://github.com/ECP-WarpX/WarpX/tree/development/Examples/Tests>`__.

Each test directory must contain a file named ``CMakeLists.txt`` where all tests associated with the input files and scripts in that directory must be listed.

A new test can be added by adding a corresponding entry in ``CMakeLists.txt`` as illustrated in the examples below:

* Add the **regular test** ``test_1d_laser_acceleration``:

  .. code-block:: cmake

       add_warpx_test(
           test_1d_laser_acceleration  # name
           1  # dims
           2  # nprocs
           inputs_test_1d_laser_acceleration  # inputs
           "analysis.py diags/diag1000100"  # analysis
           "analysis_default_regression.py --path diags/diag1000100"  # checksum
           OFF  # dependency
       )

* Add the **PICMI test** ``test_2d_laser_acceleration_picmi``:

  .. code-block:: cmake

       add_warpx_test(
           test_2d_laser_acceleration_picmi  # name
           2  # dims
           2  # nprocs
           inputs_test_2d_laser_acceleration_picmi.py  # inputs
           "analysis.py diags/diag1000100"  # analysis
           "analysis_default_regression.py --path diags/diag1000100"  # checksum
           OFF  # dependency
       )

* Add the **restart test** ``test_3d_laser_acceleration_restart``:

  .. code-block:: cmake

       add_warpx_test(
           test_3d_laser_acceleration_restart  # name
           3  # dims
           2  # nprocs
           inputs_test_3d_laser_acceleration_restart  # inputs
           "analysis_default_restart.py diags/diag1000100"  # analysis
           "analysis_default_regression.py --path diags/diag1000100"  # checksum
           test_3d_laser_acceleration  # dependency
       )

  Note that the restart has an explicit dependency, namely it can run only provided that the original test, from which the restart checkpoint files will be read, runs first.

* A more complex example. Add the **PICMI test** ``test_rz_laser_acceleration_picmi``, with custom command-line arguments ``--test`` and ``dir``, openPMD time series output, and custom command line arguments for the checksum comparison:

  .. code-block:: cmake

       add_warpx_test(
           test_rz_laser_acceleration_picmi  # name
           RZ  # dims
           2   # nprocs
           "inputs_test_rz_laser_acceleration_picmi.py --test --dir 1"  # inputs
           "analysis.py diags/diag1/"  # analysis
           "analysis_default_regression.py --path diags/diag1/ --skip-particles --rtol 1e-7"  # checksum
           OFF  # dependency
       )

The ``analysis`` and ``checksum`` commands passed as arguments to ``add_warpx_test`` can be set to ``OFF`` if the intention is to skip the respective analysis for a given test.

If you need a new Python package dependency for testing, please add it in `Regression/requirements.txt <https://github.com/ECP-WarpX/WarpX/blob/development/Regression/requirements.txt>`__.

Sometimes two or more tests share a large number of input parameters.
The shared input parameters can be collected in a "base" input file that can be passed as a runtime parameter in the actual test input files through the parameter ``FILE``.

If the new test is added in a new directory that did not exist before, please add the name of that directory with the command ``add_subdirectory`` in `Physics_applications/CMakeLists.txt <https://github.com/ECP-WarpX/WarpX/tree/development/Examples/Physics_applications/CMakeLists.txt>`__ or `Tests/CMakeLists.txt <https://github.com/ECP-WarpX/WarpX/tree/development/Examples/Tests/CMakeLists.txt>`__, depending on where the new test directory is located.

If not already present, the default regression analysis script ``analysis_default_regression.py`` in the examples above must be linked from `Examples/analysis_default_regression.py <https://github.com/ECP-WarpX/WarpX/blob/development/Examples/analysis_default_regression.py>`__, by executing once the following command from the test directory:

  .. code-block:: bash

       ln -s ../../analysis_default_regression.py analysis_default_regression.py

Here is the help message of the default regression analysis script, including usage and list of available options and arguments:

  .. code-block:: bash

       usage: analysis_default_regression.py [-h] [--path PATH] [--rtol RTOL] [--skip-fields] [--skip-particles]
       options:
         -h, --help        show this help message and exit
         --path PATH       path to output file(s)
         --rtol RTOL       relative tolerance to compare checksums
         --skip-fields     skip fields when comparing checksums
         --skip-particles  skip particles when comparing checksums

Naming conventions for automated tests
--------------------------------------

Note that we currently obey the following snake\_case naming conventions for test names and test input files (which make automation tasks easier, e.g., parsing visually, parsing through code, sorting alphabetically, filtering tests in CTest via ``-R``, etc.):

#. **Regular test names** start with the string ``test_1d_``, ``test_2d_``, ``test_3d_`` or ``test_rz_``, followed by a string that is descriptive of the test. For example, ``test_3d_laser_acceleration``.

#. **PICMI test names** start with the string ``test_1d_``, ``test_2d_``, ``test_3d_`` or ``test_rz_``, followed by a string that is descriptive of the test, and end with the string ``_picmi``. For example, ``test_3d_laser_acceleration_picmi``.

#. **Restart test names** end with the string ``_restart``. For example, ``test_3d_laser_acceleration_restart``.

#. **Test input files** start with the string ``inputs_`` followed by the test name. For example, ``inputs_test_3d_laser_acceleration`` or ``inputs_test_3d_laser_acceleration_picmi.py`` or ``inputs_test_3d_laser_acceleration_restart``.

#. **Base input files** (that is, files collecting input parameters shared between two or more tests) are typically named ``inputs_base_1d``, ``inputs_base_2d``, ``inputs_base_3d`` or ``inputs_base_rz``, possibly followed by additional strings if need be.

Useful tool for plotfile comparison: ``fcompare``
-------------------------------------------------

AMReX provides ``fcompare``, an executable that takes two ``plotfiles`` as input and returns the absolute and relative difference for each field between these two plotfiles. For some changes in the code, it is very convenient to run the same input file with an old and your current version, and ``fcompare`` the plotfiles at the same iteration. To use it:

.. code-block:: sh

   # Compile the executable
   cd <path to AMReX>/Tools/Plotfile/ # This may change
   make -j 8
   # Run the executable to compare old and new versions
   <path to AMReX>/Tools/Plotfile/fcompare.gnu.ex old/plt00200 new/plt00200

which should return something like

.. code-block:: sh

             variable name             absolute error            relative error
                                          (||A - B||)         (||A - B||/||A||)
   ----------------------------------------------------------------------------
   level = 0
   jx                                 1.044455105e+11               1.021651316
   jy                                  4.08631977e+16               7.734299273
   jz                                 1.877301764e+14               1.073458933
   Ex                                 4.196315448e+10               1.253551615
   Ey                                 3.330698083e+12               6.436470137
   Ez                                 2.598167798e+10              0.6804387128
   Bx                                     273.8687473               2.340209782
   By                                     152.3911863                1.10952567
   Bz                                     37.43212767                 2.1977289
   part_per_cell                                   15                    0.9375
   Ex_fp                              4.196315448e+10               1.253551615
   Ey_fp                              3.330698083e+12               6.436470137
   Ez_fp                              2.598167798e+10              0.6804387128
   Bx_fp                                  273.8687473               2.340209782
   By_fp                                  152.3911863                1.10952567
   Bz_fp                                  37.43212767                 2.1977289
