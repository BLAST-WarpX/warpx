.. _developers-testing:

How to test the code
====================

When you propose code changes, you want to make sure that

* the code changes do not break the behavior of the rest of the code;
* the code changes give correct results (numerics, physics, etc.).

Following the continuous integration (CI) software development practice, WarpX runs automated builds and tests after a commit is pushed to an open PR as well as after a PR is merged into the main branch.

How to run pre-commit tests locally
-----------------------------------

First, WarpX uses `pre-commit <https://pre-commit.com/>`__ to perform automated style and correctness checks.

Here is how to install ``pre-commit`` locally:

#. Install ``pre-commit``:

   .. code-block:: sh

      python -m pip install -U pre-commit

#. Install the git hook scripts:

   .. code-block:: sh

      pre-commit install

If you install ``pre-commit`` locally, the style and correctness checks will run automatically on your computer, after you commit the code changes and before you push them to the remote repository.

If you do not install ``pre-commit`` locally, these checks will run automatically as part of our CI workflows and a commit containing style and correctness changes might be added automatically to your branch after you have pushed your own commit.
In that case, you will need to pull that automated commit before pushing further commits.

The configuration options for ``pre-commit`` are set in the `pre-commit-config.yaml <https://github.com/ECP-WarpX/WarpX/blob/development/.pre-commit-config.yaml>`__ file.

How to configure the automated tests
------------------------------------

Our regression tests are run with `CTest <https://cmake.org/cmake/help/book/mastering-cmake/chapter/Testing%20With%20CMake%20and%20CTest.html#>`__, an executable that comes with CMake.

The test suite is ready to run once you have configured and built WarpX with CMake, following the instructions that you find in our :ref:`Users <install-cmake>` or :ref:`Developers <building-cmake>` sections.

A test that requires a build option that was not configured and built will be skipped automatically. For example, if you configure and build WarpX in 1D only, any test of dimensionality other than 1D, which would require WarpX to be configured and built in the corresponding dimensionality, will be skipped automatically.

How to run automated tests locally
----------------------------------

Once your code changes are ready, there are ways to check that they do not break the rest of the code.
WarpX has automated tests running every time a commit is pushed to an open pull request.
The input files and scripts used by the automated tests can be found in the `Examples <https://github.com/ECP-WarpX/WarpX/tree/development/Examples>`__ directory, either under `Physics_applications <https://github.com/ECP-WarpX/WarpX/tree/development/Examples/Physics_applications>`__ or `Tests <https://github.com/ECP-WarpX/WarpX/tree/development/Examples/Tests>`__.

For easier debugging, it can be convenient to run the tests on your local computer by executing CTest as illustrated in the examples below (where we assume that WarpX was configured and built in the directory ``build``):

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
  Note also that the escaping ``\.`` in the regular expression is necessary in order to take into account the fact that each test is automatically appended with the strings ``.run``, ``.analysis``, ``.checksum`` and possibly ``.cleanup``.

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

As mentioned above, the input files and scripts used by the automated tests can be found in the `Examples <https://github.com/ECP-WarpX/WarpX/tree/development/Examples>`__ directory, under either `Physics_applications <https://github.com/ECP-WarpX/WarpX/tree/development/Examples/Physics_applications>`__ or `Tests <https://github.com/ECP-WarpX/WarpX/tree/development/Examples/Tests>`__.

Each test directory must contain a file named ``CMakeLists.txt`` where all tests associated with the input files and scripts in that directory must be listed.

A checksum file is a file that contains reference values obtained by computing a chosen checksum for a set of fields.
More precisely, we compute the sums of the absolute values of the arrays corresponding to each field from the results produced by the automated test and compare these checksums with the reference ones stored in the checksum file of that test, with respect to specific tolerances.
This is expected to be sensitive enough to make the automated test fail if the code changes cause significant differences in the final results, thus catching possible bugs.

A new test can be added by calling the function ``add_warpx_test`` in ``CMakeLists.txt``. The function has the following signature:

.. code-block:: cmake

     function(add_warpx_test
         name        # unique test name:
                     # test_1d_example, test_2d_example_picmi, etc.
         dims        # dimensionality: 1, 2, 3, RZ
         nprocs      # number of processes: 1, 2
         inputs      # inputs file or PICMI script:
                     # inputs_test_1d_example, inputs_test_2d_example_picmi.py, "inputs_test_2d_example_picmi.py arg1 arg2", etc.
         analysis    # custom test analysis command:
                     # OFF, "analysis.py", "analysis.py arg1 arg2", etc.
         checksum    # default regression analysis command:
                     # OFF, "analysis_default_regression.py --path diags/diag1", etc.
         dependency  # name of base test that must run first (must match name exactly):
                     # OFF, test_1d_example_prepare, etc.
     )

Here's how to add an automated test:

#. Choose the test directory, either an existing one or a new one.

#. Add an input file or PICMI input script.
   The name must follow the naming conventions described in the section :ref:`developers-testing-naming` below.

#. Add a Python analysis script to analyze the results of the test.

#. Add the test to the ``CMakeLists.txt`` file (add such file if you are adding the test in a new test directory) using the function ``add_warpx_test`` mentioned above.

#. If the test directory is new, add the directory with the command ``add_subdirectory`` in `Physics_applications/CMakeLists.txt <https://github.com/ECP-WarpX/WarpX/tree/development/Examples/Physics_applications/CMakeLists.txt>`__ or `Tests/CMakeLists.txt <https://github.com/ECP-WarpX/WarpX/tree/development/Examples/Tests/CMakeLists.txt>`__, depending on where the test directory is located.

#. If the test directory is new, make a symbolic link to the default regression analysis script ``analysis_default_regression.py`` from `Examples/analysis_default_regression.py <https://github.com/ECP-WarpX/WarpX/blob/development/Examples/analysis_default_regression.py>`__, by running ``ln -s ../../analysis_default_regression.py analysis_default_regression.py`` from the test directory.

#. Run the test locally with ``ctest``, after setting the environment variable ``CHECKSUM_RESET=ON``, in order to generate automatically the checksum file.

Once you have added the test, run the test locally again, after resetting ``CHECKSUM_RESET=OFF``, to check that everything works as expected.

The ``analysis`` and ``checksum`` commands passed as arguments to ``add_warpx_test`` can be set to ``OFF`` if the intention is to skip the respective analysis for a given test.

If you need a new Python package dependency for testing, please add it in `Regression/requirements.txt <https://github.com/ECP-WarpX/WarpX/blob/development/Regression/requirements.txt>`__.

Sometimes two or more tests share a large number of input parameters.
The shared input parameters can be collected in a "base" input file that can be passed as a runtime parameter in the actual test input files through the parameter ``FILE``.

Here is the help message of the default regression analysis script, including usage and list of available options and arguments:

  .. code-block:: bash

       usage: analysis_default_regression.py [-h] [--path PATH] [--rtol RTOL] [--skip-fields] [--skip-particles]
       options:
         -h, --help        show this help message and exit
         --path PATH       path to output file(s)
         --rtol RTOL       relative tolerance to compare checksums
         --skip-fields     skip fields when comparing checksums
         --skip-particles  skip particles when comparing checksums

How to reset checksums locally
------------------------------

It is possible to reset a checksum file locally by running the corresponding test with ``ctest`` with the environment variable ``CHECKSUM_RESET=ON``. For example:

  .. code-block:: bash

       CHECKSUM_RESET=ON ctest --test-dir build -R laser_acceleration

Alternatively, it is also possible to reset multiple checksum files using the output of our Azure pipelines, which can be useful for code changes that result in resetting a large numbers of checksum files.
Here's how to do so:

#. On the GitHub page of the pull request, find (one of) the pipeline(s) failing due to checksum regressions and click on "Details" (highlighted in blue).

   .. figure:: https://gist.github.com/user-attachments/assets/09db91b9-5711-4250-8b36-c52a6049e38e

#. In the new page that opens up, click on "View more details on Azure pipelines" (highlighted in blue).

   .. figure:: https://gist.github.com/user-attachments/assets/ab0c9a24-5518-4da7-890f-d79fa1c8de4c

#. In the new page that opens up, select the group of tests for which you want to reset the checksum files (e.g., ``cartesian_3d``) and click on "View raw log".

   .. figure:: https://gist.github.com/user-attachments/assets/06c1fe27-2c13-4bd3-b6b8-8b8941b37889

#. Save the raw log as a text file on your computer.

#. Go to the directory `Tools/DevUtils <https://github.com/ECP-WarpX/WarpX/tree/development/Tools/DevUtils>`__ and run the Python script `update_benchmarks_from_azure_output.py <https://github.com/ECP-WarpX/WarpX/blob/development/Tools/DevUtils/update_benchmarks_from_azure_output.py>`__ passing the path of the raw log text file as a command line argument:

   .. code:: bash

        python update_benchmarks_from_azure_output.py path/to/raw_log.txt

   This will update the checksum files for all the tests in the raw log that did not pass the checksum analysis.

.. _developers-testing-naming:

Naming conventions for automated tests
--------------------------------------

Note that we currently obey the following snake\_case naming conventions for test names and test input files (which make automation tasks easier, e.g., parsing visually, parsing through code, sorting alphabetically, filtering tests in CTest via ``-R``, etc.):

#. **Regular test names** start with the string ``test_1d_``, ``test_2d_``, ``test_3d_`` or ``test_rz_``, followed by a string that is descriptive of the test. For example, ``test_3d_laser_acceleration``.

#. **PICMI test names** start with the string ``test_1d_``, ``test_2d_``, ``test_3d_`` or ``test_rz_``, followed by a string that is descriptive of the test, and end with the string ``_picmi``. For example, ``test_3d_laser_acceleration_picmi``.

#. **Restart test names** end with the string ``_restart``. For example, ``test_3d_laser_acceleration_restart``.

#. **Test input files** start with the string ``inputs_`` followed by the test name. For example, ``inputs_test_3d_laser_acceleration`` or ``inputs_test_3d_laser_acceleration_picmi.py`` or ``inputs_test_3d_laser_acceleration_restart``.

#. **Base input files** (that is, files collecting input parameters shared between two or more tests) are typically named ``inputs_base_1d``, ``inputs_base_2d``, ``inputs_base_3d`` or ``inputs_base_rz``, possibly followed by additional strings if need be.

Other resources
---------------

With regard to testing the code more generally, not necessarily in the context of continuous integration, AMReX provides a number of useful post-processing tools for plotfiles.
The complete list of tools can be found `here <https://amrex-codes.github.io/amrex/docs_html/Post_Processing.html>`__.
One tool that traditionally stood out as especially useful for core developers and maintainers is `fcompare <https://amrex-codes.github.io/amrex/docs_html/Post_Processing.html#fcompare>`__.
