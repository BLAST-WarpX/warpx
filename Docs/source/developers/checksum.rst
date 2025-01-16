.. _developers-checksum:

Checksums on Tests
==================

When running an automated test, we often compare the data of final time step of the test with expected values to catch accidental changes.
Instead of relying on reference files that we would have to store in their full size, we calculate an aggregate checksum.

For this purpose, the checksum Python module computes one aggregated number per field (e.g., the sum of the absolute values of the array elements) and compares it to a reference value (benchmark).
This should be sensitive enough to make the test fail if your PR causes a significant difference, print meaningful error messages, and give you a chance to fix a bug or reset the benchmark if needed.

The checksum module is located in ``Regression/Checksum/``, and the benchmarks are stored as human-readable `JSON <https://www.json.org/json-en.html>`__ files in ``Regression/Checksum/benchmarks_json/``, with one file per benchmark (for example, the test ``test_2d_langmuir_multi`` has a corresponding benchmark ``Regression/Checksum/benchmarks_json/test_2d_langmuir_multi.json``).

For more details on the implementation, please refer to the Python implementation in ``Regression/Checksum/``.

From a user point of view, you should only need to use ``checksumAPI.py``, which contains Python functions that can be imported and used from an analysis Python script or can also be executed directly as a Python script.

How to compare checksums in your analysis script
------------------------------------------------

This relies on the function ``evaluate_checksum``:

.. autofunction:: checksumAPI.evaluate_checksum

This can also be included as part of an existing analysis script.

How to evaluate checksums from the command line
-----------------------------------------------

You can execute ``checksumAPI.py`` as a Python script for that, and pass the plotfile that you want to evaluate, as well as the test name (so the script knows which benchmark to compare it to).

.. code-block:: bash

   ./checksumAPI.py --evaluate --output-file <path/to/plotfile> --output-format <'openpmd' or 'plotfile'> --test-name <test name>

See additional options

* ``--skip-fields`` if you don't want the fields to be compared (in that case, the benchmark must not have fields)
* ``--skip-particles`` same thing for particles
* ``--rtol`` relative tolerance for the comparison
* ``--atol`` absolute tolerance for the comparison (a sum of both is used by ``numpy.isclose()``)

How to create or reset checksums with local benchmark values
------------------------------------------------------------

This is using ``checksumAPI.py`` as a Python script.

.. code-block:: bash

   ./checksumAPI.py --reset-benchmark --output-file <path/to/plotfile> --output-format <'openpmd' or 'plotfile'> --test-name <test name>

See additional options

* ``--skip-fields`` if you don't want the benchmark to have fields
* ``--skip-particles`` same thing for particles

Since this will automatically change the JSON file stored on the repo, make a separate commit just for this file, and if possible commit it under the ``Tools`` name:

.. code-block:: bash

   git add <test name>.json
   git commit -m "reset benchmark for <test name> because ..." --author="Tools <warpx@lbl.gov>"

How to reset checksums for a list of tests with local benchmark values
----------------------------------------------------------------------

If you set the environment variable ``export CHECKSUM_RESET=ON`` before running tests that are compared against existing benchmarks, the test analysis will reset the benchmarks to the new values, skipping the comparison.

With `CTest <https://cmake.org/cmake/help/latest/manual/ctest.1.html>`__ (coming soon), select the test(s) to reset by `name <https://cmake.org/cmake/help/latest/manual/ctest.1.html#run-tests>`__ or `label <https://cmake.org/cmake/help/latest/manual/ctest.1.html#label-matching>`__.

.. code-block:: bash

   # regex filter: matched names
   CHECKSUM_RESET=ON ctest --test-dir build -R "Langmuir_multi|LaserAcceleration"

   # ... check and commit changes ...


How to reset checksums for a list of tests with benchmark values from the Azure pipeline output
-----------------------------------------------------------------------------------------------

Alternatively, the benchmarks can be reset using the output of the Azure continuous intergration (CI) tests on Github. The output can be accessed by following the steps below:

* On the Github page of the Pull Request, find (one of) the pipeline(s) failing due to benchmarks that need to be updated and click on "Details".

  .. figure:: https://user-images.githubusercontent.com/49716072/135133589-1fd8a626-ff93-4b9e-983f-acee028e0e4e.png
     :alt: Screen capture showing how to access Azure pipeline output on Github.

* Click on "View more details on Azure pipelines".

  .. figure:: https://user-images.githubusercontent.com/49716072/135133596-8f73afa2-969e-49a4-b4a6-184a4f478a44.png
     :alt: Screen capture showing how to access Azure pipeline output on Github.

* Click on "Build & test".

  .. figure:: https://user-images.githubusercontent.com/49716072/135133607-87324124-6145-4589-9a92-dcc8ea9432e4.png
     :alt: Screen capture showing how to access Azure pipeline output on Github.

From this output, there are two options to reset the benchmarks:

#. For each of the tests failing due to benchmark changes, the output contains the content of the new benchmark file, as shown below.
   This content can be copied and pasted into the corresponding benchmark file.
   For instance, if the failing test is ``LaserAcceleration_BTD``, this content can be pasted into the file ``Regression/Checksum/benchmarks_json/LaserAcceleration_BTD.json``.

   .. figure:: https://user-images.githubusercontent.com/49716072/244415944-3199a933-990b-4bde-94b1-162b7e8e22be.png
      :alt: Screen capture showing how to read new benchmark file from Azure pipeline output.

#. If there are many tests failing in a single Azure pipeline, it might become more convenient to update the benchmarks automatically.
   WarpX provides a script for this, located in ``Tools/DevUtils/update_benchmarks_from_azure_output.py``.
   This script can be used by following the steps below:

   * From the Azure output, click on "View raw log".

     .. figure:: https://user-images.githubusercontent.com/49716072/135133617-764b6daf-a8e4-4a50-afae-d4b3a7568b2f.png
        :alt: Screen capture showing how to download raw Azure pipeline output.

   * This should lead to a page that looks like the image below. Save it as a text file on your local computer.

     .. figure:: https://user-images.githubusercontent.com/49716072/135133624-310df207-5f87-4260-9917-26d5af665d60.png
        :alt: Screen capture showing how to download raw Azure pipeline output.

   * On your local computer, go to the WarpX folder and cd to the ``Tools/DevUtils`` folder.

   * Run the command ``python update_benchmarks_from_azure_output.py /path/to/azure_output.txt``. The benchmarks included in that Azure output should now be updated.

   * Repeat this for every Azure pipeline (e.g. ``cartesian2d``, ``cartesian3d``, ``qed``) that contains benchmarks that need to be updated.
