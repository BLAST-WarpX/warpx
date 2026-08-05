.. _developers-testing-gpus:

How to test a pull request on GPUs
==================================

In addition to the CPU tests that run automatically on every pull request (see :ref:`developers-testing`), WarpX can run a subset of its test suite on **GPUs** of multiple vendors.
These GPU tests are **optional**: they do not run by default on pull requests and they are not required to pass before a pull request can be merged.

The GPU runners cover the three GPU backends that WarpX supports:

* **NVIDIA** (CUDA), on H100 GPUs;
* **AMD** (HIP/ROCm), on MI300 GPUs;
* **Intel** (SYCL), on Data Center Max (PVC) GPUs.

The runners are provided by the `HPSF <https://hpsf.io>`__ CI working group on the `Frank <https://gitlab.spack.io/groups/hpsf/-/wikis/supported-tags>`__ cluster at the `University of Oregon <https://systems.nic.uoregon.edu/internal-wiki/index.php?title=Category:Servers>`__ and are hosted on `GitLab CI <https://gitlab.spack.io/warpx/warpx>`__.
The job definitions live in `.gitlab/ci.yaml <https://github.com/BLAST-WarpX/warpx/blob/development/.gitlab/ci.yaml>`__.

When do the GPU tests run?
--------------------------

The GPU pipeline runs in two situations:

* on every merge to the ``development`` branch;
* on a pull request, **only** once the ``bot: run GPU`` label is added to it.

WarpX lives on GitHub, while the GPU runners are on GitLab.
A scheduled job mirrors the GitHub repository (including labeled pull requests) to GitLab every 5 minutes, so expect a short delay between adding the label and the pipeline starting.
The mirror's `sync schedule <https://gitlab.spack.io/warpx/warpx/-/pipeline_schedules>`__ and `sync logic <https://gitlab.spack.io/warpx/warpx/-/blob/gh-gl-sync/.gitlab/ci.yaml?ref_type=heads>`__ are configured on GitLab.

How to trigger the GPU tests on a pull request
----------------------------------------------

To run the GPU tests on a pull request, add the ``bot: run GPU`` label to it on GitHub.
Once the change is mirrored to GitLab (see above), a GPU pipeline starts.
You can follow the status of recent runs on the `GitLab jobs page <https://gitlab.spack.io/warpx/warpx/-/jobs?sources=PUSH&kind=BUILD>`__.

To re-run the GPU tests after pushing new commits, remove and re-add the label.

.. warning::

   Adding the ``bot: run GPU`` label causes the pull request's code to be **built and executed on shared GPU CI hardware**.
   Only add this label for **trusted contributors**, and only **after** you have reviewed the pull request's code and confirmed that it is safe to run.
   Do not add it to pull requests from unknown or first-time contributors without a careful look at the changes first.

Getting help
------------

The Frank runners are operated together with the HPSF CI/CD working group.
For questions about runner availability or outages, reach out in the ``#wg-cicd`` channel of the `HPSF Slack <https://hpsf.slack.com>`__.
Current contacts are Zack Galbreath (Kitware) and Luke (University of Oregon).
The list of currently available runners and their tags is maintained on the `HPSF supported tags wiki <https://gitlab.spack.io/groups/hpsf/-/wikis/supported-tags>`__.
