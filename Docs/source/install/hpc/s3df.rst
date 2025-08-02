.. _building-s3df:

S3DF (SLAC)
=============

The S3DF cluster is located at SLAC.


Introduction
------------

If you are new to this system, **please see the following resources**:

* `S3DF documentation <https://s3df.slac.stanford.edu/#/>`__
* Batch system: `Slurm <https://github.com/slaclab/sdf-docs/blob/main/batch-compute.md>`__
* Filesystem locations:
    * User folder: ``$HOME>`` (25GB on Weka file system)
    * Scratch folder: ``/sdf/scratch/<username_initial>/<username>`` (100GB on Weka file system)
    * Group storage: Depends on your group, see e.g. information for the ATLAS group `here <https://usatlas.readthedocs.io/projects/af-docs/en/latest/sshlogin/ssh2SLAC/>`__

Through S3DF we have access to CPU and GPU nodes (the latter equipped with NVIDIA A100).



Installation
------------


1) Prerequisites
^^^^^^^^^^^^^^^^^^^^^^^^^


- S3DF login to a node with access to A100 GPUs (via SLURM job or interactive session).
- Conda available in your shell (Miniconda/Anaconda on S3DF user space).
- GPU: NVIDIA A100 (sm_80).

For other GPUs, adjust ``-DCMAKE_CUDA_ARCHITECTURES``:

- V100: `70`
- A100: `80`
- H100: `90`

.. note::

  SLURM tip (S3DF):

  -  Use srun to have GPU nodes allocated. WarpX requires a "task" per CPU-GPU pair, so if you want to run WarpX e.g. on 4 GPUs, you would do:

  .. code-block:: bash

    srun --pty --cpus-per-task=8 --ntasks-per-node=4 --gpus=4 --mem=100GB --nodes=1 --time=2:00:00 --partition=ampere --account=atlas bash




2) Create & Populate the Conda Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


  .. code-block:: bash

    # A) create env (example set; versions can vary by site repos)

    conda create -n warpx-gpu-test -y python=3.11
    conda activate warpx-gpu-test

    #Get the full CUDA toolkit in your conda env.
    conda install -c nvidia -y "cuda=12.9.*"

    # Additionally install NVTX dev tools (cuda separates from the runtime version)
    conda install -c nvidia -y "cuda-nvtx-dev=12.9.*"

    # B) Install build tools
    conda install -c conda-forge -y \
      "cmake>=3.27" ninja make pkg-config git \
      openmpi ucx mpi4py \
      gcc_linux-64=13 gxx_linux-64=13

    # C) Install ADIOS2 with OpenMPI (so we can use openpmd_backend = bp)
    conda install -c conda-forge "adios2=*=mpi_openmpi*"

    # D) Verify build tools
    which cmake && cmake --version     # should show cmake >= 3.24 from your env
    which nvcc  && nvcc  --version     # should show CUDA 12.9 from the nvidia 'cuda' metapackage
    which mpicxx && ompi_info | head   # Open MPI from conda-forge



.. note::
  - ``cuda-nvtx-dev`` provides the nvtx3 headers used by recent NVTX versions.
  - ``cuda-nvcc`` brings nvcc under ``$CONDA_PREFIX/bin/nvcc``.
  - MPI here is via ``mpich`` in the env (``mpicc``, ``mpicxx`` will be at ``$CONDA_PREFIX/bin/...``).


3) Download WarpX
^^^^^^^^^^^^^^^^^

.. code-block:: bash

  git clone https://github.com/BLAST-WarpX/warpx.git
  cd warpx




4) Environment Fixes (NVTX headers + dlopen/dlsym)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Recent NVTX installs the header under ``nvtx3/nvToolsExt.h``. Add it to your include path, and ensure ``libdl`` is linked (``CUDA/NVTX`` uses ``dlopen()/dlsym()`` dynamically):

.. code-block:: bash

  # ensure libdl is not dropped by the linker (GCC + gold/ld with --as-needed)
  export LDFLAGS="${LDFLAGS} -Wl,--no-as-needed -ldl"

  # verify the header exists (one of these should print a path)
  ls "$CONDA_PREFIX/targets/x86_64-linux/include/nvtx3/nvToolsExt.h" \
   || ls "$CONDA_PREFIX/include/nvtx3/nvToolsExt.h"

  # include & library paths from the conda CUDA toolchain
  export CUDACXX="$CONDA_PREFIX/bin/nvcc"
  export CUDA_HOME="$CONDA_PREFIX"

  export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include/nvtx3:$CONDA_PREFIX/targets/x86_64-linux/include"
  export LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib"
  export LD_LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib"

  # help CMake find ADIOS2/openPMD (if using conda-provided openPMD)
  export ADIOS2_DIR="$CONDA_PREFIX"
  export openPMD_DIR="$CONDA_PREFIX"




5) Configure with CMake (CUDA + MPI, 3D)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. code-block:: bash

  "$CONDA_PREFIX/bin/cmake" -S . -B build \
    -DWarpX_APP=ON \
    -DWarpX_DIMS="3" \
    -DWarpX_COMPUTE=CUDA \
    -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared \
    -DCMAKE_CUDA_COMPILER="$CONDA_PREFIX/bin/nvcc" \
    -DCMAKE_CUDA_HOST_COMPILER="$(which mpicxx)" \
    -DCMAKE_C_COMPILER="$(which mpicc)" \
    -DCMAKE_CXX_COMPILER="$(which mpicxx)" \
    -DWarpX_FFT=ON



.. note::
  What these do:

  - ``WarpX_APP=ON``: build the warpx application.
  - ``WarpX_DIMS="3"``: build the 3D executable.
  - ``WarpX_COMPUTE=CUDA``: enable GPU backend.
  - ``CMAKE_CUDA_ARCHITECTURES=80``: targets A100.
  - ``CMAKE_CUDA_RUNTIME_LIBRARY=Shared``: link CUDA runtime dynamically (matches conda toolchain layout).
  - Compilers explicitly point to MPI wrappers from the env.



6) Build
^^^^^^^^


.. code-block:: bash

  "$CONDA_PREFIX/bin/cmake" --build build -j 8



7) Run (MPI + GPU)
^^^^^^^^^^^^^^^^^^

From the build dir:

.. code-block:: bash

  mkdir run_test  # create run folder for each new test!
  cp build/bin/warpx.3d run_test/  # copy warpx executable to each run folder!
  cp Examples/Physics_applications/beam_beam_collision/inputs_test_3d_beam_beam_collision run_test/ #copy input file!
  cd run_test/
  mpirun -np 4 ./warpx.3d inputs_test_3d_beam_beam_collision # run!
