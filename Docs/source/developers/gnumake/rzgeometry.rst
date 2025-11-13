Building WarpX to use RZ geometry
=================================

WarpX can be built to run with RZ geometry. Both an FDTD solver (the default)
and a PSATD solver are available. Both solvers allow multiple azimuthal modes.

To select RZ geometry, set the flag USE_RZ = TRUE when compiling:
::

    make -j 4 USE_RZ=TRUE

Note that this sets DIM=2, which is required with USE_RZ=TRUE.
The executable produced will have "RZ" as a suffix.

RZ geometry with spectral solver
--------------------------------

Additional steps are needed to build the spectral solver. Some of the steps
are the same as is done for the Cartesian spectral solver, setting up the FFTW
package and setting ``USE_FFT=TRUE``.

      - Install (or load) an MPI-enabled version of FFTW.
        For instance, for Debian, this can be done with
        ::

           apt-get install libfftw3-dev libfftw3-mpi-dev

      - Set the environment variable ``FFTW_HOME`` to the path for FFTW.
        For instance, for Debian, this is done with
        ::

           export FFTW_HOME=/usr/

      - Download and build the blaspp and lapackpp packages. These can be obtained from GitHub.
        ::

           git clone https://github.com/icl-utk-edu/blaspp.git
           git clone https://github.com/icl-utk-edu/lapackpp.git

        The two packages can be built in multiple ways. A recommended method is to follow the cmake instructions
        provided in the INSTALL.md that comes with the packages. They can also be installed using spack.

      - Set the environment variables BLASPP_HOME and LAPACKPP_HOME to the locations where
        the packages libraries were installed. For example, using bash:
        ::

           export BLASPP_HOME=/location/of/installation/blaspp
           export LAPACKPP_HOME=/location/of/installation/lapackpp

      - In some case, the blas and lapack libraries need to be specified.
        If needed, this can be done by setting the BLAS_LIB and LAPACK_LIB
        environment variables appropriately. For example, using bash:
        ::

           export BLAS_LIB=-lblas

      - Set ``USE_FFT=TRUE`` when compiling:
        ::

           make -j 4 USE_RZ=TRUE USE_FFT=TRUE
