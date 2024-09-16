# Copyright 2022 Marco Garten
#
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

# requirements:
# - python

import numpy as np


class MemoryCalculator:
    """
    Memory requirement calculation tool for WarpX

    Contains calculation for fields, particles and random number generation.

    @TODO add memory complexity of diagnostics
    """

    def __init__(
        self,
        n_x,
        n_y,
        n_z,
        build_dim,
        particle_shape_order=3,
        precision="double",
    ):
        """
        Class constructor

        Parameters
        ----------

        n_x : int
            number of cells in x direction (per device)
        n_y : int
            number of cells in y direction (per device)
        n_z : int
            number of cells in z direction (per device)
        build_dim : int
            simulation build dimension [3 (default), 2, 1, "RZ"]
            (see build option WarpX_DIMS)
        particle_shape_order : int
            order of the shape factors (splines) for the macro-particles
            along all spatial directions: 1 for linear, 2 for quadratic, 3 for cubic
            (see algo.particle_shape)
        precision : str
            floating point precision for WarpX (build option, single/double)
        """
        # local device domain size
        self.n_x = n_x
        self.n_y = n_y
        self.n_z = n_z

        self.local_cells = self.n_x * self.n_y * self.n_z

        if build_dim == "RZ":
            self.sim_dim = 2
        elif build_dim in [1, 2, 3]:
            self.sim_dim = build_dim
        else:
            raise ValueError(
                "Please choose a value from [1,2,3,'RZ']. ",
                build_dim,
                " is not a supported choice.",
            )

        self.build_dim = build_dim
        self.particle_shape_order = particle_shape_order
        self.precision = precision

        if self.precision == "single":
            # value size in bytes
            self.value_size = np.float32().itemsize
        elif self.precision == "double":
            # value size in bytes
            self.value_size = np.float64().itemsize
        else:
            raise ValueError(
                "The build option WarpX_PRECISION can either be SINGLE or DOUBLE (default)"
            )

    def get_guard_cells_per_box(self, guard_cells_per_dim=[]):
        """
        Get the number of guard cells around the box.

        Parameters
        ----------

        guard_cells_per_dim : list
            only needs to be set explicitly when using the PSATD solver
            (see e.g. psatd.nx_guard) and is otherwise derived
            from the particle shape


        Returns
        -------

        guard_cells_per_box : int
        """
        if self.sim_dim == len(guard_cells_per_dim):
            pass
        elif not guard_cells_per_dim:
            pass
        else:
            raise ValueError(
                "The simulation dimension does not agree with the length of the guard cell list."
            )

        pso = self.particle_shape_order

        if not guard_cells_per_dim:
            # set guard size according to particle shape
            guard_cells_per_dim = np.tile(pso, self.sim_dim)
        else:
            guard_cells_per_dim = np.array([guard_cells_per_dim])

        if self.sim_dim == 1:
            local_cells = self.n_x + 2 * guard_cells_per_dim[0]
            guard_cells_per_box = local_cells - self.n_x

        elif self.sim_dim == 2:
            local_cells = (self.n_x + 2 * guard_cells_per_dim[0]) * (
                self.n_y + 2 * guard_cells_per_dim[1]
            )
            guard_cells_per_box = local_cells - self.n_x * self.n_y

        elif self.sim_dim == 3:
            local_cells = (
                (self.n_x + 2 * guard_cells_per_dim[0])
                * (self.n_y + 2 * guard_cells_per_dim[1])
                * (self.n_z + 2 * guard_cells_per_dim[2])
            )
            guard_cells_per_box = local_cells - (self.n_x * self.n_y * self.n_z)

        return guard_cells_per_box

    def get_local_pml_cells(self, pml_cells_per_dim=[]):
        """
        Get number of PML cells, assuming PMLs are around the whole box.

        Parameters
        ----------

        pml_cells_per_dim : list
            number of PML cells per dimension

        Returns
        -------

        local_pml_cells : int
        """
        if self.sim_dim == len(pml_cells_per_dim):
            pass
        else:
            raise ValueError(
                "The simulation dimension does not agree with the length of the PML cell list."
            )

        if self.sim_dim == 1:
            pml_n_x = list(pml_cells_per_dim)[0]
            local_pml_cells = pml_n_x

        elif self.sim_dim == 2:
            pml_n_x, pml_n_y = pml_cells_per_dim
            local_pml_cells = self.n_x * self.n_y - (self.n_x - pml_n_x) * (
                self.n_y - pml_n_y
            )

        elif self.sim_dim == 3:
            pml_n_x, pml_n_y, pml_n_z = pml_cells_per_dim
            local_pml_cells = self.n_x * self.n_y * self.n_z - (self.n_x - pml_n_x) * (
                self.n_y - pml_n_y
            ) * (self.n_z - pml_n_z)

        return local_pml_cells

    def mem_req_by_fields(
        self,
        n_x=None,
        n_y=None,
        n_z=None,
        dive_cleaning=True,
        divb_cleaning=True,
        pml_ncell=10,
        guard_cells_per_dim=[],
    ):
        """
        Memory reserved for fields on each device

        @TODO handle embedded boundaries?

        Parameters
        ----------

        n_x : int
            number of cells in x direction (per device)
        n_y : int
            number of cells in y direction (per device)
        n_z : int
            number of cells in z direction (per device)
        dive_cleaning : bool
            Correction for div E = rho in PML region.
            Requires divb_cleaning=True.
            see warpx.do_pml_dive_cleaning
        divb_cleaning : bool
            Correction for div B = 0 in PML region.
            Requires dive_cleaning=True.
            see warpx.do_pml_divb_cleaning
        pml_ncell : int
            number of PML cells in each direction
            Set to 0 if boundaries have no PMLs.


        Returns
        -------

        req_mem : int
            required memory {unit: bytes} per device
        """

        if n_x is None:
            n_x = self.n_x
        if n_y is None:
            n_y = self.n_y
        if n_z is None:
            n_z = self.n_z

        pml_cell_mem = 0

        if pml_ncell == 0:
            pass

        else:
            # PML size cannot exceed the local grid size
            # @TODO is that true for WarpX?
            pml_n_x = min(pml_ncell, n_x)
            pml_n_y = min(pml_ncell, n_y)
            pml_n_z = min(pml_ncell, n_z)

            if self.sim_dim == 1:
                local_pml_cells = self.get_local_pml_cells(pml_cells_per_dim=[pml_n_z])
            elif self.sim_dim == 2:
                local_pml_cells = self.get_local_pml_cells(
                    pml_cells_per_dim=[pml_n_x, pml_n_z]
                )
            elif self.sim_dim == 3:
                local_pml_cells = self.get_local_pml_cells(
                    pml_cells_per_dim=[pml_n_x, pml_n_y, pml_n_z]
                )
            else:
                raise ValueError("Invalid number of dimensions: ", self.sim_dim)
            # number of additional PML field components
            # @TODO figure out how many fields for each PML configuration
            # see Source/BoundaryConditions/PML.H#L214-233
            # 3x E, 3x B, 3x J, and F and G?

            # PML splitting: 2 values for E and B in each component but no diagonal
            #                          E       B   J
            num_pml_field_values = 2 * 3 + 2 * 3 + 3

            if dive_cleaning is True and divb_cleaning is True:
                # diagonal values for E, B, and 3 scalar values for F and G
                num_pml_field_values += 3 + 3 + 3

            pml_cell_mem = self.value_size * num_pml_field_values * local_pml_cells

        double_buffer_cells = self.get_guard_cells_per_box(
            guard_cells_per_dim=guard_cells_per_dim
        )

        # @TODO: find out how many temporary fields there can be and how much memory they take
        temporary_field_slots = 1
        # number of fields: 3 * 3 = x,y,z for E,B,J
        num_fields = 3 * 3 + temporary_field_slots
        # double buffer memory
        double_buffer_mem = double_buffer_cells * num_fields * self.value_size

        req_mem = (
            self.value_size * num_fields * self.local_cells
            + double_buffer_mem
            + pml_cell_mem
        )

        return req_mem

    def mem_req_by_species(
        self,
        target_n_x=None,
        target_n_y=None,
        target_n_z=None,
        num_additional_ints=0,
        num_additional_reals=0,
        particles_per_cell=2,
    ):
        """
        Memory reserved for all particles of a species on a device.
        We currently neglect the constant species memory.

        IMPORTANT NOTE
        --------------
        Do not forget to set <species>.xmin/xmax/... etc. because especially
        simulations of overdense localized targets could otherwise crash on
        initialization. Particles will be created in parallel first and then
        set to be invalid for regions outside of density thresholds.

        Parameters
        ----------

        target_n_x : int
            number of cells in x direction containing the target
        target_n_y : int
            number of cells in y direction containing the target
        target_n_z : int
            number of cells in z direction containing the target
        num_additional_ints : int
            number of additional int attributes
        num_additional_reals : int
            number of additional real attributes
        particles_per_cell : int
            number of particles of the species per cell

        Returns
        -------

        req_mem : int
            required memory {unit: bytes} per device and species
        """

        if target_n_x is None:
            target_n_x = self.n_x
        if target_n_y is None:
            target_n_y = self.n_y
        if target_n_z is None:
            target_n_z = self.n_z

        # memory required by the standard particle attributes
        standard_attribute_mem = np.array(
            [
                3 * self.value_size,  # momentum
                self.sim_dim * self.value_size,  # position
                1 * np.int32().itemsize,  # particle id
                1 * np.int32().itemsize,  # cpu id
                1 * self.value_size,  # weighting
            ]
        )

        # memory per particle for additional attributes {unit: byte}
        additional_int_mem = num_additional_ints * np.int32().itemsize
        additional_real_mem = num_additional_reals * self.value_size
        additional_mem = additional_int_mem + additional_real_mem

        target_cells = target_n_x * target_n_y * target_n_z

        req_mem = (
            target_cells
            * particles_per_cell
            * (np.sum(standard_attribute_mem) + additional_mem)
        )

        return req_mem

    def mem_req_by_rng(self, warpx_compute="CUDA", gpu_model="A100", omp_num_threads=1):
        """
        Memory reserved for the random number generator.

        Parameters
        ----------
        warpx_compute : str
            Build method WarpX_COMPUTE
            influences which random number generator is used.
        gpu_model : str
            Only valid for `warpx_compute` being either `"CUDA"`, `"HIP"` or `"SYCL"`.
            This influences the number of multiprocessors and the number of threads per multiprocessor.
            These eventually determine the size of the RNG state.
        omp_num_threads : int
            number of OpenMP threads that are working on one box

        Returns
        -------

        req_mem : int
            required memory {unit: bytes} per device
            :param gpu_model:
        """

        warpx_compute_list = ["CUDA", "HIP", "OMP", "SYCL", "NOACC"]

        if warpx_compute not in warpx_compute_list:
            raise ValueError(
                "{} is not an available option for WarpX_COMPUTE.".format(
                    warpx_compute
                ),
                "Please choose one of the following: ",
                warpx_compute_list,
            )
        else:
            pass

        generator_method_d = {
            "CUDA": "XORWOW",  # curand
            "HIP": "XORWOW",  # hiprand
            "OMP": "mt19937",
            "SYCL": "Philox4x32x10",  # oneapi math kernel library
            "NOACC": "mt19937",
        }

        generator_method = generator_method_d[warpx_compute]

        # @TODO perhaps let the user
        multProc_num_and_maxThreads_d = {"A100": [108, 2048]}

        req_mem = 0
        if generator_method == "XORWOW":
            # state size: N * sizeof(randstate_t)
            # N = 4 * numMultiProcessors() * maxThreadsPerMultiProcessors()
            # Perlmutter A100:  108 * 2048
            num_states = np.prod(multProc_num_and_maxThreads_d[gpu_model])
            state_size = 6 * 8  # bytes
            req_mem = state_size * num_states
        elif generator_method == "mt19937":
            state_size = np.ceil(19937 / 8)  # bytes
            if warpx_compute == "OMP":
                req_mem = state_size * omp_num_threads
            elif warpx_compute == "NOACC":
                # we just have a single generator state
                req_mem = state_size
            else:
                pass
        elif generator_method == "Philox4x32x10":
            # 128 Bit counter + 2 * 32 Bit Keys
            num_states = np.prod(multProc_num_and_maxThreads_d[gpu_model])
            state_size_per_engine = 16 + 2 * 4  # bytes
            req_mem = state_size_per_engine * num_states
        else:
            raise ValueError(
                "The chosen random number generator method",
                generator_method,
                "is not available. Please choose one from this list:",
                list(generator_method_d.keys()),
            )

        return req_mem
