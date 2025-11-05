# Copyright 2017-2023 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: David Grote, Roelof Groenewald, Axel Huebl
#
# License: BSD-3-Clause-LBNL


from ._libwarpx import libwarpx
from .LoadThirdParty import load_cupy


class ParticleContainerWrapper(object):
    """Wrapper around particle containers.
    This provides a convenient way to query and set data in the particle containers.

    Parameters
    ----------
    species_name: string
        The name of the species to be accessed.
    """

    def __init__(self, species_name):
        self.name = species_name
        self._particle_container = None

    @property
    def particle_container(self):
        if self._particle_container is None:
            try:
                mypc = libwarpx.warpx.multi_particle_container()
                self._particle_container = mypc.get_particle_container_from_name(
                    self.name
                )
            except AttributeError as e:
                msg = "You must initialize WarpX before accessing a ParticleContainerWrapper's particle_container."
                raise AttributeError(msg) from e

        return self._particle_container

    def add_particles(
        self,
        x=None,
        y=None,
        z=None,
        ux=None,
        uy=None,
        uz=None,
        w=None,
        unique_particles=True,
        **kwargs,
    ):
        """
        A function for adding particles to the WarpX simulation.

        Parameters
        ----------
        species_name     : str
            The type of species for which particles will be added

        x, y, z          : arrays or scalars
            The particle positions (m) (default = 0.)

        ux, uy, uz       : arrays or scalars
            The particle proper velocities (m/s) (default = 0.)

        w                : array or scalars
            Particle weights (default = 0.)

        unique_particles : bool
            True means the added particles are duplicated by each process;
            False means the number of added particles is independent of
            the number of processes (default = True)

        kwargs           : dict
            Containing an entry for all the extra particle attribute arrays. If
            an attribute is not given it will be set to 0.
        """
        self.particle_container.add_particles(
            x,
            y,
            z,
            ux,
            uy,
            uz,
            w,
            unique_particles,
            **kwargs,
        )

    def get_particle_count(self, local=False):
        """
        Get the number of particles of this species in the simulation.

        Parameters
        ----------

        local        : bool
            If True the particle count on this processor will be returned.
            Default False.

        Returns
        -------

        int
            An integer count of the number of particles
        """
        return self.particle_container.total_number_of_particles(True, local)

    nps = property(get_particle_count)

    def add_real_comp(self, pid_name, comm=True):
        """
        Add a real component to the particle data array.

        Parameters
        ----------

        pid_name       : str
            Name that can be used to identify the new component

        comm           : bool
            Should the component be communicated
        """
        self.particle_container.add_real_comp(pid_name, comm)

    def get_particle_real_arrays(self, comp_name, level, copy_to_host=False):
        """
        This returns a list of numpy or cupy arrays containing the particle real array data
        on each tile for this process.

        Unless copy_to_host is specified, the data for the arrays are not
        copied, but share the underlying memory buffer with WarpX. The
        arrays are fully writeable.

        Parameters
        ----------

        comp_name      : str
            The component of the array data that will be returned

        level          : int
            The refinement level to reference (default=0)

        copy_to_host   : bool
            For GPU-enabled runs, one can either return the GPU
            arrays (the default) or force a device-to-host copy.

        Returns
        -------

        List of arrays
            The requested particle array data
        """
        comp_idx = self.particle_container.get_real_comp_index(comp_name)

        data_array = []
        for pti in libwarpx.libwarpx_so.WarpXParIter(self.particle_container, level):
            soa = pti.soa()
            idx = soa.get_real_data(comp_idx)
            if copy_to_host:
                data_array.append(idx.to_numpy(copy=True))
            else:
                xp, cupy_status = load_cupy()
                if cupy_status is not None:
                    libwarpx.amr.Print(cupy_status)

                data_array.append(xp.array(idx, copy=False))

        return data_array

    def get_particle_int_arrays(self, comp_name, level, copy_to_host=False):
        """
        This returns a list of numpy or cupy arrays containing the particle int array data
        on each tile for this process.

        Unless copy_to_host is specified, the data for the arrays are not
        copied, but share the underlying memory buffer with WarpX. The
        arrays are fully writeable.

        Parameters
        ----------

        comp_name      : str
            The component of the array data that will be returned

        level          : int
            The refinement level to reference (default=0)

        copy_to_host   : bool
            For GPU-enabled runs, one can either return the GPU
            arrays (the default) or force a device-to-host copy.

        Returns
        -------

        List of arrays
            The requested particle array data
        """
        comp_idx = self.particle_container.get_int_comp_index(comp_name)

        data_array = []
        for pti in libwarpx.libwarpx_so.WarpXParIter(self.particle_container, level):
            soa = pti.soa()
            idx = soa.get_int_data(comp_idx)
            if copy_to_host:
                data_array.append(idx.to_numpy(copy=True))
            else:
                xp, cupy_status = load_cupy()
                if cupy_status is not None:
                    libwarpx.amr.Print(cupy_status)

                data_array.append(xp.array(idx, copy=False))

        return data_array

    def get_particle_idcpu_arrays(self, level, copy_to_host=False):
        """
        This returns a list of numpy or cupy arrays containing the particle idcpu data
        on each tile for this process.

        Unless copy_to_host is specified, the data for the arrays are not
        copied, but share the underlying memory buffer with WarpX. The
        arrays are fully writeable.

        Parameters
        ----------
        level          : int
            The refinement level to reference (default=0)

        copy_to_host   : bool
            For GPU-enabled runs, one can either return the GPU
            arrays (the default) or force a device-to-host copy.

        Returns
        -------

        List of arrays
            The requested particle array data
        """
        data_array = []
        for pti in libwarpx.libwarpx_so.WarpXParIter(self.particle_container, level):
            soa = pti.soa()
            idx = soa.get_idcpu_data()
            if copy_to_host:
                data_array.append(idx.to_numpy(copy=True))
            else:
                xp, cupy_status = load_cupy()
                if cupy_status is not None:
                    libwarpx.amr.Print(cupy_status)

                data_array.append(xp.array(idx, copy=False))

        return data_array

    def get_particle_idcpu(self, level=0, copy_to_host=False):
        """
        Return a list of numpy or cupy arrays containing the particle 'idcpu'
        numbers on each tile.

        Parameters
        ----------

        level        : int
            The refinement level to reference (default=0)

        copy_to_host : bool
            For GPU-enabled runs, one can either return the GPU
            arrays (the default) or force a device-to-host copy.

        Returns
        -------

        List of arrays
            The requested particle idcpu
        """
        return self.get_particle_idcpu_arrays(level, copy_to_host=copy_to_host)

    idcpu = property(get_particle_idcpu)

    def get_particle_id(self, level=0, copy_to_host=False):
        """
        Return a list of numpy or cupy arrays containing the particle 'id'
        numbers on each tile.

        Parameters
        ----------

        level        : int
            The refinement level to reference (default=0)

        copy_to_host : bool
            For GPU-enabled runs, one can either return the GPU
            arrays (the default) or force a device-to-host copy.

        Returns
        -------

        List of arrays
            The requested particle ids
        """
        idcpu = self.get_particle_idcpu(level, copy_to_host)
        return [libwarpx.amr.unpack_ids(tile) for tile in idcpu]

    def get_particle_cpu(self, level=0, copy_to_host=False):
        """
        Return a list of numpy or cupy arrays containing the particle 'cpu'
        numbers on each tile.

        Parameters
        ----------

        level        : int
            The refinement level to reference (default=0)

        copy_to_host : bool
            For GPU-enabled runs, one can either return the GPU
            arrays (the default) or force a device-to-host copy.

        Returns
        -------

        List of arrays
            The requested particle cpus
        """
        idcpu = self.get_particle_idcpu(level, copy_to_host)
        return [libwarpx.amr.unpack_cpus(tile) for tile in idcpu]

    def get_particle_x(self, level=0, copy_to_host=False):
        """
        Return a list of numpy or cupy arrays containing the particle 'x'
        positions on each tile.

        Parameters
        ----------

        level        : int
            The refinement level to reference (default=0)

        copy_to_host : bool
            For GPU-enabled runs, one can either return the GPU
            arrays (the default) or force a device-to-host copy.

        Returns
        -------

        List of arrays
            The requested particle x position
        """
        return self.get_particle_real_arrays("x", level, copy_to_host=copy_to_host)

    xp = property(get_particle_x)

    def get_particle_y(self, level=0, copy_to_host=False):
        """
        Return a list of numpy or cupy arrays containing the particle 'y'
        positions on each tile.

        Parameters
        ----------

        level        : int
            The refinement level to reference (default=0)

        copy_to_host : bool
            For GPU-enabled runs, one can either return the GPU
            arrays (the default) or force a device-to-host copy.

        Returns
        -------

        List of arrays
            The requested particle y position
        """
        return self.get_particle_real_arrays("y", level, copy_to_host=copy_to_host)

    yp = property(get_particle_y)

    def get_particle_r(self, level=0, copy_to_host=False):
        """
        Return a list of numpy or cupy arrays containing the particle 'r'
        positions on each tile.

        Parameters
        ----------

        level        : int
            The refinement level to reference (default=0)

        copy_to_host : bool
            For GPU-enabled runs, one can either return the GPU
            arrays (the default) or force a device-to-host copy.

        Returns
        -------

        List of arrays
            The requested particle r position
        """
        xp, cupy_status = load_cupy()

        if libwarpx.geometry_dim == "rz":
            return self.get_particle_x(level, copy_to_host)
        elif libwarpx.geometry_dim == "3d":
            x = self.get_particle_x(level, copy_to_host)
            y = self.get_particle_y(level, copy_to_host)
            return xp.sqrt(x**2 + y**2)
        elif libwarpx.geometry_dim == "2d" or libwarpx.geometry_dim == "1d":
            raise Exception(
                "get_particle_r: There is no r coordinate with 1D or 2D Cartesian"
            )

    rp = property(get_particle_r)

    def get_particle_theta(self, level=0, copy_to_host=False):
        """
        Return a list of numpy or cupy arrays containing the particle
        theta on each tile.

        Parameters
        ----------

        level        : int
            The refinement level to reference (default=0)

        copy_to_host : bool
            For GPU-enabled runs, one can either return the GPU
            arrays (the default) or force a device-to-host copy.

        Returns
        -------

        List of arrays
            The requested particle theta position
        """
        xp, cupy_status = load_cupy()

        if libwarpx.geometry_dim == "rz":
            return self.get_particle_real_arrays("theta", level, copy_to_host)
        elif libwarpx.geometry_dim == "3d":
            x = self.get_particle_x(level, copy_to_host)
            y = self.get_particle_y(level, copy_to_host)
            return xp.arctan2(y, x)
        elif libwarpx.geometry_dim == "2d" or libwarpx.geometry_dim == "1d":
            raise Exception(
                "get_particle_theta: There is no theta coordinate with 1D or 2D Cartesian"
            )

    thetap = property(get_particle_theta)

    def get_particle_z(self, level=0, copy_to_host=False):
        """
        Return a list of numpy or cupy arrays containing the particle 'z'
        positions on each tile.

        Parameters
        ----------

        level        : int
            The refinement level to reference (default=0)

        copy_to_host : bool
            For GPU-enabled runs, one can either return the GPU
            arrays (the default) or force a device-to-host copy.

        Returns
        -------

        List of arrays
            The requested particle z position
        """
        return self.get_particle_real_arrays("z", level, copy_to_host=copy_to_host)

    zp = property(get_particle_z)

    def get_particle_weight(self, level=0, copy_to_host=False):
        """
        Return a list of numpy or cupy arrays containing the particle
        weight on each tile.

        Parameters
        ----------

        level        : int
            The refinement level to reference (default=0)

        copy_to_host : bool
            For GPU-enabled runs, one can either return the GPU
            arrays (the default) or force a device-to-host copy.

        Returns
        -------

        List of arrays
            The requested particle weight
        """
        return self.get_particle_real_arrays("w", level, copy_to_host=copy_to_host)

    wp = property(get_particle_weight)

    def get_particle_ux(self, level=0, copy_to_host=False):
        """
        Return a list of numpy or cupy arrays containing the particle
        x momentum on each tile.

        Parameters
        ----------

        level        : int
            The refinement level to reference (default=0)

        copy_to_host : bool
            For GPU-enabled runs, one can either return the GPU
            arrays (the default) or force a device-to-host copy.

        Returns
        -------

        List of arrays
            The requested particle x momentum
        """
        return self.get_particle_real_arrays("ux", level, copy_to_host=copy_to_host)

    uxp = property(get_particle_ux)

    def get_particle_uy(self, level=0, copy_to_host=False):
        """
        Return a list of numpy or cupy arrays containing the particle
        y momentum on each tile.

        Parameters
        ----------

        level        : int
            The refinement level to reference (default=0)

        copy_to_host : bool
            For GPU-enabled runs, one can either return the GPU
            arrays (the default) or force a device-to-host copy.

        Returns
        -------

        List of arrays
            The requested particle y momentum
        """
        return self.get_particle_real_arrays("uy", level, copy_to_host=copy_to_host)

    uyp = property(get_particle_uy)

    def get_particle_uz(self, level=0, copy_to_host=False):
        """
        Return a list of numpy or cupy arrays containing the particle
        z momentum on each tile.

        Parameters
        ----------

        level        : int
            The refinement level to reference (default=0)

        copy_to_host : bool
            For GPU-enabled runs, one can either return the GPU
            arrays (the default) or force a device-to-host copy.

        Returns
        -------

        List of arrays
            The requested particle z momentum
        """

        return self.get_particle_real_arrays("uz", level, copy_to_host=copy_to_host)

    uzp = property(get_particle_uz)

    def get_species_charge_sum(self, local=False):
        """
        Returns the total charge in the simulation due to the given species.

        Parameters
        ----------

        local          : bool
            If True return total charge per processor
        """
        return self.particle_container.sum_particle_charge(local)

    def get_species_energy_sum(self, local=False):
        """
        Returns the total kinetic energy in the simulation due to the given species.

        Parameters
        ----------

        local          : bool
            If True return total energy per processor
        """
        return self.particle_container.sum_particle_energy(local)

    def getex(self):
        raise NotImplementedError("Particle E fields not supported")

    ex = property(getex)

    def getey(self):
        raise NotImplementedError("Particle E fields not supported")

    ey = property(getey)

    def getez(self):
        raise NotImplementedError("Particle E fields not supported")

    ez = property(getez)

    def getbx(self):
        raise NotImplementedError("Particle B fields not supported")

    bx = property(getbx)

    def getby(self):
        raise NotImplementedError("Particle B fields not supported")

    by = property(getby)

    def getbz(self):
        raise NotImplementedError("Particle B fields not supported")

    bz = property(getbz)

    def deposit_charge_density(self, level, clear_rho=True, sync_rho=True):
        """
        Deposit this species' charge density in rho_fp in order to
        access that data via pywarpx.fields.RhoFPWrapper().

        Parameters
        ----------
        species_name   : str
            The species name that will be deposited.
        level          : int
            Which AMR level to retrieve scraped particle data from.
        clear_rho      : bool
            If True, zero out rho_fp before deposition.
        sync_rho       : bool
            If True, perform MPI exchange and properly set boundary cells for rho_fp.
        """
        rho_fp = libwarpx.warpx.multifab("rho_fp", level)

        if rho_fp is None:
            raise RuntimeError("Multifab `rho_fp` is not allocated.")

        if clear_rho:
            rho_fp.set_val(0.0)

        # deposit the charge density from the desired species
        self.particle_container.deposit_charge(rho_fp, level)

        if libwarpx.geometry_dim == "rz":
            libwarpx.warpx.apply_inverse_volume_scaling_to_charge_density(rho_fp, level)

        if sync_rho:
            libwarpx.warpx.sync_rho()


class ParticleBoundaryBufferWrapper(object):
    """Wrapper around particle boundary buffer containers.
    This provides a convenient way to query data in the particle boundary
    buffer containers.
    """

    def __init__(self):
        self._particle_buffer = None

    @property
    def particle_buffer(self):
        if self._particle_buffer is None:
            try:
                self._particle_buffer = libwarpx.warpx.get_particle_boundary_buffer()
            except AttributeError as e:
                msg = "You must initialize WarpX before accessing a ParticleBoundaryBufferWrapper's particle_buffer."
                raise AttributeError(msg) from e

        return self._particle_buffer

    def get_particle_boundary_buffer_size(self, species_name, boundary, local=False):
        """
        This returns the number of particles that have been scraped so far in the simulation
        from the specified boundary and of the specified species.

        Parameters
        ----------

        species_name   : str
            Return the number of scraped particles of this species

        boundary       : str
            The boundary from which to get the scraped particle data in the
            form x/y/z_hi/lo

        local          : bool
            Whether to only return the number of particles in the current
            processor's buffer
        """
        return self.particle_buffer.get_num_particles_in_container(
            species_name, self._get_boundary_number(boundary), local=local
        )

    def get_particle_boundary_buffer(self, species_name, boundary, comp_name, level):
        """
        This returns a list of numpy or cupy arrays containing the particle array data
        for a species that has been scraped by a specific simulation boundary.

        The data for the arrays are not copied, but share the underlying
        memory buffer with WarpX. The arrays are fully writeable.

        You can find `here https://github.com/BLAST-WarpX/warpx/blob/319e55b10ad4f7c71b84a4fb21afbafe1f5b65c2/Examples/Tests/particle_boundary_interaction/PICMI_inputs_rz.py`
        an example of a simple case of particle-boundary interaction (reflection).

        Parameters
        ----------

            species_name   : str
                The species name that the data will be returned for.

            boundary       : str
                The boundary from which to get the scraped particle data in the
                form x/y/z_hi/lo or eb.

            comp_name      : str
                The component of the array data that will be returned.
                "x", "y", "z", "ux", "uy", "uz", "w"
                "stepScraped","deltaTimeScraped",
                if boundary='eb': "nx", "ny", "nz"

            level          : int
                Which AMR level to retrieve scraped particle data from.
        """
        xp, cupy_status = load_cupy()

        part_container = self.particle_buffer.get_particle_container(
            species_name, self._get_boundary_number(boundary)
        )
        data_array = []
        # loop over the real attributes
        if comp_name in part_container.real_soa_names:
            comp_idx = part_container.get_real_comp_index(comp_name)
            for ii, pti in enumerate(
                libwarpx.libwarpx_so.BoundaryBufferParIter(part_container, level)
            ):
                soa = pti.soa()
                data_array.append(xp.array(soa.get_real_data(comp_idx), copy=False))
        # loop over the integer attributes
        elif comp_name in part_container.int_soa_names:
            comp_idx = part_container.get_int_comp_index(comp_name)
            for ii, pti in enumerate(
                libwarpx.libwarpx_so.BoundaryBufferParIter(part_container, level)
            ):
                soa = pti.soa()
                data_array.append(xp.array(soa.get_int_data(comp_idx), copy=False))
        else:
            raise RuntimeError("Name %s not found" % comp_name)
        return data_array

    def get_particle_scraped_this_step(self, species_name, boundary, comp_name, level):
        """
        This returns a list of numpy or cupy arrays containing the particle array data
        for particles that have been scraped at the current timestep,
        for a specific species and simulation boundary.

        The data for the arrays is a view of the underlying boundary buffer in WarpX ;
        writing to these arrays will therefore also modify the underlying boundary buffer.

        Parameters
        ----------

            species_name   : str
                The species name that the data will be returned for.

            boundary       : str
                The boundary from which to get the scraped particle data in the
                form x/y/z_hi/lo or eb.

            comp_name      : str
                The component of the array data that will be returned.
                "x", "y", "z", "ux", "uy", "uz", "w"
                "stepScraped","deltaTimeScraped",
                if boundary='eb': "nx", "ny", "nz"

            level          : int
                Which AMR level to retrieve scraped particle data from.
        """
        # Extract the integer number of the current timestep
        current_step = libwarpx.libwarpx_so.get_instance().getistep(level)

        # Extract the data requested by the user
        data_array = self.get_particle_boundary_buffer(
            species_name, boundary, comp_name, level
        )
        step_scraped_array = self.get_particle_boundary_buffer(
            species_name, boundary, "stepScraped", level
        )

        # Select on the particles from the previous step
        data_array_this_step = []
        for data, step in zip(data_array, step_scraped_array):
            data_array_this_step.append(data[step == current_step])
        return data_array_this_step

    def clear_buffer(self):
        """

        Clear the buffer that holds the particles lost at the boundaries.

        """
        self.particle_buffer.clear_particles()

    def _get_boundary_number(self, boundary):
        """

        Utility function to find the boundary number given a boundary name.

        Parameters
        ----------

        boundary       : str
            The boundary from which to get the scraped particle data. In the
            form x/y/z_hi/lo or eb.

        Returns
        -------
        int
            Integer index in the boundary scraper buffer for the given boundary.
        """
        if libwarpx.geometry_dim == "3d":
            dimensions = {"x": 0, "y": 1, "z": 2}
        elif libwarpx.geometry_dim == "2d" or libwarpx.geometry_dim == "rz":
            dimensions = {"x": 0, "z": 1}
        elif libwarpx.geometry_dim == "1d":
            dimensions = {"z": 0}
        else:
            raise RuntimeError(f"Unknown simulation geometry: {libwarpx.geometry_dim}")

        if boundary != "eb":
            boundary_parts = boundary.split("_")
            dim_num = dimensions[boundary_parts[0]]
            if boundary_parts[1] == "lo":
                side = 0
            elif boundary_parts[1] == "hi":
                side = 1
            else:
                raise RuntimeError(f"Unknown boundary specified: {boundary}")
            boundary_num = 2 * dim_num + side
        else:
            if libwarpx.geometry_dim == "3d":
                boundary_num = 6
            elif libwarpx.geometry_dim == "2d" or libwarpx.geometry_dim == "rz":
                boundary_num = 4
            elif libwarpx.geometry_dim == "1d":
                boundary_num = 2
            else:
                raise RuntimeError(
                    f"Unknown simulation geometry: {libwarpx.geometry_dim}"
                )

        return boundary_num
