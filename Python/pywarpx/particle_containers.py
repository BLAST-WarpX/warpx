# Copyright 2017-2023 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: David Grote, Roelof Groenewald, Axel Huebl
#
# License: BSD-3-Clause-LBNL

import numpy as np

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

        # --- Get length of arrays, set to one for scalars
        lenx = np.size(x)
        leny = np.size(y)
        lenz = np.size(z)
        lenux = np.size(ux)
        lenuy = np.size(uy)
        lenuz = np.size(uz)
        lenw = np.size(w)

        # --- Find the max length of the parameters supplied
        maxlen = 0
        if x is not None:
            maxlen = max(maxlen, lenx)
        if y is not None:
            maxlen = max(maxlen, leny)
        if z is not None:
            maxlen = max(maxlen, lenz)
        if ux is not None:
            maxlen = max(maxlen, lenux)
        if uy is not None:
            maxlen = max(maxlen, lenuy)
        if uz is not None:
            maxlen = max(maxlen, lenuz)
        if w is not None:
            maxlen = max(maxlen, lenw)

        # --- Make sure that the lengths of the input parameters are consistent
        assert x is None or lenx == maxlen or lenx == 1, (
            "Length of x doesn't match len of others"
        )
        assert y is None or leny == maxlen or leny == 1, (
            "Length of y doesn't match len of others"
        )
        assert z is None or lenz == maxlen or lenz == 1, (
            "Length of z doesn't match len of others"
        )
        assert ux is None or lenux == maxlen or lenux == 1, (
            "Length of ux doesn't match len of others"
        )
        assert uy is None or lenuy == maxlen or lenuy == 1, (
            "Length of uy doesn't match len of others"
        )
        assert uz is None or lenuz == maxlen or lenuz == 1, (
            "Length of uz doesn't match len of others"
        )
        assert w is None or lenw == maxlen or lenw == 1, (
            "Length of w doesn't match len of others"
        )
        for key, val in kwargs.items():
            assert np.size(val) == 1 or len(val) == maxlen, (
                f"Length of {key} doesn't match len of others"
            )

        # --- Broadcast scalars into appropriate length arrays
        # --- If the parameter was not supplied, use the default value
        if lenx == 1:
            x = np.full(maxlen, (x or 0.0))
        if leny == 1:
            y = np.full(maxlen, (y or 0.0))
        if lenz == 1:
            z = np.full(maxlen, (z or 0.0))
        if lenux == 1:
            ux = np.full(maxlen, (ux or 0.0))
        if lenuy == 1:
            uy = np.full(maxlen, (uy or 0.0))
        if lenuz == 1:
            uz = np.full(maxlen, (uz or 0.0))
        if lenw == 1:
            w = np.full(maxlen, (w or 0.0))
        for key, val in kwargs.items():
            if np.size(val) == 1:
                kwargs[key] = np.full(maxlen, val)

        # --- The number of built in attributes
        # --- The positions
        built_in_attrs = libwarpx.dim
        # --- The three velocities
        built_in_attrs += 3
        if libwarpx.geometry_dim == "rz":
            # --- With RZ, there is also theta
            built_in_attrs += 1

        # --- The number of extra attributes (including the weight)
        nattr = self.particle_container.num_real_comps - built_in_attrs
        attr = np.zeros((maxlen, nattr))
        attr[:, 0] = w

        # --- Note that the velocities are handled separately and not included in attr
        # --- (even though they are stored as attributes in the C++)
        for key, vals in kwargs.items():
            attr[:, self.particle_container.get_comp_index(key) - built_in_attrs] = vals

        nattr_int = 0
        attr_int = np.empty([0], dtype=np.int32)

        # TODO: expose ParticleReal through pyAMReX
        # and cast arrays to the correct types, before calling add_n_particles
        # x = x.astype(self._numpy_particlereal_dtype, copy=False)
        # y = y.astype(self._numpy_particlereal_dtype, copy=False)
        # z = z.astype(self._numpy_particlereal_dtype, copy=False)
        # ux = ux.astype(self._numpy_particlereal_dtype, copy=False)
        # uy = uy.astype(self._numpy_particlereal_dtype, copy=False)
        # uz = uz.astype(self._numpy_particlereal_dtype, copy=False)

        self.particle_container.add_n_particles(
            0,
            x.size,
            x,
            y,
            z,
            ux,
            uy,
            uz,
            nattr,
            attr,
            nattr_int,
            attr_int,
            unique_particles,
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
        comp_idx = self.particle_container.get_comp_index(comp_name)

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
        comp_idx = self.particle_container.get_icomp_index(comp_name)

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

        You can find `here https://github.com/ECP-WarpX/WarpX/blob/319e55b10ad4f7c71b84a4fb21afbafe1f5b65c2/Examples/Tests/particle_boundary_interaction/PICMI_inputs_rz.py`
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
        if comp_name in part_container.real_comp_names:
            comp_idx = part_container.real_comp_names[comp_name]
            for ii, pti in enumerate(
                libwarpx.libwarpx_so.BoundaryBufferParIter(part_container, level)
            ):
                soa = pti.soa()
                data_array.append(xp.array(soa.get_real_data(comp_idx), copy=False))
        # loop over the integer attributes
        elif comp_name in part_container.int_comp_names:
            comp_idx = part_container.int_comp_names[comp_name]
            for ii, pti in enumerate(
                libwarpx.libwarpx_so.BoundaryBufferParIter(part_container, level)
            ):
                soa = pti.soa()
                data_array.append(xp.array(soa.get_int_data(comp_idx), copy=False))
        else:
            raise RuntimeError("Name %s not found" % comp_name)
        return data_array

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
