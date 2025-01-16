#!/usr/bin/env python3
#
# --- Copyright 2021 Modern Electron (DSMC test added in 2023 by TAE Technologies)
# --- Monte-Carlo Collision script to reproduce the benchmark tests from
# --- Turner et al. (2013) - https://doi.org/10.1063/1.4775084

import argparse
import sys

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse import linalg as sla

from pywarpx import callbacks, fields, libwarpx, particle_containers, picmi
from pywarpx.LoadThirdParty import load_cupy

constants = picmi.constants


class PoissonSolver1D(picmi.ElectrostaticSolver):
    """This solver is maintained as an example of the use of Python callbacks.
    However, it is not necessarily needed since the 1D code has the direct tridiagonal
    solver implemented."""

    def __init__(self, grid, **kwargs):
        """Direct solver for the Poisson equation using superLU. This solver is
        useful for 1D cases.

        Arguments:
            grid (picmi.Cartesian1DGrid): Instance of the grid on which the
            solver will be installed.
        """
        # Sanity check that this solver is appropriate to use
        if not isinstance(grid, picmi.Cartesian1DGrid):
            raise RuntimeError("Direct solver can only be used on a 1D grid.")

        super(PoissonSolver1D, self).__init__(
            grid=grid,
            method=kwargs.pop("method", "Multigrid"),
            required_precision=1,
            **kwargs,
        )

    def solver_initialize_inputs(self):
        """Grab geometrical quantities from the grid. The boundary potentials
        are also obtained from the grid using 'warpx_potential_zmin' for the
        left_voltage and 'warpx_potential_zmax' for the right_voltage.
        These can be given as floats or strings that can be parsed by the
        WarpX parser.
        """
        # grab the boundary potentials from the grid object
        self.right_voltage = self.grid.potential_zmax

        # set WarpX boundary potentials to None since we will handle it
        # ourselves in this solver
        self.grid.potential_xmin = None
        self.grid.potential_xmax = None
        self.grid.potential_ymin = None
        self.grid.potential_ymax = None
        self.grid.potential_zmin = None
        self.grid.potential_zmax = None

        super(PoissonSolver1D, self).solver_initialize_inputs()

        self.nz = self.grid.number_of_cells[0]
        self.dz = (self.grid.upper_bound[0] - self.grid.lower_bound[0]) / self.nz

        self.nxguardphi = 1
        self.nzguardphi = 1

        self.phi = np.zeros(self.nz + 1 + 2 * self.nzguardphi)

        self.decompose_matrix()

        callbacks.installpoissonsolver(self._run_solve)

    def decompose_matrix(self):
        """Function to build the superLU object used to solve the linear
        system."""
        self.nsolve = self.nz + 1

        # Set up the computation matrix in order to solve A*phi = rho
        A = np.zeros((self.nsolve, self.nsolve))
        idx = np.arange(self.nsolve)
        A[idx, idx] = -2.0
        A[idx[1:], idx[:-1]] = 1.0
        A[idx[:-1], idx[1:]] = 1.0

        A[0, 1] = 0.0
        A[-1, -2] = 0.0
        A[0, 0] = 1.0
        A[-1, -1] = 1.0

        A = csc_matrix(A, dtype=np.float64)
        self.lu = sla.splu(A)

    def _run_solve(self):
        """Function run on every step to perform the required steps to solve
        Poisson's equation."""
        # get rho from WarpX
        self.rho_data = fields.RhoFPWrapper(0, False)[...]
        # run superLU solver to get phi
        self.solve()
        # write phi to WarpX
        fields.PhiFPWrapper(0, True)[...] = self.phi[:]

    def solve(self):
        """The solution step. Includes getting the boundary potentials and
        calculating phi from rho."""

        left_voltage = 0.0
        right_voltage = eval(
            self.right_voltage,
            {"t": self.sim.extension.warpx.gett_new(0), "sin": np.sin, "pi": np.pi},
        )

        # Construct b vector
        rho = -self.rho_data / constants.ep0
        b = np.zeros(rho.shape[0], dtype=np.float64)
        b[:] = rho * self.dz**2

        b[0] = left_voltage
        b[-1] = right_voltage

        phi = self.lu.solve(b)

        self.phi[self.nzguardphi : -self.nzguardphi] = phi

        self.phi[: self.nzguardphi] = left_voltage
        self.phi[-self.nzguardphi :] = right_voltage


class CapacitiveDischargeExample(object):
    """The following runs a simulation of a parallel plate capacitor seeded
    with a plasma in the spacing between the plates. A time varying voltage is
    applied across the capacitor. The groups of 4 values below correspond to
    the 4 cases simulated by Turner et al. (2013) in their benchmarks of
    PIC-MCC codes.
    """

    gap = 0.067  # m

    freq = 13.56e6  # Hz
    voltage = [450.0, 200.0, 150.0, 120.0]  # V

    gas_density = [9.64e20, 32.1e20, 96.4e20, 321e20]  # m^-3
    gas_temp = 300.0  # K
    m_ion = 6.67e-27  # kg

    plasma_density = [2.56e14, 5.12e14, 5.12e14, 3.84e14]  # m^-3
    elec_temp = 30000.0  # K

    seed_nppc = 16 * np.array([32, 16, 8, 4])

    nz = [128, 256, 512, 512]

    dt = 1.0 / (np.array([400, 800, 1600, 3200]) * freq)

    # Total simulation time in seconds
    total_time = np.array([1280, 5120, 5120, 15360]) / freq
    # Time (in seconds) between diagnostic evaluations
    diag_interval = 32 / freq

    def __init__(self, n=0, test=False, pythonsolver=False, dsmc=False):
        """Get input parameters for the specific case (n) desired."""
        self.n = n
        self.test = test
        self.pythonsolver = pythonsolver
        self.dsmc = dsmc

        # Case specific input parameters
        self.voltage = f"{self.voltage[n]}*sin(2*pi*{self.freq:.5e}*t)"

        self.gas_density = self.gas_density[n]
        self.plasma_density = self.plasma_density[n]
        self.seed_nppc = self.seed_nppc[n]

        self.nz = self.nz[n]

        self.dt = self.dt[n]
        self.max_steps = int(self.total_time[n] / self.dt)
        self.diag_steps = int(self.diag_interval / self.dt)

        if self.test:
            self.max_steps = 50
            self.diag_steps = 5
            self.mcc_subcycling_steps = 2
            self.rng = np.random.default_rng(23094290)
        else:
            self.mcc_subcycling_steps = None
            self.rng = np.random.default_rng()

        self.ion_density_array = np.zeros(self.nz + 1)

        self.setup_run()

    def setup_run(self):
        """Setup simulation components."""

        #######################################################################
        # Set geometry and boundary conditions                                #
        #######################################################################

        self.grid = picmi.Cartesian1DGrid(
            number_of_cells=[self.nz],
            warpx_max_grid_size=128,
            lower_bound=[0],
            upper_bound=[self.gap],
            lower_boundary_conditions=["dirichlet"],
            upper_boundary_conditions=["dirichlet"],
            lower_boundary_conditions_particles=["absorbing"],
            upper_boundary_conditions_particles=["absorbing"],
            warpx_potential_hi_z=self.voltage,
        )

        #######################################################################
        # Field solver                                                        #
        #######################################################################

        if self.pythonsolver:
            self.solver = PoissonSolver1D(grid=self.grid)
        else:
            # This will use the tridiagonal solver
            self.solver = picmi.ElectrostaticSolver(grid=self.grid)

        #######################################################################
        # Particle types setup                                                #
        #######################################################################

        self.electrons = picmi.Species(
            particle_type="electron",
            name="electrons",
            initial_distribution=picmi.UniformDistribution(
                density=self.plasma_density,
                rms_velocity=[np.sqrt(constants.kb * self.elec_temp / constants.m_e)]
                * 3,
            ),
        )
        self.ions = picmi.Species(
            particle_type="He",
            name="he_ions",
            charge="q_e",
            mass=self.m_ion,
            initial_distribution=picmi.UniformDistribution(
                density=self.plasma_density,
                rms_velocity=[np.sqrt(constants.kb * self.gas_temp / self.m_ion)] * 3,
            ),
        )
        if self.dsmc:
            self.neutrals = picmi.Species(
                particle_type="He",
                name="neutrals",
                charge=0,
                mass=self.m_ion,
                warpx_reflection_model_zlo=1.0,
                warpx_reflection_model_zhi=1.0,
                warpx_do_resampling=True,
                warpx_resampling_trigger_max_avg_ppc=int(self.seed_nppc * 1.5),
                initial_distribution=picmi.UniformDistribution(
                    density=self.gas_density,
                    rms_velocity=[np.sqrt(constants.kb * self.gas_temp / self.m_ion)]
                    * 3,
                ),
            )

        #######################################################################
        # Collision initialization                                            #
        #######################################################################

        cross_sec_direc = "../../../../warpx-data/MCC_cross_sections/He/"
        electron_colls = picmi.MCCCollisions(
            name="coll_elec",
            species=self.electrons,
            background_density=self.gas_density,
            background_temperature=self.gas_temp,
            background_mass=self.ions.mass,
            ndt=self.mcc_subcycling_steps,
            scattering_processes={
                "elastic": {
                    "cross_section": cross_sec_direc + "electron_scattering.dat"
                },
                "excitation1": {
                    "cross_section": cross_sec_direc + "excitation_1.dat",
                    "energy": 19.82,
                },
                "excitation2": {
                    "cross_section": cross_sec_direc + "excitation_2.dat",
                    "energy": 20.61,
                },
                "ionization": {
                    "cross_section": cross_sec_direc + "ionization.dat",
                    "energy": 24.55,
                    "species": self.ions,
                },
            },
        )

        ion_scattering_processes = {
            "elastic": {"cross_section": cross_sec_direc + "ion_scattering.dat"},
            "back": {"cross_section": cross_sec_direc + "ion_back_scatter.dat"},
            # 'charge_exchange': {'cross_section': cross_sec_direc+'charge_exchange.dat'}
        }
        if self.dsmc:
            ion_colls = picmi.DSMCCollisions(
                name="coll_ion",
                species=[self.ions, self.neutrals],
                ndt=5,
                scattering_processes=ion_scattering_processes,
            )
        else:
            ion_colls = picmi.MCCCollisions(
                name="coll_ion",
                species=self.ions,
                background_density=self.gas_density,
                background_temperature=self.gas_temp,
                ndt=self.mcc_subcycling_steps,
                scattering_processes=ion_scattering_processes,
            )

        #######################################################################
        # Initialize simulation                                               #
        #######################################################################

        self.sim = picmi.Simulation(
            solver=self.solver,
            time_step_size=self.dt,
            max_steps=self.max_steps,
            warpx_collisions=[electron_colls, ion_colls],
            verbose=self.test,
        )
        self.solver.sim = self.sim

        self.sim.add_species(
            self.electrons,
            layout=picmi.GriddedLayout(
                n_macroparticle_per_cell=[self.seed_nppc], grid=self.grid
            ),
        )
        self.sim.add_species(
            self.ions,
            layout=picmi.GriddedLayout(
                n_macroparticle_per_cell=[self.seed_nppc], grid=self.grid
            ),
        )
        if self.dsmc:
            self.sim.add_species(
                self.neutrals,
                layout=picmi.GriddedLayout(
                    n_macroparticle_per_cell=[self.seed_nppc // 2], grid=self.grid
                ),
            )
        self.solver.sim_ext = self.sim.extension

        if self.dsmc:
            # Periodically reset neutral density to starting temperature
            callbacks.installbeforecollisions(self.rethermalize_neutrals)

        #######################################################################
        # Add diagnostics for the CI test to be happy                         #
        #######################################################################

        species = [self.electrons, self.ions]
        if self.dsmc:
            species.append(self.neutrals)
        particle_diag = picmi.ParticleDiagnostic(
            species=species,
            name="diag1",
            period=0,
        )
        field_diag = picmi.FieldDiagnostic(
            name="diag1",
            grid=self.grid,
            period=0,
            data_list=["rho_electrons", "rho_he_ions"],
        )
        self.sim.add_diagnostic(particle_diag)
        self.sim.add_diagnostic(field_diag)

    def rethermalize_neutrals(self):
        # When using DSMC the neutral temperature will change due to collisions
        # with the ions. This is not captured in the original MCC test.
        # Re-thermalize the neutrals every 1000 steps
        step = self.sim.extension.warpx.getistep(lev=0)
        if step % 1000 != 10:
            return

        if not hasattr(self, "neutral_cont"):
            self.neutral_cont = particle_containers.ParticleContainerWrapper(
                self.neutrals.name
            )

        ux_arrays = self.neutral_cont.uxp
        uy_arrays = self.neutral_cont.uyp
        uz_arrays = self.neutral_cont.uzp

        xp, _ = load_cupy()

        vel_std = np.sqrt(constants.kb * self.gas_temp / self.m_ion)
        for ii in range(len(ux_arrays)):
            nps = len(ux_arrays[ii])
            ux_arrays[ii][:] = xp.array(vel_std * self.rng.normal(size=nps))
            uy_arrays[ii][:] = xp.array(vel_std * self.rng.normal(size=nps))
            uz_arrays[ii][:] = xp.array(vel_std * self.rng.normal(size=nps))

    def _get_rho_ions(self):
        # deposit the ion density in rho_fp
        he_ions_wrapper = particle_containers.ParticleContainerWrapper("he_ions")
        he_ions_wrapper.deposit_charge_density(level=0)

        rho_data = self.rho_wrapper[...]
        self.ion_density_array += rho_data / constants.q_e / self.diag_steps

    def run_sim(self):
        self.sim.step(self.max_steps - self.diag_steps)

        self.rho_wrapper = fields.RhoFPWrapper(0, False)
        callbacks.installafterstep(self._get_rho_ions)

        self.sim.step(self.diag_steps)

        if self.pythonsolver:
            # confirm that the external solver was run
            assert hasattr(self.solver, "phi")

        if libwarpx.amr.ParallelDescriptor.MyProc() == 0:
            np.save(f"ion_density_case_{self.n + 1}.npy", self.ion_density_array)

        # query the particle z-coordinates if this is run during CI testing
        # to cover that functionality
        if self.test:
            he_ions_wrapper = particle_containers.ParticleContainerWrapper("he_ions")
            nparts = he_ions_wrapper.get_particle_count(local=True)
            z_coords = np.concatenate(he_ions_wrapper.zp)
            assert len(z_coords) == nparts
            assert np.all(z_coords >= 0.0) and np.all(z_coords <= self.gap)


##########################
# parse input parameters
##########################

parser = argparse.ArgumentParser()
parser.add_argument(
    "-t",
    "--test",
    help="toggle whether this script is run as a short CI test",
    action="store_true",
)
parser.add_argument(
    "-n", help="Test number to run (1 to 4)", required=False, type=int, default=1
)
parser.add_argument(
    "--pythonsolver",
    help="toggle whether to use the Python level solver",
    action="store_true",
)
parser.add_argument(
    "--dsmc",
    help="toggle whether to use DSMC for ions in place of MCC",
    action="store_true",
)
args, left = parser.parse_known_args()
sys.argv = sys.argv[:1] + left

if args.n < 1 or args.n > 4:
    raise AttributeError("Test number must be an integer from 1 to 4.")

run = CapacitiveDischargeExample(
    n=args.n - 1, test=args.test, pythonsolver=args.pythonsolver, dsmc=args.dsmc
)
run.run_sim()
