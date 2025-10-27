import numpy as np
from numba import jit
from scipy.sparse import csc_matrix
from scipy.sparse import linalg as sla

from pywarpx import callbacks, fields, picmi

constants = picmi.constants


@jit(nopython=True)
def _chi_accumalation_loop(z_pos, w, dz, z_min, nz, prefac, chi, theta):
    # loop over particles and add their contribution to chi
    for pid, z in enumerate(z_pos):
        # find the two nodes this particle contributes to
        # get left array index
        idx_left = int((z - z_min) // dz)
        if idx_left == nz:
            idx_left = nz - 1
        idx_right = idx_left + 1

        # get right coefficient
        coef = (z - z_min) / dz - idx_left
        coefs = [1.0 - coef, coef]

        # get the matrix A = (1 - theta x + theta theta^T) / (1 + |theta|^2), recalling that
        #          |  0  -t_3  t_2 |
        #    t x = | t_3   0  -t_1 |
        #          |-t_2  t_1   0  |
        A = np.array(
            [
                [
                    1.0 + theta[pid][0] * theta[pid][0],
                    theta[pid][2] + theta[pid][0] * theta[pid][1],
                    -theta[pid][1] + theta[pid][0] * theta[pid][2],
                ],
                [
                    -theta[pid][2] + theta[pid][0] * theta[pid][1],
                    1.0 + theta[pid][1] * theta[pid][1],
                    theta[pid][0] + theta[pid][1] * theta[pid][2],
                ],
                [
                    theta[pid][1] + theta[pid][0] * theta[pid][2],
                    -theta[pid][0] + theta[pid][1] * theta[pid][2],
                    1.0 + theta[pid][2] * theta[pid][2],
                ],
            ]
        ) / (1.0 + np.linalg.norm(theta[pid]) ** 2)

        # each particle contributes to 4 entries of chi_ii' with
        # 1) i = idx_left;  i' = idx_left
        # 2) i = idx_left;  i' = idx_right
        # 3) i = idx_right; i' = idx_left
        # 4) i = idx_right; i' = idx_right

        for i in [idx_left, idx_right]:
            coef1 = coefs[int(i == idx_right)]
            for j in [idx_left, idx_right]:
                coef2 = coefs[int(j == idx_right)]

                # populate chi array at appropriate indices
                for ii in range(3):
                    for jj in range(3):
                        chi[i + ii * (nz + 1), j + jj * (nz + 1)] += (
                            prefac * w[pid] * (coef1 * A[ii, jj] * coef2)
                        )

        # force chi to be periodic - if idx_left = 0, also add
        # the contribution to the right most node and vice-versa
        # for idx_right = nz
        if idx_left == 0 or idx_right == nz:
            if idx_left == 0:
                idx_left = nz
            else:
                idx_right = 0

            for i, j in [
                (idx_left, idx_right),
                (idx_right, idx_left),
                (idx_left, idx_left),
                (idx_right, idx_right),
            ]:
                coef1 = coefs[int(i == idx_right)]
                coef2 = coefs[int(j == idx_right)]

                # populate chi array at appropriate indices
                for ii in range(3):
                    for jj in range(3):
                        chi[i + ii * (nz + 1), j + jj * (nz + 1)] += (
                            prefac * w[pid] * (coef1 * A[ii, jj] * coef2)
                        )


class OneD_DarwinSolver(picmi.ElectrostaticSolver):
    """Prototype class for a 1d solver using the Darwin scheme from Dan
    Barnes.

    The solver generally has 5 steps:
    1) Get partial velocity update, predicted current and susceptibility.
    2) Solve MS field equation.
    3) Calculate E^n and push particles with B^n-1/2 and E_n from x^n to x^n+1
    4) Solve Poisson to get phi^n+1
    5) Update B to B^n+1/2

    In this prototype solver it is assumed that all species have
    `warpx_do_not_push=True`. The above steps are placed in the WarpX PIC loop
    as follows:
    After collisions - step 1, 2 & 3.
    Poisson solver - step 4
    AfterEsolve - step 5 (but this should eventually go in ``MagnetostaticSolve`` or equivalent)

    This algorithm requires an additional sets of multifabs, A.
    """

    def __init__(
        self, simulation, grid, dt, Csi, skip_es=False, python_ms_solve=None, **kwargs
    ):
        # Sanity check that this solver is appropriate to use
        if not isinstance(grid, picmi.Cartesian1DGrid):
            raise RuntimeError("OneD_DarwinSolver can only be used on a 1D grid.")

        self.simulation = simulation
        self.grid = grid
        self.dt = dt

        self.skip_es = skip_es
        self.python_ms_solve = python_ms_solve

        super(OneD_DarwinSolver, self).__init__(
            grid=self.grid, method="Multigrid", required_precision=1, **kwargs
        )

        if not skip_es:
            self.es_solver = picmi.ElectrostaticSolver(
                grid=self.grid,
                required_precision=1e-6,
                warpx_effective_potential=True,
                warpx_effective_potential_factor=Csi,
                warpx_self_fields_verbosity=0,
            )

    def solver_initialize_inputs(self):
        """Grab geometrical quantities from the grid."""
        super(OneD_DarwinSolver, self).solver_initialize_inputs()

        # get grid properties
        self.z_min = self.grid.lower_bound[0]
        self.z_max = self.grid.upper_bound[0]
        self.nz = self.grid.number_of_cells[0]
        self.dz = (self.z_max - self.z_min) / self.nz
        self.z_grid = np.linspace(self.z_min, self.z_max, self.nz + 1)

        # install callback to execute step 1
        callbacks.installbeforedeposition(self.accumulate_j_and_susceptibility)

        # callback for ComputeRHS - step 2 is done in C++ with GMRES
        callbacks.installafterdeposition(self.compute_rhs)

        # install callback to execute step 2 (for testing purposes)
        if self.python_ms_solve:
            callbacks.installparticlescraper(self.run_solve)

        # install SIPIC Poisson solver (step 4) to skip the ES evolution
        if self.skip_es:
            callbacks.installpoissonsolver(self.skip_poisson_solve)
        else:
            self.es_solver.solver_initialize_inputs()

        # install callback to create MF wrappers
        callbacks.installbeforeInitEsolve(self._after_init)

    def _after_init(self):
        # grab needed multifab wrappers
        self.B_x = fields.BxFPWrapper()
        self.B_y = fields.ByFPWrapper()
        self.B_z = fields.BzFPWrapper()
        self.J_x = fields.JxFPWrapper()
        self.J_y = fields.JyFPWrapper()
        self.J_z = fields.JzFPWrapper()

        # MFs for MS solve
        self.dA_x = fields.MultiFabWrapper(mf_name="dA_fp", idir=0, level=0)
        self.dA_y = fields.MultiFabWrapper(mf_name="dA_fp", idir=1, level=0)
        self.dA_z = fields.MultiFabWrapper(mf_name="dA_fp", idir=2, level=0)
        self.xi_fp = fields.MultiFabWrapper(mf_name="xi_fp", level=0)

    def accumulate_j_and_susceptibility(self):
        """
        Get partial velocity update, predicted current and susceptibility. Then
        calculate source vector for MS equation.
        """
        # define arrays to store chi
        self.chi = np.zeros((3 * (self.nz + 1), 3 * (self.nz + 1)))

        # -- Deposit J* and susceptibility
        self._deposition_step()

        # get MS operator so it doesn't have to be done at every GMRES iteration
        self.operator = self._get_MS_operator()

    def compute_rhs(self):
        """Python function that performs the task of the C++ function
        ComputeRHS. `guess` is the `a_dA` vector.
        """
        # Allocate array for the guess vector
        guess = np.zeros(3 * self.nz + 2 + self.nz + 1)
        guess[: self.nz + 1] = self.dA_x[...]
        guess[self.nz + 1 : 2 * self.nz + 2] = self.dA_y[...]
        guess[2 * self.nz + 2 : 3 * self.nz + 2] = self.dA_z[...]
        guess[3 * self.nz + 2 :] = self.xi_fp[...]

        rhs = np.dot(self.operator, guess)

        # populate MFs that are then copied to a_RHS
        self.dA_x[...] = rhs[: self.nz + 1]
        self.dA_y[...] = rhs[self.nz + 1 : 2 * self.nz + 2]
        self.dA_z[...] = rhs[2 * self.nz + 2 : 3 * self.nz + 2]
        self.xi_fp[...] = rhs[3 * self.nz + 2 :]

    def run_solve(self):
        # did we skipped the MS solve in C++?
        self._perform_MS_field_solve()

    def _deposition_step(self):
        # define arrays to store jn_star
        jn_star = np.zeros((self.nz + 1, 3))

        # grab B and interpolate to nodes
        Bx = (self.B_x[-1j:] + self.B_x[:2j]) / 2.0
        By = (self.B_y[-1j:] + self.B_y[:2j]) / 2.0
        Bz = self.B_z[:]

        # start by looping over particles, pushing their velocities to v^*,
        # depositing J* and accumulate \Theta (cell average)
        multi_pc = self.simulation.extension.warpx.multi_particle_container()
        for species in self.simulation.species:
            pc = multi_pc.get_particle_container_from_name(species.name)

            z_idx = pc.get_real_comp_index("z")
            ux_idx = pc.get_real_comp_index("ux")
            uy_idx = pc.get_real_comp_index("uy")
            uz_idx = pc.get_real_comp_index("uz")
            w_idx = pc.get_real_comp_index("w")

            # iterate over particle levels
            for lvl in range(pc.finest_level + 1):
                # get every local chunk of particles
                for pti in pc.iterator(pc, level=lvl):
                    # attributes in SoA format
                    soa = pti.soa()

                    # get position
                    z = np.array(soa.get_real_data(z_idx), copy=False)

                    # get velocities
                    ux = np.array(soa.get_real_data(ux_idx), copy=False)
                    uy = np.array(soa.get_real_data(uy_idx), copy=False)
                    uz = np.array(soa.get_real_data(uz_idx), copy=False)

                    # get weight
                    w = np.array(soa.get_real_data(w_idx), copy=False)

                    # Accumalate w_p * q_p * (u_new + u_old)
                    self._deposit_current(
                        array=jn_star,
                        z=z,
                        vx=ux,
                        vy=uy,
                        vz=uz,
                        weight=species.charge * w,
                    )

                    # get B at particle
                    bx = np.interp(z, self.z_grid, Bx)
                    by = np.interp(z, self.z_grid, By)
                    bz = np.interp(z, self.z_grid, Bz)
                    theta = (
                        0.5
                        * self.dt
                        * species.charge
                        / species.mass
                        * np.asarray([bx, by, bz]).T
                    )
                    _chi_accumalation_loop(
                        z,
                        w,
                        self.dz,
                        self.z_min,
                        self.nz,
                        constants.mu0 * species.charge**2 / species.mass,
                        self.chi,
                        theta,
                    )

        # Apply periodic boundaries to jn_star
        jn_star[0, 0] = jn_star[0, 0] + jn_star[-1, 0]
        jn_star[-1, 0] = jn_star[0, 0]
        jn_star[0, 1] = jn_star[0, 1] + jn_star[-1, 1]
        jn_star[-1, 1] = jn_star[0, 1]
        jn_star[0, 2] = jn_star[0, 2] + jn_star[-1, 2]
        jn_star[-1, 2] = jn_star[0, 2]

        # get the node to edge averaging operator
        A = self._get_node_to_edge_average()

        # Interpolate jn_star to the edges
        j_edge = np.dot(A, jn_star.T.flatten())

        # Populate the current multifab
        self.J_x[...] = j_edge[: self.nz + 1]
        self.J_y[...] = j_edge[self.nz + 1 : 2 * self.nz + 2]
        self.J_z[...] = j_edge[2 * self.nz + 2 :]

    def _deposit_current(self, array, z, vx, vy, vz, weight):
        """Function to deposit current onto nodal grid with shape function 1."""
        # get left array index
        idx_left = ((z - self.z_min) / self.dz).astype(int)
        idx_left[np.where(idx_left == self.nz)] = self.nz - 1
        idx_right = idx_left + 1

        # get right coefficients
        coef = (z - self.z_min) / self.dz - idx_left

        v = np.array([vx, vy, vz]).T

        for ii in range(3):
            array[:, ii] += np.bincount(
                idx_left,
                weights=(1.0 - coef) * weight * v[:, ii] / self.dz,
                minlength=self.nz + 1,
            )
            array[:, ii] += np.bincount(
                idx_right,
                weights=coef * weight * v[:, ii] / self.dz,
                minlength=self.nz + 1,
            )

    def _perform_MS_field_solve(self):
        # Allocate array for the source vector
        source = np.zeros(3 * self.nz + 2 + self.nz + 1)

        source[: self.nz + 1] = self.dA_x[...]
        source[self.nz + 1 : 2 * self.nz + 2] = self.dA_y[...]
        source[2 * self.nz + 2 : 3 * self.nz + 2] = self.dA_z[...]

        M = self._get_MS_operator()

        # get full linear operator as csc matrix
        Mcsc = csc_matrix(M, dtype=np.float64)

        lu = sla.splu(Mcsc)
        solution = lu.solve(source)

        self.dA_x[...] = solution[: self.nz + 1]
        self.dA_y[...] = solution[self.nz + 1 : 2 * self.nz + 2]
        self.dA_z[...] = solution[2 * self.nz + 2 : 3 * self.nz + 2]

        # force dA_z to be zero since in 1d non-zero values here are noise
        self.dA_z[...] = 0.0

    def _get_MS_operator(self):
        # Now construct the linear operator of the MS equation
        M = np.zeros((3 * self.nz + 2 + self.nz + 1, 3 * self.nz + 2 + self.nz + 1))
        M[: 3 * self.nz + 2, : 3 * self.nz + 2] = -self._get_vector_laplacian()

        # get chi -> V^-1 A chi A^T
        chi = np.dot(
            np.identity(3 * self.nz + 2) / self.dz,
            np.dot(
                self._get_node_to_edge_average(),
                np.dot(self.chi, self._get_edge_to_node_average()),
            ),
        )

        # add chi to linear operator
        M[: 3 * self.nz + 2, : 3 * self.nz + 2] += chi

        # build grad dxi part
        M[2 * self.nz + 2 : 3 * self.nz + 2, 3 * self.nz + 2 :] = (
            -constants.mu0 * self._get_gradient()
        )

        # now build the second equation and add it to M
        K = np.dot(self._get_divergence(), chi)
        M[3 * self.nz + 2 :, : 3 * self.nz + 2] = K
        M[3 * self.nz + 2 :, 3 * self.nz + 2 :] = -constants.mu0 * self._get_laplacian()
        return M

    def _get_laplacian(self):
        L = np.zeros((self.nz + 1, self.nz + 1))
        idx = np.arange(self.nz + 1)
        L[idx, idx] = -2.0 / self.dz**2
        L[idx[1:], idx[:-1]] = 1.0 / self.dz**2
        L[idx[:-1], idx[1:]] = 1.0 / self.dz**2
        # add element entries to enforce periodic boundaries
        L[self.nz, 1] = 1.0 / self.dz**2
        L[0, self.nz - 1] = 1.0 / self.dz**2
        return L

    def _get_vector_laplacian(self):
        """The vector Laplacian is (crucially!) constructed from first order
        operators."""

        L = np.zeros((3 * self.nz + 2, 3 * self.nz + 2))

        # the x and y parts are nodal so can just use the scalar Laplacian
        L[: self.nz + 1, : self.nz + 1] = self._get_laplacian()
        L[self.nz + 1 : 2 * self.nz + 2, self.nz + 1 : 2 * self.nz + 2] = (
            self._get_laplacian()
        )

        # the z part is cell centered
        idx = np.arange(self.nz)
        L[2 * self.nz + 2 + idx, 2 * self.nz + 2 + idx] = -2.0 / self.dz**2
        L[2 * self.nz + 2 + idx[1:], 2 * self.nz + 2 + idx[:-1]] = 1.0 / self.dz**2
        L[2 * self.nz + 2 + idx[:-1], 2 * self.nz + 2 + idx[1:]] = 1.0 / self.dz**2
        # add element entries to enforce periodic boundaries
        L[-1, 2 * self.nz + 2] = 1.0 / self.dz**2
        L[2 * self.nz + 2, -1] = 1.0 / self.dz**2

        return L

    def _get_gradient(self):
        D = np.zeros((self.nz, self.nz + 1))
        idx = np.arange(self.nz)
        D[idx, idx + 1] = 1.0 / self.dz
        D[idx, idx] = -1.0 / self.dz
        return D

    def _get_divergence(self):
        # get matrix to calculate divergence of edge field (edge to nodes)
        D = np.zeros((self.nz + 1, 2 * self.nz + 2 + self.nz))
        idx = np.arange(self.nz)
        D[idx, 2 * self.nz + 2 + idx] = 1.0 / self.dz
        D[idx + 1, 2 * self.nz + 2 + idx] = -1.0 / self.dz
        D[-1, 2 * self.nz + 2] = 1.0 / self.dz
        D[0, -1] = -1.0 / self.dz
        return D

    def _get_edge_to_node_average(self):
        # get A (matrix to average edges to nodes)
        A = np.zeros((3 * self.nz + 3, 3 * self.nz + 2))
        idx = np.arange(2 * self.nz + 2)
        A[idx, idx] = 1.0
        idx = np.arange(self.nz)
        A[idx + 2 * self.nz + 2, idx + 2 * self.nz + 2] = 1.0 / 2.0
        A[idx[:-1] + 2 * self.nz + 3, idx[:-1] + 2 * self.nz + 2] = 1.0 / 2.0
        A[-1, 2 * self.nz + 2] = 1.0 / 2.0
        A[-1, -1] = 1.0 / 2.0
        A[2 * self.nz + 2, -1] = 1.0 / 2.0
        return A

    def _get_node_to_edge_average(self):
        # get A (matrix to average nodes to edges)
        A = np.zeros((3 * self.nz + 2, 3 * self.nz + 3))
        idx = np.arange(2 * self.nz + 2)
        A[idx, idx] = 1.0
        idx = np.arange(self.nz)
        A[idx + 2 * self.nz + 2, idx + 2 * self.nz + 2] = 1.0 / 2.0
        A[idx + 2 * self.nz + 2, idx + 2 * self.nz + 3] = 1.0 / 2.0
        return A

    def skip_poisson_solve(self):
        """Function run on every step to perform a null solve of Poisson's
        equation."""
        pass
