# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Voltage clamp for embedded-boundary conductors in electromagnetic runs (RZ, Yee).

Method and supported configurations: ``Docs/source/usage/workflows/electrode_voltage_clamp.rst``.
"""

import numpy as np

# Largest accepted condition number of the capacitance matrix
MAX_CAPACITANCE_CONDITION = 1.0e10


class StaircaseBiasCorrector:
    """Hold embedded-boundary conductors at prescribed potentials.

    Parameters
    ----------
    sim : picmi.Simulation
    correction_interval : int
        Correct every this many steps.
    electrodes : list of dict
        ``{"name", "region", "potential"}`` per conductor. ``region`` is a parser
        expression in (x, y, z) that is 1 on one whole conductor and 0 elsewhere; in
        RZ it is evaluated at (r, 0, z), so ``x`` is the radius. Every conductor must
        be listed, grounded ones with potential 0, except conductors joined to the
        PEC wall, which may be left out: they are part of the ground. A listed one
        must have potential 0; it gets no unit solve and no row in the capacitance
        matrix.
    tolerance, max_iterations : optional
        Relative tolerance and iteration cap of the unit-potential solves (MLMG). They
        set how exactly the capacitance matrix and the voltage readback agree.
        The multigrid depth and bottom solver follow the ``warpx.self_fields_*``
        options, shared with the electrostatic solver. Unless
        ``warpx.self_fields_max_coarsening_level`` is set, these solves coarsen only
        once: deeper coarsening loses thin conductors and the solve fails to converge.
        They run once, at setup, so this favors robustness over speed. Fewer levels
        leave a larger bottom problem: on large grids, raise
        ``warpx.self_fields_bottom_max_iters`` if the solve slows down; a deeper cap or
        another bottom solver can be faster for thick conductors. See the user docs.

    Call ``setup_after_init()`` after ``sim.initialize_warpx()``, then
    ``initialize_vacuum_bias()`` for a new run or ``resume_from_checkpoint()`` on a
    restart, then ``callbacks.installafterstep(clamp.correct_after_step)``.
    """

    def __init__(
        self,
        sim,
        correction_interval,
        electrodes,
        *,
        tolerance=1.0e-12,
        max_iterations=200,
    ):
        if not isinstance(correction_interval, int) or correction_interval < 1:
            raise ValueError("correction_interval must be a positive integer")
        if len(electrodes) < 1:
            raise ValueError("Provide at least one electrode")
        self.sim = sim
        self.correction_interval = correction_interval
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.names = [e.get("name", f"electrode_{k}") for k, e in enumerate(electrodes)]
        self.regions = [e["region"] for e in electrodes]
        self.v_target = np.array([float(e["potential"]) for e in electrodes])
        self.n = len(electrodes)
        # Listed conductors with a unit solve (set up in setup_after_init); the others
        # are joined to the grounded PEC wall
        self.solved = None
        self.capacitance = None
        self._active = False
        self._last_step = None

    def _warpx(self):
        # The C++ WarpX instance, as documented for PICMI simulations
        return self.sim.extension.warpx

    def setup_after_init(self):
        """Solve the unit potentials and measure the capacitance matrix."""
        wx = self._warpx()
        if not hasattr(wx, "voltage_clamp_solve_unit"):
            raise RuntimeError(
                "The voltage clamp is implemented for RZ builds with embedded "
                "boundaries only"
            )
        self.solved = []
        for index, (name, region, potential) in enumerate(
            zip(self.names, self.regions, self.v_target)
        ):
            selected, on_ground = wx.voltage_clamp_ground_overlap(region)
            if selected == 0:
                raise ValueError(
                    f"The region of electrode {name!r} selects no conductor (no fixed "
                    "node of the staircase)"
                )
            if on_ground > 0:
                # Part of the wall-connected ground: no unit solve
                if on_ground < selected:
                    raise ValueError(
                        f"The region of electrode {name!r} covers conductors joined to "
                        "a PEC wall (the ground) and free ones; list them separately"
                    )
                if potential != 0.0:
                    raise ValueError(
                        f"Electrode {name!r} is joined to a grounded PEC wall, so it "
                        f"cannot be held at {potential} V"
                    )
                continue
            wx.voltage_clamp_solve_unit(
                len(self.solved), region, self.tolerance, self.max_iterations
            )
            self.solved.append(index)
        if not self.solved:
            raise ValueError(
                "Every listed electrode is joined to a grounded PEC wall: nothing to clamp"
            )
        n = len(self.solved)
        wx.voltage_clamp_check_selection(n)
        self.capacitance = np.asarray(wx.voltage_clamp_capacitance(n))
        # A near-singular C means a conductor without a grounded reference, or one
        # listed twice: V = C^-1 Q would then be meaningless.
        if np.linalg.cond(self.capacitance) > MAX_CAPACITANCE_CONDITION:
            raise RuntimeError("The capacitance matrix is singular")

    def initialize_vacuum_bias(self, *, rho_tolerance=0.0):
        """Start a new run from zero E and zero (or neutral) charge.

        ``rho_tolerance`` is the largest |rho| in C/m^3 accepted as zero, for
        roundoff of a neutral loading.
        """
        wx = self._warpx()
        if wx.getistep(0) != 0:
            raise RuntimeError("initialize_vacuum_bias must run before the first step")
        # Fresh charge density of all species (does not touch rho_fp)
        rho = wx.multi_particle_container().get_charge_density(
            lev=0, local=False, finalize=True
        )
        if rho.norm0(0, 0, False, False) > rho_tolerance:
            raise RuntimeError(
                "The initial charge density must be zero; a charged start needs a "
                "matched space-charge field"
            )
        wx.voltage_clamp_add_unit_fields(list(self.v_target[self.solved]))
        self._active = True

    def resume_from_checkpoint(self):
        """Continue a run restarted from a checkpoint that was clamped."""
        if self._warpx().getistep(0) <= 0:
            raise RuntimeError("resume_from_checkpoint needs a restarted run")
        self._active = True

    def measure_voltage_state(self):
        """Conductor voltages, from the field charge and the plasma-induced charge.

        One entry per listed electrode. Electrodes joined to the grounded PEC wall
        read 0 V; their charges are not measured (NaN).
        """
        field_charge, grounded = (
            np.asarray(q)
            for q in self._warpx().voltage_clamp_charge_state(len(self.solved))
        )
        state = {
            "voltage": np.zeros(self.n),
            "field_charge": np.full(self.n, np.nan),
            "grounded_charge": np.full(self.n, np.nan),
        }
        state["voltage"][self.solved] = np.linalg.solve(
            self.capacitance, field_charge - grounded
        )
        state["field_charge"][self.solved] = field_charge
        state["grounded_charge"][self.solved] = grounded
        return state

    def correct_after_step(self):
        """Install with ``callbacks.installafterstep``, after particle scraping."""
        if not self._active:
            raise RuntimeError("Call initialize_vacuum_bias or resume_from_checkpoint")
        step = self._warpx().getistep(0)
        # Skip off-interval steps, and a second call in the same step (e.g. if the
        # callback was installed twice), which would apply the correction twice
        if step % self.correction_interval or step == self._last_step:
            return
        voltage = self.measure_voltage_state()["voltage"]
        dv = (self.v_target - voltage)[self.solved]
        self._warpx().voltage_clamp_add_unit_fields(list(dv))
        self._last_step = step
