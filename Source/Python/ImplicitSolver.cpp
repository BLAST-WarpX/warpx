/* Copyright 2026 The WarpX Community
 *
 * Authors: Remi Lehe
 * License: BSD-3-Clause-LBNL
 */
#include "Python/pyWarpX.H"

#include <FieldSolver/ImplicitSolvers/ImplicitSolver.H>
#include <WarpX.H>

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_REAL.H>

#include <memory>
#include <optional>
#include <string>


void init_ImplicitSolver (py::module& m)
{
    // The implicit solver is owned by WarpX (WarpX::m_implicit_solver) and is
    // destroyed with it in WarpX::Finalize, so the Python object must never
    // delete it: py::nodelete makes the returned pointer non-owning.
    py::class_<ImplicitSolver, std::unique_ptr<ImplicitSolver, py::nodelete>>(m, "ImplicitSolver")
        .def("pre_linear_solve", &ImplicitSolver::PreLinearSolve,
            R"pbdoc(Prepare the mass matrices for a linear solve

This is what the implicit solvers call before every linear solve. When mass
matrices are in use, it deposits them from the current particle state, folds
the symmetric halves of the diagonal blocks, saves the electric field that
they were linearized around and, when a preconditioner uses them, forms the
reduced preconditioner mass matrices.

Note that this does not sum the guard cells of the mass matrices: the
theta-implicit solvers apply the mass matrices on the guard cells too and
sum the resulting current instead. Use ``WarpX.sync_mass_matrices`` for
mass matrices that are complete on the valid cells.)pbdoc"
        )
        .def("finish_mass_matrices", &ImplicitSolver::FinishMassMatrices,
            R"pbdoc(Fill the second half of the symmetric diagonal mass matrices

The deposition only fills half of the band of the diagonal blocks (``Sxx``,
``Syy``, ``Szz``), exploiting their symmetry. This mirrors the other half.)pbdoc"
        )
        .def("apply_mass_matrices",
            [](ImplicitSolver& solver,
               std::string const& out_name,
               std::string const& in_name,
               std::optional<std::string> const& in_ref_name,
               std::optional<std::string> const& baseline_name,
               amrex::Real scale,
               bool zero_out_first)
            {
                auto& warpx = WarpX::GetInstance();
                auto& fields = warpx.GetMultiFabRegister();
                const int finest_level = warpx.finestLevel();

                ablastr::fields::MultiLevelVectorField out =
                    fields.get_mr_levels_alldirs(out_name, finest_level);
                const ablastr::fields::MultiLevelVectorField in =
                    fields.get_mr_levels_alldirs(in_name, finest_level);

                std::optional<ablastr::fields::MultiLevelVectorField> in_ref;
                if (in_ref_name) {
                    in_ref = fields.get_mr_levels_alldirs(*in_ref_name, finest_level);
                }
                std::optional<ablastr::fields::MultiLevelVectorField> baseline;
                if (baseline_name) {
                    baseline = fields.get_mr_levels_alldirs(*baseline_name, finest_level);
                }

                solver.ApplyMassMatrices(
                    out, in,
                    in_ref ? &(*in_ref) : nullptr,
                    baseline ? &(*baseline) : nullptr,
                    scale, zero_out_first);
            },
            py::arg("out_name"), py::arg("in_name"),
            py::arg("in_ref_name") = py::none(), py::arg("baseline_name") = py::none(),
            py::arg("scale") = 1.0, py::arg("zero_out_first") = false,
            R"pbdoc(Apply the mass matrices to a vector field

Computes ``out += scale * S * (in - in_ref) [+ baseline]``, where ``S`` are
the mass matrices, the linear response of the deposited current density to
the electric field (``dJ = S dE``). All fields are looked up by name in the
MultiFab register, on all mesh-refinement levels. ``in`` must have its guard
cells filled: the stencil reads the neighbors of every point it writes.

Parameters
----------
out_name: str
  Name of the vector field to accumulate into, with the staggering of the current density
in_name: str
  Name of the vector field to apply the mass matrices to, with the staggering of the electric field
in_ref_name: str, optional
  Name of a vector field subtracted from ``in`` before applying the mass matrices
baseline_name: str, optional
  Name of a vector field added to the result
scale: float, optional
  Factor multiplying the mass matrices, defaults to 1
zero_out_first: bool, optional
  Whether to zero ``out`` first, so that it holds the result of this operation only;
  otherwise the result is added to its existing contents (defaults to False))pbdoc"
        )
    ;
}
