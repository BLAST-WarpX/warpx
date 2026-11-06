/* Copyright 2026 The WarpX Community
 *
 * Authors: Remi Lehe
 * License: BSD-3-Clause-LBNL
 */
#include "Python/pyWarpX.H"

#include <FieldSolver/ImplicitSolvers/ImplicitSolver.H>

#include <ablastr/fields/MultiFabRegister.H>

#include <memory>


void init_ImplicitSolver (py::module& m)
{
    // The implicit solver is owned by WarpX (WarpX::m_implicit_solver) and is
    // destroyed with it in WarpX::Finalize, so the Python object must never
    // delete it: py::nodelete makes the returned pointer non-owning.
    py::class_<ImplicitSolver, std::unique_ptr<ImplicitSolver, py::nodelete>>(m, "ImplicitSolver")
        .def("finish_mass_matrices_deposit", &ImplicitSolver::FinishMassMatricesDeposition,
            R"pbdoc(Fill the second half of the symmetric diagonal mass matrices)pbdoc"
        )
        .def("apply_mass_matrices", &ImplicitSolver::ApplyMassMatrices,
            py::arg("out"), py::arg("in"),
            py::arg("in_ref").none(true) = nullptr,
            py::arg("baseline").none(true) = nullptr,
            py::arg("scale") = 1.0,
            py::arg("zero_out_first") = false,
            R"pbdoc(Apply the mass matrices to a vector field)pbdoc"
        )
    ;
}
