/* Copyright 2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald (TAE Technologies)
 *
 * License: BSD-3-Clause-LBNL
 */

#include "PoissonBoundaryHandler.H"

using namespace amrex;

PoissonBoundaryHandler::PoissonBoundaryHandler ()
{
    ReadParameters();
    BuildParsers();
}

void PoissonBoundaryHandler::ReadParameters()
{
    // Parse the input file for domain boundary potentials
    const ParmParse pp_boundary("boundary");

    // Read potentials from input file
    m_boundary_potential_specified |= pp_boundary.query("potential_lo_x", potential_xlo_str);
    m_boundary_potential_specified |= pp_boundary.query("potential_hi_x", potential_xhi_str);
    m_boundary_potential_specified |= pp_boundary.query("potential_lo_y", potential_ylo_str);
    m_boundary_potential_specified |= pp_boundary.query("potential_hi_y", potential_yhi_str);
    m_boundary_potential_specified |= pp_boundary.query("potential_lo_z", potential_zlo_str);
    m_boundary_potential_specified |= pp_boundary.query("potential_hi_z", potential_zhi_str);

    const ParmParse pp_warpx("warpx");
    m_boundary_potential_specified |= pp_warpx.query("eb_potential(x,y,z,t)", potential_eb_str);

    if (m_boundary_potential_specified & (WarpX::electromagnetic_solver_id == ElectromagneticSolverAlgo::HybridPIC)) {
        ablastr::warn_manager::WMRecordWarning(
            "Algorithms",
            "The input script specifies the electric potential (phi) at the boundary, but \
            also uses the hybrid PIC solver based on Ohm’s law. When using this solver, the \
            electric potential does not have any impact on the simulation.",
            ablastr::warn_manager::WarnPriority::low);
    }
    else if (m_boundary_potential_specified & (WarpX::electromagnetic_solver_id != ElectromagneticSolverAlgo::None)) {
        ablastr::warn_manager::WMRecordWarning(
            "Algorithms",
            "The input script specifies the electric potential (phi) at the boundary so \
            an initial Poisson solve will be performed.",
            ablastr::warn_manager::WarnPriority::low);
    }
}

void PoissonBoundaryHandler::DefinePhiBCs (const amrex::Geometry& geom)
{
#ifdef WARPX_DIM_RZ
    if (geom.ProbLo(0) == 0){
        lobc[0] = LinOpBCType::Neumann;
        dirichlet_flag[0] = false;

        // handle the r_max boundary explicitly
        if (WarpX::field_boundary_hi[0] == FieldBoundaryType::PEC) {
            hibc[0] = LinOpBCType::Dirichlet;
            dirichlet_flag[1] = true;
        }
        else if (WarpX::field_boundary_hi[0] == FieldBoundaryType::Neumann) {
            hibc[0] = LinOpBCType::Neumann;
            dirichlet_flag[1] = false;
        }
        else {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(false,
                "Field boundary condition at the outer radius must be either PEC or neumann "
                "when using the electrostatic solver"
            );
        }
    }
    const int dim_start = 1;
#else
    const int dim_start = 0;
    amrex::ignore_unused(geom);
#endif
    for (int idim=dim_start; idim<AMREX_SPACEDIM; idim++){
        if (WarpX::poisson_solver_id == PoissonSolverAlgo::Multigrid){
            if ( WarpX::field_boundary_lo[idim] == FieldBoundaryType::Periodic
                    && WarpX::field_boundary_hi[idim] == FieldBoundaryType::Periodic ) {
                lobc[idim] = LinOpBCType::Periodic;
                hibc[idim] = LinOpBCType::Periodic;
                dirichlet_flag[idim*2] = false;
                dirichlet_flag[idim*2+1] = false;
            }
            else {
                has_non_periodic = true;
                if ( WarpX::field_boundary_lo[idim] == FieldBoundaryType::PEC ) {
                    lobc[idim] = LinOpBCType::Dirichlet;
                    dirichlet_flag[idim*2] = true;
                }
                else if ( WarpX::field_boundary_lo[idim] == FieldBoundaryType::Neumann ) {
                    lobc[idim] = LinOpBCType::Neumann;
                    dirichlet_flag[idim*2] = false;
                }
                else {
                    WARPX_ABORT_WITH_MESSAGE(
                        "Field boundary conditions have to be either periodic, PEC or neumann "
                        "when using the electrostatic Multigrid solver,  but they are " + amrex::getEnumNameString(WarpX::field_boundary_lo[idim])
                    );
                }

                if ( WarpX::field_boundary_hi[idim] == FieldBoundaryType::PEC ) {
                    hibc[idim] = LinOpBCType::Dirichlet;
                    dirichlet_flag[idim*2+1] = true;
                }
                else if ( WarpX::field_boundary_hi[idim] == FieldBoundaryType::Neumann ) {
                    hibc[idim] = LinOpBCType::Neumann;
                    dirichlet_flag[idim*2+1] = false;
                }
                else {
                    WARPX_ABORT_WITH_MESSAGE(
                        "Field boundary conditions have to be either periodic, PEC or neumann "
                        "when using the electrostatic Multigrid solver,  but they are " + amrex::getEnumNameString(WarpX::field_boundary_hi[idim])
                    );
                }
            }

            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                (WarpX::field_boundary_lo[idim] != FieldBoundaryType::Open &&
                WarpX::field_boundary_hi[idim] != FieldBoundaryType::Open &&
                WarpX::field_boundary_lo[idim] != FieldBoundaryType::PML &&
                WarpX::field_boundary_hi[idim] != FieldBoundaryType::PML) ,
                "Open and PML field boundary conditions only work with "
                "warpx.poisson_solver = fft."
            );
        }
        else if (WarpX::poisson_solver_id == PoissonSolverAlgo::IntegratedGreenFunction){
            if (WarpX::electrostatic_solver_id != ElectrostaticSolverAlgo::None){
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    (WarpX::field_boundary_lo[idim] == FieldBoundaryType::Open &&
                    WarpX::field_boundary_hi[idim] == FieldBoundaryType::Open),
                    "The FFT Poisson solver only works with field open boundary conditions "
                    "in electrostatic mode."
                );
            }
            else{ // if electromagnetic mode on with species.initialize_self_fields = 1
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    (WarpX::field_boundary_lo[idim] == FieldBoundaryType::PML &&
                    WarpX::field_boundary_hi[idim] == FieldBoundaryType::PML),
                    "The FFT Poisson solver only works with field PML boundary conditions "
                    "to initialize the self-fields of the species in electromagnetic mode."
                );
            }
        }
    }
}

void PoissonBoundaryHandler::BuildParsers ()
{
    potential_xlo_parser = utils::parser::makeParser(potential_xlo_str, {"t"});
    potential_xhi_parser = utils::parser::makeParser(potential_xhi_str, {"t"});
    potential_ylo_parser = utils::parser::makeParser(potential_ylo_str, {"t"});
    potential_yhi_parser = utils::parser::makeParser(potential_yhi_str, {"t"});
    potential_zlo_parser = utils::parser::makeParser(potential_zlo_str, {"t"});
    potential_zhi_parser = utils::parser::makeParser(potential_zhi_str, {"t"});

    potential_xlo = potential_xlo_parser.compile<1>();
    potential_xhi = potential_xhi_parser.compile<1>();
    potential_ylo = potential_ylo_parser.compile<1>();
    potential_yhi = potential_yhi_parser.compile<1>();
    potential_zlo = potential_zlo_parser.compile<1>();
    potential_zhi = potential_zhi_parser.compile<1>();

    BuildParsersEB();
}

void PoissonBoundaryHandler::BuildParsersEB ()
{
    potential_eb_parser  = utils::parser::makeParser(potential_eb_str, {"x", "y", "z", "t"});

    // check if the EB potential is a function of space or only of time
    const std::set<std::string> eb_symbols = potential_eb_parser.symbols();
    if ((eb_symbols.count("x") != 0) || (eb_symbols.count("y") != 0)
            || (eb_symbols.count("z") != 0)) {
        potential_eb = potential_eb_parser.compile<4>();
        phi_EB_only_t = false;
    }
    else {
        potential_eb_parser = utils::parser::makeParser(potential_eb_str, {"t"});
        potential_eb_t = potential_eb_parser.compile<1>();
    }
}
