/* Copyright 2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */

#include "ExternalVectorPotential.H"
#include "EmbeddedBoundary/Covered.H"
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "Fields.H"
#include "WarpX.H"

#include <ablastr/fields/MultiFabRegister.H>

using namespace amrex;
using namespace warpx::fields;

ExternalVectorPotential::ExternalVectorPotential ()
{
    ReadParameters();
}

void
ExternalVectorPotential::ReadParameters ()
{
    const ParmParse pp_ext_A("external_vector_potential");

    utils::parser::queryWithParser(pp_ext_A, "read_from_file", m_read_A_from_file);

    if (m_read_A_from_file) {
        pp_ext_A.query("path", m_external_file_path);
    } else {
        pp_ext_A.query("Ax_external_grid_function(x,y,z)", m_Ax_ext_grid_function);
        pp_ext_A.query("Ay_external_grid_function(x,y,z)", m_Ay_ext_grid_function);
        pp_ext_A.query("Az_external_grid_function(x,y,z)", m_Az_ext_grid_function);
    }

    pp_ext_A.query("A_time_external_function(t)", m_A_ext_time_function);
}

void
ExternalVectorPotential::AllocateLevelMFs (
    ablastr::fields::MultiFabRegister & fields,
    int lev, const BoxArray& ba, const DistributionMapping& dm,
    const int ncomps,
    const IntVect& ngEB,
    const IntVect& Ex_nodal_flag,
    const IntVect& Ey_nodal_flag,
    const IntVect& Ez_nodal_flag,
    const IntVect& Bx_nodal_flag,
    const IntVect& By_nodal_flag,
    const IntVect& Bz_nodal_flag)
{
    using ablastr::fields::Direction;
    fields.alloc_init(FieldType::hybrid_A_fp_external, Direction{0},
        lev, amrex::convert(ba, Ex_nodal_flag),
        dm, ncomps, ngEB, 0.0_rt);
    fields.alloc_init(FieldType::hybrid_A_fp_external, Direction{1},
        lev, amrex::convert(ba, Ey_nodal_flag),
        dm, ncomps, ngEB, 0.0_rt);
    fields.alloc_init(FieldType::hybrid_A_fp_external, Direction{2},
        lev, amrex::convert(ba, Ez_nodal_flag),
        dm, ncomps, ngEB, 0.0_rt);
    fields.alloc_init(FieldType::hybrid_E_fp_external, Direction{0},
        lev, amrex::convert(ba, Ex_nodal_flag),
        dm, ncomps, ngEB, 0.0_rt);
    fields.alloc_init(FieldType::hybrid_E_fp_external, Direction{1},
        lev, amrex::convert(ba, Ey_nodal_flag),
        dm, ncomps, ngEB, 0.0_rt);
    fields.alloc_init(FieldType::hybrid_E_fp_external, Direction{2},
        lev, amrex::convert(ba, Ez_nodal_flag),
        dm, ncomps, ngEB, 0.0_rt);
    fields.alloc_init(FieldType::hybrid_B_fp_external, Direction{0},
        lev, amrex::convert(ba, Bx_nodal_flag),
        dm, ncomps, ngEB, 0.0_rt);
    fields.alloc_init(FieldType::hybrid_B_fp_external, Direction{1},
        lev, amrex::convert(ba, By_nodal_flag),
        dm, ncomps, ngEB, 0.0_rt);
    fields.alloc_init(FieldType::hybrid_B_fp_external, Direction{2},
        lev, amrex::convert(ba, Bz_nodal_flag),
        dm, ncomps, ngEB, 0.0_rt);
}

void
ExternalVectorPotential::InitData ()
{
    using ablastr::fields::Direction;
    auto& warpx = WarpX::GetInstance();

    if (m_read_A_from_file) {
        // Read A fields from file
        for (auto lev = 0; lev <= warpx.finestLevel(); ++lev) {
#if defined(WARPX_DIM_RZ)
            warpx.ReadExternalFieldFromFile(m_external_file_path,
                warpx.m_fields.get(FieldType::hybrid_A_fp_external, Direction{0}, lev),
                "A", "r");
            warpx.ReadExternalFieldFromFile(m_external_file_path,
                warpx.m_fields.get(FieldType::hybrid_A_fp_external, Direction{1}, lev),
                "A", "t");
            warpx.ReadExternalFieldFromFile(m_external_file_path,
                warpx.m_fields.get(FieldType::hybrid_A_fp_external, Direction{2}, lev),
                "A", "z");
#else
            warpx.ReadExternalFieldFromFile(m_external_file_path,
                warpx.m_fields.get(FieldType::hybrid_A_fp_external, Direction{0}, lev),
                "A", "x");
            warpx.ReadExternalFieldFromFile(m_external_file_path,
                warpx.m_fields.get(FieldType::hybrid_A_fp_external, Direction{1}, lev),
                "A", "y");
            warpx.ReadExternalFieldFromFile(m_external_file_path,
                warpx.m_fields.get(FieldType::hybrid_A_fp_external, Direction{2}, lev),
                "A", "z");
#endif
        }
    } else {
        // Initialize the A fields from expression
        m_A_external_parser[0] = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(m_Ax_ext_grid_function,{"x","y","z","t"}));
        m_A_external_parser[1] = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(m_Ay_ext_grid_function,{"x","y","z","t"}));
        m_A_external_parser[2] = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(m_Az_ext_grid_function,{"x","y","z","t"}));
        m_A_external[0] = m_A_external_parser[0]->compile<4>();
        m_A_external[1] = m_A_external_parser[1]->compile<4>();
        m_A_external[2] = m_A_external_parser[2]->compile<4>();

        // check if the external current parsers depend on time
        for (int i=0; i<3; i++) {
            const std::set<std::string> A_ext_symbols = m_A_external_parser[i]->symbols();
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(A_ext_symbols.count("t") == 0,
                "Externally Applied Vector potential time variation must be set with A_time_external_function(t)");
        }

        // Initialize data onto grid
        for (auto lev = 0; lev <= warpx.finestLevel(); ++lev) {
            warpx.ComputeExternalFieldOnGridUsingParser(
                FieldType::hybrid_A_fp_external,
                m_A_external[0],
                m_A_external[1],
                m_A_external[2],
                lev, PatchType::fine, EB::CoverTopology::none);

            for (int idir = 0; idir < 3; ++idir) {
                warpx.m_fields.get(FieldType::hybrid_A_fp_external, Direction{idir}, lev)->
                    FillBoundary(warpx.Geom(lev).periodicity());
            }
        }
    }

    amrex::Gpu::streamSynchronize();

    m_A_external_time_parser = std::make_unique<amrex::Parser>(
        utils::parser::makeParser(m_A_ext_time_function,{"t",}));
    m_A_time_scale = m_A_external_time_parser->compile<1>();

    UpdateHybridExternalFields(warpx.gett_new(0), warpx.getdt(0));
}

AMREX_FORCE_INLINE
void
ExternalVectorPotential::ZeroFieldinEB (ablastr::fields::VectorField const& Field, EB::CoverTopology topology, const int lev)
{
    auto &warpx = WarpX::GetInstance();

    // Loop through the grids, and over the tiles within each grid
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Field[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        // Extract field data for this grid/tile
        Array4<Real> const& Fx = Field[0]->array(mfi);
        Array4<Real> const& Fy = Field[1]->array(mfi);
        Array4<Real> const& Fz = Field[2]->array(mfi);

        EB::Covered const& cov_ptr = EB::Covered(mfi, lev);

        // Extract tileboxes for which to loop
        Box const& tbx  = mfi.tilebox(Field[0]->ixType().toIntVect());
        Box const& tby  = mfi.tilebox(Field[1]->ixType().toIntVect());
        Box const& tbz  = mfi.tilebox(Field[2]->ixType().toIntVect());

        // Loop over the cells and update the fields
        amrex::ParallelFor(tbx, tby, tbz,

            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                if (cov_ptr.isCovered(0, topology, i, j, k)) Fx(i, j, k) = 0_rt;
            },

            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                if (cov_ptr.isCovered(1, topology, i, j, k)) Fy(i, j, k) = 0_rt;
            },

            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                if (cov_ptr.isCovered(2, topology, i, j, k)) Fz(i, j, k) = 0_rt;
            }
        );
    }
}

void
ExternalVectorPotential::UpdateHybridExternalFields (const amrex::Real t, const amrex::Real dt)
{
    using ablastr::fields::Direction;
    auto& warpx = WarpX::GetInstance();

    // Get B-field Scaling Factor
    amrex::Real scale_factor_B = m_A_time_scale(t);

    // Get dA/dt scaling factor based on time centered FD around t
    amrex::Real sf_l = m_A_time_scale(t-0.5_rt*dt);
    amrex::Real sf_r = m_A_time_scale(t+0.5_rt*dt);
    amrex::Real scale_factor_E = -(sf_r - sf_l)/dt;

    ablastr::fields::MultiLevelVectorField A_ext =
        warpx.m_fields.get_mr_levels_alldirs(FieldType::hybrid_A_fp_external, warpx.finestLevel());
    ablastr::fields::MultiLevelVectorField B_ext =
        warpx.m_fields.get_mr_levels_alldirs(FieldType::hybrid_B_fp_external, warpx.finestLevel());
    ablastr::fields::MultiLevelVectorField E_ext =
        warpx.m_fields.get_mr_levels_alldirs(FieldType::hybrid_E_fp_external, warpx.finestLevel());

    for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
        warpx.get_pointer_fdtd_solver_fp(lev)->ComputeCurlA(
            B_ext[lev],
            A_ext[lev],
            lev
        );

        for (int idir = 0; idir < 3; ++idir) {
            // Scale B field by the time factor
            B_ext[lev][Direction{idir}]->mult(scale_factor_B);
            B_ext[lev][Direction{idir}]->FillBoundary(warpx.Geom(lev).periodicity());

            // Copy A into E and scale by the (-) derivative of the time function
            E_ext[lev][Direction{idir}]->setVal(scale_factor_E);
            amrex::MultiFab::Multiply(*E_ext[lev][Direction{idir}], *A_ext[lev][Direction{idir}], 0, 0, 1, 0);
            E_ext[lev][Direction{idir}]->FillBoundary(warpx.Geom(lev).periodicity());
        }
        ZeroFieldinEB(B_ext[lev], EB::CoverTopology::face, lev);
        ZeroFieldinEB(E_ext[lev], EB::CoverTopology::edge, lev);
    }
    amrex::Gpu::streamSynchronize();
}
