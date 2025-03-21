/* Copyright 2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */

#include "ExternalVectorPotential.H"
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

    pp_ext_A.queryarr("fields", m_field_names);

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_field_names.empty(),
        "No external field names defined in external_vector_potential.fields");

    m_nFields = static_cast<int>(m_field_names.size());

    // Resize vectors and set defaults
    m_Ax_ext_grid_function.resize(m_nFields);
    m_Ay_ext_grid_function.resize(m_nFields);
    m_Az_ext_grid_function.resize(m_nFields);
    for (std::string & field : m_Ax_ext_grid_function) { field = "0.0"; }
    for (std::string & field : m_Ay_ext_grid_function) { field = "0.0"; }
    for (std::string & field : m_Az_ext_grid_function) { field = "0.0"; }

    m_A_external_parser.resize(m_nFields);
    m_A_external.resize(m_nFields);

    m_A_ext_time_function.resize(m_nFields);
    for (std::string & field_time : m_A_ext_time_function) { field_time = "1.0"; }

    m_A_external_time_parser.resize(m_nFields);
    m_A_time_scale.resize(m_nFields);

    m_read_A_from_file.resize(m_nFields);
    m_external_file_path.resize(m_nFields);
    for (std::string & file_name : m_external_file_path) { file_name = ""; }

    for (int i = 0; i < m_nFields; ++i) {
        bool read_from_file = false;
        utils::parser::queryWithParser(pp_ext_A,
            (m_field_names[i]+".read_from_file").c_str(), read_from_file);
        m_read_A_from_file[i] = read_from_file;

        if (m_read_A_from_file[i]) {
            pp_ext_A.query((m_field_names[i]+".path").c_str(), m_external_file_path[i]);
        } else {
            pp_ext_A.query((m_field_names[i]+".Ax_external_grid_function(x,y,z)").c_str(),
                m_Ax_ext_grid_function[i]);
            pp_ext_A.query((m_field_names[i]+".Ay_external_grid_function(x,y,z)").c_str(),
                m_Ay_ext_grid_function[i]);
            pp_ext_A.query((m_field_names[i]+".Az_external_grid_function(x,y,z)").c_str(),
                m_Az_ext_grid_function[i]);
        }

        pp_ext_A.query((m_field_names[i]+".A_time_external_function(t)").c_str(),
            m_A_ext_time_function[i]);
    }
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
    for (std::string const & field_name : m_field_names) {
        const std::string Aext_field = field_name + std::string{"_Aext"};
        fields.alloc_init(Aext_field, Direction{0},
            lev, amrex::convert(ba, Ex_nodal_flag),
            dm, ncomps, ngEB, 0.0_rt);
        fields.alloc_init(Aext_field, Direction{1},
            lev, amrex::convert(ba, Ey_nodal_flag),
            dm, ncomps, ngEB, 0.0_rt);
        fields.alloc_init(Aext_field, Direction{2},
            lev, amrex::convert(ba, Ez_nodal_flag),
            dm, ncomps, ngEB, 0.0_rt);

        const std::string curlAext_field = field_name + std::string{"_curlAext"};
        fields.alloc_init(curlAext_field, Direction{0},
            lev, amrex::convert(ba, Bx_nodal_flag),
            dm, ncomps, ngEB, 0.0_rt);
        fields.alloc_init(curlAext_field, Direction{1},
            lev, amrex::convert(ba, By_nodal_flag),
            dm, ncomps, ngEB, 0.0_rt);
        fields.alloc_init(curlAext_field, Direction{2},
            lev, amrex::convert(ba, Bz_nodal_flag),
            dm, ncomps, ngEB, 0.0_rt);
    }
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

    int A_time_dep_count = 0;

    for (int i = 0; i < m_nFields; ++i) {

        const std::string Aext_field = m_field_names[i] + std::string{"_Aext"};

        if (m_read_A_from_file[i]) {
            // Read A fields from file
            for (auto lev = 0; lev <= warpx.finestLevel(); ++lev) {
#if defined(WARPX_DIM_RZ)
                warpx.ReadExternalFieldFromFile(m_external_file_path[i],
                    warpx.m_fields.get(Aext_field, Direction{0}, lev),
                    "A", "r");
                warpx.ReadExternalFieldFromFile(m_external_file_path[i],
                    warpx.m_fields.get(Aext_field, Direction{1}, lev),
                    "A", "t");
                warpx.ReadExternalFieldFromFile(m_external_file_path[i],
                    warpx.m_fields.get(Aext_field, Direction{2}, lev),
                    "A", "z");
#else
                warpx.ReadExternalFieldFromFile(m_external_file_path[i],
                    warpx.m_fields.get(Aext_field, Direction{0}, lev),
                    "A", "x");
                warpx.ReadExternalFieldFromFile(m_external_file_path[i],
                    warpx.m_fields.get(Aext_field, Direction{1}, lev),
                    "A", "y");
                warpx.ReadExternalFieldFromFile(m_external_file_path[i],
                    warpx.m_fields.get(Aext_field, Direction{2}, lev),
                    "A", "z");
#endif
            }
        } else {
            // Initialize the A fields from expression
            m_A_external_parser[i][0] = std::make_unique<amrex::Parser>(
                utils::parser::makeParser(m_Ax_ext_grid_function[i],{"x","y","z","t"}));
            m_A_external_parser[i][1] = std::make_unique<amrex::Parser>(
                utils::parser::makeParser(m_Ay_ext_grid_function[i],{"x","y","z","t"}));
            m_A_external_parser[i][2] = std::make_unique<amrex::Parser>(
                utils::parser::makeParser(m_Az_ext_grid_function[i],{"x","y","z","t"}));
            m_A_external[i][0] = m_A_external_parser[i][0]->compile<4>();
            m_A_external[i][1] = m_A_external_parser[i][1]->compile<4>();
            m_A_external[i][2] = m_A_external_parser[i][2]->compile<4>();

            // check if the external current parsers depend on time
            for (int idim=0; idim<3; idim++) {
                const std::set<std::string> A_ext_symbols = m_A_external_parser[i][idim]->symbols();
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(A_ext_symbols.count("t") == 0,
                    "Externally Applied Vector potential time variation must be set with A_time_external_function(t)");
            }

            // Initialize data onto grid
            for (auto lev = 0; lev <= warpx.finestLevel(); ++lev) {
                warpx.ComputeExternalFieldOnGridUsingParser(
                    Aext_field,
                    m_A_external[i][0],
                    m_A_external[i][1],
                    m_A_external[i][2],
                    lev, PatchType::fine,
                    warpx.GetEBUpdateEFlag(),
                    false);

                for (int idir = 0; idir < 3; ++idir) {
                    warpx.m_fields.get(Aext_field, Direction{idir}, lev)->
                        FillBoundary(warpx.Geom(lev).periodicity());
                }
            }
        }

        amrex::Gpu::streamSynchronize();

        CalculateExternalCurlA(m_field_names[i]);

        // Generate parser for time function
        m_A_external_time_parser[i] = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(m_A_ext_time_function[i],{"t",}));
        m_A_time_scale[i] = m_A_external_time_parser[i]->compile<1>();

        const std::set<std::string> A_time_ext_symbols = m_A_external_time_parser[i]->symbols();
        A_time_dep_count += static_cast<int>(A_time_ext_symbols.count("t"));
    }

    if (A_time_dep_count > 0) {
        ablastr::warn_manager::WMRecordWarning(
            "HybridPIC ExternalVectorPotential",
            "Coulomb Gauge is Expected, please be sure to have a divergence free A. Divergence cleaning of A to be implemented soon.",
            ablastr::warn_manager::WarnPriority::low
        );
    }

    UpdateHybridExternalFields(warpx.gett_new(0), warpx.getdt(0));
}


void
ExternalVectorPotential::CalculateExternalCurlA ()
{
    for (auto fname : m_field_names) {
        CalculateExternalCurlA(fname);
    }
}

void
ExternalVectorPotential::CalculateExternalCurlA (std::string& coil_name)
{
    using ablastr::fields::Direction;
    auto & warpx = WarpX::GetInstance();

    // Compute the curl of the reference A field (unscaled by time function)
    const std::string Aext_field = coil_name + std::string{"_Aext"};
    const std::string curlAext_field = coil_name + std::string{"_curlAext"};

    ablastr::fields::MultiLevelVectorField A_ext =
        warpx.m_fields.get_mr_levels_alldirs(Aext_field, warpx.finestLevel());
    ablastr::fields::MultiLevelVectorField curlA_ext =
        warpx.m_fields.get_mr_levels_alldirs(curlAext_field, warpx.finestLevel());

    for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
        warpx.get_pointer_fdtd_solver_fp(lev)->ComputeCurlA(
            curlA_ext[lev],
            A_ext[lev],
            warpx.GetEBUpdateBFlag()[lev],
            lev);

        for (int idir = 0; idir < 3; ++idir) {
            warpx.m_fields.get(curlAext_field, Direction{idir}, lev)->
                FillBoundary(warpx.Geom(lev).periodicity());
        }
    }
}

AMREX_FORCE_INLINE
void
ExternalVectorPotential::AddExternalFieldFromVectorPotential (
    ablastr::fields::VectorField const& dstField,
    amrex::Real scale_factor,
    ablastr::fields::VectorField const& srcField,
    std::array< std::unique_ptr<amrex::iMultiFab>,3> const& eb_update)
{
    // Loop through the grids, and over the tiles within each grid
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*dstField[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        // Extract field data for this grid/tile
        Array4<Real> const& Fx = dstField[0]->array(mfi);
        Array4<Real> const& Fy = dstField[1]->array(mfi);
        Array4<Real> const& Fz = dstField[2]->array(mfi);

        Array4<const Real> const& Sx = srcField[0]->const_array(mfi);
        Array4<const Real> const& Sy = srcField[1]->const_array(mfi);
        Array4<const Real> const& Sz = srcField[2]->const_array(mfi);

        // Extract structures indicating where the fields
        // should be updated, given the position of the embedded boundaries.
        amrex::Array4<int> update_Fx_arr, update_Fy_arr, update_Fz_arr;
        if (EB::enabled()) {
            update_Fx_arr = eb_update[0]->array(mfi);
            update_Fy_arr = eb_update[1]->array(mfi);
            update_Fz_arr = eb_update[2]->array(mfi);
        }

        // Extract tileboxes for which to loop
        Box const& tbx  = mfi.tilebox(dstField[0]->ixType().toIntVect());
        Box const& tby  = mfi.tilebox(dstField[1]->ixType().toIntVect());
        Box const& tbz  = mfi.tilebox(dstField[2]->ixType().toIntVect());

        // Loop over the cells and update the fields
        amrex::ParallelFor(tbx, tby, tbz,

            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                // Skip field update in the embedded boundaries
                if (update_Fx_arr && update_Fx_arr(i, j, k) == 0) { return; }

                Fx(i,j,k) += scale_factor * Sx(i,j,k);
            },

            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                // Skip field update in the embedded boundaries
                if (update_Fy_arr && update_Fy_arr(i, j, k) == 0) { return; }

                Fy(i,j,k) += scale_factor * Sy(i,j,k);
            },

            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                // Skip field update in the embedded boundaries
                if (update_Fz_arr && update_Fz_arr(i, j, k) == 0) { return; }

                Fz(i,j,k) += scale_factor * Sz(i,j,k);
            }
        );
    }
}

void
ExternalVectorPotential::UpdateHybridExternalFields (const amrex::Real t, const amrex::Real dt)
{
    using ablastr::fields::Direction;
    auto& warpx = WarpX::GetInstance();

    ablastr::fields::MultiLevelVectorField B_ext =
        warpx.m_fields.get_mr_levels_alldirs(FieldType::hybrid_B_fp_external, warpx.finestLevel());
    ablastr::fields::MultiLevelVectorField E_ext =
        warpx.m_fields.get_mr_levels_alldirs(FieldType::hybrid_E_fp_external, warpx.finestLevel());

    // Zero E and B external fields prior to accumulating external fields
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
        for (int idir = 0; idir < 3; ++idir) {
            B_ext[lev][Direction{idir}]->setVal(0.0_rt);
            E_ext[lev][Direction{idir}]->setVal(0.0_rt);
        }
    }

    // Iterate over external fields and add together with individual time functions.
    for (int i = 0; i < m_nFields; ++i) {
        const std::string Aext_field = m_field_names[i] + std::string{"_Aext"};
        const std::string curlAext_field = m_field_names[i] + std::string{"_curlAext"};

        // Get B-field Scaling Factor
        const amrex::Real scale_factor_B = m_A_time_scale[i](t);

        // Get dA/dt scaling factor based on time centered FD around t
        const amrex::Real sf_l = m_A_time_scale[i](t-0.5_rt*dt);
        const amrex::Real sf_r = m_A_time_scale[i](t+0.5_rt*dt);
        const amrex::Real scale_factor_E = -(sf_r - sf_l)/dt;

        ablastr::fields::MultiLevelVectorField A_ext =
            warpx.m_fields.get_mr_levels_alldirs(Aext_field, warpx.finestLevel());
        ablastr::fields::MultiLevelVectorField curlA_ext =
            warpx.m_fields.get_mr_levels_alldirs(curlAext_field, warpx.finestLevel());

        for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
            AddExternalFieldFromVectorPotential(E_ext[lev], scale_factor_E, A_ext[lev], warpx.GetEBUpdateEFlag()[lev]);
            AddExternalFieldFromVectorPotential(B_ext[lev], scale_factor_B, curlA_ext[lev], warpx.GetEBUpdateBFlag()[lev]);

            for (int idir = 0; idir < 3; ++idir) {
                E_ext[lev][Direction{idir}]->FillBoundary(warpx.Geom(lev).periodicity());
                B_ext[lev][Direction{idir}]->FillBoundary(warpx.Geom(lev).periodicity());
            }
        }
    }
    amrex::Gpu::streamSynchronize();
}
