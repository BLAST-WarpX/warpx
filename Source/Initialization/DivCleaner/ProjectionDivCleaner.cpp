 /* Copyright 2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy, Inc.)
 *
 * License: BSD-3-Clause-LBNL
 */

#include "ProjectionDivCleaner.H"

#include <AMReX_MLPoisson.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_LO_BCTYPES.H>

#include <WarpX.H>
#if defined WARPX_DIM_RZ
#include <FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CylindricalYeeAlgorithm.H>
#else
#include <FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H>
#endif
#include "Fields.H"
#include <Initialization/ExternalField.H>
#include <ablastr/utils/Communication.H>
#include <Utils/WarpXProfilerWrapper.H>

#include <map>

using namespace amrex;

namespace warpx::initialization {

ProjectionDivCleaner::ProjectionDivCleaner(std::string const& a_field_name) :
    m_field_name(a_field_name)
{
    using ablastr::fields::Direction;
    ReadParameters();

    auto& warpx = WarpX::GetInstance();

    // Only div clean level 0
    if (warpx.finestLevel() > 0) {
        ablastr::warn_manager::WMRecordWarning("Projection Div Cleaner",
            "Multiple AMR levels detected, only first level has been cleaned.",
            ablastr::warn_manager::WarnPriority::low);
    }

    m_solution.resize(m_levels);
    m_source.resize(m_levels);

    const int ncomps = WarpX::ncomps;
    auto const& ng = warpx.m_fields.get(m_field_name, Direction{0}, 0)->nGrowVect();

    for (int lev = 0; lev < m_levels; ++lev)
    {
        // Default BoxArray and DistributionMap for initializing the output MultiFab, m_mf_output.
        const amrex::BoxArray& ba = warpx.boxArray(lev);
        const amrex::DistributionMapping& dmap = warpx.DistributionMap(lev);

        m_solution[lev].reset();
        m_source[lev].reset();

        const auto tag1 = amrex::MFInfo().SetTag("div_cleaner_solution");
        m_solution[lev] = std::make_unique<MultiFab>(amrex::convert(ba, IntVect::TheCellVector()),
            dmap, ncomps, ng, tag1);
        const auto tag2 = amrex::MFInfo().SetTag("div_cleaner_source");
        m_source[lev] = std::make_unique<MultiFab>(amrex::convert(ba, IntVect::TheCellVector()),
            dmap, ncomps, ng, tag2);

        m_solution[lev]->setVal(0.0, ng);
        m_source[lev]->setVal(0.0, ng);
    }

    auto cell_size = WarpX::CellSize(0);
#if defined(WARPX_DIM_RZ)
    amrex::Real const rmin = WarpX::GetInstance().Geom(0).ProbLo(0);
    CylindricalYeeAlgorithm::InitializeStencilCoefficients( cell_size, rmin,
            m_h_stencil_coefs_x, m_h_stencil_coefs_y, m_h_stencil_coefs_z );
#else
    CartesianYeeAlgorithm::InitializeStencilCoefficients( cell_size,
            m_h_stencil_coefs_x, m_h_stencil_coefs_y, m_h_stencil_coefs_z );
#endif

    m_stencil_coefs_x.resize(m_h_stencil_coefs_x.size());
#if !defined(WARPX_DIM_RZ)
    m_stencil_coefs_y.resize(m_h_stencil_coefs_y.size());
#endif
    m_stencil_coefs_z.resize(m_h_stencil_coefs_z.size());

    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                          m_h_stencil_coefs_x.begin(), m_h_stencil_coefs_x.end(),
                          m_stencil_coefs_x.begin());
#if !defined(WARPX_DIM_RZ)
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                          m_h_stencil_coefs_y.begin(), m_h_stencil_coefs_y.end(),
                          m_stencil_coefs_y.begin());
#endif
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                          m_h_stencil_coefs_z.begin(), m_h_stencil_coefs_z.end(),
                          m_stencil_coefs_z.begin());
    amrex::Gpu::synchronize();
}

void
ProjectionDivCleaner::ReadParameters ()
{
    // Initialize tolerance based on field precision
    if constexpr (std::is_same<Real, float>::value) {
        m_rtol = 5e-5;
        m_atol = 0.0;
    }
    else {
        m_rtol = 5e-12;
        m_atol = 0.0;
    }

    const ParmParse pp_divb_cleaner("projection_divb_cleaner");

    // Defaults to rtol 5e-12 for double fields and 5e-5 for single
    utils::parser::queryWithParser(pp_divb_cleaner, "atol", m_atol);
    utils::parser::queryWithParser(pp_divb_cleaner, "rtol", m_rtol);
}

void
ProjectionDivCleaner::solve ()
{
    // Get WarpX object
    auto & warpx = WarpX::GetInstance();

    const auto& ba = warpx.boxArray();
    const auto& dmap = warpx.DistributionMap();
    const auto& geom = warpx.Geom();

    // Pull boundary conditions from WarpX class
    // bogus values are overwritten.
    amrex::Array<LinOpBCType,AMREX_SPACEDIM> lobc({AMREX_D_DECL(LinOpBCType::bogus,
                                                                LinOpBCType::bogus,
                                                                LinOpBCType::bogus)});
    amrex::Array<LinOpBCType,AMREX_SPACEDIM> hibc({AMREX_D_DECL(LinOpBCType::bogus,
                                                                LinOpBCType::bogus,
                                                                LinOpBCType::bogus)});

    std::map<FieldBoundaryType, LinOpBCType> bcmap{
        {FieldBoundaryType::PEC, LinOpBCType::Dirichlet},
        {FieldBoundaryType::Neumann, LinOpBCType::Neumann},
        {FieldBoundaryType::Periodic, LinOpBCType::Periodic},
        {FieldBoundaryType::None, LinOpBCType::Neumann}
    };

    for (int idim=0; idim<AMREX_SPACEDIM; idim++){
        auto itlo = bcmap.find(WarpX::field_boundary_lo[idim]);
        auto ithi = bcmap.find(WarpX::field_boundary_hi[idim]);
        if (itlo == bcmap.end() || ithi == bcmap.end()) {
            WARPX_ABORT_WITH_MESSAGE(
                "Field boundary conditions have to be either periodic, PEC or neumann "
                "when using the MLMG projection based divergence cleaner solver."
            );
        }

        lobc[idim] = bcmap[WarpX::field_boundary_lo[idim]];
        hibc[idim] = bcmap[WarpX::field_boundary_hi[idim]];
    }

    LPInfo info;
    info.setAgglomeration(m_agglomeration);
    info.setConsolidation(m_consolidation);
    info.setMaxCoarseningLevel(m_max_coarsening_level);
#ifdef WARPX_DIM_RZ
    info.setMetricTerm(true);
#endif


    for (int ilev = 0; ilev < m_levels; ++ilev)
    {
        MLPoisson mlpoisson({geom[ilev]}, {ba[ilev]}, {dmap[ilev]}, info);

        mlpoisson.setMaxOrder(m_linop_maxorder);
        mlpoisson.setDomainBC(lobc, hibc);

        if (ilev > 0) {
            mlpoisson.setCoarseFineBC(m_solution[ilev-1].get(), m_ref_ratio);
        }

        mlpoisson.setLevelBC(ilev, m_solution[ilev].get());

        MLMG mlmg(mlpoisson);
        mlmg.setMaxIter(m_max_iter);
        mlmg.setMaxFmgIter(m_max_fmg_iter);
        mlmg.setBottomSolver(m_bottom_solver);
        mlmg.setVerbose(m_verbose);
        mlmg.setBottomVerbose(m_bottom_verbose);
        mlmg.setAlwaysUseBNorm(false);
        mlmg.solve({m_solution[ilev].get()}, {m_source[ilev].get()}, m_rtol, m_atol);

        // Synchronize the ghost cells, do halo exchange
        ablastr::utils::communication::FillBoundary(*m_solution[ilev],
                                                m_solution[ilev]->nGrowVect(),
                                                WarpX::do_single_precision_comms,
                                                geom[ilev].periodicity());
    }
}

void
ProjectionDivCleaner::setSourceFromBfield ()
{
    using ablastr::fields::Direction;

    // Get WarpX object
    auto & warpx = WarpX::GetInstance();
    const auto& geom = warpx.Geom();

    // This function will compute -divB and store it in the source multifab
    for (int ilev = 0; ilev < m_levels; ++ilev)
    {
        WarpX::ComputeDivB(
            *m_source[ilev],
            0,
            {warpx.m_fields.get(m_field_name, Direction{0}, ilev),
             warpx.m_fields.get(m_field_name, Direction{1}, ilev),
             warpx.m_fields.get(m_field_name, Direction{2}, ilev)},
            WarpX::CellSize(0)
            );

        m_source[ilev]->mult(-1._rt);

        // Synchronize the ghost cells, do halo exchange
        ablastr::utils::communication::FillBoundary(*m_source[ilev],
                                                m_source[ilev]->nGrowVect(),
                                                WarpX::do_single_precision_comms,
                                                geom[ilev].periodicity());
    }
}

void
ProjectionDivCleaner::correctBfield ()
{
    using ablastr::fields::Direction;

    // Get WarpX object
    auto & warpx = WarpX::GetInstance();
    const auto& geom = warpx.Geom();

    // This function computes the gradient of the solution and subtracts out divB component from B
    for (int ilev = 0; ilev < m_levels; ++ilev)
    {
        // Grab B-field multifabs at this level
        amrex::MultiFab* Bx = warpx.m_fields.get(m_field_name, Direction{0}, ilev);
        amrex::MultiFab* By = warpx.m_fields.get(m_field_name, Direction{1}, ilev);
        amrex::MultiFab* Bz = warpx.m_fields.get(m_field_name, Direction{2}, ilev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(*m_solution[ilev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            // Grab references to B field arrays for this grid/tile
            amrex::Array4<Real> const& Bx_arr = Bx->array(mfi);
#if !defined(WARPX_DIM_RZ)
            amrex::Array4<Real> const& By_arr = By->array(mfi);
#endif
            amrex::Array4<Real> const& Bz_arr = Bz->array(mfi);

            // Extract stencil coefficients
            Real const * const AMREX_RESTRICT coefs_x = m_stencil_coefs_x.dataPtr();
            auto const n_coefs_x = static_cast<int>(m_stencil_coefs_x.size());
#if !defined(WARPX_DIM_RZ)
            Real const * const AMREX_RESTRICT coefs_y = m_stencil_coefs_y.dataPtr();
            auto const n_coefs_y = static_cast<int>(m_stencil_coefs_y.size());
#endif
            Real const * const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
            auto const n_coefs_z = static_cast<int>(m_stencil_coefs_z.size());

            const Box& tbx = mfi.tilebox(Bx->ixType().toIntVect());
#if !defined(WARPX_DIM_RZ)
            const Box& tby = mfi.tilebox(By->ixType().toIntVect());
#endif
            const Box& tbz = mfi.tilebox(Bz->ixType().toIntVect());

            amrex::Array4<Real> const& sol_arr = m_solution[ilev]->array(mfi);

#if defined(WARPX_DIM_RZ)
            amrex::ParallelFor(tbx, tbz,
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/)
            {
                Bx_arr(i,j,0) += CylindricalYeeAlgorithm::DownwardDr(sol_arr, coefs_x, n_coefs_x, i, j, 0, 0);
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/)
            {
                Bz_arr(i,j,0) += CylindricalYeeAlgorithm::DownwardDz(sol_arr, coefs_z, n_coefs_z, i, j, 0, 0);
            });
#else
            amrex::ParallelFor(tbx, tby, tbz,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                Bx_arr(i,j,k) += CartesianYeeAlgorithm::DownwardDx(sol_arr, coefs_x, n_coefs_x, i, j, k);
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                By_arr(i,j,k) += CartesianYeeAlgorithm::DownwardDy(sol_arr, coefs_y, n_coefs_y, i, j, k);
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                Bz_arr(i,j,k) += CartesianYeeAlgorithm::DownwardDz(sol_arr, coefs_z, n_coefs_z, i, j, k);
            });
#endif
        }
        // Synchronize the ghost cells, do halo exchange
        ablastr::utils::communication::FillBoundary(*Bx,
                                                    Bx->nGrowVect(),
                                                    WarpX::do_single_precision_comms,
                                                    geom[ilev].periodicity());
        ablastr::utils::communication::FillBoundary(*By,
                                                    By->nGrowVect(),
                                                    WarpX::do_single_precision_comms,
                                                    geom[ilev].periodicity());
        ablastr::utils::communication::FillBoundary(*Bz,
                                                    Bz->nGrowVect(),
                                                    WarpX::do_single_precision_comms,
                                                    geom[ilev].periodicity());
        amrex::Gpu::synchronize();
    }
}

} // namespace warpx::initialization

void
WarpX::ProjectionCleanDivB() {
    WARPX_PROFILE("WarpX::ProjectionDivCleanB()");

    if (grid_type == GridType::Collocated) {
        ablastr::warn_manager::WMRecordWarning("Projection Div Cleaner",
            "Grid Type is collocated, so divB not cleaned. Interpolation may lead to non-zero B field divergence.",
            ablastr::warn_manager::WarnPriority::low);
    } else if ( WarpX::electromagnetic_solver_id == ElectromagneticSolverAlgo::Yee
            ||  WarpX::electromagnetic_solver_id == ElectromagneticSolverAlgo::HybridPIC
            ||  ( (WarpX::electrostatic_solver_id == ElectrostaticSolverAlgo::LabFrame
                || WarpX::electrostatic_solver_id == ElectrostaticSolverAlgo::LabFrameElectroMagnetostatic)
                && WarpX::poisson_solver_id == PoissonSolverAlgo::Multigrid)) {
        amrex::Print() << Utils::TextMsg::Info( "Starting Projection B-Field divergence cleaner.");

        if constexpr (!std::is_same<Real, double>::value) {
            ablastr::warn_manager::WMRecordWarning("Projection Div Cleaner",
                "WarpX is running with a field precision of SINGLE."
                "Convergence of projection based div cleaner is not optimal and may fail.",
                ablastr::warn_manager::WarnPriority::low);
        }

        warpx::initialization::ProjectionDivCleaner dc("Bfield_fp_external");


        dc.setSourceFromBfield();
        dc.solve();
        dc.correctBfield();

        amrex::Print() << Utils::TextMsg::Info( "Finished Projection B-Field divergence cleaner.");
    } else {
        ablastr::warn_manager::WMRecordWarning("Projection Div Cleaner",
            "Only Yee, HybridPIC, and MLMG based static Labframe solvers are currently supported, so divB not cleaned. "
            "Interpolation may lead to non-zero B field divergence.",
            ablastr::warn_manager::WarnPriority::low);
    }
}
