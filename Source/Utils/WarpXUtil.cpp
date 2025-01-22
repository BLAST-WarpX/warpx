/* Copyright 2019-2020 Andrew Myers, Burlen Loring, Luca Fedeli
 * Maxence Thevenet, Remi Lehe, Revathi Jambunathan
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "WarpX.H"

#include "Utils/Parser/ParserUtils.H"
#include "TextMsg.H"
#include "WarpXAlgorithmSelection.H"
#include "WarpXConst.H"
#include "WarpXProfilerWrapper.H"
#include "WarpXUtil.H"

#include <ablastr/fields/MultiFabRegister.H>
#include <ablastr/warn_manager/WarnManager.H>

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_BLassert.H>
#include <AMReX_Box.H>
#include <AMReX_Config.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_FabArray.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Parser.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <set>
#include <string>
#include <limits>

using namespace amrex;

void ReadBoostedFrameParameters(Real& gamma_boost, Real& beta_boost,
                                Vector<int>& boost_direction)
{
    const ParmParse pp_warpx("warpx");
    utils::parser::queryWithParser(pp_warpx, "gamma_boost", gamma_boost);
    if( gamma_boost > 1. ) {
        beta_boost = std::sqrt(1._rt-1._rt/std::pow(gamma_boost,2._rt));
        std::string s;
        pp_warpx.get("boost_direction", s);
        if (s == "x" || s == "X") {
            boost_direction[0] = 1;
        }
#if defined(WARPX_DIM_3D)
        else if (s == "y" || s == "Y") {
            boost_direction[1] = 1;
        }
#endif
        else if (s == "z" || s == "Z") {
            boost_direction[2] = 1;
        }
        else {
            WARPX_ABORT_WITH_MESSAGE("Unknown boost_dir: "+s);
        }

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE( s == "z" || s == "Z" ,
            "The boost must be in the z direction.");
    }
}

void ReadMovingWindowParameters(
    int& do_moving_window, int& start_moving_window_step, int& end_moving_window_step,
    int& moving_window_dir, amrex::Real& moving_window_v)
{
    const ParmParse pp_warpx("warpx");
    pp_warpx.query("do_moving_window", do_moving_window);
    if (do_moving_window) {
        utils::parser::queryWithParser(
            pp_warpx, "start_moving_window_step", start_moving_window_step);
        utils::parser::queryWithParser(
            pp_warpx, "end_moving_window_step", end_moving_window_step);
        std::string s;
        pp_warpx.get("moving_window_dir", s);

        if (s == "z" || s == "Z") {
            moving_window_dir = WARPX_ZINDEX;
        }
#if defined(WARPX_DIM_3D)
        else if (s == "y" || s == "Y") {
            moving_window_dir = 1;
        }
#endif
#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_3D)
        else if (s == "x" || s == "X") {
            moving_window_dir = 0;
        }
#endif
        else {
            WARPX_ABORT_WITH_MESSAGE("Unknown moving_window_dir: "+s);
        }

        utils::parser::getWithParser(
            pp_warpx, "moving_window_v", moving_window_v);
        moving_window_v *= PhysConst::c;
    }
}

void ConvertLabParamsToBoost()
{
    Real gamma_boost = 1., beta_boost = 0.;
    int max_level = 0;
    Vector<int> boost_direction {0,0,0};

    ReadBoostedFrameParameters(gamma_boost, beta_boost, boost_direction);

    if (gamma_boost <= 1.) { return; }

    Vector<Real> prob_lo(AMREX_SPACEDIM);
    Vector<Real> prob_hi(AMREX_SPACEDIM);
    Vector<Real> fine_tag_lo(AMREX_SPACEDIM);
    Vector<Real> fine_tag_hi(AMREX_SPACEDIM);
    Vector<Real> slice_lo(AMREX_SPACEDIM);
    Vector<Real> slice_hi(AMREX_SPACEDIM);

    ParmParse pp_geometry("geometry");
    ParmParse pp_warpx("warpx");
    ParmParse pp_slice("slice");
    const ParmParse pp_amr("amr");

    utils::parser::getArrWithParser(
        pp_geometry, "prob_lo", prob_lo, 0, AMREX_SPACEDIM);
    utils::parser::getArrWithParser(
        pp_geometry, "prob_hi", prob_hi, 0, AMREX_SPACEDIM);

    utils::parser::queryArrWithParser(
        pp_slice, "dom_lo", slice_lo, 0, AMREX_SPACEDIM);
    AMREX_ALWAYS_ASSERT(slice_lo.size() == AMREX_SPACEDIM);
    utils::parser::queryArrWithParser(
        pp_slice, "dom_hi", slice_hi, 0, AMREX_SPACEDIM);
    AMREX_ALWAYS_ASSERT(slice_hi.size() == AMREX_SPACEDIM);


    pp_amr.query("max_level", max_level);
    if (max_level > 0){
      utils::parser::getArrWithParser(
        pp_warpx, "fine_tag_lo", fine_tag_lo);
      utils::parser::getArrWithParser(
        pp_warpx, "fine_tag_hi", fine_tag_hi);
    }


#if defined(WARPX_DIM_3D)
    Vector<int> dim_map {0, 1, 2};
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    Vector<int> dim_map {0, 2};
#else
    Vector<int> dim_map {2};
#endif

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        if (boost_direction[dim_map[idim]]) {
            amrex::Real convert_factor;
            amrex::Real beta_window = beta_boost;
            if (WarpX::do_moving_window && idim == WarpX::moving_window_dir) {
                beta_window = WarpX::moving_window_v / PhysConst::c;
            }
            convert_factor = 1._rt/( gamma_boost * ( 1 - beta_boost * beta_window ) );
            prob_lo[idim] *= convert_factor;
            prob_hi[idim] *= convert_factor;
            if (max_level > 0){
              fine_tag_lo[idim] *= convert_factor;
              fine_tag_hi[idim] *= convert_factor;
            }
            slice_lo[idim] *= convert_factor;
            slice_hi[idim] *= convert_factor;
            break;
        }
    }

    pp_geometry.addarr("prob_lo", prob_lo);
    pp_geometry.addarr("prob_hi", prob_hi);
    if (max_level > 0){
      pp_warpx.addarr("fine_tag_lo", fine_tag_lo);
      pp_warpx.addarr("fine_tag_hi", fine_tag_hi);
    }

    pp_slice.addarr("dom_lo",slice_lo);
    pp_slice.addarr("dom_hi",slice_hi);

}

void NullifyMFinstance (
    amrex::MultiFab *mf,
    int lev,
    amrex::Real zmin,
    amrex::Real zmax
)
{
    int const ncomp = mf->nComp();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for(amrex::MFIter mfi(*mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi){
        const amrex::Box& bx = mfi.tilebox();
        // Get box lower and upper physical z bound, and dz
        const amrex::Real zmin_box = WarpX::LowerCorner(bx, lev, 0._rt).z;
        const amrex::Real zmax_box = WarpX::UpperCorner(bx, lev, 0._rt).z;
        const amrex::Real dz  = WarpX::CellSize(lev)[2];
        // Get box lower index in the z direction
#if defined(WARPX_DIM_3D)
        const int lo_ind = bx.loVect()[2];
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
        const int lo_ind = bx.loVect()[1];
#else
        const int lo_ind = bx.loVect()[0];
#endif
        // Check if box intersect with [zmin, zmax]
        if ( (zmax>zmin_box && zmin<=zmax_box) ){
            const Array4<Real> arr = (*mf)[mfi].array();
            // Set field to 0 between zmin and zmax
            ParallelFor(bx, ncomp,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept{
#if defined(WARPX_DIM_3D)
                    const Real z_gridpoint = zmin_box+(k-lo_ind)*dz;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                    const Real z_gridpoint = zmin_box+(j-lo_ind)*dz;
#else
                    const Real z_gridpoint = zmin_box+(i-lo_ind)*dz;
#endif
                    if ( (z_gridpoint >= zmin) && (z_gridpoint < zmax) ) {
                        arr(i,j,k,n) = 0.;
                    }
                }
            );
        }
    }
}

void NullifyMF (
    ablastr::fields::MultiFabRegister& multifab_map,
    std::string const& mf_name,
    int lev,
    amrex::Real zmin,
    amrex::Real zmax
)
{
    WARPX_PROFILE("WarpXUtil::NullifyMF()");
    if (!multifab_map.has(mf_name, lev)) { return; }

    auto * mf = multifab_map.get(mf_name, lev);

    NullifyMFinstance ( mf, lev, zmin, zmax);
}

void NullifyMF (
    ablastr::fields::MultiFabRegister& multifab_map,
    std::string const& mf_name,
    ablastr::fields::Direction dir,
    int lev,
    amrex::Real zmin,
    amrex::Real zmax
)
{
    WARPX_PROFILE("WarpXUtil::NullifyMF()");
    if (!multifab_map.has(mf_name, dir, lev)) { return; }

    auto * mf = multifab_map.get(mf_name, dir, lev);

    NullifyMFinstance ( mf, lev, zmin, zmax);
}

namespace WarpXUtilIO{
    bool WriteBinaryDataOnFile(const std::string& filename, const amrex::Vector<char>& data)
    {
        std::ofstream of{filename, std::ios::binary};
        of.write(data.data(), data.size());
        of.close();
        return  of.good();
    }
}

void CheckGriddingForRZSpectral ()
{
#ifdef WARPX_DIM_RZ
    const ParmParse pp_algo("algo");
    auto electromagnetic_solver_id = ElectromagneticSolverAlgo::Default;
    pp_algo.query_enum_sloppy("maxwell_solver", electromagnetic_solver_id, "-_");

    // only check for PSATD in RZ
    if (electromagnetic_solver_id != ElectromagneticSolverAlgo::PSATD) {
        return;
    }

    int max_level;
    Vector<int> n_cell(AMREX_SPACEDIM, -1);

    ParmParse pp_amr("amr");

    pp_amr.get("max_level",max_level);
    pp_amr.getarr("n_cell", n_cell, 0, AMREX_SPACEDIM);

    Vector<int> blocking_factor_x(max_level+1);
    Vector<int> max_grid_size_x(max_level+1);

    // Set the radial block size to be the power of 2 greater than or equal to
    // the number of grid cells. The blocking factor must be a power of 2
    // and the max_grid_size must be a multiple of the blocking_factor unless
    // it is less than the blocking factor.
    int k = 1;
    while (k < n_cell[0]) {
        k *= 2;
    }
    blocking_factor_x[0] = k;
    max_grid_size_x[0] = n_cell[0];

    for (int lev=1 ; lev <= max_level ; lev++) {
        // For this to be correct, this needs to read in any user specified refinement ratios.
        // But since that is messy and unlikely to be needed anytime soon, the ratio is
        // fixed to 2 which will be the most likely value.
        blocking_factor_x[lev] = blocking_factor_x[lev-1]*2; // refRatio(lev-1);
        max_grid_size_x[lev] = max_grid_size_x[lev-1]*2; // refRatio(lev-1);
    }

    // Note that any user input values for these parameters are discarded.
    pp_amr.addarr("blocking_factor_x", blocking_factor_x);
    pp_amr.addarr("max_grid_size_x", max_grid_size_x);

    // Adjust the longitudinal block sizes, making sure that there are
    // more blocks than processors.
    // The factor of 8 is there to make some room for higher order
    // shape factors and filtering.
    const int nprocs = ParallelDescriptor::NProcs();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(n_cell[1] >= 8*nprocs,
                                     "With RZ spectral, there must be at least eight z-cells per processor so that there can be at least one block per processor.");

    // Get the longitudinal blocking factor in case it was set by the user.
    // If not set, use the default value of 8.
    Vector<int> bf;
    pp_amr.queryarr("blocking_factor", bf);
    pp_amr.queryarr("blocking_factor_y", bf);
    bf.resize(std::max(static_cast<int>(bf.size()),1),8);

    // Modify the default or any user input, making sure that the blocking factor
    // is small enough so that there will be at least as many blocks as there are
    // processors. Because of the ASSERT above, bf will never be less than 8.
    while (n_cell[1] < nprocs*bf[0]) {
        bf[0] /= 2;
    }
    pp_amr.addarr("blocking_factor_y", bf);

    // Get the longitudinal max grid size in case it was set by the user.
    // If not set, use the default value of 128.
    Vector<int> mg;
    pp_amr.queryarr("max_grid_size", mg);
    pp_amr.queryarr("max_grid_size_y", mg);
    mg.resize(std::max(static_cast<int>(mg.size()),1),128);

    // Modify the default or any user input, making sure that the max grid size
    // (of the coarsest level) is small enough so that there will be at least
    // as many blocks as there are processors.
    while (n_cell[1] < nprocs*mg[0]) {
        mg[0] /= 2;
    }
    pp_amr.addarr("max_grid_size_y", mg);
#endif
}


void ReadBCParams ()
{

    amrex::Vector<int> geom_periodicity(AMREX_SPACEDIM,0);
    ParmParse pp_geometry("geometry");
    const ParmParse pp_warpx("warpx");
    const ParmParse pp_algo("algo");
    auto electromagnetic_solver_id = ElectromagneticSolverAlgo::Default;
    pp_algo.query_enum_sloppy("maxwell_solver", electromagnetic_solver_id, "-_");
    auto poisson_solver_id = PoissonSolverAlgo::Default;
    pp_warpx.query_enum_sloppy("poisson_solver", poisson_solver_id, "-_");

    if (pp_geometry.queryarr("is_periodic", geom_periodicity))
    {
        std::string const warnMsg =
            "geometry.is_periodic is only used internally. Please use `boundary.field_lo`,"
            " `boundary.field_hi` to specifiy field boundary conditions and"
            " 'boundary.particle_lo', 'boundary.particle_hi'  to specify particle"
            " boundary conditions.";
        ablastr::warn_manager::WMRecordWarning("Input", warnMsg);
    }

    // particle boundary may not be explicitly specified for some applications
    bool particle_boundary_specified = false;
    const ParmParse pp_boundary("boundary");
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        // Get field boundary type
        pp_boundary.query_enum_sloppy("field_lo",
                                      WarpX::field_boundary_lo[idim], "-_", idim);
        pp_boundary.query_enum_sloppy("field_hi",
                                      WarpX::field_boundary_hi[idim], "-_", idim);
        // Get particle boundary type
        if (pp_boundary.query_enum_sloppy("particle_lo",
                                          WarpX::particle_boundary_lo[idim], "-_", idim)) {
            particle_boundary_specified = true;
        }
        if (pp_boundary.query_enum_sloppy("particle_hi",
                                          WarpX::particle_boundary_hi[idim], "-_", idim)) {
            particle_boundary_specified = true;
        }

        if (WarpX::field_boundary_lo[idim] == FieldBoundaryType::Periodic ||
            WarpX::field_boundary_hi[idim] == FieldBoundaryType::Periodic ||
            WarpX::particle_boundary_lo[idim] == ParticleBoundaryType::Periodic ||
            WarpX::particle_boundary_hi[idim] == ParticleBoundaryType::Periodic ) {
            geom_periodicity[idim] = 1;
            // to ensure both lo and hi are set to periodic consistently for both field and particles.
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                (WarpX::field_boundary_lo[idim]  == FieldBoundaryType::Periodic) &&
                (WarpX::field_boundary_hi[idim]  == FieldBoundaryType::Periodic),
            "field boundary must be consistenly periodic in both lo and hi");
            if (particle_boundary_specified) {
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    (WarpX::particle_boundary_lo[idim] == ParticleBoundaryType::Periodic) &&
                    (WarpX::particle_boundary_hi[idim] == ParticleBoundaryType::Periodic),
               "field and particle boundary must be periodic in both lo and hi");
            } else {
                // set particle boundary to periodic
                WarpX::particle_boundary_lo[idim] = ParticleBoundaryType::Periodic;
                WarpX::particle_boundary_hi[idim] = ParticleBoundaryType::Periodic;
            }
        }

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            (electromagnetic_solver_id != ElectromagneticSolverAlgo::PSATD) ||
            (
                WarpX::field_boundary_lo[idim] != FieldBoundaryType::PEC &&
                WarpX::field_boundary_hi[idim] != FieldBoundaryType::PEC
            ),
            "PEC boundary not implemented for PSATD, yet!"
        );

        if(WarpX::field_boundary_lo[idim] == FieldBoundaryType::Open &&
           WarpX::field_boundary_hi[idim] == FieldBoundaryType::Open){
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                poisson_solver_id == PoissonSolverAlgo::IntegratedGreenFunction,
                "Field open boundary conditions are only implemented for the FFT-based Poisson solver"
            );
        }
    }

    // Appending periodicity information to input so that it can be used by amrex
    // to set parameters necessary to define geometry and perform communication
    // such as FillBoundary. The periodicity is 1 if user-define boundary condition is
    // periodic else it is set to 0.
    pp_geometry.addarr("is_periodic", geom_periodicity);
}


namespace WarpXUtilLoadBalance
{
    bool doCosts (const amrex::LayoutData<amrex::Real>* cost, const amrex::BoxArray& ba,
                  const amrex::DistributionMapping& dm)
    {
        const bool consistent = cost && (dm == cost->DistributionMap()) &&
            (ba.CellEqual(cost->boxArray())) &&
            (WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers);
        return consistent;
    }
}
