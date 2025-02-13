/* Copyright 2019-2020 Andrew Myers, Luca Fedeli, Maxence Thevenet
 * Revathi Jambunathan
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_UTILS_H_
#define WARPX_UTILS_H_

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_LayoutData.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Parser.H>
#include <AMReX_REAL.H>
#include <AMReX_Utility.H>
#include <AMReX_Vector.H>

#include <AMReX_BaseFwd.H>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

void ReadBoostedFrameParameters(amrex::Real& gamma_boost, amrex::Real& beta_boost,
                                amrex::Vector<int>& boost_direction);

void ReadMovingWindowParameters(
    int& do_moving_window, int& start_moving_window_step, int& end_moving_window_step,
    int& moving_window_dir, amrex::Real& moving_window_v);

void ConvertLabParamsToBoost();

/**
 * Reads the user-defined field and particle boundary condition parameters
 */
void ReadBCParams ();

/** Check the warpx.dims matches the binary name & set up RZ gridding
 *
 * Ensures that the blocks are setup correctly for the RZ spectral solver
 * When using the RZ spectral solver, the Hankel transform cannot be
 * divided among multiple blocks. Each block must extend over the
 * entire radial extent.
 * The grid can be divided up along z, but the number of blocks
 * must be >= the number of processors.
 */
void CheckGriddingForRZSpectral ();

/** Function that sets the value of MultiFab MF to zero.
 *
 * \param[in] mf Pointer to the MultiFab
 * \param[in] lev The mesh refinement level
 * \param[in] zmin The minimum z of the range to be mullified
 * \param[in] zmin The maximum z of the range to be mullified
 */
void NullifyMFinstance (
    amrex::MultiFab *mf,
    int lev,
    amrex::Real zmin,
    amrex::Real zmax
);

/** Function that sets the value of MultiFab MF to zero.
 *
 * \param[in] multifab_map Multifab registry
 * \param[in] nf_name Name of Multifab to modify
 * \param[in] lev The mesh refinement level
 * \param[in] zmin The minimum z of the range to be mullified
 * \param[in] zmin The maximum z of the range to be mullified
 */
void NullifyMF (
    ablastr::fields::MultiFabRegister& multifab_map,
    std::string const& mf_name,
    int lev,
    amrex::Real zmin,
    amrex::Real zmax
);

/** Function that sets the value of MultiFab MF to zero.
 *
 * \param[in] multifab_map Multifab registry
 * \param[in] nf_name Name of Multifab to modify
 * \param[in] dir Direction, for Multifabs that are components of vectors
 * \param[in] lev The mesh refinement level
 * \param[in] zmin The minimum z of the range to be mullified
 * \param[in] zmin The maximum z of the range to be mullified
 */
void NullifyMF (
    ablastr::fields::MultiFabRegister& multifab_map,
    std::string const& mf_name,
    ablastr::fields::Direction dir,
    int lev,
    amrex::Real zmin,
    amrex::Real zmax
);

namespace WarpXUtilIO{
/**
 * A helper function to write binary data on disk.
 * @param[in] filename where to write
 * @param[in] data Vector containing binary data to write on disk
 * return true if it succeeds, false otherwise
 */
bool WriteBinaryDataOnFile(const std::string& filename, const amrex::Vector<char>& data);

}

namespace WarpXUtilLoadBalance
{
    /** \brief We only want to update the cost data if the grids we are working on
     *  are the main grids, i.e. not the PML grids. This function returns whether
     *   this is the case or not.
     * @param[in] cost pointer to the cost data
     * @param[in] ba the grids to check
     * @param[in] dm the dmap to check
     * @return consistent whether the grids are consistent or not.
     */
    bool doCosts (const amrex::LayoutData<amrex::Real>* cost, const amrex::BoxArray& ba,
                  const amrex::DistributionMapping& dm);
}

#endif //WARPX_UTILS_H_
