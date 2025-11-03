/* Copyright 2021 David Grote
 *
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PML_RZ_H_
#define WARPX_PML_RZ_H_

#include "PML_RZ_fwd.H"

#include "Utils/WarpXAlgorithmSelection.H"

#ifdef WARPX_USE_FFT
#   include "FieldSolver/SpectralSolver/SpectralSolverRZ.H"
#endif

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_MultiFab.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Config.H>
#include <AMReX_REAL.H>

#include <AMReX_BaseFwd.H>

#include <array>
#include <optional>
#include <string>

class PML_RZ
{
public:
    PML_RZ (int lev, amrex::BoxArray const& grid_ba, amrex::DistributionMapping const& grid_dm,
            amrex::Geometry const* geom, ablastr::fields::MultiFabRegister& fields,
            int ncell, int do_pml_in_domain);

    void ApplyDamping(amrex::MultiFab* Et_fp, amrex::MultiFab* Ez_fp,
                      amrex::MultiFab* Bt_fp, amrex::MultiFab* Bz_fp,
                      amrex::Real dt, ablastr::fields::MultiFabRegister& fields);

#ifdef WARPX_USE_FFT
    void PushPSATD (int lev, ablastr::fields::MultiFabRegister& fields, SpectralSolverRZ& spec_solver);
#endif

    void FillBoundaryE (ablastr::fields::MultiFabRegister& fields, PatchType patch_type,
        bool do_single_precision_comms, std::optional<bool> nodal_sync=std::nullopt);
    void FillBoundaryB (ablastr::fields::MultiFabRegister& fields, PatchType patch_type,
        bool do_single_precision_comms, std::optional<bool> nodal_sync=std::nullopt);

    void CheckPoint (ablastr::fields::MultiFabRegister& fields, std::string const& dir) const;
    void Restart (ablastr::fields::MultiFabRegister& fields, std::string const& dir);

private:

    int m_ncell;
    int m_do_pml_in_domain;
    const amrex::Geometry* m_geom;
};

#endif
