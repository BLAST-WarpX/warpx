/* Copyright 2021 David Grote
 *
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "PML_RZ.H"

#include "BoundaryConditions/PML_RZ.H"
#include "Fields.H"
#ifdef WARPX_USE_FFT
#   include "FieldSolver/SpectralSolver/SpectralFieldDataRZ.H"
#endif
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <ablastr/utils/Communication.H>

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_FabArrayBase.H>
#include <AMReX_Geometry.H>
#include <AMReX_IndexType.H>
#include <AMReX_MFIter.H>
#include <AMReX_RealVect.H>
#include <AMReX_VisMF.H>

#include <cmath>
#include <memory>

using namespace amrex::literals;
using warpx::fields::FieldType;
using ablastr::fields::Direction;

PML_RZ::PML_RZ (int lev, amrex::BoxArray const& grid_ba, amrex::DistributionMapping const& grid_dm,
                amrex::Geometry const* geom, int ncell, int do_pml_in_domain)
    : m_ncell(ncell),
      m_do_pml_in_domain(do_pml_in_domain),
      m_geom(geom)
{
    auto & warpx = WarpX::GetInstance();

    bool const remake = false;
    bool const redistribute_on_remake = false;

    amrex::MultiFab const& Er_fp = *warpx.m_fields.get(FieldType::Efield_fp, Direction{0}, lev);
    amrex::MultiFab const& Et_fp = *warpx.m_fields.get(FieldType::Efield_fp, Direction{1}, lev);
    amrex::BoxArray const ba_Er = amrex::convert(grid_ba, Er_fp.ixType().toIntVect());
    amrex::BoxArray const ba_Et = amrex::convert(grid_ba, Et_fp.ixType().toIntVect());
    warpx.m_fields.alloc_init(FieldType::pml_E_fp, Direction{0}, lev, ba_Er, grid_dm, Er_fp.nComp(), Er_fp.nGrowVect(), 0.0_rt,
                              remake, redistribute_on_remake);
    warpx.m_fields.alloc_init(FieldType::pml_E_fp, Direction{1}, lev, ba_Et, grid_dm, Et_fp.nComp(), Et_fp.nGrowVect(), 0.0_rt,
                              remake, redistribute_on_remake);

    amrex::MultiFab const& Br_fp = *warpx.m_fields.get(FieldType::Bfield_fp,Direction{0},lev);
    amrex::MultiFab const& Bt_fp = *warpx.m_fields.get(FieldType::Bfield_fp,Direction{1},lev);
    amrex::BoxArray const ba_Br = amrex::convert(grid_ba, Br_fp.ixType().toIntVect());
    amrex::BoxArray const ba_Bt = amrex::convert(grid_ba, Bt_fp.ixType().toIntVect());
    warpx.m_fields.alloc_init(FieldType::pml_B_fp, Direction{0}, lev, ba_Br, grid_dm, Br_fp.nComp(), Br_fp.nGrowVect(), 0.0_rt,
                              remake, redistribute_on_remake);
    warpx.m_fields.alloc_init(FieldType::pml_B_fp, Direction{1}, lev, ba_Bt, grid_dm, Bt_fp.nComp(), Bt_fp.nGrowVect(), 0.0_rt,
                              remake, redistribute_on_remake);

}

void
PML_RZ::ApplyDamping (amrex::MultiFab* Et_fp, amrex::MultiFab* Ez_fp,
                      amrex::MultiFab* Bt_fp, amrex::MultiFab* Bz_fp,
                      amrex::Real dt, ablastr::fields::MultiFabRegister& fields)
{

    amrex::Real const dr = m_geom->CellSize(0);
    amrex::Real const cdt_over_dr = PhysConst::c*dt/dr;

    amrex::MultiFab* pml_Et = fields.get(FieldType::pml_E_fp, Direction{1}, 0);
    amrex::MultiFab* pml_Bt = fields.get(FieldType::pml_B_fp, Direction{1}, 0);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( amrex::MFIter mfi(*Et_fp, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi )
    {
        amrex::Array4<amrex::Real> const& Et_arr = Et_fp->array(mfi);
        amrex::Array4<amrex::Real> const& Ez_arr = Ez_fp->array(mfi);
        amrex::Array4<amrex::Real> const& Bt_arr = Bt_fp->array(mfi);
        amrex::Array4<amrex::Real> const& Bz_arr = Bz_fp->array(mfi);

        amrex::Array4<amrex::Real> const& pml_Et_arr = pml_Et->array(mfi);
        amrex::Array4<amrex::Real> const& pml_Bt_arr = pml_Bt->array(mfi);

        // Get the tileboxes from Efield and Bfield so that they include the guard cells
        // They are all the same, cell centered
        amrex::Box tilebox = amrex::convert((*Et_fp)[mfi].box(), Et_fp->ixType().toIntVect());

        // Box for the whole simulation domain
        amrex::Box const& domain = m_geom->Domain();
        int const nr_domain = domain.bigEnd(0);

        // Set tilebox to only include the upper radial cells
        int const nr_damp = m_ncell;
        int const nr_damp_min = (m_do_pml_in_domain)?(nr_domain - nr_damp):(nr_domain);
        tilebox.setSmall(0, nr_damp_min + 1);

        amrex::ParallelFor( tilebox, Et_fp->nComp(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int icomp)
            {
                auto const rr = static_cast<amrex::Real>(i - nr_damp_min);
                amrex::Real const wr = rr/nr_damp;
                amrex::Real const damp_factor = std::exp( -4._rt * cdt_over_dr * wr*wr );

                // Substract the theta PML fields from the regular theta fields
                Et_arr(i,j,k,icomp) -= pml_Et_arr(i,j,k,icomp);
                Bt_arr(i,j,k,icomp) -= pml_Bt_arr(i,j,k,icomp);
                // Damp the theta PML fields
                pml_Et_arr(i,j,k,icomp) *= damp_factor;
                pml_Bt_arr(i,j,k,icomp) *= damp_factor;
                // Add the theta PML fields back to the regular theta fields
                Et_arr(i,j,k,icomp) += pml_Et_arr(i,j,k,icomp);
                Bt_arr(i,j,k,icomp) += pml_Bt_arr(i,j,k,icomp);
                // Damp the z fields
                Bz_arr(i,j,k,icomp) *= damp_factor;
                Ez_arr(i,j,k,icomp) *= damp_factor;
            });
    }
}

void
PML_RZ::FillBoundaryE (ablastr::fields::MultiFabRegister& fields, PatchType patch_type, std::optional<bool> nodal_sync)
{
    amrex::MultiFab * pml_Er = fields.get(FieldType::pml_E_fp, Direction{0}, 0);
    amrex::MultiFab * pml_Et = fields.get(FieldType::pml_E_fp, Direction{1}, 0);

    if (patch_type == PatchType::fine && pml_Er->nGrowVect().max() > 0)
    {
        amrex::Periodicity const& period = m_geom->periodicity();
        const amrex::Vector<amrex::MultiFab*> mf = {pml_Er, pml_Et};
        ablastr::utils::communication::FillBoundary(mf, WarpX::do_single_precision_comms, period, nodal_sync);
    }
}

void
PML_RZ::FillBoundaryB (ablastr::fields::MultiFabRegister& fields, PatchType patch_type, std::optional<bool> nodal_sync)
{
    if (patch_type == PatchType::fine)
    {
        amrex::MultiFab * pml_Br = fields.get(FieldType::pml_B_fp, Direction{0}, 0);
        amrex::MultiFab * pml_Bt = fields.get(FieldType::pml_B_fp, Direction{1}, 0);

        amrex::Periodicity const& period = m_geom->periodicity();
        const amrex::Vector<amrex::MultiFab*> mf = {pml_Br, pml_Bt};
        ablastr::utils::communication::FillBoundary(mf, WarpX::do_single_precision_comms, period, nodal_sync);
    }
}

void
PML_RZ::CheckPoint (ablastr::fields::MultiFabRegister& fields, std::string const& dir) const
{
    if (fields.has(FieldType::pml_E_fp, Direction{0}, 0)) {
        amrex::VisMF::AsyncWrite(*fields.get(FieldType::pml_E_fp, Direction{0}, 0), dir+"_Er_fp");
        amrex::VisMF::AsyncWrite(*fields.get(FieldType::pml_E_fp, Direction{1}, 0), dir+"_Et_fp");
        amrex::VisMF::AsyncWrite(*fields.get(FieldType::pml_B_fp, Direction{0}, 0), dir+"_Br_fp");
        amrex::VisMF::AsyncWrite(*fields.get(FieldType::pml_B_fp, Direction{1}, 0), dir+"_Bt_fp");
    }
}

void
PML_RZ::Restart (ablastr::fields::MultiFabRegister& fields, std::string const& dir)
{
    if (fields.has(FieldType::pml_E_fp, Direction{0}, 0)) {
        amrex::VisMF::Read(*fields.get(FieldType::pml_E_fp, Direction{0}, 0), dir+"_Er_fp");
        amrex::VisMF::Read(*fields.get(FieldType::pml_E_fp, Direction{1}, 0), dir+"_Et_fp");
        amrex::VisMF::Read(*fields.get(FieldType::pml_B_fp, Direction{0}, 0), dir+"_Br_fp");
        amrex::VisMF::Read(*fields.get(FieldType::pml_B_fp, Direction{1}, 0), dir+"_Bt_fp");
    }
}

#ifdef WARPX_USE_FFT
void
PML_RZ::PushPSATD (int lev)
{
    // Update the fields on the fine and coarse patch
    WarpX& warpx = WarpX::GetInstance();
    SpectralSolverRZ& solver = warpx.get_spectral_solver_fp(lev);
    PushPMLPSATDSinglePatchRZ(lev, solver, warpx.m_fields);
}

void
PML_RZ::PushPMLPSATDSinglePatchRZ (
    int lev,
    SpectralSolverRZ& solver,
    ablastr::fields::MultiFabRegister& fields)
{
    SpectralFieldIndex const& Idx = solver.m_spectral_index;
    amrex::MultiFab * pml_Er = fields.get(FieldType::pml_E_fp, Direction{0}, 0);
    amrex::MultiFab * pml_Et = fields.get(FieldType::pml_E_fp, Direction{1}, 0);
    amrex::MultiFab * pml_Br = fields.get(FieldType::pml_B_fp, Direction{0}, 0);
    amrex::MultiFab * pml_Bt = fields.get(FieldType::pml_B_fp, Direction{1}, 0);

    // Perform forward Fourier transforms
    solver.ForwardTransform(lev, *pml_Er, Idx.Er_pml, *pml_Et, Idx.Et_pml);
    solver.ForwardTransform(lev, *pml_Br, Idx.Br_pml, *pml_Bt, Idx.Bt_pml);

    // Advance fields in spectral space
    bool const doing_pml = true;
    solver.pushSpectralFields(doing_pml);

    // Perform backward Fourier transforms
    solver.BackwardTransform(lev, *pml_Er, Idx.Er_pml, *pml_Et, Idx.Et_pml);
    solver.BackwardTransform(lev, *pml_Br, Idx.Br_pml, *pml_Bt, Idx.Bt_pml);
}
#endif
