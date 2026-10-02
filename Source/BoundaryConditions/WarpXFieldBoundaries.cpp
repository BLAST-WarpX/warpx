#include "WarpX.H"
#include "BoundaryConditions/PEC_Insulator.H"
#include "BoundaryConditions/PML.H"
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Utils/TextMsg.H"
#include "WarpX_PEC.H"

#include <AMReX.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <string>

using namespace amrex;
using namespace amrex::literals;
using warpx::fields::FieldType;

namespace
{
    /** Returns true if any field boundary is set to FieldBoundaryType FT, else returns false.*/
    template <FieldBoundaryType FT>
    [[nodiscard]]
    bool isAnyBoundary (const amrex::Array<FieldBoundaryType,AMREX_SPACEDIM>& field_boundary_lo,
        const amrex::Array<FieldBoundaryType,AMREX_SPACEDIM>& field_boundary_hi)
    {
        const auto isFT = [](const auto& b){
            return b == FT;};
        return std::any_of(field_boundary_lo.begin(), field_boundary_lo.end(), isFT) ||
               std::any_of(field_boundary_hi.begin(), field_boundary_hi.end(), isFT);
    }

    /** Returns true if any particle boundary is set to ParticleBoundaryType PT, else returns false.*/
    template <ParticleBoundaryType PT>
    [[nodiscard]]
    bool isAnyBoundary (const amrex::Array<ParticleBoundaryType,AMREX_SPACEDIM>& particle_boundary_lo,
        const amrex::Array<ParticleBoundaryType,AMREX_SPACEDIM>& particle_boundary_hi)
    {
        const auto isPT = [](const auto& b){
            return b == PT;};
        return std::any_of(particle_boundary_lo.begin(), particle_boundary_lo.end(), isPT) ||
               std::any_of(particle_boundary_hi.begin(), particle_boundary_hi.end(), isPT);
    }

}

void WarpX::ApplyEfieldBoundary(const int lev, PatchType patch_type, amrex::Real time)
{
    using ablastr::fields::Direction;

    if (::isAnyBoundary<FieldBoundaryType::PEC>(field_boundary_lo, field_boundary_hi)) {
        if (patch_type == PatchType::fine) {
            PEC::ApplyPECtoEfield(
                    m_fields.get_alldirs(FieldType::Efield_fp, lev),
                    field_boundary_lo, field_boundary_hi, FieldBoundaryType::PEC,
                    get_ng_fieldgather(), Geom(lev),
                    lev, patch_type, ref_ratio);
            if (::isAnyBoundary<FieldBoundaryType::PML>(field_boundary_lo, field_boundary_hi)) {
                // apply pec on split E-fields in PML region
                const bool split_pml_field = true;
                PEC::ApplyPECtoEfield(
                    m_fields.get_alldirs(FieldType::pml_E_fp, lev),
                    field_boundary_lo, field_boundary_hi, FieldBoundaryType::PEC,
                    get_ng_fieldgather(), Geom(lev),
                    lev, patch_type, ref_ratio,
                    split_pml_field);
            }
        } else {
            PEC::ApplyPECtoEfield(
                    m_fields.get_alldirs(FieldType::Efield_cp, lev),
                    field_boundary_lo, field_boundary_hi, FieldBoundaryType::PEC,
                    get_ng_fieldgather(), Geom(lev),
                    lev, patch_type, ref_ratio);
            if (::isAnyBoundary<FieldBoundaryType::PML>(field_boundary_lo, field_boundary_hi)) {
                // apply pec on split E-fields in PML region
                const bool split_pml_field = true;
                PEC::ApplyPECtoEfield(
                    m_fields.get_alldirs(FieldType::pml_E_cp, lev),
                    field_boundary_lo, field_boundary_hi, FieldBoundaryType::PEC,
                    get_ng_fieldgather(), Geom(lev),
                    lev, patch_type, ref_ratio,
                    split_pml_field);
            }
        }
    }

    if (::isAnyBoundary<FieldBoundaryType::PMC>(field_boundary_lo, field_boundary_hi)) {
        if (patch_type == PatchType::fine) {
            PEC::ApplyPECtoBfield(
                    m_fields.get_alldirs(FieldType::Efield_fp, lev),
                    field_boundary_lo, field_boundary_hi, FieldBoundaryType::PMC,
                    get_ng_fieldgather(), Geom(lev),
                    lev, patch_type, ref_ratio);
            if (::isAnyBoundary<FieldBoundaryType::PML>(field_boundary_lo, field_boundary_hi)) {
                // apply pec on split E-fields in PML region
                const bool split_pml_field = true;
                PEC::ApplyPECtoBfield(
                    m_fields.get_alldirs(FieldType::pml_E_fp, lev),
                    field_boundary_lo, field_boundary_hi, FieldBoundaryType::PMC,
                    get_ng_fieldgather(), Geom(lev),
                    lev, patch_type, ref_ratio,
                    split_pml_field);
            }
        } else {
            PEC::ApplyPECtoBfield(
                    m_fields.get_alldirs(FieldType::Efield_cp, lev),
                    field_boundary_lo, field_boundary_hi, FieldBoundaryType::PMC,
                    get_ng_fieldgather(), Geom(lev),
                    lev, patch_type, ref_ratio);
            if (::isAnyBoundary<FieldBoundaryType::PML>(field_boundary_lo, field_boundary_hi)) {
                // apply pec on split E-fields in PML region
                const bool split_pml_field = true;
                PEC::ApplyPECtoBfield(
                    m_fields.get_alldirs(FieldType::pml_E_cp, lev),
                    field_boundary_lo, field_boundary_hi, FieldBoundaryType::PMC,
                    get_ng_fieldgather(), Geom(lev),
                    lev, patch_type, ref_ratio,
                    split_pml_field);
            }
        }
    }

    if (::isAnyBoundary<FieldBoundaryType::PEC_Insulator>(field_boundary_lo, field_boundary_hi)) {
        if (patch_type == PatchType::fine) {
            pec_insulator_boundary->ApplyPEC_InsulatortoEfield(
                    m_fields.get_alldirs(FieldType::Efield_fp, lev),
                    field_boundary_lo, field_boundary_hi,
                    get_ng_fieldgather(), Geom(lev),
                    lev, patch_type, ref_ratio, time);
            if (::isAnyBoundary<FieldBoundaryType::PML>(field_boundary_lo, field_boundary_hi)) {
                // apply on split E-fields in PML region
                const bool split_pml_field = true;
                pec_insulator_boundary->ApplyPEC_InsulatortoEfield(
                    m_fields.get_alldirs(FieldType::pml_E_fp, lev),
                    field_boundary_lo, field_boundary_hi,
                    get_ng_fieldgather(), Geom(lev),
                    lev, patch_type, ref_ratio, time,
                    split_pml_field);
            }
        } else {
            pec_insulator_boundary->ApplyPEC_InsulatortoEfield(
                m_fields.get_alldirs(FieldType::Efield_cp, lev),
                field_boundary_lo, field_boundary_hi,
                get_ng_fieldgather(), Geom(lev),
                lev, patch_type, ref_ratio, time);
            if (::isAnyBoundary<FieldBoundaryType::PML>(field_boundary_lo, field_boundary_hi)) {
                // apply on split E-fields in PML region
                const bool split_pml_field = true;
                pec_insulator_boundary->ApplyPEC_InsulatortoEfield(
                    m_fields.get_alldirs(FieldType::pml_E_cp, lev),
                    field_boundary_lo, field_boundary_hi,
                    get_ng_fieldgather(), Geom(lev),
                    lev, patch_type, ref_ratio, time,
                    split_pml_field);
            }
        }
    }

#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    if (patch_type == PatchType::fine) {
        ApplyFieldBoundaryOnAxis(m_fields.get(FieldType::Efield_fp, Direction{0}, lev),
                                 m_fields.get(FieldType::Efield_fp, Direction{1}, lev),
                                 m_fields.get(FieldType::Efield_fp, Direction{2}, lev), lev);
    } else {
        ApplyFieldBoundaryOnAxis(m_fields.get(FieldType::Efield_cp, Direction{0}, lev),
                                 m_fields.get(FieldType::Efield_cp, Direction{1}, lev),
                                 m_fields.get(FieldType::Efield_cp, Direction{2}, lev), lev);
    }
#endif
}

void WarpX::ApplyBfieldBoundary (const int lev, PatchType patch_type, SubcyclingHalf subcycling_half, amrex::Real time)
{
    using ablastr::fields::Direction;

    if (::isAnyBoundary<FieldBoundaryType::PEC>(field_boundary_lo, field_boundary_hi)) {
        if (patch_type == PatchType::fine) {
            PEC::ApplyPECtoBfield(
                m_fields.get_alldirs(FieldType::Bfield_fp, lev),
                field_boundary_lo, field_boundary_hi, FieldBoundaryType::PEC,
                get_ng_fieldgather(), Geom(lev),
                lev, patch_type, ref_ratio);
        } else {
            PEC::ApplyPECtoBfield(
                m_fields.get_alldirs(FieldType::Bfield_cp, lev),
                field_boundary_lo, field_boundary_hi, FieldBoundaryType::PEC,
                get_ng_fieldgather(), Geom(lev),
                lev, patch_type, ref_ratio);
        }
    }

    if (::isAnyBoundary<FieldBoundaryType::PMC>(field_boundary_lo, field_boundary_hi)) {
        if (patch_type == PatchType::fine) {
            PEC::ApplyPECtoEfield(
                m_fields.get_alldirs(FieldType::Bfield_fp, lev),
                field_boundary_lo, field_boundary_hi, FieldBoundaryType::PMC,
                get_ng_fieldgather(), Geom(lev),
                lev, patch_type, ref_ratio);
        } else {
            PEC::ApplyPECtoEfield(
                m_fields.get_alldirs(FieldType::Bfield_cp, lev),
                field_boundary_lo, field_boundary_hi, FieldBoundaryType::PMC,
                get_ng_fieldgather(), Geom(lev),
                lev, patch_type, ref_ratio);
        }
    }

    if (::isAnyBoundary<FieldBoundaryType::PEC_Insulator>(field_boundary_lo, field_boundary_hi)) {
        if (patch_type == PatchType::fine) {
            pec_insulator_boundary->ApplyPEC_InsulatortoBfield(
                m_fields.get_alldirs(FieldType::Bfield_fp, lev),
                field_boundary_lo, field_boundary_hi,
                get_ng_fieldgather(), Geom(lev),
                lev, patch_type, ref_ratio, time);
        } else {
            pec_insulator_boundary->ApplyPEC_InsulatortoBfield(
                m_fields.get_alldirs(FieldType::Bfield_cp, lev),
                field_boundary_lo, field_boundary_hi,
                get_ng_fieldgather(), Geom(lev),
                lev, patch_type, ref_ratio, time);
        }
    }

    // Silver-Mueller boundaries are only applied on the first half-push of B
    // This is because the formula used for Silver-Mueller assumes that
    // E and B are staggered in time, which is only true after the first half-push
    if (lev == 0) {
        if (subcycling_half == SubcyclingHalf::FirstHalf) {
            if(::isAnyBoundary<FieldBoundaryType::Absorbing_Silver_Mueller>(field_boundary_lo, field_boundary_hi)){
                auto Efield_fp = m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, max_level);
                auto Bfield_fp = m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, max_level);
                m_fdtd_solver_fp[0]->ApplySilverMuellerBoundary(
                Efield_fp[lev], Bfield_fp[lev],
                Geom(lev).Domain(), dt[lev],
                field_boundary_lo, field_boundary_hi);
            }
        }
    }

#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    if (patch_type == PatchType::fine) {
        ApplyFieldBoundaryOnAxis(m_fields.get(FieldType::Bfield_fp,Direction{0},lev),
                                 m_fields.get(FieldType::Bfield_fp,Direction{1},lev),
                                 m_fields.get(FieldType::Bfield_fp,Direction{2},lev), lev);
    } else {
        ApplyFieldBoundaryOnAxis(m_fields.get(FieldType::Bfield_cp,Direction{0},lev),
                                 m_fields.get(FieldType::Bfield_cp,Direction{1},lev),
                                 m_fields.get(FieldType::Bfield_cp,Direction{2},lev), lev);
    }
#endif
}

void WarpX::ApplyRhofieldBoundary (const int lev, MultiFab* rho,
                                   PatchType patch_type)
{
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    // Fold the charge deposited in the guard cells beyond the axis
    // onto the cells above the axis.
    FoldChargeDensityOnAxis(rho, lev);
#endif

    if (::isAnyBoundary<ParticleBoundaryType::Reflecting>(particle_boundary_lo, particle_boundary_hi) ||
        ::isAnyBoundary<ParticleBoundaryType::Thermal>(particle_boundary_lo, particle_boundary_hi) ||
        ::isAnyBoundary<FieldBoundaryType::PEC>(field_boundary_lo, field_boundary_hi) ||
        ::isAnyBoundary<FieldBoundaryType::PEC_Insulator>(field_boundary_lo, field_boundary_hi) ||
        ::isAnyBoundary<FieldBoundaryType::PMC>(field_boundary_lo, field_boundary_hi))
    {
        PEC::ApplyReflectiveBoundarytoRhofield(rho,
            field_boundary_lo, field_boundary_hi,
            particle_boundary_lo, particle_boundary_hi,
            Geom(lev), lev, patch_type, ref_ratio);
    }

    if (::isAnyBoundary<FieldBoundaryType::PEC_Insulator>(field_boundary_lo, field_boundary_hi)) {
        pec_insulator_boundary->ZeroParallelScalarInConductor(rho,
            field_boundary_lo, field_boundary_hi,
            Geom(lev), lev, patch_type, ref_ratio);
    }
}

void WarpX::ApplyJfieldBoundary (const int lev, amrex::MultiFab* Jx,
                                 amrex::MultiFab* Jy, amrex::MultiFab* Jz,
                                 PatchType patch_type)
{
    BL_PROFILE("WarpX::ApplyJfieldBoundary()");

#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    // Fold the current deposited in the guard cells beyond the axis
    // onto the cells above the axis.
    FoldCurrentDensityOnAxis(Jx, Jy, Jz, lev);
#endif

    ApplyJfieldBoundaryOnWalls(lev, Jx, Jy, Jz, patch_type);
}

void WarpX::ApplyJfieldBoundaryOnWalls (const int lev, amrex::MultiFab* Jx,
                                        amrex::MultiFab* Jy, amrex::MultiFab* Jz,
                                        PatchType patch_type)
{
    if (::isAnyBoundary<ParticleBoundaryType::Reflecting>(particle_boundary_lo, particle_boundary_hi) ||
        ::isAnyBoundary<ParticleBoundaryType::Thermal>(particle_boundary_lo, particle_boundary_hi) ||
        ::isAnyBoundary<FieldBoundaryType::PEC>(field_boundary_lo, field_boundary_hi) ||
        ::isAnyBoundary<FieldBoundaryType::PEC_Insulator>(field_boundary_lo, field_boundary_hi) ||
        ::isAnyBoundary<FieldBoundaryType::PMC>(field_boundary_lo, field_boundary_hi))
    {
        PEC::ApplyReflectiveBoundarytoJfield(Jx, Jy, Jz,
            field_boundary_lo, field_boundary_hi,
            particle_boundary_lo, particle_boundary_hi,
            Geom(lev), lev, patch_type, ref_ratio);
    }

    if (::isAnyBoundary<FieldBoundaryType::PEC_Insulator>(field_boundary_lo, field_boundary_hi)) {
        pec_insulator_boundary->ZeroParallelFieldInConductor({Jx, Jy, Jz},
            field_boundary_lo, field_boundary_hi,
            get_ng_fieldgather(), Geom(lev),
            lev, patch_type, ref_ratio);
    }
}

#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
// Applies the boundary conditions that are specific to the axis when in cylindrical or spherical
void
WarpX::ApplyFieldBoundaryOnAxis (amrex::MultiFab* Er, amrex::MultiFab* Et, amrex::MultiFab* Ez, int lev) const
{
    const amrex::IntVect ngE = get_ng_fieldgather();

    constexpr int NODE = amrex::IndexType::NODE;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( amrex::MFIter mfi(*Er, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi )
    {

        amrex::Box const & tilebox = mfi.tilebox();

        // Lower corner of tile box physical domain
        // Note that this is done before the tilebox.grow so that
        // these do not include the guard cells.
        const amrex::XDim3 xyzmin = LowerCorner(tilebox, lev, 0._rt);
        const amrex::Real rmin = xyzmin.x;

        // Skip blocks that don't touch the axis
        if (rmin > 0._rt) { continue; }

        amrex::Array4<amrex::Real> const& Er_arr = Er->array(mfi);
        amrex::Array4<amrex::Real> const& Et_arr = Et->array(mfi);
        amrex::Array4<amrex::Real> const& Ez_arr = Ez->array(mfi);

        amrex::Box tbr = amrex::convert( tilebox, Er->ixType().toIntVect() );
        amrex::Box tbt = amrex::convert( tilebox, Et->ixType().toIntVect() );
        amrex::Box tbz = amrex::convert( tilebox, Ez->ixType().toIntVect() );

        // For ishift, 1 means cell centered, 0 means node centered
        int const ishift_r = (tbr.type(0) != NODE);
        int const ishift_t = (tbt.type(0) != NODE);
        int const ishift_z = (tbz.type(0) != NODE);

        // Set tileboxes to only include the axis guard cells
        // (including the corners in z).
        tbr.setRange(0, -ngE[0], ngE[0]);
        tbt.setRange(0, -ngE[0], ngE[0]);
        tbz.setRange(0, -ngE[0], ngE[0]);
#ifdef WARPX_DIM_RZ
        tbr.grow(1, ngE[1]);
        tbt.grow(1, ngE[1]);
        tbz.grow(1, ngE[1]);
#endif

        const int nmodes = n_rz_azimuthal_modes;

        amrex::ParallelFor(tbr, tbt, tbz,
        [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/)
        {
            Er_arr(i,j,0,0) = -Er_arr(-i-ishift_r,j,0,0);

            for (int imode=1 ; imode < nmodes ; imode++) {
                Er_arr(i,j,0,2*imode-1) = std::pow(-1._rt, imode+1._rt)*Er_arr(-i-ishift_r,j,0,2*imode-1);
                Er_arr(i,j,0,2*imode) = std::pow(-1._rt, imode+1._rt)*Er_arr(-i-ishift_r,j,0,2*imode);
            }
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/)
        {
            Et_arr(i,j,0,0) = -Et_arr(-i-ishift_t,j,0,0);

            for (int imode=1 ; imode < nmodes ; imode++) {
                Et_arr(i,j,0,2*imode-1) = std::pow(-1._rt, imode+1._rt)*Et_arr(-i-ishift_t,j,0,2*imode-1);
                Et_arr(i,j,0,2*imode) = std::pow(-1._rt, imode+1._rt)*Et_arr(-i-ishift_t,j,0,2*imode);
            }
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/)
        {
#if defined(WARPX_DIM_RSPHERE)
            // Ephi is anti-symmetric
            Ez_arr(i,j,0,0) = -Ez_arr(-i-ishift_z,j,0,0);

#elif defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
            Ez_arr(i,j,0,0) = Ez_arr(-i-ishift_z,j,0,0);

            for (int imode=1 ; imode < nmodes ; imode++) {
                Ez_arr(i,j,0,2*imode-1) = -std::pow(-1._rt, imode+1._rt)*Ez_arr(-i-ishift_z,j,0,2*imode-1);
                Ez_arr(i,j,0,2*imode) = -std::pow(-1._rt, imode+1._rt)*Ez_arr(-i-ishift_z,j,0,2*imode);
            }
#endif

        });
    }
}

void
WarpX::FoldChargeDensityOnAxis (amrex::MultiFab* rho, const int lev) const
{
    // Nothing to do if the domain does not touch the axis
    if (Geom(lev).ProbLo(0) != 0._rt) { return; }

    const amrex::IntVect ng = rho->nGrowVect();

    // The MultiFab may hold several copies of the charge density side by
    // side (e.g. rho_old and rho_new for PSATD, or a single copy for the
    // electrostatic and hybrid solvers and for diagnostics). Each copy
    // uses the same component layout as J: component 0 is the mode 0,
    // and components 2*m-1 and 2*m are the real and imaginary parts of
    // the mode m.
    const int ncomp = rho->nComp();
    const int ncomps_per_copy = WarpX::ncomps;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(ncomp % ncomps_per_copy == 0,
        "FoldChargeDensityOnAxis: the number of components of rho ("
        + std::to_string(ncomp) + ") must be a multiple of the number of mode components ("
        + std::to_string(ncomps_per_copy) + ")");
    const int ncopies = ncomp / ncomps_per_copy;
#if defined(WARPX_DIM_RZ)
    const int nmodes = n_rz_azimuthal_modes;
#endif

    // Index of the first cell/node along r (the same for the fine and coarse patches)
    const int domain_lo = Geom(lev).Domain().smallEnd(0);

    // For ishift, 1 means cell centered, 0 means node centered:
    // the mirror of the cell i across the axis is the cell -ishift-i.
    const int ishift = (rho->ixType().nodeCentered(0) ? 0 : 1);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    // The false flag here is to ensure that this loop does not use tiling.
    // The boxes are grown to include the guard cells in the transverse direction;
    // with tiling, neighboring tiles would fold the overlapping region multiple times.
    for (amrex::MFIter mfi(*rho, false); mfi.isValid(); ++mfi)
    {
        // Skip boxes that don't touch the axis
        const amrex::Box& validbox = mfi.validbox();
        if (validbox.smallEnd(0) != domain_lo) { continue; }

        amrex::Array4<amrex::Real> const& rho_arr = rho->array(mfi);

        amrex::Box tb = amrex::convert(validbox, rho->ixType().toIntVect());
#if defined(WARPX_DIM_RZ)
        // Include the guard cells in the transverse direction (corners)
        tb.grow(1, ng[1]);
#endif

        // Step 1: fold the guard cells at negative radius onto their mirror
        // cells above the axis (the node on the axis, if any, is its own mirror
        // and is left untouched).
        amrex::Box tb_fold = tb;
        tb_fold.setRange(0, 1-ishift, ng[0]);
        amrex::ParallelFor(tb_fold,
        [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/)
        {
            for (int icopy = 0; icopy < ncopies; icopy++) {
                const int ic0 = icopy*ncomps_per_copy;
                // The mode 0 is symmetric across the axis
                rho_arr(i,j,0,ic0) += rho_arr(-ishift-i,j,0,ic0);
#if defined(WARPX_DIM_RZ)
                for (int imode=1 ; imode < nmodes ; imode++) {
                    // The mode m has parity (-1)^m across the axis
                    rho_arr(i,j,0,ic0+2*imode-1) += static_cast<amrex::Real>(std::pow(-1, imode)*rho_arr(-ishift-i,j,0,ic0+2*imode-1));
                    rho_arr(i,j,0,ic0+2*imode) += static_cast<amrex::Real>(std::pow(-1, imode)*rho_arr(-ishift-i,j,0,ic0+2*imode));
                }
#endif
            }
        });

        // Step 2: fill the guard cells at negative radius with the image of
        // the cells above the axis.
        amrex::Box tb_fill = tb;
        tb_fill.setRange(0, -ng[0], ng[0]);
        amrex::ParallelFor(tb_fill,
        [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/)
        {
            for (int icopy = 0; icopy < ncopies; icopy++) {
                const int ic0 = icopy*ncomps_per_copy;
                rho_arr(i,j,0,ic0) = rho_arr(-ishift-i,j,0,ic0);
#if defined(WARPX_DIM_RZ)
                for (int imode=1 ; imode < nmodes ; imode++) {
                    rho_arr(i,j,0,ic0+2*imode-1) = static_cast<amrex::Real>(std::pow(-1, imode)*rho_arr(-ishift-i,j,0,ic0+2*imode-1));
                    rho_arr(i,j,0,ic0+2*imode) = static_cast<amrex::Real>(std::pow(-1, imode)*rho_arr(-ishift-i,j,0,ic0+2*imode));
                }
#endif
            }
        });
    }
}

void
WarpX::FoldCurrentDensityOnAxis (amrex::MultiFab* Jr, amrex::MultiFab* Jt, amrex::MultiFab* Jz, const int lev) const
{
    // Nothing to do if the domain does not touch the axis
    if (Geom(lev).ProbLo(0) != 0._rt) { return; }

    const amrex::IntVect ng = Jr->nGrowVect();
#if defined(WARPX_DIM_RZ)
    const int nmodes = n_rz_azimuthal_modes;
#endif

    // Index of the first cell/node along r (the same for the fine and coarse patches)
    const int domain_lo = Geom(lev).Domain().smallEnd(0);

    // For ishift, 1 means cell centered, 0 means node centered:
    // the mirror of the cell i across the axis is the cell -ishift-i.
    const int ishift_r = (Jr->ixType().nodeCentered(0) ? 0 : 1);
    const int ishift_t = (Jt->ixType().nodeCentered(0) ? 0 : 1);
    const int ishift_z = (Jz->ixType().nodeCentered(0) ? 0 : 1);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    // The false flag here is to ensure that this loop does not use tiling.
    // The boxes are grown to include the guard cells in the transverse direction;
    // with tiling, neighboring tiles would fold the overlapping region multiple times.
    for (amrex::MFIter mfi(*Jr, false); mfi.isValid(); ++mfi)
    {
        // Skip boxes that don't touch the axis
        const amrex::Box& validbox = mfi.validbox();
        if (validbox.smallEnd(0) != domain_lo) { continue; }

        amrex::Array4<amrex::Real> const& Jr_arr = Jr->array(mfi);
        amrex::Array4<amrex::Real> const& Jt_arr = Jt->array(mfi);
        amrex::Array4<amrex::Real> const& Jz_arr = Jz->array(mfi);

        amrex::Box tbr = amrex::convert(validbox, Jr->ixType().toIntVect());
        amrex::Box tbt = amrex::convert(validbox, Jt->ixType().toIntVect());
        amrex::Box tbz = amrex::convert(validbox, Jz->ixType().toIntVect());
#if defined(WARPX_DIM_RZ)
        // Include the guard cells in the transverse direction (corners)
        tbr.grow(1, ng[1]);
        tbt.grow(1, ng[1]);
        tbz.grow(1, ng[1]);
#endif

        // Step 1: fold the guard cells at negative radius onto their mirror
        // cells above the axis (the node on the axis, if any, is its own mirror
        // and is left untouched). The parities are those of
        // ApplyFieldBoundaryOnAxis: Jr and Jt are anti-symmetric for the
        // mode 0, Jz is symmetric (anti-symmetric in RSPHERE), and the
        // parity alternates with the azimuthal mode.
        amrex::Box tbr_fold = tbr;
        amrex::Box tbt_fold = tbt;
        amrex::Box tbz_fold = tbz;
        tbr_fold.setRange(0, 1-ishift_r, ng[0]);
        tbt_fold.setRange(0, 1-ishift_t, ng[0]);
        tbz_fold.setRange(0, 1-ishift_z, ng[0]);
        amrex::ParallelFor(tbr_fold, tbt_fold, tbz_fold,
        [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/)
        {
            Jr_arr(i,j,0,0) -= Jr_arr(-ishift_r-i,j,0,0);
#if defined(WARPX_DIM_RZ)
            for (int imode=1 ; imode < nmodes ; imode++) {
                Jr_arr(i,j,0,2*imode-1) += static_cast<amrex::Real>(std::pow(-1, imode+1)*Jr_arr(-ishift_r-i,j,0,2*imode-1));
                Jr_arr(i,j,0,2*imode) += static_cast<amrex::Real>(std::pow(-1, imode+1)*Jr_arr(-ishift_r-i,j,0,2*imode));
            }
#endif
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/)
        {
            Jt_arr(i,j,0,0) -= Jt_arr(-ishift_t-i,j,0,0);
#if defined(WARPX_DIM_RZ)
            for (int imode=1 ; imode < nmodes ; imode++) {
                Jt_arr(i,j,0,2*imode-1) += static_cast<amrex::Real>(std::pow(-1, imode+1)*Jt_arr(-ishift_t-i,j,0,2*imode-1));
                Jt_arr(i,j,0,2*imode) += static_cast<amrex::Real>(std::pow(-1, imode+1)*Jt_arr(-ishift_t-i,j,0,2*imode));
            }
#endif
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/)
        {
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
            Jz_arr(i,j,0,0) += Jz_arr(-ishift_z-i,j,0,0);
#elif defined(WARPX_DIM_RSPHERE)
            Jz_arr(i,j,0,0) -= Jz_arr(-ishift_z-i,j,0,0);
#endif
#if defined(WARPX_DIM_RZ)
            for (int imode=1 ; imode < nmodes ; imode++) {
                Jz_arr(i,j,0,2*imode-1) -= static_cast<amrex::Real>(std::pow(-1, imode+1)*Jz_arr(-ishift_z-i,j,0,2*imode-1));
                Jz_arr(i,j,0,2*imode) -= static_cast<amrex::Real>(std::pow(-1, imode+1)*Jz_arr(-ishift_z-i,j,0,2*imode));
            }
#endif
        });

        // Step 2: fill the guard cells at negative radius with the image of
        // the cells above the axis.
        amrex::Box tbr_fill = tbr;
        amrex::Box tbt_fill = tbt;
        amrex::Box tbz_fill = tbz;
        tbr_fill.setRange(0, -ng[0], ng[0]);
        tbt_fill.setRange(0, -ng[0], ng[0]);
        tbz_fill.setRange(0, -ng[0], ng[0]);
        amrex::ParallelFor(tbr_fill, tbt_fill, tbz_fill,
        [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/)
        {
            Jr_arr(i,j,0,0) = -Jr_arr(-ishift_r-i,j,0,0);
#if defined(WARPX_DIM_RZ)
            for (int imode=1 ; imode < nmodes ; imode++) {
                Jr_arr(i,j,0,2*imode-1) = static_cast<amrex::Real>(std::pow(-1, imode+1)*Jr_arr(-ishift_r-i,j,0,2*imode-1));
                Jr_arr(i,j,0,2*imode) = static_cast<amrex::Real>(std::pow(-1, imode+1)*Jr_arr(-ishift_r-i,j,0,2*imode));
            }
#endif
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/)
        {
            Jt_arr(i,j,0,0) = -Jt_arr(-ishift_t-i,j,0,0);
#if defined(WARPX_DIM_RZ)
            for (int imode=1 ; imode < nmodes ; imode++) {
                Jt_arr(i,j,0,2*imode-1) = static_cast<amrex::Real>(std::pow(-1, imode+1)*Jt_arr(-ishift_t-i,j,0,2*imode-1));
                Jt_arr(i,j,0,2*imode) = static_cast<amrex::Real>(std::pow(-1, imode+1)*Jt_arr(-ishift_t-i,j,0,2*imode));
            }
#endif
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/)
        {
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
            Jz_arr(i,j,0,0) = Jz_arr(-ishift_z-i,j,0,0);
#elif defined(WARPX_DIM_RSPHERE)
            Jz_arr(i,j,0,0) = -Jz_arr(-ishift_z-i,j,0,0);
#endif
#if defined(WARPX_DIM_RZ)
            for (int imode=1 ; imode < nmodes ; imode++) {
                Jz_arr(i,j,0,2*imode-1) = -static_cast<amrex::Real>(std::pow(-1, imode+1)*Jz_arr(-ishift_z-i,j,0,2*imode-1));
                Jz_arr(i,j,0,2*imode) = -static_cast<amrex::Real>(std::pow(-1, imode+1)*Jz_arr(-ishift_z-i,j,0,2*imode));
            }
#endif
        });
    }
}

void
WarpX::FoldMassMatricesOnAxis (amrex::MultiFab* Sxx, amrex::MultiFab* Syy, amrex::MultiFab* Szz, const int lev) const
{
    // Nothing to do if the domain does not touch the axis
    if (Geom(lev).ProbLo(0) != 0._rt) { return; }

    const amrex::IntVect ng = Sxx->nGrowVect();
    const int ncomp_rr = Sxx->nComp();
    const int ncomp_tt = Syy->nComp();
    const int ncomp_zz = Szz->nComp();

    // Index of the first cell/node along r
    const int domain_lo = Geom(lev).Domain().smallEnd(0);

    // For ishift, 1 means cell centered, 0 means node centered:
    // the mirror of the cell i across the axis is the cell -ishift-i.
    const int ishift_r = (Sxx->ixType().nodeCentered(0) ? 0 : 1);
    const int ishift_t = (Syy->ixType().nodeCentered(0) ? 0 : 1);
    const int ishift_z = (Szz->ixType().nodeCentered(0) ? 0 : 1);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    // The false flag here is to ensure that this loop does not use tiling.
    // The boxes are grown to include the guard cells in the transverse direction;
    // with tiling, neighboring tiles would fold the overlapping region multiple times.
    for (amrex::MFIter mfi(*Sxx, false); mfi.isValid(); ++mfi)
    {
        // Skip boxes that don't touch the axis
        const amrex::Box& validbox = mfi.validbox();
        if (validbox.smallEnd(0) != domain_lo) { continue; }

        amrex::Array4<amrex::Real> const& Srr_arr = Sxx->array(mfi);
        amrex::Array4<amrex::Real> const& Stt_arr = Syy->array(mfi);
        amrex::Array4<amrex::Real> const& Szz_arr = Szz->array(mfi);

        amrex::Box tbr = amrex::convert(validbox, Sxx->ixType().toIntVect());
        amrex::Box tbt = amrex::convert(validbox, Syy->ixType().toIntVect());
        amrex::Box tbz = amrex::convert(validbox, Szz->ixType().toIntVect());
#if defined(WARPX_DIM_RZ)
        // Include the guard cells in the transverse direction (corners)
        tbr.grow(1, ng[1]);
        tbt.grow(1, ng[1]);
        tbz.grow(1, ng[1]);
#endif

        // Step 1: fold the guard cells at negative radius onto their mirror
        // cells above the axis (the node on the axis, if any, is its own mirror
        // and is left untouched). The mass matrices are symmetric across the axis.
        amrex::Box tbr_fold = tbr;
        amrex::Box tbt_fold = tbt;
        amrex::Box tbz_fold = tbz;
        tbr_fold.setRange(0, 1-ishift_r, ng[0]);
        tbt_fold.setRange(0, 1-ishift_t, ng[0]);
        tbz_fold.setRange(0, 1-ishift_z, ng[0]);
        amrex::ParallelFor(tbr_fold, ncomp_rr,
        [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/, int icomp)
        {
            Srr_arr(i,j,0,icomp) += Srr_arr(-ishift_r-i,j,0,icomp);
        },
        tbt_fold, ncomp_tt,
        [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/, int icomp)
        {
            Stt_arr(i,j,0,icomp) += Stt_arr(-ishift_t-i,j,0,icomp);
        },
        tbz_fold, ncomp_zz,
        [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/, int icomp)
        {
            Szz_arr(i,j,0,icomp) += Szz_arr(-ishift_z-i,j,0,icomp);
        });

        // Step 2: fill the guard cells at negative radius with the image of
        // the cells above the axis.
        amrex::Box tbr_fill = tbr;
        amrex::Box tbt_fill = tbt;
        amrex::Box tbz_fill = tbz;
        tbr_fill.setRange(0, -ng[0], ng[0]);
        tbt_fill.setRange(0, -ng[0], ng[0]);
        tbz_fill.setRange(0, -ng[0], ng[0]);
        amrex::ParallelFor(tbr_fill, ncomp_rr,
        [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/, int icomp)
        {
            Srr_arr(i,j,0,icomp) = Srr_arr(-ishift_r-i,j,0,icomp);
        },
        tbt_fill, ncomp_tt,
        [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/, int icomp)
        {
            Stt_arr(i,j,0,icomp) = Stt_arr(-ishift_t-i,j,0,icomp);
        },
        tbz_fill, ncomp_zz,
        [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/, int icomp)
        {
            Szz_arr(i,j,0,icomp) = Szz_arr(-ishift_z-i,j,0,icomp);
        });
    }
}
#endif

void WarpX::ApplyElectronPressureBoundary (const int lev, PatchType patch_type)
{
    if (::isAnyBoundary<FieldBoundaryType::PEC>(field_boundary_lo, field_boundary_hi)) {
        if (patch_type == PatchType::fine) {
            ablastr::fields::ScalarField electron_pressure_fp = m_fields.get(FieldType::hybrid_electron_pressure_fp, lev);
            PEC::ApplyPECtoElectronPressure(
                electron_pressure_fp,
                field_boundary_lo, field_boundary_hi,
                Geom(lev), lev, patch_type, ref_ratio);
        } else {
            amrex::Abort(Utils::TextMsg::Err(
            "ApplyElectronPressureBoundary: Only one level implemented for hybrid solver."));
        }
    }
}
