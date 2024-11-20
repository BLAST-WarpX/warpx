#ifndef WARPX_PEC_KERNELS_H_
#define WARPX_PEC_KERNELS_H_

#include "Utils/WarpXAlgorithmSelection.H"

#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_Vector.H>

#include <AMReX_BaseFwd.H>

#include <array>
#include <memory>

namespace PEC {
    /**
     * \brief Sets the tangential electric field at the PEC boundary to zero.
     *        The guard cell values are set equal and opposite to the valid cell
     *        field value at the respective mirror locations.
     *
     * \param[in,out] Efield              Boundary values of tangential Efield are set to zero
     * \param[in]     field_boundary_lo   Boundary types of the "low" boundaries
     * \param[in]     field_boundary_hi   Boundary types of the "high" boundaries
     * \param[in]     ng_fieldgather      number of guard cells used by field gather
     * \param[in]     geom                geometry object of level "lev"
     * \param[in]     lev                 level of the Multifab
     * \param[in]     patch_type          coarse or fine
     * \param[in]     ref_ratios          vector containing the refinement ratios of the refinement levels
     * \param[in]     split_pml_field     whether pml the multifab is the regular Efield or
     *                                        split pml field
     */
    void ApplyPECtoEfield (
                        std::array<amrex::MultiFab*, 3> Efield,
                        const amrex::Array<FieldBoundaryType,AMREX_SPACEDIM>& field_boundary_lo,
                        const amrex::Array<FieldBoundaryType,AMREX_SPACEDIM>& field_boundary_hi,
                        const amrex::IntVect& ng_fieldgather, const amrex::Geometry& geom,
                        int lev, PatchType patch_type, const amrex::Vector<amrex::IntVect>& ref_ratios,
                        bool split_pml_field = false);
    /**
     * \brief Sets the normal component of the magnetic field at the PEC boundary to zero.
     *        The guard cell values are set equal and opposite to the valid cell
     *        field value at the respective mirror locations.
     *
     * \param[in,out] Bfield              Boundary values of normal Bfield are set to zero.
     * \param[in]     field_boundary_lo   Boundary types of the "low" field boundaries
     * \param[in]     field_boundary_hi   Boundary types of the "high" field boundaries
     * \param[in]     ng_fieldgather      number of guard cells used by field gather
     * \param[in]     geom                geometry object of level "lev"
     * \param[in]     lev                 level of the Multifab
     * \param[in]     patch_type          coarse or fine
     * \param[in]     ref_ratios          vector containing the refinement ratios of the refinement levels
     */
    void ApplyPECtoBfield (
                        std::array<amrex::MultiFab*, 3> Bfield,
                        const amrex::Array<FieldBoundaryType,AMREX_SPACEDIM>& field_boundary_lo,
                        const amrex::Array<FieldBoundaryType,AMREX_SPACEDIM>& field_boundary_hi,
                        const amrex::IntVect& ng_fieldgather, const amrex::Geometry& geom,
                        int lev, PatchType patch_type, const amrex::Vector<amrex::IntVect>& ref_ratios);

    /**
     * \brief Reflects charge density deposited over the PEC boundary back into
     * the simulation domain.
     *
     * \param[in,out] rho                    Multifab containing the charge density
     * \param[in]     field_boundary_lo      Boundary types of the "low" field boundaries
     * \param[in]     field_boundary_hi      Boundary types of the "high" field boundaries
     * \param[in]     particle_boundary_lo   Boundary types of the "low" particle boundaries
     * \param[in]     particle_boundary_hi   Boundary types of the "high" particle boundaries
     * \param[in]     geom                   geometry object of level "lev"
     * \param[in]     lev                    level of the Multifab
     * \param[in]     patch_type             coarse or fine
     * \param[in]     ref_ratios             vector containing the refinement ratios of the refinement levels
     */
    void ApplyReflectiveBoundarytoRhofield(
                        amrex::MultiFab* rho,
                        const amrex::Array<FieldBoundaryType,AMREX_SPACEDIM>& field_boundary_lo,
                        const amrex::Array<FieldBoundaryType,AMREX_SPACEDIM>& field_boundary_hi,
                        const amrex::Array<ParticleBoundaryType,AMREX_SPACEDIM>& particle_boundary_lo,
                        const amrex::Array<ParticleBoundaryType,AMREX_SPACEDIM>& particle_boundary_hi,
                        const amrex::Geometry& geom,
                        int lev, PatchType patch_type, const amrex::Vector<amrex::IntVect>& ref_ratios);

    /**
     * \brief Reflects current density deposited over the PEC boundary back into
     * the simulation domain.
     *
     * \param[in,out] Jx, Jy, Jz             Multifabs containing the current density
     * \param[in]     field_boundary_lo      Boundary types of the "low" field boundaries
     * \param[in]     field_boundary_hi      Boundary types of the "high" field boundaries
     * \param[in]     particle_boundary_lo   Boundary types of the "low" particle boundaries
     * \param[in]     particle_boundary_hi   Boundary types of the "high" particle boundaries
     * \param[in]     geom                   geometry object of level "lev"
     * \param[in]     lev                    level of the Multifab
     * \param[in]     patch_type             coarse or fine
     * \param[in]     ref_ratios             vector containing the refinement ratios of the refinement levels
     */
    void ApplyReflectiveBoundarytoJfield(
                        amrex::MultiFab* Jx, amrex::MultiFab* Jy,
                        amrex::MultiFab* Jz,
                        const amrex::Array<FieldBoundaryType,AMREX_SPACEDIM>& field_boundary_lo,
                        const amrex::Array<FieldBoundaryType,AMREX_SPACEDIM>& field_boundary_hi,
                        const amrex::Array<ParticleBoundaryType,AMREX_SPACEDIM>& particle_boundary_lo,
                        const amrex::Array<ParticleBoundaryType,AMREX_SPACEDIM>& particle_boundary_hi,
                        const amrex::Geometry& geom,
                        int lev, PatchType patch_type, const amrex::Vector<amrex::IntVect>& ref_ratios);

    /**
     * \brief Apply the PEC boundary to the electron pressure field.
     *
     * \param[in,out] Pefield             Multifab containing the electron pressure
     * \param[in]     field_boundary_lo   Boundary types of the "low" field boundaries
     * \param[in]     field_boundary_hi   Boundary types of the "high" field boundaries
     * \param[in]     geom                geometry object of level "lev"
     * \param[in]     lev                 level of the Multifab
     * \param[in]     patch_type          coarse or fine
     * \param[in]     ref_ratios          vector containing the refinement ratios of the refinement levels
     */
    void ApplyPECtoElectronPressure (
                                amrex::MultiFab* Pefield,
                                const amrex::Array<FieldBoundaryType,AMREX_SPACEDIM>& field_boundary_lo,
                                const amrex::Array<FieldBoundaryType,AMREX_SPACEDIM>& field_boundary_hi,
                                const amrex::Geometry& geom,
                                int lev, PatchType patch_type, const amrex::Vector<amrex::IntVect>& ref_ratios);
}
#endif // WarpX_PEC_KERNELS_H_
