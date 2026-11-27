/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_NATIVERZTHERMALLEAKAGE_H_
#define WARPX_RADIATION_NATIVERZTHERMALLEAKAGE_H_

#include "PhotonLogicalDrawKey.H"
#include "PhotonPacketTransaction.H"
#include "ResolvedRZThermalDiffusionField.H"

#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace warpx::radiation
{
/** One locally owned D-to-K face event, never an unowned nodal duplicate.
 * accepted and correction are [E,cPr,cLz/rbar_D,cPz] [J]; actual is
 * [E,cPr,cPtheta,cPz,cLz], whose last component is [J*m]. The eight stable
 * keys identify precisely this face's staged newborns. */
struct NativeRZThermalLeakageFaceAccount
{
    int grid = -1;
    int axis = 0;
    amrex::IntVect face, donor, receiver;
    int outward_sign = 0;
    std::array<std::uint64_t, 8> keys{};
    amrex::GpuArray<amrex::Real, 5> actual{};
    amrex::GpuArray<amrex::Real, 4> accepted{}, correction{};
};

struct NativeRZThermalLeakageResult
{
    bool valid = false;
    std::string reason;
    /** Nested candidate retains endpoint's original live provenance. Caller
     * must jointly accept it with source, material and caloric candidates. */
    std::unique_ptr<PhotonPacketTransaction> photons;
    /** Owner-face actual stored-ray [E,cPr,cPtheta,cPz,cLz]. Nodal duplicates
     * may be synchronized for readback; face_accounts identify unique events. */
    amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM> actual_face_inventory;
    /** D-owner-cell correction and independently positive operand/bound [J].
     * This replaces field traction/leakage with actual packet transport in
     * the SAME total material kick/work iteration, without a second kick. */
    amrex::MultiFab donor_radiation_debit_correction;
    amrex::MultiFab donor_operand_scale;
    amrex::MultiFab donor_roundoff_bound;
    std::vector<NativeRZThermalLeakageFaceAccount> face_accounts;
    amrex::Long staged_births = 0;
};

/** Private accepted-face leakage adapter. The field's LAB face energy and
 * stress are authoritative; the existing normalized low-beta LAB-hemisphere
 * law only redistributes the prescribed leakage into eight rays. It is a
 * bounded Milne/asymptotic approximation, not an exact moving boundary law.
 * No second Doppler energy debit, source heat repair, interval streaming,
 * normal-runtime activation, live ID assignment, or live write occurs here.
 * On any failure discard the entire returned private trial. */
[[nodiscard]] NativeRZThermalLeakageResult TryStageNativeRZThermalLeakage (
    WarpXParticleContainer& live, PhotonPacketTransaction const& endpoint,
    ResolvedRZThermalDiffusionFieldResult const& accepted_field,
    amrex::iMultiFab const& diffusion_mask, amrex::MultiFab const& work_velocity,
    amrex::Geometry const& geometry, amrex::Real dt, amrex::Real photon_energy,
    LogicalPhotonBirthKeyContext const& birth_context,
    bool reflecting_axial_endpoint_trial = false,
    /** PRIVATE one-fixed-charge CPU TABLE composition; requires the closed
     * selector above. Default retains all existing endpoint admissions. */
    bool reflecting_axial_table_trial = false,
    bool single_selected_table_trial = false,
    bool single_selected_table_vacuum = false);
} // namespace warpx::radiation

#endif
