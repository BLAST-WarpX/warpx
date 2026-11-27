/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_NATIVECOMPOSITETABLECORNERCAPTURE_H_
#define WARPX_RADIATION_NATIVECOMPOSITETABLECORNERCAPTURE_H_

#include "CertifiedCompositeTableCaloric.H"

#include <AMReX.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include <array>
#include <memory>
#include <string>
#include <vector>

class HybridPICModel;

namespace warpx::radiation::detail
{
using UnitIonChargeFields =
    amrex::GpuArray<amrex::MultiFab const*, ElectronThermodynamicsExecutor::max_materials>;

struct NativeTableCornerCaptureOptions
{
    amrex::Real minimum_electron_density = 0;
    amrex::Real thermodynamic_electron_density_floor = 0;
    // RZ requires the actual native FV nodal shell, including the axis and
    // outer-wall correction. Cartesian captures must leave this false.
    bool rz_native_fv = false;
    bool rz_verboncoeur_axis_correction = true;
    // Explicit CPU-only clipped axial support for a PRIVATE table caloric
    // capture. Model adapters require actual stationary PEC/reflecting z.
    // This does not admit closed-table source, flight, history or runtime.
    bool rz_reflecting_axial_trial = false;
    /** Private CPU periodic one-selected-table proof; retain the actual
     * selector and require every consumed physical corner classify material0. */
    bool require_single_selected_material0 = false;
    /** Separate explicit single-selected policy: an exactly zero raw charge
     * requires exactly zero unit-ion mass and the actual Vacuum selector.
     * No floor, absent metadata or positive-density cutoff creates vacuum. */
    bool allow_single_selected_vacuum = false;
};

struct NativeTableCapturedNode
{
    amrex::IntVect index;
    int fab = -1;
    std::array<CompositeTableCorner, FrozenCompositeTableCaloricNode::max_corners> corners{};
    int corner_count = 0;
    FrozenCompositeTableCaloricNode caloric;
};

/** Frozen physical-corner incidence for a future reciprocal table source.
 * Every input field is copied before private ghost fill; actual valid-cell
 * corner copies are never OverrideSync'ed. Only distinct extensive corner
 * records are summed across boxes/periodic aliases. The caller's EOS table
 * owner must remain alive and immutable while the capture is evaluated.
 */
class NativeCompositeTableCornerCapture
{
public:
    [[nodiscard]] bool Valid () const noexcept { return m_valid; }
    [[nodiscard]] std::string const& Reason () const noexcept { return m_reason; }
    [[nodiscard]] std::vector<NativeTableCapturedNode> const& Owners () const noexcept
    {
        AMREX_ALWAYS_ASSERT(m_valid);
        return m_owners;
    }
    [[nodiscard]] amrex::Long GlobalOwnerCount () const noexcept
    {
        AMREX_ALWAYS_ASSERT(m_valid);
        return m_global_owners;
    }
    [[nodiscard]] amrex::MultiFab const& FrozenCharge () const noexcept
    {
        AMREX_ALWAYS_ASSERT(m_valid && m_charge);
        return *m_charge;
    }
    [[nodiscard]] amrex::MultiFab const& FrozenTemperature () const noexcept
    {
        AMREX_ALWAYS_ASSERT(m_valid && m_temperature);
        return *m_temperature;
    }
    [[nodiscard]] amrex::MultiFab const& FrozenUnitIonCharge (int material) const noexcept
    {
        AMREX_ALWAYS_ASSERT(m_valid && material >= 0 && material < m_material_count &&
                            m_unit_ion_charge[material]);
        return *m_unit_ion_charge[material];
    }
    [[nodiscard]] amrex::Real FrozenMassPerUnitCharge (int material) const noexcept
    {
        AMREX_ALWAYS_ASSERT(m_valid && material >= 0 && material < m_material_count);
        return m_mass_per_unit_charge[material];
    }

private:
    friend NativeCompositeTableCornerCapture CaptureNativeCompositeTableCorners (
        ElectronThermodynamicsExecutor const&, amrex::MultiFab const&,
        amrex::MultiFab const&, UnitIonChargeFields const&, amrex::Geometry const&,
        NativeTableCornerCaptureOptions const&);
    friend NativeCompositeTableCornerCapture CaptureNativeCompositeTableCornersFromModel (
        HybridPICModel const&, amrex::MultiFab const&, amrex::MultiFab const&,
        amrex::Geometry const&, NativeTableCornerCaptureOptions const&);

    bool m_valid = false;
    std::string m_reason = "unsupported native table corner capture";
    std::unique_ptr<amrex::MultiFab> m_charge, m_temperature;
    std::array<std::unique_ptr<amrex::MultiFab>,
               ElectronThermodynamicsExecutor::max_materials> m_unit_ion_charge{};
    amrex::GpuArray<amrex::Real, ElectronThermodynamicsExecutor::max_materials>
        m_mass_per_unit_charge{};
    std::vector<NativeTableCapturedNode> m_owners;
    amrex::Long m_global_owners = 0;
    int m_material_count = 0;
};

/** Generic capture from explicitly supplied deposited unit-ion charge fields.
 * This does not invoke the reciprocal source, change any caller field, or
 * equate a physical corner's table composition to an owner-node average.
 */
[[nodiscard]] NativeCompositeTableCornerCapture CaptureNativeCompositeTableCorners (
    ElectronThermodynamicsExecutor const& eos,
    amrex::MultiFab const& frozen_charge,
    amrex::MultiFab const& frozen_temperature,
    UnitIonChargeFields const& unit_ion_charge,
    amrex::Geometry const& geometry,
    NativeTableCornerCaptureOptions const& options);

/** Adapter for a future SourceContext: resolve the model's named
 * `ni_charge_fp_<species>` fields, then deep-copy them through the generic
 * capture. It intentionally does not lift existing table-source guards.
 */
[[nodiscard]] NativeCompositeTableCornerCapture CaptureNativeCompositeTableCornersFromModel (
    HybridPICModel const& model,
    amrex::MultiFab const& frozen_charge,
    amrex::MultiFab const& frozen_temperature,
    amrex::Geometry const& geometry,
    NativeTableCornerCaptureOptions const& options);

/** Read-only collective actual-model prerequisite. No executor rewrite,
 * capture, caloric solve or runtime activation. */
[[nodiscard]] bool HasNativeSingleSelectedTableMaterial0 (
    HybridPICModel const& model, amrex::Geometry const& geometry);

/** Separate exact-vacuum extension; the strict positive-corner entry remains
 * unchanged. Raw charge and unit-ion mass must both vanish on a vacant node. */
[[nodiscard]] bool HasNativeSingleSelectedTableMaterial0OrExactVacuum (
    HybridPICModel const& model, amrex::Geometry const& geometry);
} // namespace warpx::radiation::detail

#endif
