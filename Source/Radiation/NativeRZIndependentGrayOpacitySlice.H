/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_NATIVERZINDEPENDENTGRAYOPACITYSLICE_H_
#define WARPX_RADIATION_NATIVERZINDEPENDENTGRAYOPACITYSLICE_H_

#include "NativeRZThermalHistoryAdmission.H"
#include "ResolvedRZThermalDiffusionField.H"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace warpx::radiation
{
namespace detail { class NativeCompositeTableCornerCapture; }

/** Engine-local synchronous left-state R/P/S owner. Only the actual History
 * entry may issue it, using its genuine context and either an all-D/no-stored-photon
 * witness or the live mixed packet/mask owners. Raw charge/unit/left T and channels are privately
 * copied. Trial T belongs solely to the bath/caloric solve, never this epoch.
 *
 * The real context/witness/source/table are BORROWED and must outlive every
 * call. Destroy this owner before its left source owners or context change;
 * never return it in a candidate, retain it between slices, or checkpoint it.
 * No public constructor/factory, copied context, caller executor, arbitrary
 * channel fields or caller-provided validity/authority bool exists. The
 * opt-in History/runtime policy uses this owner only within its current slice. */
class NativeRZIndependentGrayOpacitySlice
{
    struct FrozenPhotonOwners;
public:
    ~NativeRZIndependentGrayOpacitySlice ();
    NativeRZIndependentGrayOpacitySlice (NativeRZIndependentGrayOpacitySlice const&) = delete;
    NativeRZIndependentGrayOpacitySlice& operator= (NativeRZIndependentGrayOpacitySlice const&) = delete;
    [[nodiscard]] IndependentGrayMovingFieldOpacity const& Opacity () const noexcept
    { return *m_opacity; }
    [[nodiscard]] bool IsMixed () const noexcept { return m_live_photons != nullptr; }
    [[nodiscard]] bool ValidatePacketFrame (
        amrex::MultiFab const& velocity, amrex::Real scattering_limit,
        amrex::Real absorption_limit, std::string& reason) const;
    [[nodiscard]] amrex::MultiFab const& FrozenCharge () const noexcept { return *m_charge; }
    [[nodiscard]] amrex::MultiFab const& FrozenLeftTemperature () const noexcept
    { return *m_temperature; }
    [[nodiscard]] amrex::MultiFab const& FrozenUnitIonCharge () const noexcept { return *m_unit; }
    [[nodiscard]] MaterialOpacityTable const& LoadedTable () const noexcept { return *m_table; }
    [[nodiscard]] IndependentGrayMovingAllDFieldOptions const& Options () const noexcept
    { return m_options; }
    /** Collective exact source/table/native-EOS/context and packet/mask check. Passing
     * some other genuine context does not rebind this owner. */
    [[nodiscard]] bool Validate (
        NativeRZThermalHistoryStepContext const& context,
        PhotonPacketTransaction const& current_photons,
        amrex::iMultiFab const& consumed_mask, std::string& reason) const;
    /** Compares actual physical consumed nodal corner bytes and native mass
     * conversion with the common Source's genuine frozen caloric capture. */
    [[nodiscard]] bool ValidateCaloricOwners (
        detail::NativeCompositeTableCornerCapture const& caloric, std::string& reason) const;
    [[nodiscard]] ResolvedRZThermalDiffusionFieldResult Advance (
        NativeRZThermalHistoryStepContext const& context,
        PhotonPacketTransaction const& current_photons,
        amrex::MultiFab const& lab_energy, amrex::MultiFab const& lab_momentum,
        amrex::MultiFab const& proper_bath, amrex::iMultiFab const& consumed_mask,
        amrex::MultiFab const& velocity, amrex::MultiFab const& actual_work) const;
    /** Uses the ONE frozen numerical epoch for initial/trial/actual post-kick
     * E/Q/bath/frame/work. Revalidates ownership before and after the read-only
     * screen; it does not advance radiation or stage material. */
    [[nodiscard]] bool Preflight (
        NativeRZThermalHistoryStepContext const& context,
        PhotonPacketTransaction const& current_photons,
        amrex::MultiFab const& lab_energy, amrex::MultiFab const& lab_momentum,
        amrex::MultiFab const& proper_bath, amrex::iMultiFab const& consumed_mask,
        amrex::MultiFab const& velocity, amrex::MultiFab const& actual_work,
        std::string& reason) const;

private:
    NativeRZIndependentGrayOpacitySlice (
        NativeRZThermalHistoryStepContext const& context,
        NativeRZThermalAllDNoKEventWitness const* witness,
        PhotonPacketTransaction const& source_photons,
        amrex::MultiFab const& raw_charge, amrex::MultiFab const& left_temperature,
        amrex::MultiFab const& unit_ion_charge, MaterialOpacityTable const& table,
        HybridPICModel const& native_model, ElectronThermodynamicsExecutor const& native_eos,
        amrex::Geometry const& geometry, IndependentGrayMovingAllDFieldOptions const& options,
        std::unique_ptr<amrex::MultiFab const> charge,
        std::unique_ptr<amrex::MultiFab const> temperature,
        std::unique_ptr<amrex::MultiFab const> unit,
        std::unique_ptr<IndependentGrayMovingFieldOpacity const> opacity,
        std::vector<std::uint8_t>&& table_identity,
        WarpXParticleContainer const* live_photons,
        amrex::iMultiFab const* original_mask,
        std::unique_ptr<amrex::iMultiFab const> frozen_mask,
        std::unique_ptr<FrozenPhotonOwners const> source_photons_capture);
    [[nodiscard]] static std::unique_ptr<NativeRZIndependentGrayOpacitySlice const> Issue (
        NativeRZThermalHistoryStepContext const& context,
        NativeRZThermalAllDNoKEventWitness const& witness,
        PhotonPacketTransaction const& source_photons,
        amrex::MultiFab const& raw_charge, amrex::MultiFab const& left_temperature,
        detail::ResolvedMaterialChargeFields const& unit_ion_charge,
        MaterialOpacityTable const& table, amrex::Geometry const& geometry,
        NativeRZThermalPhotonTimePolicy photon_time_policy,
        IndependentGrayMovingAllDFieldOptions const& options, std::string& reason);
    [[nodiscard]] static std::unique_ptr<NativeRZIndependentGrayOpacitySlice const> IssueMixed (
        NativeRZThermalHistoryStepContext const& context, WarpXParticleContainer const& live,
        PhotonPacketTransaction const& source_photons, amrex::iMultiFab const& fixed_mask,
        amrex::MultiFab const& raw_charge, amrex::MultiFab const& left_temperature,
        detail::ResolvedMaterialChargeFields const& unit_ion_charge,
        MaterialOpacityTable const& table, amrex::Geometry const& geometry,
        NativeRZThermalPhotonTimePolicy photon_time_policy,
        IndependentGrayMovingAllDFieldOptions const& options, std::string& reason);
    [[nodiscard]] static std::unique_ptr<NativeRZIndependentGrayOpacitySlice const> IssueEpoch (
        NativeRZThermalHistoryStepContext const& context,
        NativeRZThermalAllDNoKEventWitness const* witness, WarpXParticleContainer const* live,
        PhotonPacketTransaction const& source_photons, amrex::iMultiFab const* fixed_mask,
        amrex::MultiFab const& raw_charge, amrex::MultiFab const& left_temperature,
        detail::ResolvedMaterialChargeFields const& unit_ion_charge,
        MaterialOpacityTable const& table, amrex::Geometry const& geometry,
        NativeRZThermalPhotonTimePolicy photon_time_policy,
        IndependentGrayMovingAllDFieldOptions const& options, std::string& reason);
    friend NativeResolvedRZThermalHistoryResult TryAdvanceNativeResolvedRZThermalHistory (
        WarpXParticleContainer&, PhotonPacketTransaction const&, ParticleImpulseMaterial const&,
        std::string const&, amrex::MultiFab const&, amrex::iMultiFab const&,
        amrex::MultiFab const&, HybridPICModel const&, amrex::Geometry const&,
        amrex::Real, amrex::Real,
        std::function<bool(amrex::Real, amrex::MultiFab const&, amrex::MultiFab const&,
                           amrex::MultiFab&, amrex::MultiFab&, std::string&)> const&,
        NativeResolvedRZThermalHistoryOptions const&);

    NativeRZThermalHistoryStepContext const* const m_context;
    NativeRZThermalAllDNoKEventWitness const* const m_witness;
    PhotonPacketTransaction const* const m_source_photons;
    amrex::MultiFab const* const m_original_charge;
    amrex::MultiFab const* const m_original_temperature;
    amrex::MultiFab const* const m_original_unit;
    MaterialOpacityTable const* const m_table;
    HybridPICModel const* const m_model;
    ElectronThermodynamicsExecutor const m_eos;
    amrex::Geometry const m_geometry;
    IndependentGrayMovingAllDFieldOptions const m_options;
    std::unique_ptr<amrex::MultiFab const> const m_charge, m_temperature, m_unit;
    std::unique_ptr<IndependentGrayMovingFieldOpacity const> const m_opacity;
    std::vector<std::uint8_t> const m_table_identity;
    WarpXParticleContainer const* const m_live_photons;
    amrex::iMultiFab const* const m_original_mask;
    std::unique_ptr<amrex::iMultiFab const> const m_frozen_mask;
    std::unique_ptr<FrozenPhotonOwners const> const m_source_photons_capture;
};
} // namespace warpx::radiation
#endif
