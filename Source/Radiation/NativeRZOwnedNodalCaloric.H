/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_NATIVERZOWNEDNODALCALORIC_H_
#define WARPX_RADIATION_NATIVERZOWNEDNODALCALORIC_H_

#include "NativeCompositeTableCornerCapture.H"

#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFab.H>

#include <memory>
#include <string>

class HybridPICModel;

namespace warpx::radiation
{
struct NativeRZOwnedNodalCaloricResult;

/** Private caloric/helper scope only; does not activate any radiation runtime,
 * source, census or history boundary. Closed z is CPU, PEC/reflecting only. */
struct NativeRZCaloricOptions
{
    bool reflecting_axial_trial = false;
    bool single_selected_table_trial = false;
    bool single_selected_table_vacuum = false;
};

/** Caller-private old RZ electron state. Valid components are:
 * 0 effective charge density; 1 reference Cv density; 2 reference U density;
 * 3 available physical old energy density (including component 5);
 * 4 physical finite-volume nodal volume; 5 centered old-corner minus reference
 * U density; 6 canonical reference temperature. Components 2 and 5 MUST NOT be
 * added before taking a small energy difference: that would round away the
 * old-corner offset. Component 1 is the reference inverse capacity; a future
 * emission incidence must use each frozen corner's actual old Cv and FV
 * corner measure. Each unique physical/periodic node has one owner.
 */
class NativeRZOwnedNodalCaloricState
{
public:
    [[nodiscard]] bool Valid () const noexcept { return m_valid; }
    [[nodiscard]] std::string const& Reason () const noexcept { return m_reason; }
    [[nodiscard]] amrex::MultiFab const& Nodes () const noexcept { return m_nodes; }
    [[nodiscard]] amrex::iMultiFab const& Owners () const noexcept { return *m_owners; }
    [[nodiscard]] amrex::MultiFab const& FrozenCharge () const noexcept { return m_charge; }
    [[nodiscard]] amrex::MultiFab const& FrozenTemperature () const noexcept
    {
        return m_temperature;
    }

private:
    friend NativeRZOwnedNodalCaloricState PrepareNativeRZOwnedNodalCaloric (
        HybridPICModel const&, amrex::MultiFab const&, amrex::MultiFab const&, amrex::Real,
        NativeRZCaloricOptions const&);
    friend NativeRZOwnedNodalCaloricResult EvaluateNativeRZOwnedNodalCaloric (
        HybridPICModel const&, NativeRZOwnedNodalCaloricState const&,
        amrex::MultiFab const&, amrex::MultiFab const*);

    bool m_valid = false;
    std::string m_reason;
    amrex::MultiFab m_charge, m_temperature, m_reference, m_nodes;
    std::unique_ptr<amrex::iMultiFab> m_owners;
    // CPU-only physical-corner table provider. Never replace its density by a
    // nodal average; the EOS owner must outlive this frozen private capture.
    std::unique_ptr<detail::NativeCompositeTableCornerCapture> m_table;
    NativeRZCaloricOptions m_options;
};

/** Prepare fixed-charge ideal/builtin-latent, or CPU one-material periodic-z
 * table RZ caloric data from ACTUAL
 * frozen cell-corner rho/T. Input fields and live simulation remain untouched.
 * Occupied corners must exceed minimum_density; positive charge copies may
 * differ and are weighted conservatively. Exact vacuum corners retain their
 * temperature. Temperature copies may differ only at roundoff scale.
 * Returned data is for a level-zero, m=0, static-grid FV electron source.
 * With the explicit closed-z helper option, endpoint nodes are distinct owners
 * with dz/2 dual support; the outer-r/z corner has both half-volume factors.
 * Raw native reflective rho is consumed with this native transport measure,
 * not divided by two or reinterpreted as a cell-centered fluid density.
 * The explicit closed-z helper also admits one fixed-charge table material
 * on CPU. Its physical-corner capture retains the actual raw/unit-ion density,
 * original EOS energy and certified inverse. No closed-table radiation source,
 * packet flight, wall account, history or normal runtime is activated here.
 */
[[nodiscard]] NativeRZOwnedNodalCaloricState PrepareNativeRZOwnedNodalCaloric (
    HybridPICModel const& model, amrex::MultiFab const& frozen_charge,
    amrex::MultiFab const& frozen_temperature, amrex::Real minimum_density,
    NativeRZCaloricOptions const& options = {});

/** Candidate temperature and independent physical energy account. Heat input
 * is J per unique OWNER node; nonowners must contain zero. Nodal residual
 * components are signed J, source scale J, arithmetic bound J. Realized cell
 * components are independently reconstructed physical J/cell and its bound.
 * A prescribed trial returns its residual even when nonzero; callers must test
 * that equation before accepting it. No live field, particle, carry, or native
 * birth ID is changed. Failure leaves only discardable caller-private output.
 */
struct NativeRZOwnedNodalCaloricResult
{
    bool valid = false;
    std::string reason;
    amrex::MultiFab candidate_temperature;
    amrex::MultiFab cell_realized_energy;
    amrex::MultiFab nodal_residual;
};

[[nodiscard]] NativeRZOwnedNodalCaloricResult EvaluateNativeRZOwnedNodalCaloric (
    HybridPICModel const& model, NativeRZOwnedNodalCaloricState const& old,
    amrex::MultiFab const& owner_heat_joules,
    amrex::MultiFab const* prescribed_temperature = nullptr);
} // namespace warpx::radiation

#endif
