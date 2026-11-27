/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_CERTIFIEDCOMPOSITETABLECALORIC_H_
#define WARPX_RADIATION_CERTIFIEDCOMPOSITETABLECALORIC_H_

#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/ElectronThermodynamics.H"

#include <array>
#include <cmath>
#include <limits>

namespace warpx::radiation::detail
{
/** One actual physical cell-corner contribution to a unique nodal EOS.
 * The caller owns deposition and geometry: mass_density is the frozen
 * per-material kg/m^3 state; measure is its physical cell-corner m^3.
 * effective_charge includes the caller's declared occupied-corner floor.
 * A zero raw charge is vacuum and must have zero mass and effective charge.
 */
struct CompositeTableCorner
{
    amrex::Real raw_charge = 0;
    amrex::Real effective_charge = 0;
    ElectronThermodynamicsExecutor::MaterialMassDensities mass_density{};
    amrex::Real old_temperature = 0;
    amrex::Real measure = 0;
};

enum class CompositeTableCaloricFailure
{
    None,
    UnsupportedExecutor,
    InvalidCorner,
    InconsistentOldCopies,
    InvalidEOS,
    ChangedExecutor,
    Unaffordable,
    UncertifiedInverse
};

struct CompositeTableInverseImage
{
    amrex::Real lower_temperature = 0;
    amrex::Real upper_temperature = 0;
    amrex::Real lower_centered_joules = 0;
    amrex::Real upper_centered_joules = 0;
    bool valid = false;
};

struct CompositeTableCaloricResult
{
    CompositeTableCaloricFailure failure = CompositeTableCaloricFailure::InvalidEOS;
    amrex::Real temperature = 0;
    amrex::Real realized_joules = 0;
    amrex::Real residual_joules = 0;
    amrex::Real arithmetic_bound_joules = 0;
    bool converged = false;
    CompositeTableInverseImage inverse_image;
    [[nodiscard]] bool Valid () const noexcept
    {
        return failure == CompositeTableCaloricFailure::None;
    }
};

class FrozenCompositeTableCaloricNode
{
public:
    static constexpr int max_corners = 8;
    [[nodiscard]] bool Valid () const noexcept
    {
        return m_failure == CompositeTableCaloricFailure::None;
    }
    [[nodiscard]] CompositeTableCaloricFailure Failure () const noexcept { return m_failure; }
    [[nodiscard]] amrex::Real OldOffsetJoules () const noexcept { return m_old_offset; }
    [[nodiscard]] amrex::Real AvailableJoules () const noexcept { return m_available; }
    [[nodiscard]] amrex::Real ReferenceTemperature () const noexcept { return m_reference; }
    [[nodiscard]] int CornerCount () const noexcept { return m_count; }

private:
    friend FrozenCompositeTableCaloricNode PrepareCompositeTableCaloricNode (
        ElectronThermodynamicsExecutor const&,
        std::array<CompositeTableCorner, max_corners> const&, int, amrex::Real);
    friend CompositeTableCaloricResult EvaluateCompositeTableCaloricNode (
        ElectronThermodynamicsExecutor const&, FrozenCompositeTableCaloricNode const&,
        amrex::Real, amrex::Real const*);

    CompositeTableCaloricFailure m_failure = CompositeTableCaloricFailure::InvalidCorner;
    std::array<CompositeTableCorner, max_corners> m_corners{};
    int m_count = 0;
    int m_materials = 0;
    bool m_resolved = false;
    void const* m_tables = nullptr; // borrowed from the caller's EOS owner
    warpx::materials::MaterialRegistry::ResolvedCellSelector m_selector{};
    amrex::GpuArray<amrex::Real, ElectronThermodynamicsExecutor::max_materials>
        m_minimum_mass_density{}, m_maximum_mass_density{};
    amrex::GpuArray<amrex::Real, ElectronThermodynamicsExecutor::max_materials>
        m_atomic_mass{}, m_atomic_number{};
    amrex::Real m_floor = 0, m_ceiling = 0;
    amrex::Real m_reference = 0, m_old_offset = 0, m_old_offset_bound = 0, m_available = 0;
};

/** Freeze a bounded unique-node table contract without touching live fields.
 * The caller's ElectronThermodynamics owner must outlive this borrowed state.
 * This is a private mathematical provider, not a live material-field capture
 * or normal reciprocal-source activation. The existing table inverse assumes
 * monotone U(T); the composite inverse has that same table precondition.
 */
[[nodiscard]] FrozenCompositeTableCaloricNode PrepareCompositeTableCaloricNode (
    ElectronThermodynamicsExecutor const& eos,
    std::array<CompositeTableCorner, FrozenCompositeTableCaloricNode::max_corners> const& corners,
    int count, amrex::Real reference_temperature);

/** Invert Q+old-copy offset against Σcorner V[U(T)-U(Tref)] using 64 finite
 * bisections, and certify the energy image of its returned temperature by
 * replaying both equality branches. No endpoint clipping or hidden source
 * reconciliation is performed. A prescribed T returns its signed residual
 * without claiming it solves the nonlinear equation. Realized heat is always
 * independently re-evaluated from every original old corner.
 */
[[nodiscard]] CompositeTableCaloricResult EvaluateCompositeTableCaloricNode (
    ElectronThermodynamicsExecutor const& eos, FrozenCompositeTableCaloricNode const& old,
    amrex::Real requested_joules, amrex::Real const* prescribed_temperature = nullptr);
} // namespace warpx::radiation::detail

#endif
