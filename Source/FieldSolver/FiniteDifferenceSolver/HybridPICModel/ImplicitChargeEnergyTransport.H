/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#pragma once

#include "Utils/TextMsg.H"

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>

#include <cmath>
#include <limits>
#include <string>

namespace warpx::hybrid
{
/** Backward-Euler upwind remap for specific energy e=U/rho.
 *
 * (rho_new + dt*outflow/V)*e_i - sum(dt*inflow_ji/V*e_j) = U_old.
 *
 * Positive face amounts give an M-matrix and permit through-flow across a
 * node with zero old charge. Output U=rho_new*e is exactly zero at evacuated
 * nodes. Integrated conservation requires the caller's two representations
 * of each face amount to use their respective native node volumes. The
 * caller must independently validate endpoint charge continuity.
 *
 * outflow and inflow already contain dt and division by the recipient node
 * volume. Inflow components are low/high faces in each active direction.
 * No energy clipping or density floor is used. This distinct implicit method
 * does not rely on an explicit donor CFL bound.
 */
inline int
implicitChargeEnergyRemap (amrex::MultiFab& energy, amrex::MultiFab const& old_energy,
                           amrex::MultiFab const& new_rho, amrex::MultiFab const& outflow,
                           amrex::MultiFab const& inflow, amrex::MultiFab const& node_volume,
                           amrex::Geometry const& geometry,
                           char const* const model_name = "Implicit charge-energy remap")
{
    using namespace amrex::literals;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        inflow.nComp() == 2 * AMREX_SPACEDIM,
        std::string(model_name) + " needs one incoming coefficient per face.");
    amrex::MultiFab specific(energy.boxArray(), energy.DistributionMap(), 1, 1);
    amrex::MultiFab next(energy.boxArray(), energy.DistributionMap(), 1, 1);
    amrex::MultiFab errors(energy.boxArray(), energy.DistributionMap(), 1, 0);
    specific.setVal(0.0_rt);
    next.setVal(0.0_rt);
    constexpr int maximum_iterations = 512;
    constexpr amrex::Real tolerance = 128.0_rt * std::numeric_limits<amrex::Real>::epsilon();
    int iterations = 0;
    amrex::Real relative_residual = 1.0_rt;
    for (; iterations < maximum_iterations; ++iterations) {
        specific.FillBoundary(geometry.periodicity());
        for (amrex::MFIter mfi(energy); mfi.isValid(); ++mfi) {
            auto const old = old_energy.const_array(mfi);
            auto const rho = new_rho.const_array(mfi);
            auto const outgoing = outflow.const_array(mfi);
            auto const incoming = inflow.const_array(mfi);
            auto const volume = node_volume.const_array(mfi);
            auto const previous = specific.const_array(mfi);
            auto const updated = next.array(mfi);
            auto const error = errors.array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                auto const diagonal = rho(i, j, k) + outgoing(i, j, k);
                auto rhs = old(i, j, k);
                bool valid = std::isfinite(rhs) && rhs >= 0.0_rt && std::isfinite(rho(i, j, k)) &&
                             rho(i, j, k) >= 0.0_rt && std::isfinite(outgoing(i, j, k)) &&
                             outgoing(i, j, k) >= 0.0_rt && std::isfinite(diagonal) &&
                             std::isfinite(volume(i, j, k)) && volume(i, j, k) > 0.0_rt;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    for (int side = 0; side < 2; ++side) {
                        auto const coefficient = incoming(i, j, k, 2 * d + side);
                        valid = valid && std::isfinite(coefficient) && coefficient >= 0.0_rt;
                        if (coefficient > 0.0_rt) {
                            int const offset = side == 0 ? -1 : 1;
                            rhs += coefficient * previous(i + (d == 0 ? offset : 0),
                                                          j + (d == 1 ? offset : 0),
                                                          k + (d == 2 ? offset : 0));
                        }
                    }
                }
                valid = valid && std::isfinite(rhs) && rhs >= 0.0_rt &&
                        (diagonal > 0.0_rt || rhs == 0.0_rt);
                auto const value = diagonal > 0.0_rt ? rhs / diagonal : 0.0_rt;
                valid = valid && std::isfinite(value) && value >= 0.0_rt;
                updated(i, j, k) = valid ? value : 0.0_rt;
                auto const lhs = diagonal * previous(i, j, k);
                auto const scale = amrex::max(std::abs(lhs) + std::abs(rhs),
                                              std::numeric_limits<amrex::Real>::min());
                error(i, j, k) =
                    valid ? std::abs(lhs - rhs) / scale : std::numeric_limits<amrex::Real>::max();
            });
        }
        relative_residual = errors.norm0(0, 0, false);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            std::isfinite(relative_residual) && relative_residual <= 1.0_rt,
            std::string(model_name) + " found invalid coefficients or unsupported old energy.");
        // Residual was measured for specific, not for the newly proposed iterate.
        if (relative_residual <= tolerance) {
            break;
        }
        amrex::MultiFab::Swap(specific, next, 0, 0, 1, 1);
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        relative_residual <= tolerance,
        std::string(model_name) + " did not converge; reduce the transport timestep.");
    for (amrex::MFIter mfi(energy); mfi.isValid(); ++mfi) {
        auto const output = energy.array(mfi);
        auto const rho = new_rho.const_array(mfi);
        auto const value = specific.const_array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            output(i, j, k) = rho(i, j, k) * value(i, j, k);
        });
    }
    // An independent integrated inventory catches incorrect face metrics even
    // if the algebraic system itself converges. Count each nodal alias once.
    auto const owner = energy.OwnerMask(geometry.periodicity());
    amrex::MultiFab inventory(energy.boxArray(), energy.DistributionMap(), 3, 0);
    for (amrex::MFIter mfi(inventory); mfi.isValid(); ++mfi) {
        auto const old = old_energy.const_array(mfi);
        auto const updated = energy.const_array(mfi);
        auto const specific_energy = specific.const_array(mfi);
        auto const outgoing = outflow.const_array(mfi);
        auto const volume = node_volume.const_array(mfi);
        auto const owned = owner->const_array(mfi);
        auto const values = inventory.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            auto const measure = owned(i, j, k) ? volume(i, j, k) : 0.0_rt;
            values(i, j, k, 0) = old(i, j, k) * measure;
            values(i, j, k, 1) = updated(i, j, k) * measure;
            values(i, j, k, 2) = outgoing(i, j, k) * specific_energy(i, j, k) * measure;
        });
    }
    auto const before = inventory.sum(0, false);
    auto const after = inventory.sum(1, false);
    auto const transported = inventory.sum(2, false);
    auto const roundoff_bound = 4096.0_rt * std::numeric_limits<amrex::Real>::epsilon() *
                                (std::abs(before) + std::abs(after) + std::abs(transported));
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        std::isfinite(before) && std::isfinite(after) && std::isfinite(transported) &&
            std::isfinite(roundoff_bound) && std::abs(after - before) <= roundoff_bound,
        std::string(model_name) + " violated its native-volume energy inventory.");
    return iterations;
}
} // namespace warpx::hybrid
