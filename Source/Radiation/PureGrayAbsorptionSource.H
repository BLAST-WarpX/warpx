/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_PUREGRAYABSORPTIONSOURCE_H_
#define WARPX_RADIATION_PUREGRAYABSORPTIONSOURCE_H_

#include "IsotropicDipoleFrameMatrices.H"

#include <cmath>
#include <limits>

namespace warpx::radiation
{
namespace gray_absorption_detail
{
AMREX_GPU_HOST_DEVICE inline bool
Solve (StressEnergy matrix, FourVector rhs, FourVector& result)
{
    for (int row = 0; row < 4; ++row) {
        amrex::Real scale = 0;
        for (auto const value : matrix[row]) { scale = amrex::max(scale, std::abs(value)); }
        if (!(scale > 0) || !amrex::Math::isfinite(scale)) { return false; }
        for (auto& value : matrix[row]) { value /= scale; }
        rhs[row] /= scale;
    }
    for (int column = 0; column < 4; ++column) {
        int pivot = column;
        for (int row = column + 1; row < 4; ++row) {
            if (std::abs(matrix[row][column]) > std::abs(matrix[pivot][column])) { pivot = row; }
        }
        if (matrix[pivot][column] == 0) { return false; }
        auto const saved_row = matrix[column];
        auto const saved_rhs = rhs[column];
        matrix[column] = matrix[pivot]; matrix[pivot] = saved_row;
        rhs[column] = rhs[pivot]; rhs[pivot] = saved_rhs;
        for (int row = column + 1; row < 4; ++row) {
            auto const factor = matrix[row][column] / matrix[column][column];
            for (int k = column + 1; k < 4; ++k) { matrix[row][k] -= factor * matrix[column][k]; }
            rhs[row] -= factor * rhs[column];
        }
    }
    for (int row = 3; row >= 0; --row) {
        auto value = rhs[row];
        for (int k = row + 1; k < 4; ++k) { value -= matrix[row][k] * result[k]; }
        result[row] = value / matrix[row][row];
        if (!amrex::Math::isfinite(result[row])) { return false; }
    }
    return true;
}

AMREX_GPU_HOST_DEVICE inline bool
PositiveDipole (FourVector const& rest)
{
    if (rest[0] < 0 || !amrex::Math::isfinite(rest[0])) { return false; }
    if (rest[0] == 0) { return rest[1] == 0 && rest[2] == 0 && rest[3] == 0; }
    amrex::Real norm = 0;
    for (int d = 1; d < 4; ++d) {
        auto const ratio = rest[d] / rest[0];
        norm += ratio * ratio;
    }
    auto const epsilon = std::numeric_limits<amrex::Real>::epsilon();
    return amrex::Math::isfinite(norm) && norm <= (1 + 128 * epsilon) / 9;
}
} // namespace gray_absorption_detail

struct PureGrayAbsorptionResult
{
    bool valid = false;
    FourVector radiation{};
    FourVector rest_radiation{};
    FourVector physical_transfer{};
    FourVector raw_transfer{};
    FourVector storage_residual{};
    FourVector storage_bound{};
    FourVector equation_residual{};
    FourVector equation_bound{};
    amrex::Real physical_energy_minus_work = 0;
};

/** Backward-Euler pure gray absorption/emission with explicit COMOVING
 * isotropic pressure P0=E0 I/3 and a positive angular dipole |q0|<=E0/3.
 * LAB extensive state is M(beta)*rest[E0,q0]. Source force is N(beta)*
 * [E0-B,q0], so (M+rate*N)*rest_new=lab_old+rate*N*[B,0]. Here rate=c*dt*chi_a
 * and B=V*a_R*T^4. No coherent scattering, spatial pressure work or M1 closure
 * is included. Solve the state and increment independently: retained physical
 * transfer survives a native storage ULP, while raw transfer/residual audit the
 * actual LAB fields. No stored state or transfer is corrected. */
AMREX_GPU_HOST_DEVICE inline PureGrayAbsorptionResult
TryPureGrayAbsorptionSource (FourVector const& old,
    amrex::GpuArray<amrex::Real, 3> const& beta, amrex::Real rate, amrex::Real bath)
{
    PureGrayAbsorptionResult result;
    result.radiation = old;
    StressEnergy moment{}, source{};
    if (!BuildIsotropicDipoleFrameMatrices(beta, moment, source) || rate < 0 || bath < 0 ||
        !amrex::Math::isfinite(rate) || !amrex::Math::isfinite(bath)) { return result; }
    amrex::Real beta2 = 0, scale = bath;
    for (auto const value : beta) { beta2 += value * value; }
    if (beta2 > amrex::Real(1.e-4)) { return result; }
    for (auto const value : old) {
        if (!amrex::Math::isfinite(value)) { return result; }
        scale = amrex::max(scale, std::abs(value));
    }
    if (scale == 0) { result.valid = true; return result; }
    FourVector initial{}, rest_old{};
    for (int d = 0; d < 4; ++d) { initial[d] = old[d] / scale; }
    if (!gray_absorption_detail::Solve(moment, initial, rest_old) ||
        !gray_absorption_detail::PositiveDipole(rest_old)) { return result; }
    if (rate == 0) {
        for (int d = 0; d < 4; ++d) { result.rest_radiation[d] = scale * rest_old[d]; }
        result.valid = true;
        return result;
    }
    auto const inverse = 1 / (1 + rate), fraction = rate * inverse;
    auto const equilibrium = bath / scale;
    StressEnergy matrix{};
    FourVector rhs{}, delta_rhs{}, rest{}, delta{};
    for (int row = 0; row < 4; ++row) {
        rhs[row] = inverse * initial[row] + fraction * source[row][0] * equilibrium;
        for (int column = 0; column < 4; ++column) {
            matrix[row][column] = inverse * moment[row][column] + fraction * source[row][column];
            delta_rhs[row] -= fraction * source[row][column] *
                (rest_old[column] - (column == 0 ? equilibrium : 0));
        }
    }
    if (!gray_absorption_detail::Solve(matrix, rhs, rest) ||
        !gray_absorption_detail::Solve(matrix, delta_rhs, delta) ||
        !gray_absorption_detail::PositiveDipole(rest)) { return result; }
    auto const epsilon = std::numeric_limits<amrex::Real>::epsilon();
    auto const gamma = 96 * epsilon / (1 - 96 * epsilon);
    for (int row = 0; row < 4; ++row) {
        amrex::Real next = 0, transfer = 0, storage_scale = 0, residual = 0, equation_scale = 0;
        for (int column = 0; column < 4; ++column) {
            next += moment[row][column] * rest[column];
            transfer -= moment[row][column] * delta[column];
            storage_scale += std::abs(moment[row][column]) *
                (std::abs(rest_old[column]) + std::abs(rest[column]) + std::abs(delta[column]));
            auto const change = inverse * moment[row][column] * delta[column];
            auto const sink = fraction * source[row][column] *
                (rest[column] - (column == 0 ? equilibrium : 0));
            residual += change + sink;
            equation_scale += std::abs(change) + fraction * std::abs(source[row][column]) *
                (std::abs(rest[column]) + (column == 0 ? equilibrium : 0));
        }
        result.radiation[row] = scale * next;
        result.rest_radiation[row] = scale * rest[row];
        result.physical_transfer[row] = scale * transfer;
        result.raw_transfer[row] = old[row] - result.radiation[row];
        result.storage_residual[row] = result.raw_transfer[row] - result.physical_transfer[row];
        result.storage_bound[row] = gamma * scale * (storage_scale + std::abs(initial[row]));
        result.equation_residual[row] = scale * residual;
        result.equation_bound[row] = gamma * scale * equation_scale;
        if (!amrex::Math::isfinite(result.radiation[row]) ||
            !amrex::Math::isfinite(result.physical_transfer[row]) ||
            !amrex::Math::isfinite(result.storage_bound[row] + result.equation_bound[row]) ||
            std::abs(result.storage_residual[row]) > result.storage_bound[row] ||
            std::abs(result.equation_residual[row]) > result.equation_bound[row]) { return result; }
    }
    amrex::Real projection = delta[0];
    for (int d = 0; d < 3; ++d) { projection += beta[d] * delta[d + 1]; }
    result.physical_energy_minus_work = -scale * projection;
    result.valid = amrex::Math::isfinite(result.physical_energy_minus_work);
    return result;
}
} // namespace warpx::radiation

#endif
