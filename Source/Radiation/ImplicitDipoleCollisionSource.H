/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_IMPLICITDIPOLECOLLISIONSOURCE_H_
#define WARPX_RADIATION_IMPLICITDIPOLECOLLISIONSOURCE_H_

#include "PureGrayAbsorptionSource.H"

namespace warpx::radiation
{
/** Fixed-frame backward-Euler gray absorption plus coherent scattering.
 * Rates are c*dt*chi_a and c*dt*(chi_a+chi_s), not inverse lengths.
 * The closure is COMOVING P0=E0 I/3 with a positive angular dipole, NOT M1.
 * This local source retains temporal flux relaxation; it supplies no spatial
 * transport, native material/EOS iteration or dynamic-diffusion authority.
 *
 * Coordinates are (h=E0+beta.q0,q0), so h is LAB E-beta.q. The projected
 * energy source contains ONLY absorption: chi_a*(E0-B)/gamma. Normalize that
 * row by its own absorption rate and the momentum rows by the total rate.
 * Neither a stiff scattering cancellation nor chi_a/chi_t underflow may
 * erase a small caloric transfer. Solve stored state and increment separately.
 * Optional transport is dt*div(LAB flux) in the same extensive units. It is
 * a signed implicit forcing, not a material source. The returned physical/raw
 * transfers exclude it; only the incoming and jointly solved states must be
 * in the positive dipole cone. No spatial operator is evaluated here.
 * The existing pure-absorption entry and all its operation order are unchanged.
 */
AMREX_GPU_HOST_DEVICE inline PureGrayAbsorptionResult
TryImplicitDipoleCollisionSource (
    FourVector const& old, amrex::GpuArray<amrex::Real, 3> const& beta,
    amrex::Real absorption_rate, amrex::Real total_rate, amrex::Real bath,
    FourVector const& transport = {})
{
    PureGrayAbsorptionResult result;
    result.radiation = old;
    StressEnergy moment{}, source{};
    if (!BuildIsotropicDipoleFrameMatrices(beta, moment, source) ||
        absorption_rate < 0 || total_rate < absorption_rate || bath < 0 ||
        !amrex::Math::isfinite(absorption_rate + bath) ||
        !amrex::Math::isfinite(total_rate))
    {
        return result;
    }
    amrex::Real beta2 = 0;
    for (auto value : beta) { beta2 += value * value; }
    if (beta2 > amrex::Real(1.e-4)) { return result; }
    amrex::Real scale = absorption_rate > 0 ? bath : 0;
    for (auto value : old)
    {
        if (!amrex::Math::isfinite(value)) { return result; }
        scale = amrex::max(scale, std::abs(value));
    }
    for (auto value : transport)
    {
        if (!amrex::Math::isfinite(value)) { return result; }
        scale = amrex::max(scale, std::abs(value));
    }
    if (scale == 0) { result.valid = true; return result; }
    FourVector initial{}, rest_old{};
    for (int d = 0; d < 4; ++d) { initial[d] = old[d] / scale; }
    if (!gray_absorption_detail::Solve(moment, initial, rest_old) ||
        !gray_absorption_detail::PositiveDipole(rest_old))
    {
        return result;
    }
    // The incoming state must be physical. The implicit spatial forcing need
    // not be: pressure transport before a stiff collision can lie outside
    // the dipole cone even when the jointly solved state is admissible.
    for (int d = 0; d < 4; ++d) { initial[d] -= transport[d] / scale; }
    if (!gray_absorption_detail::Solve(moment, initial, rest_old)) { return result; }
    if (total_rate == 0)
    {
        if (!gray_absorption_detail::PositiveDipole(rest_old)) { return result; }
        for (int d = 0; d < 4; ++d)
        {
            result.radiation[d] = old[d] - transport[d];
            result.rest_radiation[d] = scale * rest_old[d];
        }
        result.valid = true;
        return result;
    }
    auto const inverse_gamma = std::sqrt(1 - beta2);
    auto const inverse_a = 1 / (1 + absorption_rate);
    auto const fraction_a = absorption_rate * inverse_a;
    auto const inverse_t = 1 / (1 + total_rate);
    auto const fraction_t = total_rate * inverse_t;
    auto const a_in_t = absorption_rate * inverse_t;
    auto const equilibrium = absorption_rate > 0 ? bath / scale : 0;
    FourVector coordinates_old = rest_old;
    for (int d = 0; d < 3; ++d) { coordinates_old[0] += beta[d] * rest_old[d + 1]; }
    StressEnergy coordinate_moment = moment;
    for (int row = 0; row < 4; ++row)
    {
        for (int d = 0; d < 3; ++d)
        {
            coordinate_moment[row][d + 1] -= beta[d] * moment[row][0];
        }
    }
    StressEnergy matrix{};
    FourVector rhs{}, delta_rhs{}, coordinates{}, delta{};
    matrix[0][0] = inverse_a + fraction_a * inverse_gamma;
    rhs[0] = inverse_a * coordinates_old[0] + fraction_a * inverse_gamma * equilibrium;
    delta_rhs[0] = -fraction_a * inverse_gamma * (rest_old[0] - equilibrium);
    for (int d = 0; d < 3; ++d)
    {
        matrix[0][d + 1] = -fraction_a * inverse_gamma * beta[d];
    }
    for (int row = 1; row < 4; ++row)
    {
        matrix[row][0] = inverse_t * moment[row][0] + a_in_t * source[row][0];
        rhs[row] = inverse_t * initial[row] + a_in_t * source[row][0] * equilibrium;
        delta_rhs[row] = -a_in_t * source[row][0] * (rest_old[0] - equilibrium);
        for (int d = 0; d < 3; ++d)
        {
            matrix[row][d + 1] = inverse_t * coordinate_moment[row][d + 1] +
                fraction_t * source[row][d + 1] - a_in_t * source[row][0] * beta[d];
            delta_rhs[row] -= fraction_t * source[row][d + 1] * rest_old[d + 1];
        }
    }
    StressEnergy state_matrix = matrix;
    if (absorption_rate == 0)
    {
        // Eliminate the exactly invariant h row before solving the state.
        // Forming rhs_i - M_i0*h/(1+t) subtracts the large boosted LTE
        // current and loses a weak proper flux after repeated relaxation.
        // The independent collision-increment solve remains unchanged.
        for (int row = 1; row < 4; ++row)
        {
            state_matrix[row][0] = 0;
            rhs[row] = 0;
            for (int d = 0; d < 3; ++d)
            {
                rhs[row] += inverse_t * coordinate_moment[row][d + 1] * rest_old[d + 1];
            }
        }
    }
    if (!gray_absorption_detail::Solve(state_matrix, rhs, coordinates) ||
        !gray_absorption_detail::Solve(matrix, delta_rhs, delta))
    {
        return result;
    }
    FourVector rest = coordinates;
    for (int d = 0; d < 3; ++d) { rest[0] -= beta[d] * coordinates[d + 1]; }
    if (!gray_absorption_detail::PositiveDipole(rest)) { return result; }
    auto const epsilon = std::numeric_limits<amrex::Real>::epsilon();
    auto const gamma = 256 * epsilon / (1 - 256 * epsilon);
    for (int row = 0; row < 4; ++row)
    {
        amrex::Real next = 0, transfer = 0, storage_scale = 0;
        for (int column = 0; column < 4; ++column)
        {
            auto const coefficient = coordinate_moment[row][column];
            next += coefficient * coordinates[column];
            transfer -= coefficient * delta[column];
            storage_scale += std::abs(coefficient) *
                (std::abs(coordinates_old[column]) + std::abs(coordinates[column]) +
                 std::abs(delta[column]));
        }
        amrex::Real residual = 0, equation_scale = 0;
        if (row == 0)
        {
            auto const change = inverse_a * delta[0];
            auto const source_value = fraction_a * inverse_gamma * (rest[0] - equilibrium);
            residual = change + source_value;
            equation_scale = std::abs(change) + fraction_a * inverse_gamma *
                (std::abs(rest[0]) + std::abs(equilibrium));
        }
        else
        {
            for (int column = 0; column < 4; ++column)
            {
                auto const change = inverse_t * coordinate_moment[row][column] * delta[column];
                auto const rate = column == 0 ? a_in_t : fraction_t;
                auto const sink = rate * source[row][column] *
                    (rest[column] - (column == 0 ? equilibrium : 0));
                residual += change + sink;
                equation_scale += std::abs(change) + rate * std::abs(source[row][column]) *
                    (std::abs(rest[column]) + (column == 0 ? std::abs(equilibrium) : 0));
            }
        }
        result.radiation[row] = scale * next;
        result.rest_radiation[row] = scale * rest[row];
        result.physical_transfer[row] = scale * transfer;
        result.raw_transfer[row] = (old[row] - transport[row]) - result.radiation[row];
        result.storage_residual[row] = result.raw_transfer[row] - result.physical_transfer[row];
        result.storage_bound[row] = gamma * scale * (storage_scale + std::abs(initial[row]));
        result.equation_residual[row] = scale * residual;
        result.equation_bound[row] = gamma * scale * equation_scale;
        if (!amrex::Math::isfinite(result.radiation[row]) ||
            !amrex::Math::isfinite(result.physical_transfer[row]) ||
            !amrex::Math::isfinite(result.storage_bound[row] + result.equation_bound[row]) ||
            std::abs(result.storage_residual[row]) > result.storage_bound[row] ||
            std::abs(result.equation_residual[row]) > result.equation_bound[row])
        {
            return result;
        }
    }
    result.physical_energy_minus_work = -scale * delta[0];
    result.valid = amrex::Math::isfinite(result.physical_energy_minus_work);
    return result;
}
} // namespace warpx::radiation
#endif
