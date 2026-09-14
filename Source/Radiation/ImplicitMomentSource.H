/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_IMPLICITMOMENTSOURCE_H_
#define WARPX_RADIATION_IMPLICITMOMENTSOURCE_H_

#include "MomentClosure.H"

#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <limits>

namespace warpx::radiation
{
    enum class MomentSourceFailure { None, InvalidInput, LinearSolve, LineSearch, IterationBudget };

    struct ImplicitMomentSourceResult {
        FourVector radiation{};
        // Material energy and c*momentum gained per volume; divide spatial
        // entries by c to obtain physical material impulse per volume.
        FourVector material_transfer{};
        // old radiation - new radiation - material_transfer, evaluated with
        // compensated arithmetic. This is a signed diagnostic, not clipping.
        FourVector numerical_residual{};
        // Integrated material energy minus beta dot c*impulse, evaluated
        // directly rather than subtracting a large mechanical work afterward.
        amrex::Real material_energy_minus_work = 0;
        // material_transfer[0] - material_energy_minus_work - beta dot
        // material_transfer[1:4], including product/summation roundoff.
        amrex::Real energy_projection_residual = 0;
        // Row zero is the energy-minus-beta*momentum equation; rows 1..3
        // are lab momentum equations. This equivalent basis avoids the large
        // scattering terms cancelling numerically in the energy equation.
        FourVector equation_residual{};
        FourVector roundoff_bound{};
        MomentSourceFailure failure = MomentSourceFailure::InvalidInput;
        int iterations = 0;
        bool valid = false;
    };

    namespace detail
    {
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
        SolveMomentLinearSystem (StressEnergy matrix, FourVector rhs, FourVector& solution) noexcept
        {
            // Row scaling and partial pivoting; no absolute pivot tolerance
            // that would reject an otherwise representable stiff coefficient.
            for (int i = 0; i < 4; ++i) {
                amrex::Real scale = 0;
                for (int j = 0; j < 4; ++j) {
                    scale = amrex::max(scale, std::abs(matrix[i][j]));
                }
                if (!(scale > 0) || !amrex::Math::isfinite(scale)) {
                    return false;
                }
                for (int j = 0; j < 4; ++j) {
                    matrix[i][j] /= scale;
                }
                rhs[i] /= scale;
            }
            for (int k = 0; k < 4; ++k) {
                int pivot = k;
                for (int i = k + 1; i < 4; ++i) {
                    if (std::abs(matrix[i][k]) > std::abs(matrix[pivot][k])) {
                        pivot = i;
                    }
                }
                if (matrix[pivot][k] == 0) {
                    return false;
                }
                auto const row = matrix[k];
                matrix[k] = matrix[pivot];
                matrix[pivot] = row;
                auto const value = rhs[k];
                rhs[k] = rhs[pivot];
                rhs[pivot] = value;
                for (int i = k + 1; i < 4; ++i) {
                    amrex::Real const factor = matrix[i][k] / matrix[k][k];
                    for (int j = k + 1; j < 4; ++j) {
                        matrix[i][j] -= factor * matrix[k][j];
                    }
                    rhs[i] -= factor * rhs[k];
                }
            }
            for (int i = 3; i >= 0; --i) {
                auto value = rhs[i];
                for (int j = i + 1; j < 4; ++j) {
                    value -= matrix[i][j] * solution[j];
                }
                solution[i] = value / matrix[i][i];
                if (!amrex::Math::isfinite(solution[i])) {
                    return false;
                }
            }
            return true;
        }

        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
        MomentExcess (FourVector const& value, FourVector const& bounds) noexcept
        {
            amrex::Real norm = 0;
            for (int d = 0; d < 4; ++d) {
                norm = amrex::max(norm, std::abs(value[d]) - bounds[d]);
            }
            return norm;
        }
    } // namespace detail

    /** Internal driven backward-Euler gray M1 source. The algebraic RHS may
     * be outside the realizability cone after subtracting a transport flux;
     * initial_guess and every physical Newton trial must remain realizable.
     * Public source-only calls retain their physical-input guard below.
     *
     * Backward-Euler gray M1 source with prescribed material velocity and
     * temperature. Rates are c*alpha*dt (dimensionless), not opacities alone.
     * Uses physical c in the caller's conversion of energy and momentum.
     * Native finite-mass/thermal response must be supplied by an outer coupled
     * iteration; this function alone is not radiation hydrodynamics.
     *
     * Solve for the increment as well as the stored state, retaining a source
     * below one ULP of a large radiation background. Independent state updates
     * also avoid loss of a stiffly attenuated beam to old+increment cancellation.
     * Failure returns unchanged initial guess and zero accepted transfers.
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE ImplicitMomentSourceResult
    TryDrivenImplicitGreyMomentSource (FourVector const& old_radiation,
                                       FourVector const& initial_guess,
                                       amrex::GpuArray<amrex::Real, 3> const& beta,
                                       amrex::Real absorption_dt, amrex::Real scattering_dt,
                                       amrex::Real equilibrium, int max_iterations = 60,
                                       amrex::Real tolerance = amrex::Real(1.e-12)) noexcept
    {
        ImplicitMomentSourceResult result;
        result.radiation = initial_guess;
        auto const old_closure = EvaluateM1Closure(initial_guess);
        if (!old_closure.valid || max_iterations < 1 ||
            !amrex::Math::isfinite(tolerance) || tolerance <= 0 || tolerance >= 1 ||
            !EvaluateGreyFourForce(old_closure.tensor, beta, absorption_dt, scattering_dt,
                                   equilibrium)
                 .valid) {
            return result;
        }
        amrex::Real scale = amrex::max(initial_guess[0], equilibrium);
        for (auto value : old_radiation) {
            if (!amrex::Math::isfinite(value)) {
                return result;
            }
            scale = amrex::max(scale, std::abs(value));
        }
        if (scale == 0) {
            result.valid = true;
            result.failure = MomentSourceFailure::None;
            return result;
        }
        // Increment coordinates are (delta E - beta dot delta q, delta q).
        // Store the first independently so weak caloric exchange is not lost
        // when the mechanical work is much larger.
        FourVector old{}, state{}, increment{};
        bool driven_guess = false;
        for (int d = 0; d < 4; ++d) {
            driven_guess = driven_guess || initial_guess[d] != old_radiation[d];
            old[d] = old_radiation[d] / scale;
            state[d] = initial_guess[d] / scale;
            increment[d] = state[d] - old[d];
        }
        for (int d = 0; d < 3; ++d) {
            increment[0] -= beta[d] * increment[d + 1];
        }
        amrex::Real const bath = equilibrium / scale;
        amrex::GpuArray<amrex::Real, 3> inverse{};
        amrex::Real b2 = 0;
        for (int d = 0; d < 3; ++d) {
            inverse[d] = -beta[d];
            b2 += beta[d] * beta[d];
        }
        auto const inverse_gamma = std::sqrt(1 - b2);
        if (!driven_guess && scattering_dt == 0 && equilibrium == 0 && old[0] > 0) {
            amrex::Real flux_norm2 = 0, beta_direction = 0;
            for (int d = 0; d < 3; ++d) {
                auto const direction = old[d + 1] / old[0];
                flux_norm2 += direction * direction;
                beta_direction += beta[d] * direction;
            }
            if (std::abs(flux_norm2 - 1) <= 8 * std::numeric_limits<amrex::Real>::epsilon()) {
                // Pure absorption preserves a beam's direction exactly. Start
                // on this invariant ray instead of taking a rounded Newton
                // step across the realizability boundary at extreme stiffness.
                // This is an initial guess, still subject to every equation gate.
                auto const rate = absorption_dt / inverse_gamma * (1 - beta_direction);
                if (amrex::Math::isfinite(rate)) {
                    auto const attenuation = 1 / (1 + rate);
                    auto const transfer = rate / (1 + rate);
                    for (int d = 0; d < 4; ++d) {
                        state[d] = old[d] * attenuation;
                        increment[d] = -old[d] * transfer;
                    }
                    for (int d = 0; d < 3; ++d) {
                        increment[0] -= beta[d] * increment[d + 1];
                    }
                }
            }
        }
        auto emission = GreyFourForceFromComoving({}, beta, absorption_dt, scattering_dt, bath);
        emission[0] = -absorption_dt * inverse_gamma * bath;
        constexpr amrex::Real epsilon = std::numeric_limits<amrex::Real>::epsilon();
        bool refined_driven_increment = false;
        for (int iteration = 0; iteration < max_iterations; ++iteration) {
            result.iterations = iteration + 1;
            auto const closure = EvaluateM1Closure(state);
            auto const force =
                EvaluateGreyFourForce(closure.tensor, beta, absorption_dt, scattering_dt, bath);
            if (!closure.valid || !force.valid) {
                return result;
            }
            auto const energy_minus_work =
                GreyEnergyMinusWork(closure.tensor, beta, absorption_dt, bath);
            StressEnergy source_jacobian{}, jacobian{};
            for (int column = 0; column < 4; ++column) {
                auto const derivative =
                    BoostStressEnergy(M1ClosureDerivative(state, column), inverse);
                auto const tangent =
                    GreyFourForceFromComoving(derivative, beta, absorption_dt, scattering_dt, 0);
                for (int row = 0; row < 4; ++row) {
                    source_jacobian[row][column] =
                        row == 0 ? absorption_dt * inverse_gamma * derivative[0][0] : tangent[row];
                }
            }
            for (int row = 0; row < 4; ++row) {
                for (int column = 0; column < 4; ++column) {
                    jacobian[row][column] =
                        source_jacobian[row][column] +
                        (column == 0 ? 0 : beta[column - 1] * source_jacobian[row][0]) +
                        (row == column ? 1 : 0);
                }
            }
            FourVector residual{}, noise{}, bounds{};
            bool converged = true;
            for (int i = 0; i < 4; ++i) {
                auto const increment_component = increment[i];
                auto const source_component = i == 0 ? energy_minus_work : force.material_force[i];
                residual[i] = increment_component + source_component;
                amrex::Real arithmetic_scale = std::abs(increment[i]) + std::abs(emission[i]);
                for (int j = 0; j < 4; ++j) {
                    arithmetic_scale += std::abs(source_jacobian[i][j] * state[j]);
                }
                noise[i] = 16 * epsilon * arithmetic_scale;
                result.equation_residual[i] = scale * residual[i];
                result.roundoff_bound[i] = scale * noise[i];
                auto const source_scale =
                    amrex::max(std::abs(increment_component), std::abs(source_component));
                bounds[i] = tolerance * source_scale + noise[i];
                converged = converged && amrex::Math::isfinite(residual[i]) &&
                            amrex::Math::isfinite(noise[i]) && std::abs(residual[i]) <= bounds[i];
            }
            // A spatial solver's already rounded state can satisfy the stiff
            // equation's arithmetic bound while its inferred source increment
            // still contains stiffness-amplified storage noise. Refine that
            // increment once; retain its sub-ULP correction independently of
            // the stored state, just as for weak source-only exchange.
            bool const refine_driven_increment = converged && !refined_driven_increment &&
                                                 driven_guess && iteration + 1 < max_iterations &&
                                                 (absorption_dt > 0 || scattering_dt > 0);
            if (converged && !refine_driven_increment) {
                auto candidate = result;
                candidate.material_energy_minus_work = -scale * increment[0];
                if (!amrex::Math::isfinite(candidate.material_energy_minus_work)) {
                    return result;
                }
                amrex::Real material_energy = candidate.material_energy_minus_work;
                amrex::Real projection_error = 0;
                for (int d = 1; d < 4; ++d) {
                    candidate.material_transfer[d] = -scale * increment[d];
                    auto const term = beta[d - 1] * candidate.material_transfer[d];
                    auto const sum = material_energy + term;
                    auto const recovered = sum - material_energy;
                    projection_error +=
                        (material_energy - (sum - recovered)) + (term - recovered) +
                        std::fma(beta[d - 1], candidate.material_transfer[d], -term);
                    material_energy = sum;
                }
                candidate.material_transfer[0] = material_energy;
                candidate.energy_projection_residual = -projection_error;
                if (!amrex::Math::isfinite(candidate.energy_projection_residual)) {
                    return result;
                }
                for (int d = 0; d < 4; ++d) {
                    candidate.radiation[d] =
                        candidate.material_transfer[d] == 0 && state[d] == old[d]
                            ? old_radiation[d]
                            : scale * state[d];
                    // Error-free two-sum of old + accepted increment, then
                    // compare to the independently stored radiation state.
                    auto const delta = -candidate.material_transfer[d];
                    auto const sum = old_radiation[d] + delta;
                    auto const recovered = sum - old_radiation[d];
                    auto const error = (old_radiation[d] - (sum - recovered)) + (delta - recovered);
                    candidate.numerical_residual[d] = (sum - candidate.radiation[d]) + error;
                    candidate.equation_residual[d] = scale * residual[d];
                    candidate.roundoff_bound[d] = scale * noise[d];
                    if (!amrex::Math::isfinite(candidate.radiation[d]) ||
                        !amrex::Math::isfinite(candidate.material_transfer[d]) ||
                        !amrex::Math::isfinite(candidate.numerical_residual[d]) ||
                        !amrex::Math::isfinite(candidate.equation_residual[d]) ||
                        !amrex::Math::isfinite(candidate.roundoff_bound[d])) {
                        return result;
                    }
                }
                candidate.failure = MomentSourceFailure::None;
                candidate.valid = true;
                return candidate;
            }
            FourVector direction{}, rhs{};
            for (int d = 0; d < 4; ++d) {
                rhs[d] = -residual[d];
            }
            if (!detail::SolveMomentLinearSystem(jacobian, rhs, direction)) {
                result.failure = MomentSourceFailure::LinearSolve;
                return result;
            }
            amrex::Real fraction = 1;
            bool accepted = false;
            for (int search = 0; search < 24; ++search) {
                FourVector trial{}, trial_increment{}, trial_residual{};
                for (int d = 0; d < 4; ++d) {
                    trial[d] = state[d] + fraction * direction[d];
                    trial_increment[d] = increment[d] + fraction * direction[d];
                }
                amrex::Real energy_direction = direction[0];
                for (int d = 0; d < 3; ++d) {
                    energy_direction += beta[d] * direction[d + 1];
                }
                trial[0] = state[0] + fraction * energy_direction;
                auto const trial_closure = EvaluateM1Closure(trial);
                auto const trial_force = EvaluateGreyFourForce(trial_closure.tensor, beta,
                                                               absorption_dt, scattering_dt, bath);
                if (trial_closure.valid && trial_force.valid) {
                    for (int d = 0; d < 4; ++d) {
                        trial_residual[d] = trial_increment[d] + trial_force.material_force[d];
                    }
                    trial_residual[0] = trial_increment[0];
                    trial_residual[0] +=
                        GreyEnergyMinusWork(trial_closure.tensor, beta, absorption_dt, bath);
                    // Keep this iteration's physical/roundoff thresholds fixed
                    // during line search. Noise in an already resolved stiff
                    // component must not block improvement of other components.
                    // The accepted state recomputes all bounds next iteration.
                    if ((refine_driven_increment &&
                         detail::MomentExcess(trial_residual, bounds) == 0) ||
                        detail::MomentExcess(trial_residual, bounds) <
                            detail::MomentExcess(residual, bounds)) {
                        state = trial;
                        increment = trial_increment;
                        refined_driven_increment =
                            refined_driven_increment || refine_driven_increment;
                        accepted = true;
                        break;
                    }
                }
                fraction *= amrex::Real(0.5);
            }
            if (!accepted) {
                result.failure = MomentSourceFailure::LineSearch;
                return result;
            }
        }
        result.failure = MomentSourceFailure::IterationBudget;
        return result;
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE ImplicitMomentSourceResult
    TryImplicitGreyMomentSource (FourVector const& old_radiation,
                                 amrex::GpuArray<amrex::Real, 3> const& beta,
                                 amrex::Real absorption_dt, amrex::Real scattering_dt,
                                 amrex::Real equilibrium, int max_iterations = 60,
                                 amrex::Real tolerance = amrex::Real(1.e-12)) noexcept
    {
        // Preserve the original source-only API's physical input requirement.
        if (!EvaluateM1Closure(old_radiation).valid) {
            ImplicitMomentSourceResult result;
            result.radiation = old_radiation;
            return result;
        }
        return TryDrivenImplicitGreyMomentSource(old_radiation, old_radiation, beta, absorption_dt,
                                                 scattering_dt, equilibrium, max_iterations,
                                                 tolerance);
    }

    /** Finalize a joint transport/source iterate while retaining independently
     * represented weak heat and impulse. transport_increment is dt*div(flux),
     * not a physical intermediate radiation state. Both the original radiation
     * and the supplied Newton guess must be physically realizable. */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE ImplicitMomentSourceResult
    TryImplicitGreyMomentSourceWithTransport (FourVector const& old_radiation,
                                              FourVector const& transport_increment,
                                              FourVector const& initial_guess,
                                              amrex::GpuArray<amrex::Real, 3> const& beta,
                                              amrex::Real absorption_dt, amrex::Real scattering_dt,
                                              amrex::Real equilibrium, int max_iterations = 60,
                                              amrex::Real tolerance = amrex::Real(1.e-12)) noexcept
    {
        if (!EvaluateM1Closure(old_radiation).valid) {
            ImplicitMomentSourceResult result;
            result.radiation = initial_guess;
            return result;
        }
        FourVector rhs{};
        for (int d = 0; d < 4; ++d) {
            rhs[d] = old_radiation[d] - transport_increment[d];
        }
        return TryDrivenImplicitGreyMomentSource(rhs, initial_guess, beta, absorption_dt,
                                                 scattering_dt, equilibrium, max_iterations,
                                                 tolerance);
    }
} // namespace warpx::radiation
#endif
