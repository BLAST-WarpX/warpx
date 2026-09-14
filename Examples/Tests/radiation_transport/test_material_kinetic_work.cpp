/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "Radiation/MaterialKineticWork.H"

#include <AMReX.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <cmath>
#include <limits>

using namespace amrex::literals;
using Velocity = amrex::GpuArray<amrex::Real, 3>;

struct Kick
{
    Velocity before;
    Velocity after;
};

AMREX_GPU_HOST_DEVICE Kick
make_kick (int index)
{
    // Rest, slow material and relativistic drift; diagonal and transverse kicks.
    amrex::Real const speeds[] = {0, 1.e-4_rt, 1.e3_rt, 0.1_rt * PhysConst::c,
                                  3 * PhysConst::c};
    amrex::Real const sizes[] = {0, 1.e-16_rt, 1.e-12_rt, 1.e-4_rt, 0.1_rt, 2};
    amrex::Real const speed = speeds[index % 5];
    amrex::Real const increment = sizes[(index / 5) % 6] * amrex::max(speed, 1._rt);
    amrex::Real const sign = (index / 30) % 2 == 0 ? 1 : -1;
    Kick result{{speed, -0.3_rt * speed, 0.7_rt * speed}, {}};
    result.after = result.before;
    int const direction = (index / 60) % 3;
    result.after[direction] += sign * increment;
    return result;
}

AMREX_GPU_HOST_DEVICE amrex::Real
check_particle_carry (int index)
{
    using namespace warpx::radiation;
    amrex::Real const initial_speed = std::ldexp(1._rt, index % 20);
    int const direction = index % 3;
    amrex::GpuArray<amrex::ParticleReal, 3> velocity{};
    velocity[direction] = static_cast<amrex::ParticleReal>(initial_speed);
    Velocity initial{};
    initial[direction] = initial_speed;
    Velocity carry{};
    amrex::Real energy_carry = 0;
    amrex::Real requested = 0;
    amrex::Real absolute_impulse = 0;
    amrex::Real work = 0;
    amrex::Real absolute_work = 0;
    amrex::Real const epsilon = std::numeric_limits<amrex::ParticleReal>::epsilon();
    constexpr int steps = 4096;
    for (int step = 0; step < steps; ++step)
    {
        // Individual requests are below one velocity ULP. Half of the cases
        // reverse the force: use absolute transfers, not cancelling net work.
        amrex::Real const sign = index % 2 == 0 || step < steps / 2 ? 1 : -1;
        Velocity increment{};
        increment[direction] = sign * initial_speed * epsilon / 8;
        auto const candidate = EvaluateMaterialImpulseCandidate(
            velocity, carry, energy_carry, increment);
        if (!candidate.valid)
        {
            return 1;
        }
        requested += increment[direction];
        absolute_impulse += std::abs(increment[direction]);
        work += candidate.requested_work;
        absolute_work += std::abs(candidate.requested_work);
        velocity = candidate.velocity;
        carry = candidate.momentum_carry;
        energy_carry = candidate.energy_carry;
        if (std::abs(carry[direction]) > 2 * epsilon * initial_speed ||
            std::abs(energy_carry) > 8 * epsilon * initial_speed * initial_speed)
        {
            return 1;
        }
    }
    Velocity final{};
    for (int d = 0; d < 3; ++d)
    {
        final[d] = velocity[d];
    }
    amrex::Real const delta_k = MaterialSpecificKineticWork(initial, final);
    amrex::Real const impulse_error = std::abs(
        requested - ((final[direction] - initial_speed) + carry[direction])) / absolute_impulse;
    amrex::Real const energy_error = std::abs(work - (delta_k + energy_carry)) / absolute_work;
    return amrex::max(impulse_error, energy_error);
}

int
main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        constexpr int cases = 180;
        amrex::Gpu::DeviceVector<amrex::Real> device(cases);
        auto* output = device.dataPtr();
        amrex::ParallelFor(cases, [=] AMREX_GPU_DEVICE (int i)
        {
            auto const kick = make_kick(i);
            output[i] = warpx::radiation::MaterialSpecificKineticWork(kick.before, kick.after);
        });
        amrex::Gpu::HostVector<amrex::Real> host(cases);
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, device.begin(), device.end(), host.begin());
        long double worst = 0;
        for (int i = 0; i < cases; ++i)
        {
            auto const kick = make_kick(i);
            // Independent extended-precision difference-of-squares identity.
            long double old_squared = 0;
            long double new_squared = 0;
            long double numerator = 0;
            long double absolute_numerator = 0;
            for (int d = 0; d < 3; ++d)
            {
                long double const a = kick.before[d];
                long double const b = kick.after[d];
                old_squared += a * a;
                new_squared += b * b;
                numerator += (b - a) * (b + a);
                absolute_numerator += std::abs((b - a) * (b + a));
            }
            long double const c2 = PhysConst::c2;
            long double const denominator = std::sqrt(1 + old_squared / c2)
                + std::sqrt(1 + new_squared / c2);
            long double const reference = numerator / denominator;
            long double const scale = absolute_numerator / denominator;
            amrex::Real const cpu =
                warpx::radiation::MaterialSpecificKineticWork(kick.before, kick.after);
            amrex::Real const reverse =
                warpx::radiation::MaterialSpecificKineticWork(kick.after, kick.before);
            AMREX_ALWAYS_ASSERT(std::isfinite(cpu) && std::isfinite(host[i]));
            if (scale == 0)
            {
                AMREX_ALWAYS_ASSERT(cpu == 0 && host[i] == 0 && reverse == 0);
            }
            else
            {
                long double const error = std::max(
                    std::max(std::abs(cpu - reference), std::abs(host[i] - reference)),
                    std::abs(reverse + reference)) / scale;
                worst = std::max(worst, error);
            }
        }
        AMREX_ALWAYS_ASSERT(worst < 32 * std::numeric_limits<amrex::Real>::epsilon());
        // A finite transverse kick at rest does work even though old v dot dp=0.
        AMREX_ALWAYS_ASSERT(warpx::radiation::MaterialSpecificKineticWork(
            {0, 0, 0}, {1, 0, 0}) > 0);
        constexpr int carry_cases = 32;
        amrex::Gpu::DeviceVector<amrex::Real> device_carry(carry_cases);
        auto* carry_output = device_carry.dataPtr();
        amrex::ParallelFor(carry_cases, [=] AMREX_GPU_DEVICE (int i)
        {
            carry_output[i] = check_particle_carry(i);
        });
        amrex::Gpu::HostVector<amrex::Real> host_carry(carry_cases);
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, device_carry.begin(), device_carry.end(),
                         host_carry.begin());
        amrex::Real carry_worst = 0;
        for (int i = 0; i < carry_cases; ++i)
        {
            carry_worst = amrex::max(carry_worst,
                amrex::max(host_carry[i], check_particle_carry(i)));
        }
        AMREX_ALWAYS_ASSERT(carry_worst < 1.e-10_rt);
        AMREX_ALWAYS_ASSERT(!warpx::radiation::EvaluateMaterialImpulseCandidate(
            {0, 0, 0}, {0, 0, 0}, std::numeric_limits<amrex::Real>::infinity(),
            {0, 0, 0}).valid);
        amrex::Print() << "Actual finite-kick work: " << cases
                       << " cases; worst scaled error " << static_cast<double>(worst) << '\n';
        amrex::Print() << "Particle-owned candidate carry: " << carry_cases
                       << " sequences of 4096 kicks; worst balance " << carry_worst << '\n';
    }
    amrex::Finalize();
}
