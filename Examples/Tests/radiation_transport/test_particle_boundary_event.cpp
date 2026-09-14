/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "Particles/ParticleBoundaries_K.H"
#include "Radiation/MaterialKineticWork.H"

#include <AMReX.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_Print.H>

#include <array>
#include <cmath>

namespace
{
    struct Result {
        ApplyParticleBoundaries::BoundaryEvent event;
        amrex::GpuArray<amrex::ParticleReal, 3> position, velocity, legacy_velocity;
        warpx::radiation::MaterialCarryReflection carry;
        amrex::Real pending_work_before = 0, pending_work_after = 0;
        bool legacy_lost = false;
    };
} // namespace

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        constexpr int count = 9;
#if defined(WARPX_DIM_1D_Z)
        constexpr int axis = 2;
#else
        constexpr int axis = 0;
#endif
        std::array<ParticleBoundaries, count> settings;
        amrex::Gpu::HostVector<ParticleBoundaries::ParticleBoundariesData> host_settings(count);
        for (int i = 0; i < count; ++i) {
            settings[i].SetAll(i == 4             ? ParticleBoundaryType::Open
                               : i == 5 || i == 6 ? ParticleBoundaryType::Absorbing
                               : i >= 7           ? ParticleBoundaryType::Thermal
                                                  : ParticleBoundaryType::Reflecting);
            settings[i].Set_reflect_all_velocities(i == 3);
            settings[i].SetThermalVelocity(i == 8 ? 1.e-4 : 0);
            if (i == 6) {
                settings[i].reflection_model_xlo_str = "1";
                settings[i].reflection_model_zlo_str = "1";
            }
            settings[i].BuildReflectionModelParsers();
            host_settings[i] = settings[i].data;
        }
        amrex::Gpu::DeviceVector<ParticleBoundaries::ParticleBoundariesData> device_settings(count);
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, host_settings.begin(), host_settings.end(),
                         device_settings.begin());
        amrex::Gpu::DeviceVector<Result> device_results(count);
        auto const* boundaries = device_settings.data();
        auto* results = device_results.data();
        amrex::ParallelForRNG(
            count, [=] AMREX_GPU_DEVICE(int i, amrex::RandomEngine const& engine) {
                amrex::GpuArray<amrex::ParticleReal, 3> position{0.5, 0.5, 0.5};
                position[axis] = i == 0 ? 0.5 : (i == 2 ? 1.125 : -0.125);
                amrex::GpuArray<amrex::ParticleReal, 3> velocity{1, 2, 3};
                if (i == 1 || i == 3 || i == 6) {
                    velocity[axis] =
                        0; // A reflected zero velocity cannot reveal the event by sign tests.
                }
                auto legacy_position = position;
                auto legacy_velocity = velocity;
                amrex::GpuArray<amrex::Real, 3> before{velocity[0], velocity[1], velocity[2]};
                bool legacy_lost = false, lost = false;
                ApplyParticleBoundaries::apply_boundaries(
                    legacy_position[0], legacy_position[1], legacy_position[2], {0, 0, 0},
                    {1, 1, 1}, legacy_velocity[0], legacy_velocity[1], legacy_velocity[2],
                    legacy_lost, boundaries[i], engine);
                ApplyParticleBoundaries::BoundaryEvent event{{true, true, true}, true, true};
                ApplyParticleBoundaries::apply_boundaries(
                    position[0], position[1], position[2], {0, 0, 0}, {1, 1, 1}, velocity[0],
                    velocity[1], velocity[2], lost, boundaries[i], engine, &event);
                amrex::GpuArray<amrex::Real, 4> const pending{1.e-30, -2.e-30, 3.e-30, -4.e-30};
                auto const carry = warpx::radiation::EvaluateMaterialCarryReflection(
                    pending, event.coordinate_reflection, event.thermalized, event.lost);
                amrex::GpuArray<amrex::Real, 3> const old_delta{pending[0], pending[1], pending[2]};
                amrex::GpuArray<amrex::Real, 3> const new_delta{carry.carry[0], carry.carry[1],
                                                                carry.carry[2]};
                amrex::GpuArray<amrex::Real, 3> const after{velocity[0], velocity[1], velocity[2]};
                results[i] = {event,
                              position,
                              velocity,
                              legacy_velocity,
                              carry,
                              warpx::radiation::MaterialSpecificRequestedWork(before, old_delta),
                              warpx::radiation::MaterialSpecificRequestedWork(after, new_delta),
                              legacy_lost};
            });
        amrex::Gpu::HostVector<Result> results_host(count);
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, device_results.begin(), device_results.end(),
                         results_host.begin());
        for (int i = 0; i < count; ++i) {
            auto const& result = results_host[i];
            bool const reflected = i == 1 || i == 2 || i == 3 || i == 6;
            AMREX_ALWAYS_ASSERT(result.event.lost == (i == 4 || i == 5));
            AMREX_ALWAYS_ASSERT(result.event.lost == result.legacy_lost);
            AMREX_ALWAYS_ASSERT(result.event.thermalized == (i >= 7));
            bool const elastic = !result.event.lost && !result.event.thermalized;
            AMREX_ALWAYS_ASSERT(result.carry.valid == elastic);
            if (elastic) {
                amrex::GpuArray<amrex::Real, 4> const pending{1.e-30, -2.e-30, 3.e-30, -4.e-30};
                for (int d = 0; d < 4; ++d) {
                    AMREX_ALWAYS_ASSERT(result.carry.carry[d] + result.carry.boundary_transfer[d] ==
                                        pending[d]);
                }
                AMREX_ALWAYS_ASSERT(result.carry.carry[3] == pending[3]);
                AMREX_ALWAYS_ASSERT(result.carry.boundary_transfer[3] == 0);
                AMREX_ALWAYS_ASSERT(result.pending_work_before == result.pending_work_after);
                if (reflected) {
                    AMREX_ALWAYS_ASSERT(result.carry.boundary_transfer[axis] != 0);
                }
            }
            for (int d = 0; d < 3; ++d) {
                AMREX_ALWAYS_ASSERT(result.event.coordinate_reflection[d] ==
                                    (reflected && (d == axis || i == 3)));
                AMREX_ALWAYS_ASSERT(std::isfinite(result.velocity[d]));
                if (i < 8) {
                    AMREX_ALWAYS_ASSERT(result.velocity[d] == result.legacy_velocity[d]);
                    AMREX_ALWAYS_ASSERT(std::signbit(result.velocity[d]) ==
                                        std::signbit(result.legacy_velocity[d]));
                }
            }
            auto const expected = i == 0              ? 0.5
                                  : i == 2            ? 0.875
                                  : result.event.lost ? -0.125
                                                      : 0.125;
            AMREX_ALWAYS_ASSERT(result.position[axis] == expected);
        }
        amrex::Print() << "Native particle boundary event decisions and legacy behavior pass\n";
    }
    amrex::Finalize();
}
