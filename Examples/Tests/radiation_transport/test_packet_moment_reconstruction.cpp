/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "Radiation/PacketMomentReconstruction.H"
#include "Utils/WarpXConst.H"

#include <AMReX.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_Print.H>

#include <cmath>
#include <limits>

using namespace amrex::literals;

namespace
{
void
check (warpx::radiation::FourVector const& desired, amrex::Real photon_energy)
{
    amrex::Gpu::DeviceScalar<warpx::radiation::PacketMomentReconstructionResult> storage;
    auto* output = storage.dataPtr();
    amrex::ParallelFor(
        1, [=] AMREX_GPU_DEVICE(int)
        { *output = warpx::radiation::ReconstructPacketMoments(desired, photon_energy); });
    auto const result = storage.dataValue();
    AMREX_ALWAYS_ASSERT(result.valid);
    long double actual[4]{}, pressure[3][3]{}, angular[3]{};
    long double const mec = static_cast<long double>(PhysConst::m_e) * PhysConst::c;
    long double const position[3] = {1.3L, -0.7L, 2.1L};
    for (int ray = 0; ray < 8; ++ray)
    {
        auto const& photon = result.photons[ray];
        long double const weight = photon.energy_weight;
        AMREX_ALWAYS_ASSERT(std::isfinite(weight) && weight >= 0);
        long double norm2 = 0, q[3]{};
        for (int d = 0; d < 3; ++d)
        {
            long double const u = photon.proper_velocity[d];
            norm2 += u * u;
            q[d] = mec * u;
        }
        long double const energy = mec * std::sqrt(norm2);
        AMREX_ALWAYS_ASSERT(std::abs(energy - photon_energy) < 2.e-13L * photon_energy);
        actual[0] += weight * energy;
        for (int d = 0; d < 3; ++d)
        {
            actual[d + 1] += weight * q[d];
            angular[d] +=
                weight *
                (position[(d + 1) % 3] * q[(d + 2) % 3] - position[(d + 2) % 3] * q[(d + 1) % 3]) /
                PhysConst::c;
            for (int e = 0; e < 3; ++e)
            {
                pressure[d][e] += weight * q[d] * q[e] / energy;
            }
            int const opposite = ray < 4 ? ray + 4 : ray - 4;
            AMREX_ALWAYS_ASSERT(photon.proper_velocity[d] ==
                                -result.photons[opposite].proper_velocity[d]);
        }
    }
    long double const energy = desired[0];
    long double norm = std::sqrt(static_cast<long double>(desired[1]) * desired[1] +
                                 static_cast<long double>(desired[2]) * desired[2] +
                                 static_cast<long double>(desired[3]) * desired[3]);
    long double const f = norm / energy;
    long double const chi = (3 + 4 * f * f) / (5 + 2 * std::sqrt(4 - 3 * f * f));
    long double n[3] = {0, 0, 1};
    if (norm > 0)
    {
        for (int d = 0; d < 3; ++d)
        {
            n[d] = desired[d + 1] / norm;
        }
    }
    constexpr long double tolerance = 3.e-13L;
    for (int d = 0; d < 4; ++d)
    {
        AMREX_ALWAYS_ASSERT(std::abs(actual[d] - desired[d]) < tolerance * energy);
        AMREX_ALWAYS_ASSERT(std::abs(actual[d] - desired[d]) <=
                            128 * std::numeric_limits<amrex::Real>::epsilon() * energy);
        AMREX_ALWAYS_ASSERT(std::abs(actual[d] - result.reconstructed_four_momentum[d]) <
                            tolerance * energy);
        AMREX_ALWAYS_ASSERT(std::abs(result.signed_residual[d] -
                                     (result.reconstructed_four_momentum[d] - desired[d])) == 0);
        AMREX_ALWAYS_ASSERT(std::abs(result.signed_residual[d]) <=
                            128 * std::numeric_limits<amrex::Real>::epsilon() * energy);
    }
    for (int d = 0; d < 3; ++d)
    {
        auto const expected_angular = (position[(d + 1) % 3] * desired[(d + 2) % 3 + 1] -
                                       position[(d + 2) % 3] * desired[(d + 1) % 3 + 1]) /
                                      PhysConst::c;
        AMREX_ALWAYS_ASSERT(std::abs(angular[d] - expected_angular) <
                            4 * tolerance * energy / PhysConst::c);
        for (int e = 0; e < 3; ++e)
        {
            auto const expected =
                energy * ((1 - chi) / 2 * (d == e ? 1 : 0) + (3 * chi - 1) / 2 * n[d] * n[e]);
            AMREX_ALWAYS_ASSERT(std::abs(pressure[d][e] - expected) < tolerance * energy);
            AMREX_ALWAYS_ASSERT(std::abs(pressure[d][e] - expected) <=
                                128 * std::numeric_limits<amrex::Real>::epsilon() * energy);
        }
    }
}
} // namespace

int
main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        constexpr amrex::Real photon_energy = 1.e-19_rt;
        for (auto energy : {1.e-20_rt, 1._rt, 1.e20_rt})
        {
            for (auto fraction : {0._rt, 0.4_rt, 0.5_rt, 0.9_rt, 0.999999_rt})
            {
                auto const scale = energy * fraction / std::sqrt(14._rt);
                check({energy, scale, 2 * scale, 3 * scale}, photon_energy);
                check({energy, energy * fraction, 0, 0}, photon_energy);
            }
        }
        check({1, 0, 0, 1}, photon_energy);
        // Actual transparent-cell output of the limited cold-front transport.
        // Its long-double norm excess is ~4.54 epsilon, not a physical deficit.
        check({0.015689286526961186696_rt, 0.0047067859580883601722_rt,
               0.0062757146107844706887_rt, 0.013587320699601333415_rt}, photon_energy);
        constexpr auto epsilon = std::numeric_limits<amrex::Real>::epsilon();
        check({1, 0, 0, 1 + 8 * epsilon}, photon_energy);
        using warpx::radiation::ReconstructPacketMoments;
        AMREX_ALWAYS_ASSERT(!ReconstructPacketMoments(
            {1, 0, 0, std::nextafter(1 + 8 * epsilon,
                                    std::numeric_limits<amrex::Real>::infinity())},
            photon_energy).valid);
        AMREX_ALWAYS_ASSERT(ReconstructPacketMoments({0, 0, 0, 0}, photon_energy).valid);
        AMREX_ALWAYS_ASSERT(!ReconstructPacketMoments({0, 1, 0, 0}, photon_energy).valid);
        AMREX_ALWAYS_ASSERT(!ReconstructPacketMoments({1, 1.1_rt, 0, 0}, photon_energy).valid);
        AMREX_ALWAYS_ASSERT(!ReconstructPacketMoments({-1, 0, 0, 0}, photon_energy).valid);
        AMREX_ALWAYS_ASSERT(!ReconstructPacketMoments({1, 0, 0, 0}, 0).valid);
        AMREX_ALWAYS_ASSERT(
            !ReconstructPacketMoments({1, std::numeric_limits<amrex::Real>::quiet_NaN(), 0, 0},
                                      photon_energy)
                 .valid);
        amrex::Print() << "Eight-ray native-unit energy, momentum, stress and "
                          "angular gates passed.\n";
    }
    amrex::Finalize();
}
