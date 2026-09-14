/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "Initialization/WarpXInit.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/ParticleBoundaries.H"
#include "Radiation/ParticleImpulse.H"
#include "Radiation/ParticleImpulseBoundary.H"
#include "WarpX.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace
{
using Row = std::array<amrex::ParticleReal, 10>;
std::vector<Row>
Read (WarpXParticleContainer& species)
{
    std::array<int, 10> const indices{
        PIdx::r,
        PIdx::theta,
        PIdx::ux,
        PIdx::uy,
        PIdx::uz,
        PIdx::w,
        species.GetRealCompIndex("radiation_impulse_angular_wall_ux"),
        species.GetRealCompIndex("radiation_impulse_angular_wall_uy"),
        species.GetRealCompIndex("radiation_impulse_angular_wall_uz"),
        species.GetRealCompIndex("radiation_impulse_angular_wall_work")};
    std::vector<Row> rows;
    for (WarpXParIter it(species, 0); it.isValid(); ++it)
    {
        auto const first = rows.size();
        rows.resize(first + it.numParticles());
        for (int component = 0; component < 10; ++component)
        {
            auto const& values = it.GetStructOfArrays().GetRealData(indices[component]);
            amrex::Gpu::HostVector<amrex::ParticleReal> host(it.numParticles());
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, values.begin(),
                             values.begin() + it.numParticles(), host.begin());
            for (long p = 0; p < it.numParticles(); ++p)
            {
                rows[first + p][component] = host[p];
            }
        }
    }
    return rows;
}
} // namespace

int
main (int argc, char* argv[])
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        using namespace warpx::radiation;
        auto& simulation = WarpX::GetInstance();
        auto& species = simulation.GetPartContainer().GetParticleContainerFromName("ions");
        RegisterParticleImpulseState(species, "angular_wall");
        species.AddRealComp("prev_x");
        species.AddRealComp("prev_y");
        simulation.InitData();
        auto const radius = species.Geom(0).ProbHi(0);
        for (WarpXParIter it(species, 0); it.isValid(); ++it)
        {
            auto const data = it.GetParticleTile().getParticleTileData();
            auto* previous_x = it.GetAttribs("prev_x").dataPtr();
            auto* previous_y = it.GetAttribs("prev_y").dataPtr();
            std::array<char const*, 4> const suffix{"ux", "uy", "uz", "work"};
            amrex::GpuArray<amrex::ParticleReal*, 4> carry{};
            for (int d = 0; d < 4; ++d)
            {
                carry[d] = it.GetAttribs("radiation_impulse_angular_wall_" + std::string(suffix[d]))
                               .dataPtr();
            }
            amrex::ParallelFor(it.numParticles(),
                               [=] AMREX_GPU_DEVICE(long p)
                               {
                                   previous_x[p] = static_cast<amrex::ParticleReal>(0.8 * radius);
                                   previous_y[p] = 0;
                                   data.m_rdata[PIdx::r][p] = static_cast<amrex::ParticleReal>(
                                       std::hypot(1.2, 0.2) * radius);
                                   data.m_rdata[PIdx::theta][p] =
                                       static_cast<amrex::ParticleReal>(std::atan2(0.2, 1.2));
                                   data.m_rdata[PIdx::ux][p] = 2000;
                                   data.m_rdata[PIdx::uy][p] = 1000;
                                   data.m_rdata[PIdx::uz][p] = 300;
                                   carry[0][p] = 100;
                                   carry[1][p] = -200;
                                   carry[2][p] = 300;
                                   carry[3][p] = 400;
                               });
        }
        ParticleBoundaries boundaries;
        boundaries.SetAll(ParticleBoundaryType::Periodic);
        boundaries.SetBoundsX(ParticleBoundaryType::None, ParticleBoundaryType::Reflecting);
        boundaries.BuildReflectionModelParsers();
        auto const before = Read(species);
        std::vector<ParticleImpulseBoundaryTransfer> transfers{{"sentinel", {1, 2, 3}}};
        bool inspected = false;
        auto const reject = [&] (std::vector<ParticleImpulseBoundaryTransfer> const& proposed)
        {
            inspected = true;
            AMREX_ALWAYS_ASSERT(proposed.size() == 1 && proposed[0].path == "angular_wall");
            return false;
        };
        AMREX_ALWAYS_ASSERT(
            !TryReflectParticleImpulseState(species, boundaries, transfers, reject, true));
        AMREX_ALWAYS_ASSERT(inspected && Read(species) == before && transfers.size() == 1 &&
                            transfers[0].path == "sentinel" && transfers[0].momentum[0] == 1);
        AMREX_ALWAYS_ASSERT(
            TryReflectParticleImpulseState(species, boundaries, transfers, {}, true));
        auto const after = Read(species);
        AMREX_ALWAYS_ASSERT(after.size() == before.size());
        long double transfer[3]{};
        long double const eps = std::numeric_limits<amrex::ParticleReal>::epsilon();
        for (std::size_t p = 0; p < before.size(); ++p)
        {
            auto const& a = before[p];
            auto const& b = after[p];
            auto const angular = [] (Row const& row)
            {
                long double const theta = row[1];
                return row[0] * (-std::sin(theta) * row[2] + std::cos(theta) * row[3]);
            };
            AMREX_ALWAYS_ASSERT(std::abs(angular(b) - angular(a)) <= 512 * eps * radius * 3000);
            AMREX_ALWAYS_ASSERT(b[0] <= radius && b[1] != a[1] && b[5] == a[5] && b[9] == a[9]);
            long double old_speed = 0, new_speed = 0, old_carry = 0, new_carry = 0;
            for (int d = 0; d < 3; ++d)
            {
                old_speed += static_cast<long double>(a[2 + d]) * a[2 + d];
                new_speed += static_cast<long double>(b[2 + d]) * b[2 + d];
                old_carry += static_cast<long double>(a[6 + d]) * a[6 + d];
                new_carry += static_cast<long double>(b[6 + d]) * b[6 + d];
                transfer[d] +=
                    static_cast<long double>(a[5]) * species.getMass() * (a[6 + d] - b[6 + d]);
            }
            AMREX_ALWAYS_ASSERT(std::abs(new_speed - old_speed) <= 512 * eps * old_speed);
            AMREX_ALWAYS_ASSERT(std::abs(new_carry - old_carry) <= 512 * eps * old_carry);
        }
#ifdef AMREX_USE_MPI
        MPI_Allreduce(MPI_IN_PLACE, transfer, 3, MPI_LONG_DOUBLE, MPI_SUM,
                      amrex::ParallelDescriptor::Communicator());
#endif
        for (int d = 0; d < 3; ++d)
        {
            AMREX_ALWAYS_ASSERT(std::abs(transfer[d] - transfers[0].momentum[d]) <=
                                1024 * eps * std::abs(transfer[d]));
        }
        amrex::Print() << "Native angular wall: finite carry, particle torque/work and atomic "
                          "rejection passed.\n";
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
