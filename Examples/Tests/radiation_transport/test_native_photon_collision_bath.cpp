/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "Initialization/WarpXInit.H"
#include "Particles/MultiParticleContainer.H"
#include "Radiation/NativePhotonCollisionReplay.H"
#include "Radiation/PhotonPacketTransaction.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

#include <array>
#include <cmath>
#include <map>
#include <vector>

namespace
{
using TileKey = std::pair<int, int>;
using Moments = std::array<long double, 5>; // E, cpz, Pzz, debit, cpz debit
using Records = std::map<TileKey, std::vector<std::array<long double, 2>>>;

Records Inventory (WarpXParticleContainer::Base const& photons, Moments& sum)
{
    Records result;
    for (auto const& [key, tile] : photons.GetParticles(0))
    {
        auto const count = tile.numParticles();
        std::array<std::vector<amrex::ParticleReal>, 4> data;
        std::array<int, 4> const components{PIdx::w, PIdx::ux, PIdx::uy, PIdx::uz};
        for (int c = 0; c < 4; ++c)
        {
            data[c].resize(count);
            auto const& values = tile.GetStructOfArrays().GetRealData(components[c]);
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, values.begin(), values.begin()+count,
                             data[c].begin());
        }
        auto& values = result[key];
        for (long p = 0; p < count; ++p)
        {
            long double const scale = static_cast<long double>(data[0][p])*
                PhysConst::m_e*PhysConst::c;
            long double const ux = data[1][p], uy = data[2][p], uz = data[3][p];
            long double const energy = scale*std::sqrt(ux*ux+uy*uy+uz*uz);
            long double const cpz = scale*uz;
            values.push_back({energy, cpz});
            sum[0] += energy;
            sum[1] += cpz;
            sum[2] += cpz*cpz/energy;
        }
    }
    return result;
}

template <std::size_t N>
void GlobalSum (std::array<long double, N>& values)
{
#ifdef AMREX_USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, values.data(), static_cast<int>(N), MPI_LONG_DOUBLE,
                  MPI_SUM, amrex::ParallelDescriptor::Communicator());
#endif
}

// Independent prescribed-bath angular equation in the comoving cosine mu:
// dq/dtau = -q/D + (1/2) integral(q/D dmu), D=gamma*(1+beta*mu).
// For a lab-isotropic bath q(mu,0)=1/(2 D^3). E' is conserved at every
// collision. This reference evolves the expected distribution, not RNG draws
// or production boosts. Midpoint angular quadrature and RK4 are refined below.
long double ReferenceDrag (int angles, int steps, long double beta, long double tau)
{
    auto const gamma = 1/std::sqrt(1-beta*beta);
    auto const h = tau/steps;
    std::vector<long double> q(angles), doppler(angles), momentum(angles);
    std::array<std::vector<long double>, 4> slope;
    for (auto& values : slope) { values.resize(angles); }
    std::vector<long double> trial(angles);
    long double initial = 0;
    for (int i = 0; i < angles; ++i)
    {
        auto const mu = -1+2*(i+0.5L)/angles;
        doppler[i] = gamma*(1+beta*mu);
        momentum[i] = gamma*(mu+beta);
        q[i] = 1/(angles*doppler[i]*doppler[i]*doppler[i]);
        initial += momentum[i]*q[i];
    }
    for (int step = 0; step < steps; ++step)
    {
        for (int stage = 0; stage < 4; ++stage)
        {
            long double source = 0;
            for (int i = 0; i < angles; ++i)
            {
                trial[i] = q[i] + (stage == 0 ? 0 :
                    h*(stage == 3 ? 1 : 0.5L)*slope[stage-1][i]);
                source += trial[i]/doppler[i];
            }
            for (int i = 0; i < angles; ++i)
            {
                slope[stage][i] = -trial[i]/doppler[i]+source/angles;
            }
        }
        for (int i = 0; i < angles; ++i)
        {
            q[i] += h*(slope[0][i]+2*slope[1][i]+2*slope[2][i]+slope[3][i])/6;
        }
    }
    long double final = 0;
    for (int i = 0; i < angles; ++i) { final += momentum[i]*q[i]; }
    return initial-final;
}
}

int main (int argc, char* argv[])
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        using namespace warpx::radiation;
        auto& simulation = WarpX::GetInstance();
        simulation.InitData();
        auto& photons = simulation.GetPartContainer().GetParticleContainerFromName("photons");
        auto const& ba = photons.ParticleBoxArray(0);
        auto const& dm = photons.ParticleDistributionMap(0);
        amrex::iMultiFab mask(ba, dm, 1, 0);
        amrex::MultiFab opacity(ba, dm, 1, 0), velocity(ba, dm, 3, 0);
        amrex::MultiFab impulse(ba, dm, 3, 0), debit(ba, dm, 1, 0), scale(ba, dm, 1, 0);
        mask.setVal(0);
        velocity.setVal(0);
        constexpr long double beta = 0.4L;
        long double const gamma = 1/std::sqrt(1-beta*beta);
        constexpr amrex::Real tau = 0.1;
        auto const dt = simulation.getdt()[0];
        opacity.setVal(tau/(PhysConst::c*dt));
        velocity.setVal(static_cast<amrex::Real>(beta*PhysConst::c), 2, 1);
        auto const count = photons.TotalNumberOfParticles();
        AMREX_ALWAYS_ASSERT(count >= 262144);
        auto const coarse_drag = ReferenceDrag(1024, 64, beta, tau);
        auto const reference_drag = ReferenceDrag(2048, 128, beta, tau);
        auto const reference_error = 4*std::abs(reference_drag-coarse_drag);
        AMREX_ALWAYS_ASSERT(reference_error < 1.e-5L*std::abs(reference_drag));
        std::array<long double, 2> equilibrium_force{};
        for (int bath = 0; bath < 2; ++bath)
        {
            // Deterministic equal-area sphere; a boosted bath transforms BOTH
            // photon energy and lab-volume number weight, not just directions.
            for (auto& [key, tile] : photons.GetParticles(0))
            {
                amrex::ignore_unused(key);
                auto const n = tile.numParticles();
                std::array<std::vector<amrex::ParticleReal>, 4> host;
                for (auto& values : host) { values.resize(n); }
                for (long p = 0; p < n; ++p)
                {
                    auto const mu = -1+2*(static_cast<long double>(p)+0.5L)/n;
                    auto const phi = 2.399963229728653322L*p;
                    auto const sine = std::sqrt(1-mu*mu);
                    auto const doppler = bath == 0 ? 1 : gamma*(1+beta*mu);
                    host[0][p] = static_cast<amrex::ParticleReal>(doppler/
                        (count*static_cast<long double>(PhysConst::m_e)*PhysConst::c*PhysConst::c));
                    host[1][p] = static_cast<amrex::ParticleReal>(PhysConst::c*sine*std::cos(phi));
                    host[2][p] = static_cast<amrex::ParticleReal>(PhysConst::c*sine*std::sin(phi));
                    host[3][p] = static_cast<amrex::ParticleReal>(PhysConst::c*
                        (bath == 0 ? mu : gamma*(mu+beta)));
                }
                std::array<int, 4> const components{PIdx::w, PIdx::ux, PIdx::uy, PIdx::uz};
                for (int c = 0; c < 4; ++c)
                {
                    auto& values = tile.GetStructOfArrays().GetRealData(components[c]);
                    amrex::Gpu::copy(amrex::Gpu::hostToDevice, host[c].begin(), host[c].end(),
                                     values.begin());
                }
            }
            Moments initial_moments{};
            auto const initial = Inventory(photons, initial_moments);
            GlobalSum(initial_moments);
            long double const expected_energy = bath == 0 ? 1 : gamma*gamma*(1+beta*beta/3);
            long double const expected_cpz = bath == 0 ? 0 : 4.L/3*gamma*gamma*beta;
            long double const expected_pressure = bath == 0 ? 1.L/3 : gamma*gamma*(1.L/3+beta*beta);
            AMREX_ALWAYS_ASSERT(std::abs(initial_moments[0]-expected_energy) < 1.e-8L);
            AMREX_ALWAYS_ASSERT(std::abs(initial_moments[1]-expected_cpz) < 1.e-8L);
            AMREX_ALWAYS_ASSERT(std::abs(initial_moments[2]-expected_pressure) < 1.e-8L);
            for (int proper = 0; proper < 2; ++proper)
            {
                NativePhotonCollisionOptions options;
                options.seed = 7391;
                options.interval = 27;
                options.moving_cross_sections = proper != 0;
                NativePhotonCollisionReport report;
                NativePhotonCollisionReplay replay;
                bool const prepared = proper != 0
                    ? replay.Prepare(photons, mask, opacity, simulation.Geom(0), dt,
                                     velocity, options, report)
                    : replay.Prepare(photons, mask, opacity, simulation.Geom(0), dt,
                                     options, report);
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(prepared, report.reason.c_str());
                PhotonPacketTransaction transaction(photons);
                AMREX_ALWAYS_ASSERT(replay.Replay(velocity, transaction.Particles(),
                    impulse, debit, scale, report));
                Moments final_moments{};
                auto const final = Inventory(transaction.Particles(), final_moments);
                std::array<long double, 3> measured{}; // debit, cpz debit, sum squares
                for (auto const& [key, values] : initial)
                {
                    auto const& next = final.at(key);
                    for (std::size_t p = 0; p < values.size(); ++p)
                    {
                        auto const transfer = values[p][1]-next[p][1];
                        measured[0] += values[p][0]-next[p][0];
                        measured[1] += transfer;
                        measured[2] += transfer*transfer;
                    }
                }
                GlobalSum(measured);
                // Six standard errors, conservatively retaining the squared
                // mean in the second moment. No relative floor at equilibrium.
                auto const noise_bound = 6*std::sqrt(measured[2]);
                AMREX_ALWAYS_ASSERT(std::abs(measured[1]-PhysConst::c*
                    report.material_impulse[2]) < 1.e-11L*expected_energy);
                AMREX_ALWAYS_ASSERT(std::abs(measured[0]-report.photon_energy_debit)
                                    < 1.e-11L*expected_energy);
                if (bath == 0)
                {
                    if (proper != 0)
                    {
                        AMREX_ALWAYS_ASSERT(std::abs(measured[1]-reference_drag) <
                            reference_error+noise_bound);
                    }
                    else
                    {
                        auto const lab_rate_drag = -gamma*gamma*beta*(1-std::exp(-tau));
                        AMREX_ALWAYS_ASSERT(std::abs(measured[1]-lab_rate_drag) < noise_bound);
                        AMREX_ALWAYS_ASSERT(std::abs(measured[1]-reference_drag) >
                            reference_error+noise_bound);
                    }
                }
                if (bath == 1)
                {
                    equilibrium_force[proper] = measured[1];
                    if (proper != 0) { AMREX_ALWAYS_ASSERT(std::abs(measured[1]) < noise_bound); }
                    else
                    {
                        auto const spurious = gamma*gamma*beta*(1-std::exp(-tau))/3;
                        AMREX_ALWAYS_ASSERT(std::abs(measured[1]-spurious) < noise_bound);
                        AMREX_ALWAYS_ASSERT(std::abs(measured[1]) > noise_bound);
                    }
                }
                amrex::Print() << "bath=" << bath << " proper=" << proper
                    << " cp source=" << measured[1] << " 6sigma=" << noise_bound
                    << " events=" << report.collision_events << '\n';
            }
        }
        AMREX_ALWAYS_ASSERT(std::abs(equilibrium_force[1]) < std::abs(equilibrium_force[0]));
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
