/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "Initialization/WarpXInit.H"
#include "Particles/MultiParticleContainer.H"
#include "Radiation/PhotonPacketTransaction.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Random.H>

#include <array>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace
{
void Require (bool valid, char const* message)
{
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if (!valid) { throw std::runtime_error(message); }
}
template<typename Data>
auto Host (Data const& source)
{
    using Value = typename Data::value_type;
    std::vector<Value> host(source.size());
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, source.begin(), source.end(), host.begin());
    return host;
}
template<typename Value>
void Bytes (std::vector<unsigned char>& result, std::vector<Value> const& values)
{
    if (!values.empty()) {
        auto const* begin = reinterpret_cast<unsigned char const*>(values.data());
        result.insert(result.end(), begin, begin+values.size()*sizeof(Value));
    }
}
auto Snapshot (WarpXParticleContainer::Base const& particles)
{
    std::vector<unsigned char> result;
    for (auto const& [key,tile] : particles.GetParticles(0)) {
        Bytes(result, std::vector<long>{key.first,key.second,tile.numParticles()});
        auto const& soa = tile.GetStructOfArrays();
        Bytes(result, Host(soa.GetIdCPUData()));
        for (int n = 0; n < tile.NumRealComps(); ++n) { Bytes(result, Host(soa.GetRealData(n))); }
        for (int n = 0; n < tile.NumIntComps(); ++n) { Bytes(result, Host(soa.GetIntData(n))); }
    }
    return result;
}
auto Inventory (WarpXParticleContainer::Base const& particles)
{
    // Physical Cartesian four-inventory and axial angular momentum times c.
    // No radial momentum sum is asserted conserved in cylindrical geometry.
    std::array<amrex::Real,5> result{};
    for (auto const& [key,tile] : particles.GetParticles(0)) {
        amrex::ignore_unused(key);
        auto const& soa = tile.GetStructOfArrays();
        auto const radius = Host(soa.GetRealData(PIdx::r));
        auto const theta = Host(soa.GetRealData(PIdx::theta));
        auto const weight = Host(soa.GetRealData(PIdx::w));
        auto const ux = Host(soa.GetRealData(PIdx::ux));
        auto const uy = Host(soa.GetRealData(PIdx::uy));
        auto const uz = Host(soa.GetRealData(PIdx::uz));
        for (std::size_t p = 0; p < weight.size(); ++p) {
            long double const factor = static_cast<long double>(weight[p])*PhysConst::m_e*PhysConst::c;
            result[0] += factor*std::hypot(ux[p],uy[p],uz[p]);
            result[1] += factor*ux[p];
            result[2] += factor*uy[p];
            result[3] += factor*uz[p];
            result[4] += factor*radius[p]*(std::cos(theta[p])*uy[p]-std::sin(theta[p])*ux[p]);
        }
    }
    amrex::ParallelDescriptor::ReduceRealSum(result.data(),5);
    return result;
}
std::string RandomState ()
{
    std::ostringstream state;
    amrex::SaveRandomState(state);
    return state.str();
}
}

int main (int argc, char* argv[])
{
    warpx::initialization::initialize_external_libraries(argc,argv);
    int status = 0;
    try {
        auto& sim = WarpX::GetInstance();
        sim.InitData();
        auto& photons = sim.GetPartContainer().GetParticleContainerFromName("photons");
        using Transaction = warpx::radiation::PhotonPacketTransaction;
        using Particle = WarpXParticleContainer::ParticleType;
        Require(sim.Geom(0).Coord() == 0 || sim.Geom(0).Coord() == 1,
                "fixture must use supported native RZ metric convention");
        int grid = 0;
        int const rank = amrex::ParallelDescriptor::MyProc();
        bool owns_grid = false;
        auto const& distribution = photons.ParticleDistributionMap(0);
        for (int g = 0; g < photons.ParticleBoxArray(0).size(); ++g) {
            if (distribution[g] == rank) { grid = g; owns_grid = true; break; }
        }
        bool const emitter = rank == 0 && owns_grid;
        auto const radius = .5*(sim.Geom(0).ProbLo(0)+sim.Geom(0).ProbHi(0));
        auto const axial = .5*(sim.Geom(0).ProbLo(1)+sim.Geom(0).ProbHi(1));
        std::array<double,3> const radii{sim.Geom(0).ProbLo(0) == 0 ? 0 : radius, radius, radius};
        std::array<double,3> const angles{0,2.2,-1.1};
        amrex::Vector<amrex::ParticleReal> x,y,z,ux,uy,uz,w;
        std::array<double,3> const vx{3,-2,1}, vy{4,1,-4}, vz{0,2,2};
        std::array<double,3> const weights{2.e20,3.e20,5.e20};
        std::array<amrex::Real,5> expected{};
        for (int p = 0; p < 3; ++p) {
            auto const xp = radii[p]*std::cos(angles[p]);
            auto const yp = radii[p]*std::sin(angles[p]);
            if (emitter) {
                x.push_back(xp); y.push_back(yp); z.push_back(axial);
                ux.push_back(vx[p]); uy.push_back(vy[p]); uz.push_back(vz[p]);
                w.push_back(weights[p]);
            }
            auto const factor = weights[p]*PhysConst::m_e*PhysConst::c;
            expected[0] += factor*std::hypot(vx[p],vy[p],vz[p]);
            expected[1] += factor*vx[p]; expected[2] += factor*vy[p];
            expected[3] += factor*vz[p]; expected[4] += factor*(xp*vy[p]-yp*vx[p]);
        }
        auto birth = [&](Transaction& trial) {
            return trial.AddProvisionalBirths(grid, static_cast<long>(x.size()),
                x,y,z,ux,uy,uz,{w},{},true);
        };
        auto const baseline = Snapshot(photons);
        auto const initial = Inventory(photons);
        auto const random_before = RandomState();
        auto const rollback_id = Particle::NextID();
        {
            Transaction trial(photons);
            Require(birth(trial), "RZ provisional birth rejected");
            trial.Particles().Redistribute();
            auto const staged = Inventory(trial.Particles());
            for (int n = 0; n < 5; ++n) {
                Require(std::abs(staged[n]-initial[n]-expected[n]) <= 2.e-13*expected[0],
                        "RZ Cartesian E/cp/axial-torque inventory is incorrect");
            }
            Require(trial.ValidateLive(), "RZ private validation failed");
        }
        Require(Snapshot(photons) == baseline, "RZ rollback changed native state or theta");
        Require(Particle::NextID() == rollback_id+1, "RZ rollback consumed native IDs");
        Require(RandomState() == random_before, "RZ staging changed host random state");
        auto const commit_id = Particle::NextID();
        {
            Transaction trial(photons);
            Require(birth(trial), "second RZ provisional birth rejected");
            trial.Particles().Redistribute();
            long local_births = 0;
            for (auto const& [key,tile] : trial.Particles().GetParticles(0)) {
                amrex::ignore_unused(key);
                auto const marks = Host(tile.GetStructOfArrays().GetIntData(tile.NumIntComps()-1));
                for (auto const marked : marks) { local_births += marked; }
            }
            Require(trial.ValidateLive(), "RZ final validation failed");
            trial.Commit();
            Require(Particle::NextID() == commit_id+1+local_births,
                    "RZ commit allocated wrong native ID count");
        }
        auto const final = Inventory(photons);
        for (int n = 0; n < 5; ++n) {
            Require(std::abs(final[n]-initial[n]-expected[n]) <= 2.e-13*expected[0],
                    "RZ committed physical inventory differs from birth request");
        }
        Require(RandomState() == random_before, "RZ commit changed host random state");
        // Explicitly reject random-default initialization without consuming RNG.
        photons.AddRealComp("opticalDepthBW",1);
        auto const rejected_baseline = Snapshot(photons);
        {
            Transaction trial(photons);
            Require(!birth(trial), "RZ random default was unexpectedly initialized");
        }
        Require(Snapshot(photons) == rejected_baseline && RandomState() == random_before,
                "RZ rejected random-default birth mutated live state or RNG");
        amrex::Print() << "Native RZ transaction births, rollback, IDs, host RNG and physical inventories pass.\n";
    } catch (std::exception const& error) {
        amrex::Print() << error.what() << '\n';
        status = 1;
    }
    WarpX::Finalize();
    warpx::initialization::finalize_external_libraries();
    return status;
}
