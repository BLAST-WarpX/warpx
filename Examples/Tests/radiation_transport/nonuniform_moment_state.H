/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_TEST_NONUNIFORM_MOMENT_STATE_H_
#define WARPX_TEST_NONUNIFORM_MOMENT_STATE_H_

#include "Particles/MultiParticleContainer.H"
#include "Utils/WarpXConst.H"

#include <AMReX_BoxIterator.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_ParallelDescriptor.H>

#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <utility>
#include <vector>

namespace qualification
{
    // Raw particles, not deposited moments: mass, three mesh coordinates
    // (unused coordinates zero), proper velocity[3], carry[4]. Positions are
    // stationary in this source test, so the oracle can match them exactly.
    inline void
    WriteParticles (WarpXParticleContainer& ions, char const* filename)
    {
        std::vector<long double> local;
        std::array<int, 8> components{PIdx::w, PIdx::ux, PIdx::uy, PIdx::uz,
            ions.GetRealCompIndex("radiation_impulse_coupled_ux"),
            ions.GetRealCompIndex("radiation_impulse_coupled_uy"),
            ions.GetRealCompIndex("radiation_impulse_coupled_uz"),
            ions.GetRealCompIndex("radiation_impulse_coupled_work")};
        for (WarpXParIter iterator(ions, 0); iterator.isValid(); ++iterator)
        {
            std::array<amrex::Gpu::HostVector<amrex::ParticleReal>, 8 + AMREX_SPACEDIM> host;
            for (int d = 0; d < 8 + AMREX_SPACEDIM; ++d)
            {
                int const component = d < 8 ? components[d] : d - 8;
                auto const& values = iterator.GetStructOfArrays().GetRealData(component);
                host[d].resize(iterator.numParticles());
                amrex::Gpu::copy(amrex::Gpu::deviceToHost, values.begin(),
                    values.begin() + iterator.numParticles(), host[d].begin());
            }
            for (long p = 0; p < iterator.numParticles(); ++p)
            {
                local.push_back(static_cast<long double>(host[0][p]) * ions.getMass());
                for (int d = 0; d < 3; ++d)
                {
                    local.push_back(d < AMREX_SPACEDIM ? host[8 + d][p] : 0);
                }
                for (int d = 1; d < 8; ++d) { local.push_back(host[d][p]); }
            }
        }
#ifdef AMREX_USE_MPI
        int const count = static_cast<int>(local.size());
        std::vector<int> counts(amrex::ParallelDescriptor::NProcs());
        MPI_Allgather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT,
            amrex::ParallelDescriptor::Communicator());
        std::vector<int> offsets(counts.size());
        int total = 0;
        for (std::size_t i = 0; i < counts.size(); ++i)
        {
            offsets[i] = total;
            total += counts[i];
        }
        std::vector<long double> all(total);
        MPI_Allgatherv(local.data(), count, MPI_LONG_DOUBLE, all.data(), counts.data(),
            offsets.data(), MPI_LONG_DOUBLE, amrex::ParallelDescriptor::Communicator());
        local = std::move(all);
#endif
        if (!amrex::ParallelDescriptor::IOProcessor()) { return; }
        std::ofstream stream(filename);
        stream << std::setprecision(std::numeric_limits<long double>::max_digits10);
        for (std::size_t i = 0; i < local.size(); ++i)
        {
            stream << local[i] << (i % 11 == 10 ? '\n' : ' ');
        }
        AMREX_ALWAYS_ASSERT(stream.good());
    }

    // Independent host inventory indexed by unique periodic cells/lower nodes:
    // E,q[3],T,M,M*delta_u[3],delta_K,carry_P[3],carry_E,count.
    inline std::vector<long double>
    CollectState (amrex::MultiFab const& radiation, amrex::MultiFab const& temperature,
                  WarpXParticleContainer& ions, amrex::Geometry const& geometry)
    {
        int const nx = geometry.Domain().length(0);
        int const ny = AMREX_SPACEDIM == 2 ? geometry.Domain().length(1) : 1;
        std::vector<long double> state(15 * nx * ny, 0);
        for (amrex::MFIter iterator(radiation); iterator.isValid(); ++iterator) {
            auto const box = iterator.validbox();
            amrex::FArrayBox packed(box, 5, amrex::The_Arena());
            amrex::FArrayBox host(box, 5, amrex::The_Pinned_Arena());
            auto const r = radiation.const_array(iterator);
            auto const t = temperature.const_array(iterator);
            auto const output = packed.array();
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                for (int d = 0; d < 4; ++d) {
                    output(i, j, k, d) = r(i, j, k, d);
                }
                output(i, j, k, 4) = t(i, j, k);
            });
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, packed.dataPtr(),
                             packed.dataPtr() + packed.size(), host.dataPtr());
            auto const values = host.const_array();
            for (amrex::BoxIterator cell(box); cell.ok(); ++cell) {
                auto const index = cell();
                int const flat = index[0] + (AMREX_SPACEDIM == 2 ? nx * index[1] : 0);
                for (int d = 0; d < 5; ++d) {
                    state[15 * flat + d] = values(index, d);
                }
            }
        }
        std::array<int, 8> components{PIdx::w,
                                      PIdx::ux,
                                      PIdx::uy,
                                      PIdx::uz,
                                      ions.GetRealCompIndex("radiation_impulse_coupled_ux"),
                                      ions.GetRealCompIndex("radiation_impulse_coupled_uy"),
                                      ions.GetRealCompIndex("radiation_impulse_coupled_uz"),
                                      ions.GetRealCompIndex("radiation_impulse_coupled_work")};
        for (WarpXParIter iterator(ions, 0); iterator.isValid(); ++iterator) {
            std::array<amrex::Gpu::HostVector<amrex::ParticleReal>, 8 + AMREX_SPACEDIM> host;
            for (int d = 0; d < 8 + AMREX_SPACEDIM; ++d) {
                int const component = d < 8 ? components[d] : d - 8;
                auto const& data = iterator.GetStructOfArrays().GetRealData(component);
                host[d].resize(iterator.numParticles());
                amrex::Gpu::copy(amrex::Gpu::deviceToHost, data.begin(),
                                 data.begin() + iterator.numParticles(), host[d].begin());
            }
            for (long p = 0; p < iterator.numParticles(); ++p) {
                int flat = 0;
                int stride = 1;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    int const index = static_cast<int>(
                        std::floor((host[8 + d][p] - geometry.ProbLo(d)) / geometry.CellSize(d)));
                    AMREX_ALWAYS_ASSERT(index >= 0 && index < geometry.Domain().length(d));
                    flat += stride * index;
                    stride *= geometry.Domain().length(d);
                }
                long double const mass = static_cast<long double>(host[0][p]) * ions.getMass();
                long double squared = 0;
                long double difference = 0;
                long double const c2 = PhysConst::c2;
                for (int d = 0; d < 3; ++d) {
                    long double const u = host[1 + d][p];
                    long double const old = d == 2 ? 1.e5L : 0;
                    squared += u * u;
                    difference += (u - old) * (u + old);
                    state[15 * flat + 6 + d] += mass * (u - old);
                    state[15 * flat + 10 + d] += mass * host[4 + d][p];
                }
                state[15 * flat + 5] += mass;
                state[15 * flat + 9] +=
                    mass * difference / (std::sqrt(1 + 1.e10L / c2) + std::sqrt(1 + squared / c2));
                state[15 * flat + 13] += mass * host[7][p];
                state[15 * flat + 14] += 1;
            }
        }
#ifdef AMREX_USE_MPI
        MPI_Allreduce(MPI_IN_PLACE, state.data(), static_cast<int>(state.size()), MPI_LONG_DOUBLE,
                      MPI_SUM, amrex::ParallelDescriptor::Communicator());
#endif
        return state;
    }

    inline void
    WriteState (std::vector<long double> const& before, std::vector<long double> const& after,
                amrex::Geometry const& geometry, amrex::Real volume, amrex::Real absorption_step,
                bool latent, amrex::Real dt, bool spatial)
    {
        if (!amrex::ParallelDescriptor::IOProcessor()) {
            return;
        }
        std::ofstream stream("nonuniform_state.txt");
        stream << std::setprecision(std::numeric_limits<long double>::max_digits10);
        stream << geometry.Domain().length(0) << ' '
               << (AMREX_SPACEDIM == 2 ? geometry.Domain().length(1) : 1) << ' ' << volume << ' '
               << absorption_step << ' ' << PhysConst::c << ' ' << PhysConst::kb << ' '
               << PhysConst::q_e << ' ' << latent << ' ' << dt << ' ' << spatial << ' '
               << geometry.CellSize(0) << ' ' << (AMREX_SPACEDIM == 2 ? geometry.CellSize(1) : 1)
               << '\n';
        for (std::size_t i = 0; i < before.size() / 15; ++i) {
            for (int d = 0; d < 15; ++d) {
                stream << before[15 * i + d] << ' ';
            }
            for (int d = 0; d < 15; ++d) {
                stream << after[15 * i + d] << ' ';
            }
            stream << '\n';
        }
        AMREX_ALWAYS_ASSERT(stream.good());
    }
} // namespace qualification
#endif
