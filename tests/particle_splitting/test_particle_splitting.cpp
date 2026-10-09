/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 * License: BSD-3-Clause-LBNL
 */
#include "Initialization/WarpXInit.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/PhysicalParticleContainer.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "WarpX.H"

#include <AMReX_Gpu.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <array>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace amrex::literals;

int main (int argc, char* argv[]) try
{
    // Initialize a small, empty simulation, then insert only four markers
    // needed by the test. Force device memory on GPU builds.
    std::vector<std::string> args{argv[0], "max_step=0", "warpx.verbose=0",
        "amr.max_level=0", "amr.max_grid_size=8", "amr.blocking_factor=8",
        "algo.maxwell_solver=yee", "algo.particle_shape=1", "warpx.use_filter=0",
        "amrex.the_arena_is_managed=0", "amrex.the_arena_init_size=0",
        "amrex.signal_handling=0", "amrex.throw_exception=1",
        "particles.species_names=test", "test.charge=q_e", "test.mass=m_p",
        "test.injection_style=NUniformPerCell", "test.profile=constant",
        "test.density=0", "test.momentum_distribution_type=at_rest"};
#if defined(WARPX_DIM_RZ)
    args.insert(args.end(), {"geometry.dims=RZ", "amr.n_cell=16 16",
        "geometry.prob_lo=0 0", "geometry.prob_hi=1 1",
        "boundary.field_lo=none periodic", "boundary.field_hi=pec periodic",
        "boundary.particle_lo=none periodic", "boundary.particle_hi=reflecting periodic",
        "test.num_particles_per_cell_each_dim=1 1 1"});
#elif defined(WARPX_DIM_3D)
    args.insert(args.end(), {"geometry.dims=3", "amr.n_cell=16 16 16",
        "geometry.prob_lo=0 0 0", "geometry.prob_hi=1 1 1",
        "boundary.field_lo=periodic periodic periodic",
        "boundary.field_hi=periodic periodic periodic",
        "test.num_particles_per_cell_each_dim=1 1 1"});
#elif defined(WARPX_DIM_XZ)
    args.insert(args.end(), {"geometry.dims=2", "amr.n_cell=16 16",
        "geometry.prob_lo=0 0", "geometry.prob_hi=1 1",
        "boundary.field_lo=periodic periodic", "boundary.field_hi=periodic periodic",
        "test.num_particles_per_cell_each_dim=1 1"});
#elif defined(WARPX_DIM_1D_Z)
    args.insert(args.end(), {"geometry.dims=1", "amr.n_cell=16",
        "geometry.prob_lo=0", "geometry.prob_hi=1",
        "boundary.field_lo=periodic", "boundary.field_hi=periodic",
        "test.num_particles_per_cell_each_dim=1"});
#else
    // The splitter displaces Cartesian x for the radial 1D geometries.
#ifdef WARPX_DIM_RCYLINDER
    args.emplace_back("geometry.dims=RCYLINDER");
    args.emplace_back("test.num_particles_per_cell_each_dim=1 1");
#else
    args.emplace_back("geometry.dims=RSPHERE");
    args.emplace_back("test.num_particles_per_cell_each_dim=1 1 1");
#endif
    args.insert(args.end(), {"amr.n_cell=16", "geometry.prob_lo=0",
        "geometry.prob_hi=1", "boundary.field_lo=none", "boundary.field_hi=pec",
        "boundary.particle_lo=none", "boundary.particle_hi=reflecting"});
#endif
    for (int i = 1; i < argc; ++i) { args.emplace_back(argv[i]); }
    std::vector<char*> ptrs;
    ptrs.reserve(args.size() + 1);
    for (auto& arg : args) { ptrs.push_back(arg.data()); }
    const int nargs = static_cast<int>(args.size());
    // MPI initialization requires the same argv[argc] null sentinel as main().
    ptrs.push_back(nullptr);
    char** pargs = ptrs.data();
    warpx::initialization::initialize_external_libraries(nargs, pargs);
    {
        auto& warpx = WarpX::GetInstance();
        warpx.InitData();
        auto& pc = dynamic_cast<PhysicalParticleContainer&>(
            warpx.GetPartContainer().GetParticleContainer(0));
        int split_type = 0;
        int attributes = 0;
        const amrex::ParmParse pp("test");
        pp.get("split_type", split_type);
        pp.get("add_attributes", attributes);
        if (attributes) {
            pc.AddRealComp("split_real", true);
            pc.AddIntComp("split_int", true);
        }
        pc.SplitParticles(0); // Empty containers are valid inputs.
        amrex::Vector<amrex::ParticleReal> x{0.375_prt, 0.49_prt, 0.625_prt, 0.75_prt};
        amrex::Vector<amrex::ParticleReal> y(4, 0.0_prt);
        amrex::Vector<amrex::ParticleReal> z{0.375_prt, 0.49_prt, 0.625_prt, 0.75_prt};
#ifdef WARPX_DIM_3D
        y.assign(4, 0.375_prt);
#elif defined(WARPX_DIM_RSPHERE)
        // Keep the final marker inside r < 1 while retaining a nonzero polar angle.
        x[3] = 0.5_prt;
#endif
        amrex::Vector<amrex::ParticleReal> ux{1.0e5_prt, 4.0e5_prt, -2.0e5_prt, 3.0e5_prt};
        amrex::Vector<amrex::ParticleReal> uy{2.0e5_prt, -2.0e5_prt, 3.0e5_prt, -1.0e5_prt};
        amrex::Vector<amrex::ParticleReal> uz{-3.0e5_prt, 4.0e5_prt, 1.0e5_prt, 2.0e5_prt};
        amrex::Vector<amrex::Vector<amrex::ParticleReal>> weights{{8.0_prt, 16.0_prt, 4.0_prt, 2.0_prt}};
        pc.AddNParticles(0, 4, x, y, z, ux, uy, uz, 1, weights, 0, {}, 0);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(pc.TotalNumberOfParticles() == 4,
            "All four test markers must be inside the domain before splitting");
        for (WarpXParIter pti(pc, 0); pti.isValid(); ++pti) {
            auto data = pti.GetParticleTile().getParticleTileData();
            amrex::ParallelFor(pti.numParticles(), [=] AMREX_GPU_DEVICE (int i) {
                if (data.m_rdata[PIdx::w][i] >= 8.0_prt) {
                    amrex::ParticleIDWrapper{data.m_idcpu[i]} = amrex::LongParticleIds::DoSplitParticleID;
                    amrex::ParticleCPUWrapper{data.m_idcpu[i]} = 7;
                } else if (data.m_rdata[PIdx::w][i] == 2.0_prt) {
                    amrex::ParticleIDWrapper{data.m_idcpu[i]} = amrex::LongParticleIds::NoSplitParticleID;
                }
                if (attributes) {
                    data.m_runtime_rdata[0][i] = data.m_rdata[PIdx::ux][i];
                    data.m_runtime_idata[0][i] = static_cast<int>(data.m_rdata[PIdx::w][i]);
                }
            });
        }
        amrex::Gpu::streamSynchronize();
        const auto weight_before = pc.sumParticleWeight(false);
        const auto energy_before = pc.sumParticleEnergy(false);
        pc.SplitParticles(0);
        pc.Redistribute();
        const int children = split_type == 0 ? (1 << AMREX_SPACEDIM) : 2*AMREX_SPACEDIM;
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(pc.TotalNumberOfParticles() == 2*children + 2,
            "Splitting must replace exactly two tagged parents and retain untagged particles");
        const auto tol = 128 * std::numeric_limits<amrex::ParticleReal>::epsilon();
        AMREX_ALWAYS_ASSERT(std::abs(pc.sumParticleWeight(false) - weight_before) <= tol*weight_before);
        AMREX_ALWAYS_ASSERT(std::abs(pc.sumParticleEnergy(false) - energy_before) <= tol*energy_before);
        amrex::Real momentum[3]{0, 0, 0};
        amrex::Real child_displacement[3]{0, 0, 0};
        for (WarpXParIter pti(pc, 0); pti.isValid(); ++pti) {
            auto const& soa = pti.GetStructOfArrays();
            const auto np = pti.numParticles();
            std::array<amrex::Gpu::PinnedVector<amrex::ParticleReal>, 4> host;
            for (int c = 0; c < 4; ++c) {
                host[c].resize(np);
                auto const& src = soa.GetRealData(PIdx::w + c);
                amrex::Gpu::copy(amrex::Gpu::deviceToHost, src.begin(), src.end(), host[c].begin());
            }
            amrex::Gpu::DeviceVector<amrex::ParticleReal> positions(3*np);
            auto* pos = positions.data();
            const auto get_position = GetParticlePosition<PIdx>(pti);
            amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) {
                get_position(i, pos[3*i], pos[3*i+1], pos[3*i+2]);
            });
            amrex::Gpu::PinnedVector<amrex::ParticleReal> host_pos(3*np);
            amrex::Gpu::PinnedVector<std::uint64_t> ids(np);
            amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                positions.begin(), positions.end(), host_pos.begin());
            auto const& src_ids = soa.GetIdCPUData();
            amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                src_ids.begin(), src_ids.end(), ids.begin());
            for (long i = 0; i < np; ++i) {
                for (int c = 0; c < 3; ++c) { momentum[c] += host[0][i]*host[c+1][i]; }
                const int parent = host[1][i] == ux[0] ? 0 :
                    (host[1][i] == ux[1] ? 1 : (host[1][i] == ux[2] ? 2 : 3));
                AMREX_ALWAYS_ASSERT(host[1][i] == ux[parent] &&
                    host[2][i] == uy[parent] && host[3][i] == uz[parent]);
                std::array<amrex::ParticleReal, 3> center{x[parent], y[parent], z[parent]};
#if defined(WARPX_DIM_1D_Z)
                center[0] = center[1] = 0;
#elif defined(WARPX_DIM_XZ)
                center[1] = 0;
#elif defined(WARPX_DIM_RCYLINDER)
                center[2] = 0;
#endif
                int shifted_axes = 0;
                for (int c = 0; c < 3; ++c) {
                    const auto delta = host_pos[3*i+c] - center[c];
                    AMREX_ALWAYS_ASSERT(std::isfinite(delta));
                    if (parent < 2) {
                        const auto offset = WarpX::CellSize(0)[c]/2;
#if defined(WARPX_DIM_1D_Z)
                        const bool active_axis = c == 2;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                        const bool active_axis = c != 1;
#elif defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
                        const bool active_axis = c == 0;
#else
                        const bool active_axis = true;
#endif
                        if (active_axis && std::abs(delta) > tol) {
                            AMREX_ALWAYS_ASSERT(std::abs(std::abs(delta)-offset) <= tol);
                            ++shifted_axes;
                        } else {
                            AMREX_ALWAYS_ASSERT(std::abs(delta) <= tol);
                        }
                        child_displacement[c] += delta;
                    } else {
                        AMREX_ALWAYS_ASSERT(std::abs(delta) <= tol);
                    }
                }
                if (parent < 2) {
                    AMREX_ALWAYS_ASSERT(std::abs(host[0][i] - weights[0][parent]/children)
                        <= tol*weights[0][parent]/children);
#if AMREX_SPACEDIM == 1
                    AMREX_ALWAYS_ASSERT(shifted_axes == 1);
#else
                    AMREX_ALWAYS_ASSERT(shifted_axes == (split_type == 0 ? AMREX_SPACEDIM : 1));
#endif
                    AMREX_ALWAYS_ASSERT(amrex::ConstParticleIDWrapper{ids[i]} ==
                        amrex::LongParticleIds::NoSplitParticleID);
                } else {
                    AMREX_ALWAYS_ASSERT(host[0][i] == weights[0][parent]);
                }
            }
            if (attributes) {
                amrex::Gpu::PinnedVector<amrex::ParticleReal> reals(np);
                amrex::Gpu::PinnedVector<int> ints(np);
                auto const& src_real = soa.GetRealData(pc.GetRealCompIndex("split_real"));
                auto const& src_int = soa.GetIntData(pc.GetIntCompIndex("split_int"));
                amrex::Gpu::copy(amrex::Gpu::deviceToHost, src_real.begin(), src_real.end(), reals.begin());
                amrex::Gpu::copy(amrex::Gpu::deviceToHost, src_int.begin(), src_int.end(), ints.begin());
                for (long i = 0; i < np; ++i) {
                    AMREX_ALWAYS_ASSERT(reals[i] == host[1][i]);
                    const int parent = host[1][i] == ux[0] ? 0 :
                        (host[1][i] == ux[1] ? 1 : (host[1][i] == ux[2] ? 2 : 3));
                    AMREX_ALWAYS_ASSERT(ints[i] == static_cast<int>(weights[0][parent]));
                }
            }
        }
        amrex::ParallelDescriptor::ReduceRealSum(momentum, 3);
        amrex::ParallelDescriptor::ReduceRealSum(child_displacement, 3);
        for (int c = 0; c < 3; ++c) {
            AMREX_ALWAYS_ASSERT(std::abs(child_displacement[c]) <= tol*2*children);
            const auto& u = c == 0 ? ux : (c == 1 ? uy : uz);
            amrex::Real expected = 0;
            for (int i = 0; i < 4; ++i) { expected += weights[0][i]*u[i]; }
            AMREX_ALWAYS_ASSERT(std::abs(momentum[c]-expected) <= tol*std::abs(expected));
        }
        // No selected parents remain; this must be a no-op, including for
        // empty local tiles and containers with runtime attributes.
        pc.SplitParticles(0);
        pc.Redistribute();
        AMREX_ALWAYS_ASSERT(pc.TotalNumberOfParticles() == 2*children + 2);
        amrex::Print() << "Particle splitting conservation and attributes: PASS\n";
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
catch (const std::exception& error)
{
    std::cerr << "Particle splitting test failed: " << error.what() << '\n';
    return 1;
}
catch (...)
{
    return 1;
}
