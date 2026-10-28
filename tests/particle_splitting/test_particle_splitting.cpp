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
    std::string geometry_case = "interior";
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        const std::string prefix = "test.geometry_case=";
        if (arg.starts_with(prefix)) { geometry_case = arg.substr(prefix.size()); }
    }
    const bool roundoff = geometry_case == "boundary_roundoff";
    const bool pending_boundary = geometry_case == "pending_boundary";
    const bool contained_boundary = geometry_case == "contained_boundary";
    const bool boundary = geometry_case == "boundary" || roundoff || pending_boundary
        || contained_boundary;
    const bool periodic_crossing = geometry_case == "periodic";
    const bool spacing = geometry_case == "spacing";
    const bool nongridded = geometry_case == "nongridded";
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
    // Radial 1D geometries split the stored radius.
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
    if (spacing) {
#if defined(WARPX_DIM_3D)
        args.emplace_back("test.num_particles_per_cell_each_dim=2 3 5");
#elif defined(WARPX_DIM_RZ)
        args.emplace_back("test.num_particles_per_cell_each_dim=2 7 5");
#elif defined(WARPX_DIM_XZ)
        args.emplace_back("test.num_particles_per_cell_each_dim=2 5");
#elif defined(WARPX_DIM_1D_Z)
        args.emplace_back("test.num_particles_per_cell_each_dim=5");
#elif defined(WARPX_DIM_RCYLINDER)
        args.emplace_back("test.num_particles_per_cell_each_dim=2 7");
#else
        args.emplace_back("test.num_particles_per_cell_each_dim=2 7 5");
#endif
    }
    if (nongridded) { args.emplace_back("test.injection_style=none"); }
    if (boundary) {
#if defined(WARPX_DIM_3D)
        args.insert(args.end(), {"boundary.field_lo=pec pec pec",
            "boundary.field_hi=pec pec pec",
            "boundary.particle_lo=reflecting reflecting reflecting",
            "boundary.particle_hi=reflecting reflecting reflecting"});
#elif defined(WARPX_DIM_XZ)
        args.insert(args.end(), {"boundary.field_lo=pec pec", "boundary.field_hi=pec pec",
            "boundary.particle_lo=reflecting reflecting",
            "boundary.particle_hi=reflecting reflecting"});
#elif defined(WARPX_DIM_1D_Z)
        args.insert(args.end(), {"boundary.field_lo=pec", "boundary.field_hi=pec",
            "boundary.particle_lo=reflecting", "boundary.particle_hi=reflecting"});
#elif defined(WARPX_DIM_RZ)
        args.insert(args.end(), {"boundary.field_lo=none pec", "boundary.field_hi=pec pec",
            "boundary.particle_lo=none reflecting", "boundary.particle_hi=reflecting reflecting"});
#endif
    }
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
        amrex::Vector<amrex::ParticleReal> x{0.375_prt, 0.499_prt, 0.625_prt, 0.75_prt};
        amrex::Vector<amrex::ParticleReal> y(4, 0.0_prt);
        amrex::Vector<amrex::ParticleReal> z{0.375_prt, 0.499_prt, 0.625_prt, 0.75_prt};
#ifdef WARPX_DIM_3D
        y.assign(4, 0.375_prt);
#endif
        if (boundary || periodic_crossing) {
            for (int i = 0; i < 2; ++i) {
                const auto face = contained_boundary ? (i == 0 ? 0.0625_prt : 0.9375_prt)
                                                     : (i == 0 ? 0.001_prt : 0.999_prt);
                x[i] = face;
                z[i] = face;
#ifdef WARPX_DIM_3D
                y[i] = face;
#elif defined(WARPX_DIM_RZ)
                if (periodic_crossing) { x[i] = i == 0 ? 0.375_prt : 0.625_prt; }
#endif
            }
        }
        if (roundoff) {
            // AMReX deliberately excludes some physical-face-adjacent floating
            // point values. Exercise its actual inclusive particle bounds.
            for (int i = 0; i < 2; ++i) {
                const auto face = i == 0 ? warpx.Geom(0).ProbLoArrayInParticleReal()
                                        : warpx.Geom(0).ProbHiArrayInParticleReal();
#if defined(WARPX_DIM_3D)
                x[i] = face[0]; y[i] = face[1]; z[i] = face[2];
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                x[i] = face[0]; z[i] = face[1];
#elif defined(WARPX_DIM_1D_Z)
                z[i] = face[0];
#else
                x[i] = face[0];
#endif
            }
        }
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
        const bool angles = geometry_case == "angles";
        for (int i = 0; i < 4; ++i) {
            const auto radius = x[i];
            const auto theta = angles ? (i%2 == 0 ? 0.7_prt : -1.2_prt) : 0.0_prt;
            x[i] = radius*std::cos(theta);
            y[i] = radius*std::sin(theta);
#if defined(WARPX_DIM_RSPHERE)
            const auto phi = angles ? (i%2 == 0 ? 0.4_prt : -0.6_prt) : 0.0_prt;
            x[i] *= std::cos(phi);
            y[i] *= std::cos(phi);
            z[i] = radius*std::sin(phi);
#elif defined(WARPX_DIM_RCYLINDER)
            z[i] = 0.0_prt;
#endif
        }
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
                    if (pending_boundary) {
                        // Emulate a pushed parent that still awaits physical BC.
                        data.pos(AMREX_SPACEDIM-1, i) = 1.001_prt;
                    }
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
        if (pending_boundary) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(pc.TotalNumberOfParticles() == 4,
                "Parents awaiting physical BC must not be replaced by split children");
            AMREX_ALWAYS_ASSERT(pc.sumParticleWeight(false) == weight_before);
            AMREX_ALWAYS_ASSERT(pc.sumParticleEnergy(false) == energy_before);
            // Reflect the two outside parents through the normal species handler.
            pc.ApplyBoundaryConditions();
            pc.Redistribute();
            AMREX_ALWAYS_ASSERT(pc.TotalNumberOfParticles() == 4);
            AMREX_ALWAYS_ASSERT(pc.sumParticleWeight(false) == weight_before);
            amrex::Print() << "Pending-boundary parent preservation: PASS\n";
            WarpX::Finalize();
            warpx::initialization::finalize_external_libraries();
            return 0;
        }
        pc.Redistribute();
        const int children = split_type == 0 ? (1 << AMREX_SPACEDIM) : 2*AMREX_SPACEDIM;
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(pc.TotalNumberOfParticles() == 2*children + 2,
            "Splitting must replace exactly two tagged parents and retain untagged particles");
        const auto tol = 128 * std::numeric_limits<amrex::ParticleReal>::epsilon();
        AMREX_ALWAYS_ASSERT(std::abs(pc.sumParticleWeight(false) - weight_before) <= tol*weight_before);
        AMREX_ALWAYS_ASSERT(std::abs(pc.sumParticleEnergy(false) - energy_before) <= tol*energy_before);
        amrex::Real momentum[3]{0, 0, 0};
        amrex::Real child_displacement[6]{};
        amrex::Real cartesian_displacement[6]{};
        std::vector<int> ppc;
        pp.getarr("num_particles_per_cell_each_dim", ppc);
        const auto& geom = warpx.Geom(0);
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
                get_position.AsStored(i, pos[3*i], pos[3*i+1], pos[3*i+2]);
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
#elif defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER)
                center[0] = std::hypot(x[parent], y[parent]);
                center[1] = std::atan2(y[parent], x[parent]);
#elif defined(WARPX_DIM_RSPHERE)
                center[0] = std::sqrt(x[parent]*x[parent] + y[parent]*y[parent]
                    + z[parent]*z[parent]);
                center[1] = std::atan2(y[parent], x[parent]);
                center[2] = std::atan2(z[parent], std::hypot(x[parent], y[parent]));
#endif
                int shifted_axes = 0;
                for (int c = 0; c < 3; ++c) {
                    auto delta = host_pos[3*i+c] - center[c];
                    AMREX_ALWAYS_ASSERT(std::isfinite(delta));
                    if (periodic_crossing && std::abs(delta) > 0.5_prt) {
                        delta -= std::round(delta); // Unwrap the unit periodic domain.
                    }
                    if (parent < 2) {
#if defined(WARPX_DIM_1D_Z)
                        const int ppc_axis = 0;
#elif defined(WARPX_DIM_XZ)
                        const int ppc_axis = c/2;
#else
                        const int ppc_axis = c;
#endif
                        const auto count = nongridded || ppc_axis >= static_cast<int>(ppc.size())
                            ? 1 : ppc[ppc_axis];
                        const auto offset = WarpX::CellSize(0)[c]/(2*count);
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
                            if (boundary && !contained_boundary) {
                                AMREX_ALWAYS_ASSERT(std::abs(delta) <= offset + tol);
                            } else {
                                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                                    std::abs(std::abs(delta)-offset) <= tol,
                                    "Split spacing must use the physical axis particle count");
                            }
                            ++shifted_axes;
                        } else {
                            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(delta) <= tol,
                                "Splitting must preserve inactive and angular coordinates");
                        }
                        child_displacement[3*parent+c] += delta;
                    } else {
                        AMREX_ALWAYS_ASSERT(std::abs(delta) <= tol);
                    }
                }
                for (int axis = 0; axis < AMREX_SPACEDIM; ++axis) {
#if defined(WARPX_DIM_1D_Z)
                    const int c = 2;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                    const int c = 2*axis;
#else
                    const int c = axis;
#endif
                    AMREX_ALWAYS_ASSERT(host_pos[3*i+c] >= geom.ProbLo(axis));
                    AMREX_ALWAYS_ASSERT(host_pos[3*i+c] < geom.ProbHi(axis));
                }
                if (parent < 2 && !periodic_crossing) {
                    std::array<amrex::Real, 3> xyz{host_pos[3*i], host_pos[3*i+1], host_pos[3*i+2]};
                    std::array<amrex::Real, 3> original{x[parent], y[parent], z[parent]};
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
                    xyz[0] = host_pos[3*i]*std::cos(host_pos[3*i+1]);
                    xyz[1] = host_pos[3*i]*std::sin(host_pos[3*i+1]);
#if defined(WARPX_DIM_RSPHERE)
                    xyz[0] *= std::cos(host_pos[3*i+2]);
                    xyz[1] *= std::cos(host_pos[3*i+2]);
                    xyz[2] = host_pos[3*i]*std::sin(host_pos[3*i+2]);
#endif
#elif defined(WARPX_DIM_1D_Z)
                    original[0] = original[1] = 0;
#elif defined(WARPX_DIM_XZ)
                    original[1] = 0;
#endif
                    for (int c = 0; c < 3; ++c) {
                        cartesian_displacement[3*parent+c] += xyz[c]-original[c];
                    }
                }
                if (parent < 2) {
                    AMREX_ALWAYS_ASSERT(std::abs(host[0][i] - weights[0][parent]/children)
                        <= tol*weights[0][parent]/children);
                    if (!roundoff) {
#if AMREX_SPACEDIM == 1
                        AMREX_ALWAYS_ASSERT(shifted_axes == 1);
#else
                        AMREX_ALWAYS_ASSERT(shifted_axes == (split_type == 0 ? AMREX_SPACEDIM : 1));
#endif
                    }
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
        amrex::ParallelDescriptor::ReduceRealSum(child_displacement, 6);
        amrex::ParallelDescriptor::ReduceRealSum(cartesian_displacement, 6);
        for (int c = 0; c < 6; ++c) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(child_displacement[c]) <= tol*children,
                "Children must preserve each parent's centroid");
            AMREX_ALWAYS_ASSERT(std::abs(cartesian_displacement[c]) <= tol*children);
        }
        for (int c = 0; c < 3; ++c) {
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
