/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "Particles/Deposition/BorisWorkCurrentDeposition.H"
#include "Particles/Gather/FieldGather.H"

#include <AMReX.H>
#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_IntVect.H>
#include <AMReX_Reduce.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <type_traits>

namespace
{
    struct WorkParticle
    {
        amrex::ParticleReal x;
        amrex::ParticleReal y;
        amrex::ParticleReal z;
        amrex::ParticleReal q_weight;
        BorisElectricWorkVelocity work_velocity;
    };

    struct OperatorResult
    {
        amrex::Real grid_inner_product;
        amrex::ParticleReal particle_inner_product;
        amrex::Real relative_residual;
    };

    char const* dimension_name () noexcept
    {
#if defined(WARPX_DIM_1D_Z)
        return "1D_Z";
#elif defined(WARPX_DIM_XZ)
        return "2D_XZ";
#elif defined(WARPX_DIM_3D)
        return "3D";
#else
        return "unsupported";
#endif
    }

    std::array<WorkParticle, 48> make_particles (
        amrex::XDim3 const& xyzmin,
        amrex::XDim3 const& dinv)
    {
        std::array<WorkParticle, 48> particles{};
        for (int ip = 0; ip < static_cast<int>(particles.size()); ++ip) {
            // Non-grid-aligned positions keep every shape order active while
            // remaining at least four cells from the FArrayBox edge.
            double const gx = 4.125
                + 13.25 * static_cast<double>((37 * ip + 11) % 127) / 127.0;
            double const gy = 4.375
                + 12.75 * static_cast<double>((53 * ip + 17) % 131) / 131.0;
            double const gz = 4.625
                + 12.25 * static_cast<double>((71 * ip + 23) % 137) / 137.0;

            WorkParticle& particle = particles[ip];
            particle.x = static_cast<amrex::ParticleReal>(
                xyzmin.x + static_cast<amrex::Real>(gx) / dinv.x);
            particle.y = static_cast<amrex::ParticleReal>(
                xyzmin.y + static_cast<amrex::Real>(gy) / dinv.y);
            particle.z = static_cast<amrex::ParticleReal>(
                xyzmin.z + static_cast<amrex::Real>(gz) / dinv.z);

            amrex::ParticleReal const sign =
                (ip % 2 == 0) ? amrex::ParticleReal(1.0)
                              : amrex::ParticleReal(-1.0);
            amrex::ParticleReal const weight = amrex::ParticleReal(0.15)
                + amrex::ParticleReal(0.19) * static_cast<amrex::ParticleReal>(
                    1 + (ip % 9));
            particle.q_weight = sign * weight;

            // Correlating the work-velocity sign with the charge sign keeps
            // the total inner product away from cancellation while still
            // exercising both signed atomic updates and nonuniform weights.
            particle.work_velocity.x = sign * (
                amrex::ParticleReal(0.31)
                + amrex::ParticleReal(0.017) * static_cast<amrex::ParticleReal>(ip % 7));
            particle.work_velocity.y = sign * (
                amrex::ParticleReal(0.23)
                + amrex::ParticleReal(0.013) * static_cast<amrex::ParticleReal>(ip % 5));
            particle.work_velocity.z = sign * (
                amrex::ParticleReal(0.19)
                + amrex::ParticleReal(0.011) * static_cast<amrex::ParticleReal>(ip % 11));
        }
        return particles;
    }

    template <int gather_order, int scatter_order>
    OperatorResult run_operator_case (bool const nodal)
    {
        constexpr int num_particles = 48;
        amrex::IntVect const cell_lo(AMREX_D_DECL(-5, -4, -3));
        amrex::IntVect const cell_hi(AMREX_D_DECL(20, 21, 22));
        amrex::Box const cell_box(cell_lo, cell_hi);
        amrex::IntVect const field_stagger = nodal
            ? amrex::IntVect::TheNodeVector()
            : amrex::IntVect::TheZeroVector();
        amrex::Box const field_box = amrex::convert(cell_box, field_stagger);
        amrex::IntVect const field_type = field_box.type();
        amrex::Dim3 const lo = amrex::lbound(field_box);

        amrex::XDim3 const xyzmin{
            amrex::Real(-0.37), amrex::Real(0.29), amrex::Real(1.13)};
        amrex::XDim3 const dinv{
            amrex::Real(1.0 / 0.19),
            amrex::Real(1.0 / 0.23),
            amrex::Real(1.0 / 0.17)};
        amrex::Real const invvol = dinv.x * dinv.y * dinv.z;
        amrex::Real const cell_volume = amrex::Real(1.0) / invvol;

        amrex::FArrayBox ex_fab(field_box, 1);
        amrex::FArrayBox ey_fab(field_box, 1);
        amrex::FArrayBox ez_fab(field_box, 1);
        amrex::FArrayBox jx_fab(field_box, 1);
        amrex::FArrayBox jy_fab(field_box, 1);
        amrex::FArrayBox jz_fab(field_box, 1);
        jx_fab.setVal<amrex::RunOn::Device>(0.0);
        jy_fab.setVal<amrex::RunOn::Device>(0.0);
        jz_fab.setVal<amrex::RunOn::Device>(0.0);

        amrex::Array4<amrex::Real> const ex = ex_fab.array();
        amrex::Array4<amrex::Real> const ey = ey_fab.array();
        amrex::Array4<amrex::Real> const ez = ez_fab.array();
        amrex::ParallelFor(field_box,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                amrex::Real const phase = amrex::Real(0.73) * i
                    - amrex::Real(0.47) * j + amrex::Real(0.61) * k;
                amrex::Real const phase_two = amrex::Real(1.37) * i
                    + amrex::Real(0.89) * j - amrex::Real(0.43) * k;
                ex(i, j, k) = amrex::Real(1.8)
                    + amrex::Real(0.55) * std::sin(phase)
                    + amrex::Real(0.21) * std::cos(phase_two);
                ey(i, j, k) = amrex::Real(1.4)
                    + amrex::Real(0.49) * std::cos(amrex::Real(1.17) * phase)
                    - amrex::Real(0.17) * std::sin(amrex::Real(0.83) * phase_two);
                ez(i, j, k) = amrex::Real(1.1)
                    - amrex::Real(0.43) * std::sin(amrex::Real(0.91) * phase)
                    + amrex::Real(0.19) * std::cos(amrex::Real(1.21) * phase_two);
            });

        std::array<WorkParticle, num_particles> const host_particles =
            make_particles(xyzmin, dinv);
        amrex::Gpu::DeviceVector<WorkParticle> device_particles(num_particles);
        amrex::Gpu::DeviceVector<amrex::ParticleReal> particle_products(
            num_particles);
        amrex::Gpu::copy(
            amrex::Gpu::hostToDevice,
            host_particles.begin(), host_particles.end(),
            device_particles.begin());

        WorkParticle const* const particles = device_particles.data();
        amrex::ParticleReal* const products = particle_products.data();
        amrex::Array4<amrex::Real const> const ex_const = ex_fab.const_array();
        amrex::Array4<amrex::Real const> const ey_const = ey_fab.const_array();
        amrex::Array4<amrex::Real const> const ez_const = ez_fab.const_array();
        amrex::Array4<amrex::Real> const jx = jx_fab.array();
        amrex::Array4<amrex::Real> const jy = jy_fab.array();
        amrex::Array4<amrex::Real> const jz = jz_fab.array();
        amrex::IndexType const gather_type = field_box.ixType();

        auto const particle_operator =
            [=] AMREX_GPU_DEVICE (int ip) noexcept
        {
            WorkParticle const particle = particles[ip];
            amrex::ParticleReal exp = 0.0;
            amrex::ParticleReal eyp = 0.0;
            amrex::ParticleReal ezp = 0.0;
            amrex::ParticleReal bxp = 0.0;
            amrex::ParticleReal byp = 0.0;
            amrex::ParticleReal bzp = 0.0;
            doGatherShapeN<gather_order, 0>(
                particle.x, particle.y, particle.z,
                exp, eyp, ezp, bxp, byp, bzp,
                ex_const, ey_const, ez_const,
                ex_const, ey_const, ez_const,
                gather_type, gather_type, gather_type,
                gather_type, gather_type, gather_type,
                dinv, xyzmin, lo, 1);

            products[ip] = particle.q_weight * (
                particle.work_velocity.x * exp
                + particle.work_velocity.y * eyp
                + particle.work_velocity.z * ezp);

            doBorisWorkCurrentDepositionShapeN<scatter_order>(
                particle.x, particle.y, particle.z,
                particle.q_weight, particle.work_velocity,
                jx, jy, jz, field_type,
                dinv, xyzmin, invvol, lo);
        };

        // Exercise actual host-thread contention in the OpenMP build.  GPU
        // builds use the same ParallelFor launch as the fused particle pusher.
#if defined(AMREX_USE_OMP) && !defined(AMREX_USE_GPU)
#pragma omp parallel for schedule(static)
        for (int ip = 0; ip < num_particles; ++ip) {
            particle_operator(ip);
        }
#else
        amrex::ParallelFor(num_particles, particle_operator);
#endif

        auto const particle_inner_product =
            amrex::Reduce::Sum<amrex::ParticleReal>(
                num_particles, particle_products.data());

        amrex::ReduceOps<amrex::ReduceOpSum> reduce_ops;
        amrex::ReduceData<amrex::Real> reduce_data(reduce_ops);
        using ReduceTuple = typename decltype(reduce_data)::Type;
        amrex::Array4<amrex::Real const> const jx_const = jx_fab.const_array();
        amrex::Array4<amrex::Real const> const jy_const = jy_fab.const_array();
        amrex::Array4<amrex::Real const> const jz_const = jz_fab.const_array();
        reduce_ops.eval(field_box, reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> ReduceTuple
            {
                return {
                    ex_const(i, j, k) * jx_const(i, j, k)
                    + ey_const(i, j, k) * jy_const(i, j, k)
                    + ez_const(i, j, k) * jz_const(i, j, k)};
            });
        amrex::Real const grid_inner_product = cell_volume
            * amrex::get<0>(reduce_data.value());

        auto const particle_as_real =
            static_cast<amrex::Real>(particle_inner_product);
        amrex::Real const scale = std::max({
            std::abs(grid_inner_product), std::abs(particle_as_real),
            amrex::Real(1.0)});
        amrex::Real const relative_residual =
            std::abs(grid_inner_product - particle_as_real) / scale;
        return {grid_inner_product, particle_inner_product, relative_residual};
    }

    template <int order>
    bool check_exact_case (bool const nodal)
    {
        OperatorResult const result = run_operator_case<order, order>(nodal);
        amrex::Real const arithmetic_epsilon = std::max(
            std::numeric_limits<amrex::Real>::epsilon(),
            static_cast<amrex::Real>(
                std::numeric_limits<amrex::ParticleReal>::epsilon()));
        amrex::Real const tolerance = amrex::Real(2048.0)
            * arithmetic_epsilon;
        bool const pass = result.relative_residual <= tolerance;
        std::cout << dimension_name() << " order=" << order
                  << " staggering=" << (nodal ? "nodal" : "cell")
                  << " H/H* residual=" << result.relative_residual
                  << " tolerance=" << tolerance
                  << (pass ? " PASS" : " FAIL") << '\n';
        return pass;
    }
}

int main (int argc, char* argv[])
{
    static_assert(std::is_trivially_copyable_v<WorkParticle>);

    amrex::Initialize(argc, argv);
    bool pass = true;
    {
#if defined(WARPX_DIM_1D_Z) || defined(WARPX_DIM_XZ) || defined(WARPX_DIM_3D)
        pass = check_exact_case<1>(true) && pass;
        pass = check_exact_case<2>(true) && pass;
        pass = check_exact_case<3>(true) && pass;
        pass = check_exact_case<4>(true) && pass;
        pass = check_exact_case<1>(false) && pass;
        pass = check_exact_case<2>(false) && pass;
        pass = check_exact_case<3>(false) && pass;
        pass = check_exact_case<4>(false) && pass;

        // A reduced-order scatter is not the transpose of the full-order
        // direct gather.  This negative control makes the test sensitive to
        // the common but non-conservative Galerkin/order-mismatch error.
        OperatorResult const mismatch = run_operator_case<4, 0>(true);
        auto const minimum_mismatch = amrex::Real(1.0e-3);
        bool const negative_control_pass =
            mismatch.relative_residual > minimum_mismatch;
        std::cout << dimension_name()
                  << " gather-order=4 scatter-order=0 negative-control residual="
                  << mismatch.relative_residual
                  << " required>" << minimum_mismatch
                  << (negative_control_pass ? " PASS" : " FAIL") << '\n';
        pass = negative_control_pass && pass;
#else
        std::cerr << "This test supports Cartesian 1D_Z, 2D_XZ, and 3D only.\n";
        pass = false;
#endif
    }
    amrex::Finalize();
    return pass ? 0 : 1;
}
