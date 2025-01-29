/* Copyright 2021 Axel Huebl, Remi Lehe
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef ABLASTR_PARTICLE_MOMENTS_H
#define ABLASTR_PARTICLE_MOMENTS_H

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParticleReduce.H>
#include <AMReX_REAL.H>
#include <AMReX_Reduce.H>
#include <AMReX_Tuple.H>

#include <tuple>
#include <vector>


namespace ablastr {
namespace particles {

    /** Compute the min and max of the particle position in each dimension
     *
     * \tparam T_PC a type of amrex::ParticleContainer
     *
     * \param pc the particle container to operate on
     * \returns x_min, y_min, z_min, x_max, y_max, z_max
     */
    template< typename T_PC >
    static
    std::tuple<
            amrex::ParticleReal, amrex::ParticleReal,
            amrex::ParticleReal, amrex::ParticleReal,
            amrex::ParticleReal, amrex::ParticleReal>
    MinAndMaxPositions (T_PC const & pc)
    {
        using ConstParticleTileDataType = typename T_PC::ParticleTileType::ConstParticleTileDataType;

        // Get min and max for the local rank
        amrex::ReduceOps<
            amrex::ReduceOpMin, amrex::ReduceOpMin, amrex::ReduceOpMin,
            amrex::ReduceOpMax, amrex::ReduceOpMax, amrex::ReduceOpMax> reduce_ops;
        auto r = amrex::ParticleReduce<
            amrex::ReduceData<amrex::ParticleReal, amrex::ParticleReal, amrex::ParticleReal,
                              amrex::ParticleReal, amrex::ParticleReal, amrex::ParticleReal>
        >(
            pc,
            [=] AMREX_GPU_DEVICE(const ConstParticleTileDataType& ptd, const int i) noexcept
            {
                const amrex::ParticleReal x = ptd.rdata(0)[i];
                const amrex::ParticleReal y = ptd.rdata(1)[i];
                const amrex::ParticleReal z = ptd.rdata(2)[i];

                return amrex::makeTuple(x, y, z, x, y, z);
            },
            reduce_ops
        );

        // Get min and max across all ranks
        std::vector< amrex::ParticleReal > xyz_min = {
            amrex::get<0>(r),
            amrex::get<1>(r),
            amrex::get<2>(r)
        };
        amrex::ParallelDescriptor::ReduceRealMin(xyz_min.data(), xyz_min.size());
        std::vector< amrex::ParticleReal > xyz_max = {
            amrex::get<3>(r),
            amrex::get<4>(r),
            amrex::get<5>(r)
        };
        amrex::ParallelDescriptor::ReduceRealMax(xyz_max.data(), xyz_max.size());

        return {xyz_min[0], xyz_min[1], xyz_min[2], xyz_max[0], xyz_max[1], xyz_max[2]};
    }

    /** Compute the mean and std of the particle position in each dimension
     *
     * \tparam T_PC a type of amrex::ParticleContainer
     * \tparam T_RealSoAWeight
     *
     * \param pc the particle container to operate on
     * \returns x_mean, x_std, y_mean, y_std, z_mean, z_std
     */
    template< typename T_PC, int T_RealSoAWeight >
    static
    std::tuple<
            amrex::ParticleReal, amrex::ParticleReal,
            amrex::ParticleReal, amrex::ParticleReal,
            amrex::ParticleReal, amrex::ParticleReal>
    MeanAndStdPositions (T_PC const & pc)
    {

        using ConstParticleTileDataType = typename T_PC::ParticleTileType::ConstParticleTileDataType;

        amrex::ReduceOps<
            amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
            amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
            amrex::ReduceOpSum> reduce_ops;
        auto r = amrex::ParticleReduce<
            amrex::ReduceData<
                amrex::ParticleReal, amrex::ParticleReal, amrex::ParticleReal,
                amrex::ParticleReal, amrex::ParticleReal, amrex::ParticleReal,
                amrex::ParticleReal>
        >(
            pc,
            [=] AMREX_GPU_DEVICE(const ConstParticleTileDataType& ptd, const int i) noexcept
            {

                const amrex::ParticleReal x = ptd.rdata(0)[i];
                const amrex::ParticleReal y = ptd.rdata(1)[i];
                const amrex::ParticleReal z = ptd.rdata(2)[i];

                const amrex::ParticleReal w = ptd.rdata(T_RealSoAWeight)[i];

                return amrex::makeTuple(x, x*x, y, y*y, z, z*z, w);
            },
            reduce_ops
        );

        // Reduce across MPI ranks
        std::vector<amrex::ParticleReal> data_vector = {
            amrex::get<0>(r),
            amrex::get<1>(r),
            amrex::get<2>(r),
            amrex::get<3>(r),
            amrex::get<4>(r),
            amrex::get<5>(r),
            amrex::get<6>(r)
        };
        amrex::ParallelDescriptor::ReduceRealSum(data_vector.data(), data_vector.size());

        amrex::ParticleReal w_sum = data_vector[6];
        amrex::ParticleReal x_mean = data_vector[0] / w_sum;
        amrex::ParticleReal x_std = data_vector[1] / w_sum- x_mean * x_mean;
        amrex::ParticleReal y_mean = data_vector[2] / w_sum;
        amrex::ParticleReal y_std = data_vector[3] / w_sum- x_mean * x_mean;
        amrex::ParticleReal z_mean = data_vector[4] / w_sum;
        amrex::ParticleReal z_std = data_vector[5] / w_sum- x_mean * x_mean;

        return {x_mean, x_std, y_mean, y_std, z_mean, z_std};
    }

} // namespace particles
} // namespace ablastr

#endif // ABLASTR_PARTICLE_MOMENTS_H
