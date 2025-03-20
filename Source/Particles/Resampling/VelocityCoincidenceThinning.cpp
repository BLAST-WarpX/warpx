/* Copyright 2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald (TAE Technologies)
 *
 * License: BSD-3-Clause-LBNL
 */

#include "VelocityCoincidenceThinning.H"


VelocityCoincidenceThinning::VelocityCoincidenceThinning (const std::string& species_name)
{
    using namespace amrex::literals;

    const amrex::ParmParse pp_species_name(species_name);

    utils::parser::queryWithParser(
        pp_species_name, "resampling_min_ppc", m_min_ppc
    );
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_min_ppc >= 1,
        "Resampling min_ppc should be greater than or equal to 1"
    );

    amrex::ParticleReal target_weight = 0;
    if (utils::parser::queryWithParser(
        pp_species_name, "resampling_algorithm_target_weight", target_weight
    )) {
        // factor of 2 since each cluster is reduced to 2 particles
        m_cluster_weight = target_weight * 2.0_prt;
    }

    std::string velocity_grid_type_str = "spherical";
    pp_species_name.query(
        "resampling_algorithm_velocity_grid_type", velocity_grid_type_str
    );
    if (velocity_grid_type_str == "spherical") {
        m_velocity_grid_type = VelocityGridType::Spherical;
        utils::parser::getWithParser(
            pp_species_name, "resampling_algorithm_delta_ur", m_delta_ur
        );
        utils::parser::getWithParser(
            pp_species_name, "resampling_algorithm_n_theta", m_ntheta
        );
        utils::parser::getWithParser(
            pp_species_name, "resampling_algorithm_n_phi", m_nphi
        );
    }
    else if (velocity_grid_type_str == "cartesian") {
        m_velocity_grid_type = VelocityGridType::Cartesian;
        utils::parser::getArrWithParser(
            pp_species_name, "resampling_algorithm_delta_u", m_delta_u
        );
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_delta_u.size() == 3,
            "resampling_algorithm_delta_u must have three components.");
    }
    else {
        WARPX_ABORT_WITH_MESSAGE("Unkown velocity grid type.");
    }
}

void VelocityCoincidenceThinning::operator() (WarpXParIter& pti, const int lev,
                                   WarpXParticleContainer * const pc) const
{
    using namespace amrex::literals;

    auto& ptile = pc->ParticlesAt(lev, pti);
    const auto n_parts_in_tile = pti.numParticles();
    auto& soa = ptile.GetStructOfArrays();
#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_3D)
    auto * const AMREX_RESTRICT x = soa.GetRealData(PIdx::x).data();
#elif defined(WARPX_DIM_RZ)
    auto * const AMREX_RESTRICT x = soa.GetRealData(PIdx::x).data(); // rename to PIdx::r after PR #4667
#endif
#if defined(WARPX_DIM_3D)
    auto * const AMREX_RESTRICT y = soa.GetRealData(PIdx::y).data();
#endif
    auto * const AMREX_RESTRICT z = soa.GetRealData(PIdx::z).data();
    auto * const AMREX_RESTRICT ux = soa.GetRealData(PIdx::ux).data();
    auto * const AMREX_RESTRICT uy = soa.GetRealData(PIdx::uy).data();
    auto * const AMREX_RESTRICT uz = soa.GetRealData(PIdx::uz).data();
    auto * const AMREX_RESTRICT w = soa.GetRealData(PIdx::w).data();
    auto * const AMREX_RESTRICT idcpu = soa.GetIdCPUData().data();

    // Using this function means that we must loop over the cells in the ParallelFor.
    auto bins = ParticleUtils::findParticlesInEachCell(lev, pti, ptile);

    const auto n_cells = static_cast<int>(bins.numBins());
    auto *const indices = bins.permutationPtr();
    auto *const cell_offsets = bins.offsetsPtr();

    const auto min_ppc = m_min_ppc;
    const auto cluster_weight = m_cluster_weight;
    const auto mass = pc->getMass();

    // check if species mass > 0
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        mass > 0,
        "VelocityCoincidenceThinning does not yet work for massless particles."
    );

    // create a GPU vector to hold the momentum cluster index for each particle
    amrex::Gpu::DeviceVector<int> momentum_bin_number(n_parts_in_tile);
    auto* momentum_bin_number_data = momentum_bin_number.data();

    // create a GPU vector to hold the index sorting for the momentum bins
    amrex::Gpu::DeviceVector<int> sorted_indices(n_parts_in_tile);
    auto* sorted_indices_data = sorted_indices.data();

    constexpr auto c2 = PhysConst::c * PhysConst::c;

    auto velocityBinCalculator = VelocityBinCalculator();
    velocityBinCalculator.velocity_grid_type = m_velocity_grid_type;
    if (m_velocity_grid_type == VelocityGridType::Spherical) {
        velocityBinCalculator.dur = m_delta_ur;
        velocityBinCalculator.n1 = m_ntheta;
        velocityBinCalculator.n2 = m_nphi;
        velocityBinCalculator.dutheta = 2.0_prt * MathConst::pi / m_ntheta;
        velocityBinCalculator.duphi = MathConst::pi / m_nphi;
    }
    else if (m_velocity_grid_type == VelocityGridType::Cartesian) {
        velocityBinCalculator.dux = m_delta_u[0];
        velocityBinCalculator.duy = m_delta_u[1];
        velocityBinCalculator.duz = m_delta_u[2];

        // get the minimum and maximum velocities to determine the velocity space
        // grid boundaries
        {
            using ReduceOpsT = amrex::TypeMultiplier<amrex::ReduceOps,
                                                     amrex::ReduceOpMin[3],
                                                     amrex::ReduceOpMax[2]>;
            using ReduceDataT = amrex::TypeMultiplier<amrex::ReduceData, amrex::ParticleReal[5]>;
            ReduceOpsT reduce_op;
            ReduceDataT reduce_data(reduce_op);
            using ReduceTuple = typename ReduceDataT::Type;
            reduce_op.eval(n_parts_in_tile, reduce_data, [=] AMREX_GPU_DEVICE(int i) -> ReduceTuple {
                return {ux[i], uy[i], uz[i], ux[i], uy[i]};
            });
            auto hv = reduce_data.value(reduce_op);
            velocityBinCalculator.ux_min = amrex::get<0>(hv);
            velocityBinCalculator.uy_min = amrex::get<1>(hv);
            velocityBinCalculator.uz_min = amrex::get<2>(hv);
            velocityBinCalculator.ux_max = amrex::get<3>(hv);
            velocityBinCalculator.uy_max = amrex::get<4>(hv);
        }

        velocityBinCalculator.n1 = static_cast<int>(
            std::ceil((velocityBinCalculator.ux_max - velocityBinCalculator.ux_min) / m_delta_u[0])
        );
        velocityBinCalculator.n2 = static_cast<int>(
            std::ceil((velocityBinCalculator.uy_max - velocityBinCalculator.uy_min) / m_delta_u[1])
        );
    }
    auto heapSort = HeapSort();

    // Loop over cells
    amrex::ParallelForRNG( n_cells,
        [=] AMREX_GPU_DEVICE (int i_cell, amrex::RandomEngine const& engine) noexcept
        {
            // The particles that are in the cell `i_cell` are
            // given by the `indices[cell_start:cell_stop]`
            const auto cell_start = static_cast<int>(cell_offsets[i_cell]);
            const auto cell_stop  = static_cast<int>(cell_offsets[i_cell+1]);
            const auto cell_numparts = cell_stop - cell_start;

            // do nothing for cells with less particles than min_ppc
            // (this intentionally includes skipping empty cells, too)
            if (cell_numparts < min_ppc) {
                return;
            }

            // Loop over particles and label them with the appropriate momentum bin
            // number. Also assign initial ordering to the sorted_indices array.
            velocityBinCalculator(
                ux, uy, uz, indices, momentum_bin_number_data, sorted_indices_data,
                cell_start, cell_stop
            );

            // sort indices based on comparing values in momentum_bin_number
            heapSort(sorted_indices_data, momentum_bin_number_data, cell_start, cell_numparts);

            // initialize variables used to hold cluster totals
            int particles_in_bin = 0;
            amrex::ParticleReal total_weight = 0._prt, total_energy = 0._prt;
#if !defined(WARPX_DIM_1D_Z)
            amrex::ParticleReal cluster_x = 0._prt;
#endif
#if defined(WARPX_DIM_3D)
            amrex::ParticleReal cluster_y = 0._prt;
#endif
            amrex::ParticleReal cluster_z = 0._prt;
            amrex::ParticleReal cluster_ux = 0._prt, cluster_uy = 0._prt, cluster_uz = 0._prt;

            // Finally, loop through the particles in the cell and merge
            // ones in the same momentum bin
            for (int i = cell_start; i < cell_stop; ++i)
            {
                particles_in_bin += 1;
                const auto part_idx = indices[sorted_indices_data[i]];

#if !defined(WARPX_DIM_1D_Z)
                cluster_x += w[part_idx]*x[part_idx];
#endif
#if defined(WARPX_DIM_3D)
                cluster_y += w[part_idx]*y[part_idx];
#endif
                cluster_z += w[part_idx]*z[part_idx];
                cluster_ux += w[part_idx]*ux[part_idx];
                cluster_uy += w[part_idx]*uy[part_idx];
                cluster_uz += w[part_idx]*uz[part_idx];
                total_weight += w[part_idx];
                total_energy += w[part_idx] * Algorithms::KineticEnergy(
                    ux[part_idx], uy[part_idx], uz[part_idx], mass
                );

                // check if this is the last particle in the current momentum bin,
                // or if the next particle would push the current cluster weight
                // to exceed the maximum specified cluster weight
                if (
                    (i == cell_stop - 1)
                    || (momentum_bin_number_data[sorted_indices_data[i]] != momentum_bin_number_data[sorted_indices_data[i + 1]])
                    || (total_weight + w[indices[sorted_indices_data[i+1]]] > cluster_weight)
                ) {
                    // check if the bin has more than 2 particles in it
                    if ( particles_in_bin > 2 && total_weight > std::numeric_limits<amrex::ParticleReal>::min() ){
                        // get average quantities for the bin
#if !defined(WARPX_DIM_1D_Z)
                        cluster_x /= total_weight;
#endif
#if defined(WARPX_DIM_3D)
                        cluster_y /= total_weight;
#endif
                        cluster_z /= total_weight;
                        cluster_ux /= total_weight;
                        cluster_uy /= total_weight;
                        cluster_uz /= total_weight;

                        // perform merging of momentum bin particles
                        auto u_perp2 = cluster_ux*cluster_ux + cluster_uy*cluster_uy;
                        auto u_perp = std::sqrt(u_perp2);
                        auto cluster_u_mag2 = u_perp2 + cluster_uz*cluster_uz;
                        auto cluster_u_mag = std::sqrt(cluster_u_mag2);

                        // calculate required velocity magnitude to achieve
                        // energy conservation
                        auto v_mag2 = total_energy / total_weight * (
                            (total_energy / total_weight + 2._prt * mass * c2 )
                            / (mass * mass * c2)
                        );
                        auto v_perp = (v_mag2 > cluster_u_mag2) ? std::sqrt(v_mag2 - cluster_u_mag2) : 0_prt;

                        // choose random angle for new velocity vector
                        auto phi = amrex::Random(engine) * MathConst::pi;

                        // set new velocity components based on chosen phi
                        auto vx = v_perp * std::cos(phi);
                        auto vy = v_perp * std::sin(phi);

                        // calculate rotation angles to parallel coord. frame
                        auto cos_theta = (cluster_u_mag > 0._prt) ? cluster_uz / cluster_u_mag : 0._prt;
                        auto sin_theta = (cluster_u_mag > 0._prt) ? u_perp / cluster_u_mag : 0._prt;
                        auto cos_phi = (u_perp > 0._prt) ? cluster_ux / u_perp : 0._prt;
                        auto sin_phi = (u_perp > 0._prt) ? cluster_uy / u_perp : 0._prt;

                        // rotate new velocity vector to labframe
                        auto ux_new = (
                            vx * cos_theta * cos_phi - vy * sin_phi
                            + cluster_u_mag * sin_theta * cos_phi
                        );
                        auto uy_new = (
                            vx * cos_theta * sin_phi + vy * cos_phi
                            + cluster_u_mag * sin_theta * sin_phi
                        );
                        auto uz_new = -vx * sin_theta + cluster_u_mag * cos_theta;

                        // set the last two particles' attributes according to
                        // the bin's aggregate values
                        const auto part_idx2 = indices[sorted_indices_data[i - 1]];

                        w[part_idx] = total_weight / 2._prt;
                        w[part_idx2] = total_weight / 2._prt;
#if !defined(WARPX_DIM_1D_Z)
                        x[part_idx] = cluster_x;
                        x[part_idx2] = cluster_x;
#endif
#if defined(WARPX_DIM_3D)
                        y[part_idx] = cluster_y;
                        y[part_idx2] = cluster_y;
#endif
                        z[part_idx] = cluster_z;
                        z[part_idx2] = cluster_z;

                        ux[part_idx] = ux_new;
                        uy[part_idx] = uy_new;
                        uz[part_idx] = uz_new;
                        ux[part_idx2] = 2._prt * cluster_ux - ux_new;
                        uy[part_idx2] = 2._prt * cluster_uy - uy_new;
                        uz[part_idx2] = 2._prt * cluster_uz - uz_new;

                        // set ids of merged particles so they will be removed
                        for (int j = 2; j < particles_in_bin; ++j){
                            idcpu[indices[sorted_indices_data[i - j]]] = amrex::ParticleIdCpus::Invalid;
                        }
                    }

                    // restart the tallies
                    particles_in_bin = 0;
                    total_weight = 0._prt;
                    total_energy = 0._prt;
#if !defined(WARPX_DIM_1D_Z)
                    cluster_x = 0_prt;
#endif
#if defined(WARPX_DIM_3D)
                    cluster_y = 0_prt;
#endif
                    cluster_z = 0_prt;
                    cluster_ux = 0_prt;
                    cluster_uy = 0_prt;
                    cluster_uz = 0_prt;
                }
            }
        }
    );
}
