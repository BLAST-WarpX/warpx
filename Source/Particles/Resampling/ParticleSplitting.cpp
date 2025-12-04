/* Copyright 2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "ParticleSplitting.H"
#include "Particles/PhysicalParticleContainer.H"
#include "Utils/Parser/ParserUtils.H"
#include "WarpX.H"
#include <AMReX_REAL.H>
#include <AMReX_Gpu.H>
#include <AMReX_Random.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_ParmParse.H>

using namespace amrex;
using warpx::fields::FieldType;

ParticleSplitting::ParticleSplitting (const std::string& species_name)
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

    utils::parser::queryWithParser(
        pp_species_name, "resampling_random_splitting_angle", m_resampling_random_splitting_angle);

    pp_species_name.query("resampling_splitting_type", m_splitting_type);

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_splitting_type == "position_axes_aligned_split" || m_splitting_type == "position_velocity_aligned_split",
        "Invalid resampling_splitting_type specified.\n"
        "Valid options are:\n"
        "  - position_axes_aligned_split\n"
        "  - position_velocity_aligned_split.\n");

    m_splitting_type_id = (m_splitting_type == "position_axes_aligned_split") ? 1 : 0;

    utils::parser::queryWithParser(
        pp_species_name, "resampling_splitting_angle", m_splitting_angle);
}
void ParticleSplitting::operator() (
    const amrex::Geometry& geom_lev, WarpXParIter& pti,
    const int lev, WarpXParticleContainer * const pc) const
{
    using namespace amrex::literals;

    auto& ptile = pc->ParticlesAt(lev, pti);
    const auto num_particles_tile = ptile.numParticles();

    if (num_particles_tile == 0) return;
    // Bin particles by cell
    auto bins = ParticleUtils::findParticlesInEachCell(geom_lev, pti, ptile);
    const auto n_cells = static_cast<int>(bins.numBins());
    auto *const indices = bins.permutationPtr();
    auto *const cell_offsets = bins.offsetsPtr();

    const std::array<amrex::Real,3>& dx = WarpX::CellSize(lev);

    int np_split_per_parent = 2;

    if (m_splitting_type_id == 1) {
#if defined(WARPX_DIM_1D_Z)
        np_split_per_parent = 2;
#endif
#if defined(WARPX_DIM_3D)
        np_split_per_parent = 6;
#endif
#if defined(WARPX_DIM_XZ)
        np_split_per_parent = 4;
#endif
    } else {
        np_split_per_parent = 2;
    }
    int splitting_type_id = m_splitting_type_id;

    const auto min_ppc = m_min_ppc;
    const auto resampling_random_splitting_angle = m_resampling_random_splitting_angle;
    const amrex::Real splitting_angle_fixed = m_splitting_angle;

    amrex::Gpu::DeviceVector<int> n_new_children_per_cell(n_cells);
    int* num_new_children_ptr = n_new_children_per_cell.data();

    amrex::ParallelFor(n_cells,
        [=] AMREX_GPU_DEVICE (int i_cell) noexcept
        {
            const auto cell_start = static_cast<int>(cell_offsets[i_cell]);
            const auto cell_stop  = static_cast<int>(cell_offsets[i_cell+1]);
            const auto cell_numparts = cell_stop - cell_start;

            // Skip cells with enough particles or empty cells
            if (cell_numparts == 0 || cell_numparts >= min_ppc) {
                num_new_children_ptr[i_cell] = 0;
                return;
            }

            // Calculate how many particles need to be split per cell
            const int deficit = min_ppc - cell_numparts;
            const int particles_to_split = amrex::min(
                cell_numparts,
                static_cast<int>(amrex::Math::ceil(static_cast<amrex::Real>(deficit) / (np_split_per_parent - 1.0_prt)))
            );

            // Each parent that splits creates np_split_per_parent children per cell, or particles_to_split * np_split_per_parent per tile
            num_new_children_ptr[i_cell] = particles_to_split * np_split_per_parent;
        }
    );

    amrex::Gpu::DeviceVector<int> offsets(n_cells);
    int* offset_ptr = offsets.data();

    int num_new_children_tile = amrex::Scan::ExclusiveSum(n_cells, num_new_children_ptr, offset_ptr);

    if (num_new_children_tile == 0) return;

    ptile.resize(num_particles_tile + num_new_children_tile);

    auto& soa = ptile.GetStructOfArrays();

#if !defined(WARPX_DIM_1D_Z)
    auto * const AMREX_RESTRICT x = soa.GetRealData(PIdx::x).data();
#endif
#if defined(WARPX_DIM_3D)
    auto * const AMREX_RESTRICT y = soa.GetRealData(PIdx::y).data();
#endif
#if defined(WARPX_ZINDEX)
    auto * const AMREX_RESTRICT z = soa.GetRealData(PIdx::z).data();
#endif

    auto * const AMREX_RESTRICT ux = soa.GetRealData(PIdx::ux).data();
    auto * const AMREX_RESTRICT uy = soa.GetRealData(PIdx::uy).data();
    auto * const AMREX_RESTRICT uz = soa.GetRealData(PIdx::uz).data();
    auto * const AMREX_RESTRICT w = soa.GetRealData(PIdx::w).data();
    auto * const AMREX_RESTRICT idcpu = soa.GetIdCPUData().data();

    amrex::ParallelForRNG(n_cells,
        [=] AMREX_GPU_DEVICE (int i_cell, amrex::RandomEngine const& engine) noexcept
        {
            const auto cell_start = static_cast<int>(cell_offsets[i_cell]);
            const auto cell_stop  = static_cast<int>(cell_offsets[i_cell+1]);
            const auto cell_numparts = cell_stop - cell_start;

            // Skip cells with enough particles or empty cells
            if (cell_numparts == 0 || cell_numparts >= min_ppc) return;

            // Calculate how many particles need to be split
            const int deficit = min_ppc - cell_numparts;
            const int particles_to_split = amrex::min(
                cell_numparts,
                static_cast<int>(amrex::Math::ceil(static_cast<amrex::Real>(deficit) / (np_split_per_parent - 1.0_prt)))
            );

            // Calculate cell-dependent position offset
            const amrex::Real inv_split = 1.0_prt / (2.0_prt * np_split_per_parent * particles_to_split);
#if !defined(WARPX_DIM_1D_Z)
            amrex::ParticleReal offset_x = dx[0] * inv_split;
#endif
#if defined(WARPX_DIM_3D)
            amrex::ParticleReal offset_y = dx[1] * inv_split;
#endif
#if defined(WARPX_ZINDEX)
            amrex::ParticleReal offset_z = dx[2] * inv_split;
#endif

            auto phi = amrex::Random(engine) * 2.0_rt * MathConst::pi;

            amrex::Real splitting_angle = resampling_random_splitting_angle ? phi : splitting_angle_fixed;
            amrex::ignore_unused(splitting_angle);

            // Starting index for new children particles for i_cell
            const int new_particle_start = num_particles_tile + offset_ptr[i_cell];

            int split_count = 0;
            for (int i = cell_start; i < cell_stop && split_count < particles_to_split; ++i) {
                const int parent_idx = indices[i];

#if !defined(WARPX_DIM_1D_Z)
                amrex::ParticleReal xp  =  x[parent_idx];
#endif
#if defined(WARPX_DIM_3D)
                amrex::ParticleReal yp  =  y[parent_idx];
#endif
#if defined(WARPX_ZINDEX)
                amrex::ParticleReal zp  =  z[parent_idx];
#endif
                // Get parent particle properties
                const amrex::Real parent_weight = w[parent_idx];
                const amrex::Real child_weight = parent_weight / static_cast<amrex::Real>(np_split_per_parent);
                const int child_base = new_particle_start + split_count * np_split_per_parent;

                if (splitting_type_id == 1) {
#if defined(WARPX_DIM_1D_Z)
                    amrex::Print() << "Splitting particle along z axis (WARPX_DIM_1D_Z)" << std::endl;
                    // Split particle in 2 along z axis
                    for (int k = 0; k < 2; ++k) {
                        const int sign_offset = (k == 0) ? -1 : 1;
                        const int idx = child_base + k;
                        z[idx] = zp + sign_offset * offset_z;
                        ux[idx] = ux[parent_idx];
                        uy[idx] = uy[parent_idx];
                        uz[idx] = uz[parent_idx];
                        w[idx] = child_weight;
                        idcpu[idx] = amrex::SetParticleIDandCPU(
                            amrex::LongParticleIds::NoSplitParticleID,
                            amrex::ParallelDescriptor::MyProc()
                        );
                    }
#elif defined(WARPX_DIM_XZ)
                    // Split particle in 4 particles: split in the x–z plane with a rotation by splitting_angle around the z-axis.
                    for (int k = 0; k < 4; ++k)  {
                        const int idx = child_base + k;
                        const int sign_offset = (k % 2 == 0) ? -1 : 1;
                        if (k < 2) {
                            // Split 2 of 4 particles
                            x[idx] = xp + std::cos(splitting_angle) * sign_offset * offset_x;
                            z[idx] = zp - std::sin(splitting_angle) * sign_offset * offset_z;
                        } else {
                            // Split other 2 of 4 particles
                            x[idx] = xp - std::sin(splitting_angle) * sign_offset * offset_x;
                            z[idx] = zp + std::cos(splitting_angle) * sign_offset * offset_z;
                        }
                        ux[idx] = ux[parent_idx];
                        uy[idx] = uy[parent_idx];
                        uz[idx] = uz[parent_idx];
                        w[idx] = child_weight;
                        idcpu[idx] = amrex::SetParticleIDandCPU(
                            amrex::LongParticleIds::NoSplitParticleID,
                            amrex::ParallelDescriptor::MyProc()
                        );
                    }
#elif defined(WARPX_DIM_3D)
                    // Split parent particle in 6 particles
                    for (int k = 0; k < 6; ++k) {
                        const int idx = child_base + k;
                        const int sign_offset = (k % 2 == 0) ? -1 : 1;

                        if (k < 4) {
                            //  Split 4 of 6 particles in the x–y plane with a rotation by splitting_angle around the z-axis.
                            if (k < 2) {
                                x[idx] = xp + std::cos(splitting_angle) * sign_offset * offset_x;
                                y[idx] = yp + std::sin(splitting_angle) * sign_offset * offset_y;
                            } else {
                                x[idx] = xp - std::sin(splitting_angle) * sign_offset * offset_x;
                                y[idx] = yp + std::cos(splitting_angle) * sign_offset * offset_y;
                            }
                            z[idx] = zp;
                        } else {
                            // Split 2 of 6 particles along z-axis
                            x[idx] = xp;
                            y[idx] = yp;
                            z[idx] = zp + sign_offset * offset_z;
                        }
                        ux[idx] = ux[parent_idx];
                        uy[idx] = uy[parent_idx];
                        uz[idx] = uz[parent_idx];
                        w[idx] = child_weight;
                        idcpu[idx] = amrex::SetParticleIDandCPU(
                            amrex::LongParticleIds::NoSplitParticleID,
                            amrex::ParallelDescriptor::MyProc()
                        );
                    }
#elif defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
                    // Split particle in 2 along x axis
                    for (int k = 0; k < 2; ++k) {
                        const int sign_offset = (k == 0) ? -1 : 1;
                        const int idx = child_base + k;
                        x[idx] = xp + sign_offset * offset_x;
                        ux[idx] = ux[parent_idx];
                        uy[idx] = uy[parent_idx];
                        uz[idx] = uz[parent_idx];
                        w[idx] = child_weight;
                        idcpu[idx] = amrex::SetParticleIDandCPU(
                            amrex::LongParticleIds::NoSplitParticleID,
                            amrex::ParallelDescriptor::MyProc()
                        );
                    }
#endif
                }
                else if (splitting_type_id == 0) {
                    // Split particle in 2 along the velocity direction
                    const amrex::Real u2 = ux[parent_idx] * ux[parent_idx] +
                                           uy[parent_idx] * uy[parent_idx] +
                                           uz[parent_idx] * uz[parent_idx];
                    const amrex::Real u_norm = (std::sqrt(u2) > 0.0_rt) ? std::sqrt(u2) : 1.0_rt;

                    for (int k = 0; k < 2; ++k) {
                        const int sign_offset = (k == 0) ? -1 : 1;
                        const int idx = child_base + k;
#if !defined(WARPX_DIM_1D_Z)
                        x[idx] = xp + sign_offset * offset_x * ux[parent_idx] / u_norm;
#endif
#if defined(WARPX_DIM_3D)
                        y[idx] = yp + sign_offset * offset_y * uy[parent_idx] / u_norm;
#endif
#if defined(WARPX_ZINDEX)
                        z[idx] = zp + sign_offset * offset_z * uz[parent_idx] / u_norm;
#endif
                        ux[idx] = ux[parent_idx];
                        uy[idx] = uy[parent_idx];
                        uz[idx] = uz[parent_idx];
                        w[idx] = child_weight;
                        idcpu[idx] = amrex::SetParticleIDandCPU(
                            amrex::LongParticleIds::NoSplitParticleID,
                            amrex::ParallelDescriptor::MyProc()
                        );
                    }
                }

                // Mark parent particles as invalid
                idcpu[parent_idx] = amrex::ParticleIdCpus::Invalid;
                ++split_count;
            }
        }
    );
}
