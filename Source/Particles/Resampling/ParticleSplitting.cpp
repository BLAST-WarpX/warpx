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
        pp_species_name, "resampling_split_weight_koef", m_split_weight_koef);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_split_weight_koef > 0._prt,
        "resampling_split_weight_koef must be positive."
    );

    utils::parser::queryWithParser(
        pp_species_name, "splitting_max_ppc", m_split_max_ppc);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_split_max_ppc >= 0,
        "splitting_max_ppc must be non-negative (0 = no limit)."
    );

    utils::parser::queryWithParser(
        pp_species_name, "do_random_splitting_angle", m_do_random_splitting_angle);

    pp_species_name.query("resampling_splitting_type", m_splitting_type);

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_splitting_type == "axis_aligned" || m_splitting_type == "velocity_aligned" || m_splitting_type == "trivial",
        "Invalid resampling_splitting_type specified.\n"
        "Valid options are:\n"
        "  - trivial\n"
        "  - axis_aligned\n"
        "  - velocity_aligned.\n");

    if (m_splitting_type == "trivial") {
        m_splitting_type_id = 0;
    } else if (m_splitting_type == "axis_aligned") {
        m_splitting_type_id = 1;
    } else if (m_splitting_type == "velocity_aligned") {
        m_splitting_type_id = 2;
    }

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

    if (m_splitting_type_id == 0 || m_splitting_type_id == 1) {
    #if defined(WARPX_DIM_3D)
        np_split_per_parent = 6;
    #elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
        np_split_per_parent = 4;
    #else
        np_split_per_parent = 2; // cases: WARPX_DIM_1D_Z, WARPX_DIM_RCYLINDER, WARPX_DIM_RSPHERE
    #endif
    }
    else if (m_splitting_type_id == 2) {
        np_split_per_parent = 2;
    }

    int splitting_type_id = m_splitting_type_id;
    const auto split_min_ppc = m_min_ppc;
    const auto do_random_splitting_angle = m_do_random_splitting_angle;
    const amrex::Real splitting_angle_fixed = m_splitting_angle;
    const amrex::Real split_weight_koef = m_split_weight_koef;
    const int split_max_ppc = m_split_max_ppc;

    amrex::Gpu::DeviceVector<int> n_new_children_per_cell(n_cells);
    int* num_new_children_ptr = n_new_children_per_cell.data();

    amrex::Gpu::DeviceVector<amrex::Real> w_avg_per_cell(static_cast<std::size_t>(n_cells));
    amrex::Gpu::DeviceVector<int> n_split_parents_per_cell(static_cast<std::size_t>(n_cells));
    amrex::Real* w_avg_ptr = w_avg_per_cell.data();
    int* n_split_parents_ptr = n_split_parents_per_cell.data();
    {
        auto& soa = ptile.GetStructOfArrays();
        auto const* const AMREX_RESTRICT w = soa.GetRealData(PIdx::w).data();

        amrex::ParallelFor(n_cells,
            [=] AMREX_GPU_DEVICE (int i_cell) noexcept
            {
                const auto cell_start = static_cast<int>(cell_offsets[i_cell]);
                const auto cell_stop  = static_cast<int>(cell_offsets[i_cell+1]);
                const int cell_numparts = cell_stop - cell_start;

                // do nothing for cells with fewer than split_min_ppc macroparticles
                if (cell_numparts < split_min_ppc) {
                    num_new_children_ptr[i_cell] = 0;
                    w_avg_ptr[i_cell] = 0._prt;
                    n_split_parents_ptr[i_cell] = 0;
                    return;
                }
                // if already too many macroparticles per cell — do not split
                if (split_max_ppc > 0 &&
                    cell_numparts > split_max_ppc) {
                    num_new_children_ptr[i_cell] = 0;
                    w_avg_ptr[i_cell] = 0._prt;
                    n_split_parents_ptr[i_cell] = 0;
                    return;
                }
                // get averaged weight in the cell
                amrex::Real average_weight = 0._prt;
                for (int i = cell_start; i < cell_stop; ++i) {
                    average_weight += w[indices[i]];
                }
                average_weight /= static_cast<amrex::Real>(cell_numparts);

                int n_split_parents = 0;
                for (int i = cell_start; i < cell_stop; ++i) {
                    if (w[indices[i]] > split_weight_koef * average_weight) {
                        ++n_split_parents;
                    }
                }

                w_avg_ptr[i_cell] = average_weight;
                n_split_parents_ptr[i_cell] = n_split_parents;
                num_new_children_ptr[i_cell] = n_split_parents * np_split_per_parent;
            }
        );
    }

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
#if defined(WARPX_DIM_RZ)|| defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    auto * const AMREX_RESTRICT theta = soa.GetRealData(PIdx::theta).data();
#endif
#if defined(WARPX_DIM_RSPHERE)
    auto * const AMREX_RESTRICT phi = soa.GetRealData(PIdx::phi).data();
#endif
    auto * const AMREX_RESTRICT ux = soa.GetRealData(PIdx::ux).data();
    auto * const AMREX_RESTRICT uy = soa.GetRealData(PIdx::uy).data();
    auto * const AMREX_RESTRICT uz = soa.GetRealData(PIdx::uz).data();
    auto * const AMREX_RESTRICT w = soa.GetRealData(PIdx::w).data();
    auto * const AMREX_RESTRICT idcpu = soa.GetIdCPUData().data();

    const int cpu_rank = amrex::ParallelDescriptor::MyProc();
    amrex::ParallelForRNG(n_cells,
        [=] AMREX_GPU_DEVICE (int i_cell, amrex::RandomEngine const& engine) noexcept
        {
            const auto cell_start = static_cast<int>(cell_offsets[i_cell]);
            const auto cell_stop  = static_cast<int>(cell_offsets[i_cell+1]);
            const int cell_numparts = cell_stop - cell_start;

            if (cell_numparts == 0 || cell_numparts < split_min_ppc) { return; }

            if (split_max_ppc > 0 &&
                cell_numparts > split_max_ppc) {
                return;
            }

            const amrex::Real w_avg = w_avg_ptr[i_cell];
            const int n_split_parents = n_split_parents_ptr[i_cell];

            if (n_split_parents == 0) { return; }

            // calculate cell-dependent position offset
            const amrex::Real offset_fraction = 1.0_prt / (5.0_prt);
#if !defined(WARPX_DIM_1D_Z)
            amrex::ParticleReal offset_x = dx[0] * offset_fraction;
#endif
#if defined(WARPX_DIM_3D)
            amrex::ParticleReal offset_y = dx[1] * offset_fraction;
#endif
#if defined(WARPX_ZINDEX)
            amrex::ParticleReal offset_z = dx[2] * offset_fraction;
#endif
            const amrex::Real splitting_angle =
            do_random_splitting_angle
            ? amrex::Random(engine) * 2.0_rt * MathConst::pi
            : splitting_angle_fixed;

            amrex::ignore_unused(splitting_angle);

            // starting index for new children particles for i_cell
            const int new_particle_start_ind = num_particles_tile + offset_ptr[i_cell];

            int split_count = 0;
            for (int i = cell_start; i < cell_stop; ++i) {
                if (split_count >= n_split_parents) { break; }

                const int parent_idx = indices[i];
                // splitting threshold condition: split if particle weight is above the splitting threshold
                // defined as split_weight_koef * (cell average weight)
                bool split_heavy_particle = (w[parent_idx] > split_weight_koef * w_avg);
                if (!split_heavy_particle ) {
                    continue;
                }

#if !defined(WARPX_DIM_1D_Z)
                amrex::ParticleReal xp  =  x[parent_idx];
#endif
#if defined(WARPX_DIM_3D)
                amrex::ParticleReal yp  =  y[parent_idx];
#endif
#if defined(WARPX_ZINDEX)
                amrex::ParticleReal zp  =  z[parent_idx];
#endif
                // get parent particle properties
                const amrex::Real parent_weight = w[parent_idx];
                const amrex::Real child_weight = parent_weight / static_cast<amrex::Real>(np_split_per_parent);
                const int child_base = new_particle_start_ind + split_count * np_split_per_parent;
                if (splitting_type_id == 0) {
                    // create np_split_per_parent trivial children (positions unchanged)
                    for (int k = 0; k < np_split_per_parent; ++k) {
                        const int idx = child_base + k;
#if !defined(WARPX_DIM_1D_Z)
                        x[idx] = xp;
#endif
#if defined(WARPX_DIM_3D)
                        y[idx] = yp;
#endif
#if defined(WARPX_ZINDEX)
                        z[idx] = zp;
#endif
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
                        theta[idx] = theta[parent_idx];
#endif
#if defined(WARPX_DIM_RSPHERE)
                        phi[idx] = phi[parent_idx];
#endif
                        ux[idx] = ux[parent_idx];
                        uy[idx] = uy[parent_idx];
                        uz[idx] = uz[parent_idx];
                        w[idx] = child_weight;
                        idcpu[idx] = amrex::SetParticleIDandCPU(
                            amrex::LongParticleIds::NoSplitParticleID,
                            cpu_rank
                        );
                    }
                }
                else if (splitting_type_id == 1) {
                    // split particle in 2 along z axis
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
                    bool do_trivial_split = false;
                    for (int k = 0; k < np_split_per_parent; ++k) {
                        const int sx = (k & 1) ? 1 : -1;
                        amrex::Real r_child = xp + sx * offset_x;
                        if (r_child < 0.0) {
                            do_trivial_split = true;
                            break;
                        }
                    }
#elif defined(WARPX_DIM_XZ)
                    // Cartesian XZ: no radial coordinate; trivial path in XZ||RZ block is never taken.
                    bool const do_trivial_split = false;
#endif
                    for (int k = 0; k < np_split_per_parent; ++k) {
                        const int idx = child_base + k;
#if defined(WARPX_DIM_1D_Z)
                        const int sz = (k == 0) ? -1 : 1;
                        z[idx] = zp + sz * offset_z;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                    // split particle in 4 particles: split in the x–z plane with a rotation by splitting_angle around the z-axis.
                        const int sx = (k & 1) ? 1 : -1;
                        const int sz = (k & 2) ? 1 : -1;
                        if (!do_trivial_split) {
                            x[idx] = xp + sx * offset_x;
                            z[idx] = zp + sz * offset_z;
                        } else {
                            x[idx] = xp;
                            z[idx] = zp; // if split would produce negative radius, do trivial split instead
                        }
#elif defined(WARPX_DIM_3D)
                    // split parent particle in 6 particles
                        const int sign_offset = (k % 2 == 0) ? -1 : 1;
                        const int child_pair_index = k / 2;
                        const amrex::ParticleReal c = std::cos(splitting_angle);
                        const amrex::ParticleReal s = std::sin(splitting_angle);
                        if (child_pair_index == 0) {
                            x[idx] = xp + sign_offset * c * offset_x;
                            y[idx] = yp + sign_offset * s * offset_y;
                            z[idx] = zp;
                        } else if (child_pair_index == 1) {
                            x[idx] = xp - sign_offset * s * offset_x;
                            y[idx] = yp + sign_offset * c * offset_y;
                            z[idx] = zp;
                        } else {
                            x[idx] = xp;
                            y[idx] = yp;
                            z[idx] = zp + sign_offset * offset_z;
                        }
#elif defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
                        const int sx = (k == 0) ? -1 : 1;
                        if (!do_trivial_split) {
                            x[idx] = xp + sx * offset_x;
                        } else {
                            x[idx] = xp;
                        }
#endif
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
                        theta[idx] = theta[parent_idx];
#endif
#if defined(WARPX_DIM_RSPHERE)
                        phi[idx] = phi[parent_idx];
#endif
                        ux[idx] = ux[parent_idx];
                        uy[idx] = uy[parent_idx];
                        uz[idx] = uz[parent_idx];
                        w[idx] = child_weight;
                        idcpu[idx] = amrex::SetParticleIDandCPU(
                            amrex::LongParticleIds::NoSplitParticleID,
                            cpu_rank
                        );
                    }
                }
                else if (splitting_type_id == 2) {
                    // split particle in 2 along the particle momentum direction
                    amrex::Real u_norm = 0._rt;
                    amrex::Real u2 = 0._rt;
                    amrex::ParticleReal offset = std::numeric_limits<amrex::Real>::max();

#if !defined(WARPX_DIM_1D_Z)
                    u2  += ux[parent_idx] * ux[parent_idx];
                    offset = amrex::min(offset, offset_x);
#endif
#if defined(WARPX_DIM_3D)
                    u2  += uy[parent_idx] * uy[parent_idx];
                    offset = amrex::min(offset, offset_y);
#endif
#if defined(WARPX_ZINDEX)
                    u2  += uz[parent_idx] * uz[parent_idx];
                    offset = amrex::min(offset, offset_z);
#endif
                    u_norm = std::sqrt(u2);
                    bool do_trivial_split = (u_norm == 0._rt);
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
                    if (!do_trivial_split) {
                        for (int k = 0; k < np_split_per_parent; ++k) {
                            const int sx = (k & 1) ? 1 : -1;
                            amrex::Real r_child = xp + sx * offset * ux[parent_idx] / u_norm;
                            if (r_child< 0.0_rt) {
                                do_trivial_split = true;
                                break;
                            }
                        }
                    }
#endif
                    for (int k = 0; k < 2; ++k) {
                        const int sign_offset = (k == 0) ? -1 : 1;
                        const int idx = child_base + k;
                        if (!do_trivial_split) {
#if !defined(WARPX_DIM_1D_Z)
                            x[idx] = xp + sign_offset * offset * ux[parent_idx] / u_norm;
#endif
#if defined(WARPX_DIM_3D)
                            y[idx] = yp + sign_offset * offset * uy[parent_idx] / u_norm;
#endif
#if defined(WARPX_ZINDEX)
                            z[idx] = zp + sign_offset * offset * uz[parent_idx] / u_norm;
#endif
                        }
                        else {
#if !defined(WARPX_DIM_1D_Z)
                            x[idx] = xp;
#endif
#if defined(WARPX_DIM_3D)
                            y[idx] = yp;
#endif
#if defined(WARPX_ZINDEX)
                            z[idx] = zp; // if velocity is zero, split is trivial
#endif
                        }
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
                        theta[idx] = theta[parent_idx];
#endif
#if defined(WARPX_DIM_RSPHERE)
                        phi[idx] = phi[parent_idx];
#endif
                        ux[idx] = ux[parent_idx];
                        uy[idx] = uy[parent_idx];
                        uz[idx] = uz[parent_idx];
                        w[idx] = child_weight;
                        idcpu[idx] = amrex::SetParticleIDandCPU(
                            amrex::LongParticleIds::NoSplitParticleID,
                            cpu_rank
                        );
                    }
                }
                // mark parent particles as invalid
                idcpu[parent_idx] = amrex::ParticleIdCpus::Invalid;
                ++split_count;
            }
        }
    );
}
