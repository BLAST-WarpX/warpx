/* Copyright 2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald (TAE Technologies)
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_VELOCITY_COINCIDENCE_THINNING_H_
#define WARPX_VELOCITY_COINCIDENCE_THINNING_H_

#include "Particles/Algorithms/KineticEnergy.H"
#include "Resampling.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/ParticleUtils.H"

/**
 * \brief This class implements a particle merging scheme wherein particles
 * are clustered in phase space and particles in the same cluster is merged
 * into two remaining particles. The scheme conserves linear momentum and
 * kinetic energy within each cluster.
 */
class VelocityCoincidenceThinning: public ResamplingAlgorithm {
public:

    /**
     * \brief Default constructor of the VelocityCoincidenceThinning class.
     */
    VelocityCoincidenceThinning () = default;

    /**
     * \brief Constructor of the VelocityCoincidenceThinning class
     *
     * @param[in] species_name the name of the resampled species
     */
    VelocityCoincidenceThinning (const std::string& species_name);

    enum struct VelocityGridType {
        Spherical = 0,
        Cartesian = 1
    };

    /**
     * \brief A method that performs merging for the considered species.
     *
     * @param[in] pti WarpX particle iterator of the particles to resample.
     * @param[in] lev the index of the refinement level.
     * @param[in] pc a pointer to the particle container.
     */
    void operator() (WarpXParIter& pti, int lev, WarpXParticleContainer* pc) const final;

    /**
     * \brief This merging routine requires functionality to sort a GPU vector
     * based on another GPU vector's values. The heap-sort functions below were
     * obtained from https://www.geeksforgeeks.org/iterative-heap-sort/ and
     * modified for the current purpose. It achieves the same as
     * ```
     * std::sort(
     *    sorted_indices_data + cell_start, sorted_indices_data + cell_stop,
     *    [&momentum_bin_number_data](size_t i1, size_t i2) {
     *        return momentum_bin_number_data[i1] < momentum_bin_number_data[i2];
     *    }
     * );
     * ```
     * but with support for device execution.
    */
    struct HeapSort {

        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        void swap(int &a, int &b) const
        {
            const auto temp = b;
            b = a;
            a = temp;
        }

        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        void operator() (int index_array[], const int bin_array[], const int start, const int n) const
        {
            // sort index_array into a max heap structure
            for (int i = 1; i < n; i++)
            {
                auto j = i;
                // move child through heap if it is bigger than its parent
                while (j > 0 && bin_array[index_array[j+start]] > bin_array[index_array[(j - 1)/2 + start]]) {
                    // swap child and parent until branch is properly ordered
                    swap(index_array[j+start], index_array[(j - 1)/2 + start]);
                    j = (j - 1) / 2;
                }
            }

            for (int i = n - 1; i > 0; i--)
            {
                // swap value of first (now the largest value) to the new end point
                swap(index_array[start], index_array[i+start]);

                // remake the max heap
                int j = 0, index;
                while (j < i) {
                    index = 2 * j + 1;

                    // if left child is smaller than right child, point index variable to right child
                    if (index + 1 < i && bin_array[index_array[index+start]] < bin_array[index_array[index+1+start]]) {
                        index++;
                    }
                    // if parent is smaller than child, swap parent with child having higher value
                    if (index < i && bin_array[index_array[j+start]] < bin_array[index_array[index+start]]) {
                        swap(index_array[j+start], index_array[index+start]);
                    }
                    j = index;
                }
            }
        }
    };

    /**
     * \brief Struct used to assign velocity space bin numbers to a given set
     * of particles.
    */
    struct VelocityBinCalculator {

        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        void labelOnSphericalVelocityGrid (const amrex::ParticleReal ux[],
                                           const amrex::ParticleReal uy[],
                                           const amrex::ParticleReal uz[],
                                           const int indices[],
                                           int bin_array[], int index_array[],
                                           const int cell_start, const int cell_stop ) const
        {
            for (int i = cell_start; i < cell_stop; ++i)
            {
                // get polar components of the velocity vector
                auto u_mag = std::sqrt(
                    ux[indices[i]]*ux[indices[i]] +
                    uy[indices[i]]*uy[indices[i]] +
                    uz[indices[i]]*uz[indices[i]]
                );
                auto u_theta = std::atan2(uy[indices[i]], ux[indices[i]]) + MathConst::pi;
                auto u_phi = std::acos(uz[indices[i]]/u_mag);

                const int ii = static_cast<int>(u_theta / dutheta);
                const int jj = static_cast<int>(u_phi / duphi);
                const int kk = static_cast<int>(u_mag / dur);

                bin_array[i] = ii + jj * n1 + kk * n1 * n2;
                index_array[i] = i;
            }
        }

        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        void labelOnCartesianVelocityGrid (const amrex::ParticleReal ux[],
                                           const amrex::ParticleReal uy[],
                                           const amrex::ParticleReal uz[],
                                           const int indices[],
                                           int bin_array[], int index_array[],
                                           const int cell_start, const int cell_stop ) const
        {
            for (int i = cell_start; i < cell_stop; ++i)
            {
                const int ii = static_cast<int>((ux[indices[i]] - ux_min) / dux);
                const int jj = static_cast<int>((uy[indices[i]] - uy_min) / duy);
                const int kk = static_cast<int>((uz[indices[i]] - uz_min) / duz);

                bin_array[i] = ii + jj * n1 + kk * n1 * n2;
                index_array[i] = i;
            }
        }

        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        void operator() (const amrex::ParticleReal ux[], const amrex::ParticleReal uy[],
                         const amrex::ParticleReal uz[], const int indices[],
                         int bin_array[], int index_array[],
                         const int cell_start, const int cell_stop) const
        {
            if (velocity_grid_type == VelocityGridType::Spherical) {
                labelOnSphericalVelocityGrid(
                    ux, uy, uz, indices, bin_array, index_array, cell_start,
                    cell_stop
                );
            }
            else if (velocity_grid_type == VelocityGridType::Cartesian) {
                labelOnCartesianVelocityGrid(
                    ux, uy, uz, indices, bin_array, index_array, cell_start,
                    cell_stop
                );
            }
        }

        VelocityGridType velocity_grid_type;
        int n1, n2;
        amrex::ParticleReal dur, dutheta, duphi;
        amrex::ParticleReal dux, duy, duz;
        amrex::ParticleReal ux_min, uy_min, uz_min, ux_max, uy_max;
    };

private:
    VelocityGridType m_velocity_grid_type;

    int m_min_ppc = 1;
    int m_ntheta, m_nphi;
    amrex::ParticleReal m_delta_ur;
    amrex::Vector<amrex::ParticleReal> m_delta_u;
    amrex::ParticleReal m_cluster_weight = std::numeric_limits<amrex::ParticleReal>::max();
};
#endif // WARPX_VELOCITY_COINCIDENCE_THINNING_H_
