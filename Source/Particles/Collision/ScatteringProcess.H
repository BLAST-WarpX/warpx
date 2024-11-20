/* Copyright 2021-2023 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Modern Electron, Roelof Groenewald (TAE Technologies)
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_COLLISION_SCATTERING_PROCESS_H_
#define WARPX_PARTICLES_COLLISION_SCATTERING_PROCESS_H_

#include <AMReX_Math.H>
#include <AMReX_Vector.H>
#include <AMReX_RandomEngine.H>
#include <AMReX_GpuContainers.H>

enum class ScatteringProcessType {
    INVALID,
    ELASTIC,
    BACK,
    CHARGE_EXCHANGE,
    EXCITATION,
    IONIZATION,
};

class ScatteringProcess
{
public:
    ScatteringProcess (
                const std::string& scattering_process,
                const std::string& cross_section_file,
                amrex::ParticleReal energy
                );

    template <typename InputVector>
    ScatteringProcess (
                const std::string& scattering_process,
                const InputVector&& energies,
                const InputVector&& sigmas,
                amrex::ParticleReal energy
                );

    ~ScatteringProcess() = default;

    ScatteringProcess (ScatteringProcess const&)            = delete;
    ScatteringProcess& operator= (ScatteringProcess const&) = delete;
    ScatteringProcess (ScatteringProcess &&)                = default;
    ScatteringProcess& operator= (ScatteringProcess &&)     = default;

    /** Read the given cross-section data file to memory.
     *
     * @param cross_section_file the path to the file containing the cross-
     *        section data
     * @param energies vector storing energy values in eV
     * @param sigmas vector storing cross-section values
     *
     */
    static
    void readCrossSectionFile (
                               const std::string& cross_section_file,
                               amrex::Vector<amrex::ParticleReal>& energies,
                               amrex::Gpu::HostVector<amrex::ParticleReal>& sigmas
                               );

    static
    void sanityCheckEnergyGrid (
                                const amrex::Vector<amrex::ParticleReal>& energies,
                                amrex::ParticleReal dE
                                );

    struct Executor {
        /** Get the collision cross-section using a simple linear interpolator. If
         * the energy value is lower (higher) than the given energy range the
         * first (last) cross-section value is used.
         *
         * @param E_coll collision energy in eV
         *
         */
        [[nodiscard]]
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        amrex::ParticleReal getCrossSection (amrex::ParticleReal E_coll) const
        {
            if (E_coll < m_energy_lo) {
                return m_sigma_lo;
            } else if (E_coll > m_energy_hi) {
                return m_sigma_hi;
            } else {
                using amrex::Math::floor;
                using amrex::Math::ceil;
                // calculate index of bounding energy pairs
                amrex::ParticleReal temp = (E_coll - m_energy_lo) / m_dE;
                const int idx_1 = static_cast<int>(floor(temp));
                const int idx_2 = static_cast<int>(ceil(temp));

                // linearly interpolate to the given energy value
                temp -= idx_1;
                return m_sigmas_data[idx_1] + (m_sigmas_data[idx_2] - m_sigmas_data[idx_1]) * temp;
            }
        }

        amrex::ParticleReal* m_sigmas_data = nullptr;
        amrex::ParticleReal m_energy_lo, m_energy_hi, m_sigma_lo, m_sigma_hi, m_dE;
        amrex::ParticleReal m_energy_penalty;
        ScatteringProcessType m_type;
    };

    [[nodiscard]]
    Executor const& executor () const {
#ifdef AMREX_USE_GPU
        return m_exe_d;
#else
        return m_exe_h;
#endif
    }

    [[nodiscard]] amrex::ParticleReal getCrossSection (amrex::ParticleReal E_coll) const
    {
        return m_exe_h.getCrossSection(E_coll);
    }

    [[nodiscard]] amrex::ParticleReal getEnergyPenalty () const { return m_exe_h.m_energy_penalty; }
    [[nodiscard]] amrex::ParticleReal getMinEnergyInput () const { return m_exe_h.m_energy_lo; }
    [[nodiscard]] amrex::ParticleReal getMaxEnergyInput () const { return m_exe_h.m_energy_hi; }
    [[nodiscard]] amrex::ParticleReal getEnergyInputStep () const { return m_exe_h.m_dE; }

    [[nodiscard]] ScatteringProcessType type () const { return m_exe_h.m_type; }

private:

    static
    ScatteringProcessType parseProcessType(const std::string& process);

    void init (const std::string& scattering_process, amrex::ParticleReal energy);

    amrex::Vector<amrex::ParticleReal> m_energies;

#ifdef AMREX_USE_GPU
    amrex::Gpu::DeviceVector<amrex::ParticleReal> m_sigmas_d;
    Executor m_exe_d;
#endif
    amrex::Gpu::HostVector<amrex::ParticleReal> m_sigmas_h;
    Executor m_exe_h;

    int m_grid_size;
};

#endif // WARPX_PARTICLES_COLLISION_SCATTERING_PROCESS_H_
