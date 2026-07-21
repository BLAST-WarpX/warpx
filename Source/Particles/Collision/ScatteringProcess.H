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

#include "Utils/WarpXAlgorithmSelection.H"

#include <AMReX_Algorithm.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>
#include <AMReX_RandomEngine.H>
#include <AMReX_Vector.H>

#include <cmath>
#include <string>

enum class ScatteringProcessType {
    INVALID,
    ELASTIC,
    TWOPRODUCT_REACTION,
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
                amrex::ParticleReal energy,
                ScatteringAngleModel scattering_angle_model,
                const std::string& cross_section_file_mt = ""
                );

    template <typename InputVector>
    ScatteringProcess (
                const std::string& scattering_process,
                const InputVector&& energies,
                const InputVector&& sigmas,
                amrex::ParticleReal energy,
                ScatteringAngleModel scattering_angle_model,
                const InputVector&& sigmas_mt = InputVector()
                );

    ~ScatteringProcess() = default;

    ScatteringProcess (ScatteringProcess const&)            = delete;
    ScatteringProcess& operator= (ScatteringProcess const&) = delete;
    ScatteringProcess (ScatteringProcess &&)                = default;
    ScatteringProcess& operator= (ScatteringProcess &&)     = default;

    /** Read the given cross-section data file(s) to memory.
     *
     * @param cross_section_file the path to the file containing the integral cross-
     *        section data
     * @param energies vector storing energy values in eV
     * @param sigmas vector storing (integral) cross-section values
     * @param cross_section_file_mt optional path to the file containing the momentum-
     *        transfer cross-section data (used by the screened Rutherford angle model).
     *        When non-empty, it must be tabulated on the same energy grid as
     *        \p cross_section_file.
     * @param sigmas_mt optional vector storing momentum-transfer cross-section values;
     *        only filled when \p cross_section_file_mt is non-empty (may be nullptr)
     *
     */
    static
    void readCrossSectionFile (
                               const std::string& cross_section_file,
                               amrex::Vector<amrex::ParticleReal>& energies,
                               amrex::Gpu::HostVector<amrex::ParticleReal>& sigmas,
                               const std::string& cross_section_file_mt = "",
                               amrex::Gpu::HostVector<amrex::ParticleReal>* sigmas_mt = nullptr
                               );

    static
    void sanityCheckEnergyGrid (
                                const amrex::Vector<amrex::ParticleReal>& energies
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
                // Locate the pair of energy grid points bracketing E_coll using a
                // bisection search. This does not assume an evenly spaced energy grid.
                // `amrex::bisect` returns idx_1 such that
                // m_energies_data[idx_1] <= E_coll < m_energies_data[idx_1 + 1].
                const int idx_1 = amrex::bisect(m_energies_data, 0, m_grid_size - 1, E_coll);
                const int idx_2 = idx_1 + 1;

                // linearly interpolate to the given energy value
                const amrex::ParticleReal temp =
                    (E_coll - m_energies_data[idx_1])
                    / (m_energies_data[idx_2] - m_energies_data[idx_1]);
                return m_sigmas_data[idx_1] + (m_sigmas_data[idx_2] - m_sigmas_data[idx_1]) * temp;
            }
        }

        /** Get the screening parameter eta of the screened Rutherford model at the given
         * collision energy. This is only meaningful when the scattering angle model is
         * `Screened_Rutherford` (otherwise `m_log_eta_data` is nullptr). The tabulated
         * `ln(eta)` is linearly interpolated on the same (possibly non-uniform) energy grid
         * as the cross-section, using the same bisection search as `getCrossSection`. Outside
         * the energy range, the first (last) tabulated value is used.
         *
         * @param E_coll collision energy in eV
         *
         */
        [[nodiscard]]
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        amrex::ParticleReal getEta (amrex::ParticleReal E_coll) const
        {
            if (E_coll < m_energy_lo) {
                return std::exp(m_log_eta_lo);
            } else if (E_coll > m_energy_hi) {
                return std::exp(m_log_eta_hi);
            } else {
                // Same bracketing as getCrossSection (does not assume an evenly spaced grid).
                const int idx_1 = amrex::bisect(m_energies_data, 0, m_grid_size - 1, E_coll);
                const int idx_2 = idx_1 + 1;

                // linearly interpolate ln(eta) to the given energy value
                const amrex::ParticleReal temp =
                    (E_coll - m_energies_data[idx_1])
                    / (m_energies_data[idx_2] - m_energies_data[idx_1]);
                const amrex::ParticleReal log_eta =
                    m_log_eta_data[idx_1] + (m_log_eta_data[idx_2] - m_log_eta_data[idx_1]) * temp;
                return std::exp(log_eta);
            }
        }

        amrex::ParticleReal* m_energies_data = nullptr;
        amrex::ParticleReal* m_sigmas_data = nullptr;
        /** Tabulated ln(eta) on the energy grid, for the screened Rutherford angle model
         *  (nullptr for all other angle models). **/
        amrex::ParticleReal* m_log_eta_data = nullptr;
        int m_grid_size = 0;
        amrex::ParticleReal m_energy_lo, m_energy_hi, m_sigma_lo, m_sigma_hi, m_dE;
        amrex::ParticleReal m_log_eta_lo = 0, m_log_eta_hi = 0;
        amrex::ParticleReal m_energy_penalty;
        ScatteringProcessType m_type;
        ScatteringAngleModel m_scattering_angle_model;
        /** True if this process creates new product particles (ionization, charge exchange,
         *  two-product reaction); false for purely scattering processes (elastic, excitation). **/
        bool m_produces_products = false;
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

    static
    ScatteringProcessType parseProcessType(const std::string& process);

private:

    void init (const std::string& scattering_process, amrex::ParticleReal energy,
               ScatteringAngleModel scattering_angle_model);

    /** Build the lookup table of ln(eta) vs. energy for the screened Rutherford angle
     *  model, by inverting sigma_mt/sigma = 2*eta*[(eta+1)*ln(1+1/eta) - 1] at each energy
     *  node. Requires `m_energies`, `m_sigmas_h` and `m_sigmas_mt_h` to already be filled;
     *  fills `m_log_eta_h`. **/
    void buildScreenedRutherfordTable ();

    amrex::Vector<amrex::ParticleReal> m_energies;

    // Momentum-transfer cross-section and tabulated ln(eta) for the screened Rutherford
    // angle model. The momentum-transfer cross-section is only used at initialization
    // (to build the ln(eta) table) and is therefore kept host-only.
    amrex::Gpu::HostVector<amrex::ParticleReal> m_sigmas_mt_h;
    amrex::Gpu::HostVector<amrex::ParticleReal> m_log_eta_h;

#ifdef AMREX_USE_GPU
    amrex::Gpu::DeviceVector<amrex::ParticleReal> m_energies_d;
    amrex::Gpu::DeviceVector<amrex::ParticleReal> m_sigmas_d;
    amrex::Gpu::DeviceVector<amrex::ParticleReal> m_log_eta_d;
    Executor m_exe_d;
#endif
    amrex::Gpu::HostVector<amrex::ParticleReal> m_sigmas_h;
    Executor m_exe_h;
};

#endif // WARPX_PARTICLES_COLLISION_SCATTERING_PROCESS_H_
