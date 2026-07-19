/* Copyright 2021-2023 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Modern Electron, Roelof Groenewald (TAE Technologies)
 *
 * License: BSD-3-Clause-LBNL
 */
#include "ScatteringProcess.H"

#include "Utils/TextMsg.H"

ScatteringProcess::ScatteringProcess (
                        const std::string& scattering_process,
                        const std::string& cross_section_file,
                        const amrex::ParticleReal energy,
                        const std::string& cross_section_file_momentum )
{
    // read the cross-section data file into memory
    readCrossSectionFile(cross_section_file, m_energies, m_sigmas_h,cross_section_file_momentum, m_sigmas_mt_h);

    init(scattering_process, energy);
}

template <typename InputVector>
ScatteringProcess::ScatteringProcess (
                        const std::string& scattering_process,
                        const InputVector&& energies,
                        const InputVector&& sigmas,
                        const amrex::ParticleReal energy,
                        const InputVector&& sigmas_mt )
{
    m_energies.insert(m_energies.begin(), std::begin(energies), std::end(energies));
    m_sigmas_h.insert(m_sigmas_h.begin(), std::begin(sigmas),   std::end(sigmas));
    if(!sigmas_mt.empty())
    {
        m_sigmas_mt_h.insert(m_sigmas_mt_h.begin(), std::begin(sigmas_mt),   std::end(sigmas_mt));
    }
    init(scattering_process, energy);
}


void
ScatteringProcess::init (const std::string& scattering_process, const amrex::ParticleReal energy)
{
    using namespace amrex::literals;

    // m_grid_size doit être initialisé EN PREMIER
    m_grid_size = static_cast<int>(m_energies.size());

    m_exe_h.m_sigmas_data = m_sigmas_h.data();
    m_exe_h.m_energies_data = m_energies.data();

    if (!m_sigmas_mt_h.empty()) {
    m_exe_h.m_sigmas_mt_data = m_sigmas_mt_h.data();
    m_exe_h.m_sigma_mt_lo = m_sigmas_mt_h[0];
    m_exe_h.m_sigma_mt_hi = m_sigmas_mt_h[m_grid_size-1];

    const int N_case = 1000;
    const amrex::ParticleReal eta_min = 1e-5;
    const amrex::ParticleReal eta_max = 0.5;

    amrex::Vector<amrex::ParticleReal> log_eta_i(N_case);
    amrex::Vector<amrex::ParticleReal> Ri(N_case);

    const amrex::ParticleReal log_eta_min = std::log(eta_min);
    const amrex::ParticleReal log_eta_max = std::log(eta_max);
    const amrex::ParticleReal d_log_eta = (log_eta_max - log_eta_min) / (N_case - 1);

    for (int i = 0; i < N_case; ++i) {
        log_eta_i[i] = log_eta_min + i * d_log_eta;
        amrex::ParticleReal eta = std::exp(log_eta_i[i]);
        
        amrex::ParticleReal internal_term = (eta + 1.0) * std::log1p(1.0 / eta) - 1.0;
        Ri[i] = std::log(2.0 * eta * internal_term);
    }

    m_log_etan_h.resize(m_grid_size);
    for (int i = 0; i < m_grid_size; ++i) {
        amrex::ParticleReal Rn = std::log(m_sigmas_mt_h[i] / m_sigmas_h[i]);

        if (Rn <= Ri[0]) {
            m_log_etan_h[i] = log_eta_i[0];
            continue;
        }
        if (Rn >= Ri[N_case - 1]) {
            m_log_etan_h[i] = log_eta_i[N_case - 1];
            continue;
        }

        int lo = 0;
        int hi = N_case - 2;
        while (lo < hi) {
            int mid = lo + (hi - lo + 1) / 2;
            if (Ri[mid] <= Rn) { lo = mid; }
            else               { hi = mid - 1; }
        }

        const amrex::ParticleReal Rn_lo = Ri[lo];
        const amrex::ParticleReal Rn_hi = Ri[lo + 1];
        const amrex::ParticleReal ln_eta_lo = log_eta_i[lo];
        const amrex::ParticleReal ln_eta_hi = log_eta_i[lo + 1];
        
        m_log_etan_h[i] = ln_eta_lo + (ln_eta_hi - ln_eta_lo) / (Rn_hi - Rn_lo) * (Rn - Rn_lo);
}

m_exe_h.m_log_etan    = m_log_etan_h.data();
m_exe_h.m_log_etan_lo = m_log_etan_h[0];
m_exe_h.m_log_etan_hi = m_log_etan_h[m_grid_size - 1];
    }

    // Paramètres de grille
    m_exe_h.m_energy_lo      = m_energies[0];
    m_exe_h.m_energy_hi      = m_energies[m_grid_size-1];
    m_exe_h.m_sigma_lo       = m_sigmas_h[0];
    m_exe_h.m_sigma_hi       = m_sigmas_h[m_grid_size-1];
    m_exe_h.m_dE             = (m_exe_h.m_energy_hi - m_exe_h.m_energy_lo) / (m_grid_size - 1._prt);
    m_exe_h.m_energy_penalty = energy;
    m_exe_h.m_type           = parseProcessType(scattering_process);

    sanityCheckEnergyGrid(m_energies, m_exe_h.m_dE);

    if (m_exe_h.m_energy_penalty > 0) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            (getCrossSection(m_exe_h.m_energy_penalty) == 0),
            "Cross-section > 0 at energy cost for collision."
        );
    }

#ifdef AMREX_USE_GPU
    m_exe_d = m_exe_h;
    m_sigmas_d.resize(m_sigmas_h.size());
    m_sigmas_mt_d.resize(m_sigmas_mt_h.size());
    m_energies_d.resize(m_energies.size());
    m_log_etan_d.resize(m_log_etan_h.size());

    m_exe_d.m_energies_data = m_energies_d.data();
    m_exe_d.m_sigmas_data = m_sigmas_d.data();
    m_exe_d.m_sigmas_mt_data = m_sigmas_mt_d.data();
    m_exe_d.m_ln_etan = m_ln_etan_d.data();

    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
        m_sigmas_h.begin(), m_sigmas_h.end(), m_sigmas_d.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
        m_sigmas_mt_h.begin(), m_sigmas_mt_h.end(), m_sigmas_mt_d.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
        m_energies.begin(), m_energies.end(), m_energies_d.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
        m_log_etan_h.begin(), m_log_etan_h.end(), m_log_etan_d.begin());
    amrex::Gpu::streamSynchronize();

#endif
}

ScatteringProcessType
ScatteringProcess::parseProcessType(const std::string& scattering_process)
{
    if (scattering_process == "elastic") {
        return ScatteringProcessType::ELASTIC;
    } else if (scattering_process == "rutherford") {
        return ScatteringProcessType::RUTHERFORD;
    } else if (scattering_process == "back") {
        return ScatteringProcessType::BACK;
    } else if (scattering_process == "charge_exchange") {
        return ScatteringProcessType::CHARGE_EXCHANGE;
    } else if (scattering_process == "two_product_reaction") {
        return ScatteringProcessType::TWOPRODUCT_REACTION;
    } else if (scattering_process == "ionization") {
        return ScatteringProcessType::IONIZATION;
    } else if (scattering_process.find("excitation") != std::string::npos) {
        return ScatteringProcessType::EXCITATION;
    } else if (scattering_process.find("forward") != std::string::npos) {
        return ScatteringProcessType::FORWARD;
    } else {
        return ScatteringProcessType::INVALID;
    }
}


void
ScatteringProcess::readCrossSectionFile (
                                  const std::string& cross_section_file,
                                  amrex::Vector<amrex::ParticleReal>& energies,
                                  amrex::Gpu::HostVector<amrex::ParticleReal>& sigmas,
                                  const std::string& cross_section_file_momentum,
                                  amrex::Gpu::HostVector<amrex::ParticleReal>& sigmas_mt    
                                  )
{
    std::ifstream infile(cross_section_file);
    if(!infile.is_open()) { WARPX_ABORT_WITH_MESSAGE("Failed to open cross-section data file"); }

    amrex::ParticleReal energy, sigma;
    while (infile >> energy >> sigma) {
        energies.push_back(energy);
        sigmas.push_back(sigma);
    };
    if (infile.bad()) { WARPX_ABORT_WITH_MESSAGE("Failed to read cross-section data from file."); }
    infile.close();

    if(!cross_section_file_momentum.empty()) {
    std::ifstream infile2(cross_section_file_momentum);
    if(!infile2.is_open()) { WARPX_ABORT_WITH_MESSAGE("Failed to open cross-section momentum data file"); }

    amrex::ParticleReal sigma_mt;
    while (infile2 >> energy >> sigma_mt) {
        sigmas_mt.push_back(sigma_mt);
    };
    if (infile2.bad()) { WARPX_ABORT_WITH_MESSAGE("Failed to read cross-section momentum data from file."); }
    infile2.close();
    };

}


void
ScatteringProcess::sanityCheckEnergyGrid (
                                   const amrex::Vector<amrex::ParticleReal>& energies,
                                   amrex::ParticleReal dE
                                   )
{
    // confirm that the input data for the cross-section was provided with
    // equal energy steps, otherwise the linear interpolation will fail
    for (unsigned i = 1; i < energies.size(); i++) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                                         (std::abs(energies[i] - energies[i-1] - dE) < dE / 100.0),
                                         "Energy grid not evenly spaced."
                                         );
    }
}
