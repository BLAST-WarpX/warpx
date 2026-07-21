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

#include <AMReX_Algorithm.H>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>

ScatteringProcess::ScatteringProcess (
                        const std::string& scattering_process,
                        const std::string& cross_section_file,
                        const amrex::ParticleReal energy,
                        const ScatteringAngleModel scattering_angle_model,
                        const std::string& cross_section_file_mt )
{
    // read the cross-section data file(s) into memory
    readCrossSectionFile(cross_section_file, m_energies, m_sigmas_h,
                         cross_section_file_mt, &m_sigmas_mt_h);

    init(scattering_process, energy, scattering_angle_model);
}

template <typename InputVector>
ScatteringProcess::ScatteringProcess (
                        const std::string& scattering_process,
                        const InputVector&& energies,
                        const InputVector&& sigmas,
                        const amrex::ParticleReal energy,
                        const ScatteringAngleModel scattering_angle_model,
                        const InputVector&& sigmas_mt )
{
    m_energies.insert(m_energies.begin(), std::begin(energies), std::end(energies));
    m_sigmas_h.insert(m_sigmas_h.begin(), std::begin(sigmas),   std::end(sigmas));
    m_sigmas_mt_h.insert(m_sigmas_mt_h.begin(), std::begin(sigmas_mt), std::end(sigmas_mt));

    init(scattering_process, energy, scattering_angle_model);
}

void
ScatteringProcess::init (const std::string& scattering_process, const amrex::ParticleReal energy,
                         const ScatteringAngleModel scattering_angle_model)
{
    using namespace amrex::literals;
    m_exe_h.m_energies_data = m_energies.data();
    m_exe_h.m_sigmas_data = m_sigmas_h.data();

    // save energy grid parameters for easy use
    const int grid_size = static_cast<int>(m_energies.size());
    m_exe_h.m_grid_size = grid_size;
    m_exe_h.m_energy_lo = m_energies[0];
    m_exe_h.m_energy_hi = m_energies[grid_size-1];
    m_exe_h.m_sigma_lo = m_sigmas_h[0];
    m_exe_h.m_sigma_hi = m_sigmas_h[grid_size-1];
    // The energy grid does not need to be evenly spaced; `m_dE` is only used as a
    // representative energy step (e.g. to set the scan resolution when computing the
    // maximum collision frequency). Use the smallest spacing so that finely resolved
    // regions of a non-uniform grid are not skipped over.
    m_exe_h.m_dE = m_energies[grid_size-1] - m_energies[0];
    for (int i = 1; i < grid_size; i++) {
        m_exe_h.m_dE = std::min(m_exe_h.m_dE, m_energies[i] - m_energies[i-1]);
    }
    m_exe_h.m_energy_penalty = energy;
    m_exe_h.m_type = parseProcessType(scattering_process);
    m_exe_h.m_scattering_angle_model = scattering_angle_model;
    m_exe_h.m_produces_products = (
        m_exe_h.m_type == ScatteringProcessType::IONIZATION ||
        m_exe_h.m_type == ScatteringProcessType::TWOPRODUCT_REACTION ||
        m_exe_h.m_type == ScatteringProcessType::CHARGE_EXCHANGE);

    // sanity check cross-section energy grid
    sanityCheckEnergyGrid(m_energies);

    // check that the cross-section is 0 at the energy cost if the energy
    // cost is > 0 - this is to prevent the possibility of negative left
    // over energy after a collision event
    if (m_exe_h.m_energy_penalty > 0) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            (getCrossSection(m_exe_h.m_energy_penalty) == 0),
            "Cross-section > 0 at energy cost for collision."
        );
    }

    // For the screened Rutherford angle model, build the lookup table of the screening
    // parameter eta as a function of energy from the momentum-transfer and integral
    // cross-sections (see buildScreenedRutherfordTable).
    if (scattering_angle_model == ScatteringAngleModel::Screened_Rutherford) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            (!m_sigmas_mt_h.empty()),
            "The 'screened_rutherford' scattering angle model requires the momentum-transfer "
            "cross-section to be provided via '<process>_cross_section_mt'."
        );
        buildScreenedRutherfordTable();
        m_exe_h.m_log_eta_data = m_log_eta_h.data();
        m_exe_h.m_log_eta_lo   = m_log_eta_h[0];
        m_exe_h.m_log_eta_hi   = m_log_eta_h[grid_size-1];
    }

#ifdef AMREX_USE_GPU
    m_exe_d = m_exe_h;
    m_energies_d.resize(m_energies.size());
    m_sigmas_d.resize(m_sigmas_h.size());
    m_exe_d.m_energies_data = m_energies_d.data();
    m_exe_d.m_sigmas_data = m_sigmas_d.data();
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_energies.begin(), m_energies.end(),
                          m_energies_d.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_sigmas_h.begin(), m_sigmas_h.end(),
                          m_sigmas_d.begin());
    // Copy the ln(eta) table to the device (only needed for the screened Rutherford model;
    // the momentum-transfer cross-section itself is not needed on the device, since the
    // eta table shares the cross-section energy grid).
    if (!m_log_eta_h.empty()) {
        m_log_eta_d.resize(m_log_eta_h.size());
        m_exe_d.m_log_eta_data = m_log_eta_d.data();
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_log_eta_h.begin(), m_log_eta_h.end(),
                              m_log_eta_d.begin());
    }
    amrex::Gpu::streamSynchronize();
#endif
}

void
ScatteringProcess::buildScreenedRutherfordTable ()
{
    using namespace amrex::literals;

    // In the screened Rutherford model, the ratio of the momentum-transfer to the integral
    // cross-section fixes the mean scattering cosine through the screening parameter eta:
    //     g(eta) := sigma_mt/sigma = 2*eta*[(eta+1)*ln(1+1/eta) - 1] = 1 - <cos(theta)>
    // g(eta) increases monotonically from 0 (as eta -> 0, strongly forward-peaked) to 1
    // (as eta -> infinity, isotropic). At each energy node we solve g(eta) = sigma_mt/sigma
    // for eta by bisection (in log(eta), which scales well over the wide search range).

    auto g = [](amrex::ParticleReal eta) {
        return 2._prt * eta * ((eta + 1._prt) * std::log1p(1._prt / eta) - 1._prt);
    };

    // Representable range for eta. Ratios sigma_mt/sigma outside (g(eta_lo), g(eta_hi)),
    // i.e. essentially outside (0, 1), are clamped to the corresponding endpoint.
    const amrex::ParticleReal log_eta_lo = std::log(1.e-8_prt);
    const amrex::ParticleReal log_eta_hi = std::log(1.e8_prt);
    const amrex::ParticleReal g_lo = g(std::exp(log_eta_lo));
    const amrex::ParticleReal g_hi = g(std::exp(log_eta_hi));

    const int grid_size = static_cast<int>(m_energies.size());
    m_log_eta_h.resize(grid_size);
    for (int i = 0; i < grid_size; ++i) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            (m_sigmas_h[i] > 0._prt && m_sigmas_mt_h[i] > 0._prt),
            "Screened Rutherford model requires strictly positive integral and "
            "momentum-transfer cross-sections."
        );
        const amrex::ParticleReal ratio = m_sigmas_mt_h[i] / m_sigmas_h[i];
        if (ratio <= g_lo) {
            m_log_eta_h[i] = log_eta_lo;
        } else if (ratio >= g_hi) {
            m_log_eta_h[i] = log_eta_hi;
        } else {
            // Solve g(exp(t)) - ratio = 0 for t = log(eta)
            m_log_eta_h[i] = amrex::bisect(log_eta_lo, log_eta_hi,
                [=] (amrex::ParticleReal t) { return g(std::exp(t)) - ratio; });
        }
    }
}

ScatteringProcessType
ScatteringProcess::parseProcessType(const std::string& scattering_process)
{
    if (scattering_process.find("elastic") != std::string::npos) {
        // `elastic` is matched as a prefix (like `excitationX`) so that several distinct
        // elastic channels (e.g. with different cross-sections and/or scattering angle
        // models) can be included in the same collision under unique names.
        return ScatteringProcessType::ELASTIC;
    } else if (scattering_process == "charge_exchange") {
        return ScatteringProcessType::CHARGE_EXCHANGE;
    } else if (scattering_process == "two_product_reaction") {
        return ScatteringProcessType::TWOPRODUCT_REACTION;
    } else if (scattering_process == "ionization") {
        return ScatteringProcessType::IONIZATION;
    } else if (scattering_process.find("excitation") != std::string::npos) {
        return ScatteringProcessType::EXCITATION;
    } else {
        return ScatteringProcessType::INVALID;
    }
}

void
ScatteringProcess::readCrossSectionFile (
                                  const std::string& cross_section_file,
                                  amrex::Vector<amrex::ParticleReal>& energies,
                                  amrex::Gpu::HostVector<amrex::ParticleReal>& sigmas,
                                  const std::string& cross_section_file_mt,
                                  amrex::Gpu::HostVector<amrex::ParticleReal>* sigmas_mt )
{
    std::ifstream infile(cross_section_file);
    if(!infile.is_open()) { WARPX_ABORT_WITH_MESSAGE("Failed to open cross-section data file"); }

    amrex::ParticleReal energy, sigma;
    while (infile >> energy >> sigma) {
        energies.push_back(energy);
        sigmas.push_back(sigma);
    }
    if (infile.bad()) { WARPX_ABORT_WITH_MESSAGE("Failed to read cross-section data from file."); }
    infile.close();

    // Optionally read the momentum-transfer cross-section, which must be tabulated on the
    // same energy grid as the integral cross-section.
    if (!cross_section_file_mt.empty() && sigmas_mt != nullptr) {
        std::ifstream infile_mt(cross_section_file_mt);
        if(!infile_mt.is_open()) {
            WARPX_ABORT_WITH_MESSAGE("Failed to open momentum-transfer cross-section data file");
        }
        amrex::ParticleReal energy_mt, sigma_mt;
        std::size_t idx = 0;
        while (infile_mt >> energy_mt >> sigma_mt) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                (idx < energies.size() &&
                 std::abs(energy_mt - energies[idx]) <= 1.e-6 * std::abs(energies[idx])),
                "The momentum-transfer cross-section must be tabulated on the same energy "
                "grid as the integral cross-section."
            );
            sigmas_mt->push_back(sigma_mt);
            ++idx;
        }
        if (infile_mt.bad()) {
            WARPX_ABORT_WITH_MESSAGE("Failed to read momentum-transfer cross-section data from file.");
        }
        infile_mt.close();
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            (sigmas_mt->size() == sigmas.size()),
            "The momentum-transfer cross-section must have the same number of energy points "
            "as the integral cross-section."
        );
    }
}

void
ScatteringProcess::sanityCheckEnergyGrid (
                                   const amrex::Vector<amrex::ParticleReal>& energies
                                   )
{
    // The energy grid does not need to be evenly spaced, but it must be sorted in
    // strictly increasing order for the bisection search and linear interpolation
    // used in `Executor::getCrossSection` to work correctly.
    for (unsigned i = 1; i < energies.size(); i++) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                                         (energies[i] > energies[i-1]),
                                         "Cross-section energy grid must be sorted in "
                                         "strictly increasing order."
                                         );
    }
}
