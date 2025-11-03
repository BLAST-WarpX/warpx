/* Copyright 2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald (TAE Technologies)
 *
 * License: BSD-3-Clause-LBNL
 */
#include "DSMCFunc.H"
#include "Utils/TextMsg.H"

/**
 * \brief Constructor of the DSMCFunc class
 *
 * @param[in] collision_name the name of the collision
 * @param[in] mypc pointer to the MultiParticleContainer
 * @param[in] isSameSpecies whether the two colliding species are the same
 */
DSMCFunc::DSMCFunc (
    const std::string& collision_name,
    [[maybe_unused]] MultiParticleContainer const * const mypc,
    const bool isSameSpecies ): m_isSameSpecies{isSameSpecies}
{
    using namespace amrex::literals;

    const amrex::ParmParse pp_collision_name(collision_name);

    // query for a list of collision processes
    // these could be elastic, excitation, charge_exchange, back, etc.
    amrex::Vector<std::string> scattering_process_names;
    pp_collision_name.queryarr("scattering_processes", scattering_process_names);

    // create a vector of ScatteringProcess objects from each scattering
    // process name
    bool reaction_produces_new_species = false;
    for (const auto& scattering_process : scattering_process_names) {
        const std::string kw_cross_section = scattering_process + "_cross_section";
        std::string cross_section_file;
        pp_collision_name.query(kw_cross_section.c_str(), cross_section_file);

        // if the scattering process is excitation, ionization, forward or
        // two-product reaction get the energy associated with that process
        // (note that this allows forward scattering to be used both with and
        // without a fixed energy loss)
        amrex::ParticleReal energy = 0._prt;
        if (scattering_process.find("excitation") != std::string::npos ||
            scattering_process.find("ionization") != std::string::npos ||
            scattering_process.find("forward") != std::string::npos ||
            scattering_process.find("two_product_reaction") != std::string::npos ) {
            const std::string kw_energy = scattering_process + "_energy";
            utils::parser::getWithParser(
                pp_collision_name, kw_energy.c_str(), energy);
        }

        ScatteringProcess process(scattering_process, cross_section_file, energy);

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(process.type() != ScatteringProcessType::EXCITATION,
                                        "Excitation collisions are not yet supported in DSMC");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(process.type() != ScatteringProcessType::FORWARD,
                                        "Forward scattering collisions are not yet supported in DSMC");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(process.type() != ScatteringProcessType::INVALID,
                                        "Cannot add an unknown scattering process type");

        if (process.type() == ScatteringProcessType::IONIZATION || process.type() == ScatteringProcessType::TWOPRODUCT_REACTION) {
            // Only one ionization process is currently supported as part of a given
            // collision set.
            if (reaction_produces_new_species) {
                amrex::Abort("Multiple reactions that produce new species were specified in " + collision_name +
                ".scattering_processes, but DSMC only supports a single reaction that produces new species.");
            }
            reaction_produces_new_species = true;
        }

        if (scattering_process.find("ionization") != std::string::npos) {
            // Ensure that the first product species is always an electron (which is assumed
            // during the scattering operation).
            amrex::Vector<std::string> product_species_names;
            pp_collision_name.getarr("product_species", product_species_names);
            // Check that the charge is consistent with an electron species
            auto& species1 = mypc->GetParticleContainerFromName(product_species_names[0]);
            if( species1.getCharge() >= 0._prt ) {
                amrex::Abort("The first species in " + collision_name + ".product_species must be an electron.");
            }

            // TODO: add a check that the ionization species has the same mass
            // (and a positive charge), compared to the target species
        } else if (scattering_process.find("charge_exchange") != std::string::npos) {
            // Ensure that the order of the product species in the charge exchange process is correct.
            // One product must have gained an electron (charge difference ≈ -q_e),
            // and the other must have lost an electron (charge difference ≈ +q_e).
            // This check verifies that the user did not swap the product_species in the input file.
            amrex::Vector<std::string> species_names;
            pp_collision_name.getarr("species", species_names);
            auto& species1 = mypc->GetParticleContainerFromName(species_names[0]);
            auto& species2 = mypc->GetParticleContainerFromName(species_names[1]);
            amrex::Vector<std::string> product_species_name;
            pp_collision_name.getarr("product_species", product_species_name);
            auto& product_species1 = mypc->GetParticleContainerFromName(product_species_name[0]);
            auto& product_species2 = mypc->GetParticleContainerFromName(product_species_name[1]);
            amrex::ParticleReal constexpr q_e = PhysConst::q_e;
            const amrex::ParticleReal delta_charge1 = species1.getCharge() - product_species1.getCharge();
            const amrex::ParticleReal delta_charge2 = species2.getCharge() - product_species2.getCharge();
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            (( std::abs(delta_charge1 - q_e) <= 0.01_prt*q_e && std::abs(delta_charge2 + q_e) <= 0.01_prt*q_e) ||
            ( std::abs(delta_charge1 + q_e) <= 0.01_prt*q_e && std::abs(delta_charge2 - q_e) <= 0.01_prt*q_e)),
            std::string("Charge exchange product species are not in the correct order.\n") +
            "Please reverse the order to:\n" +
             "  " + collision_name + ".product_species = " + product_species_name[1] + " " + product_species_name[0]
            );
        }

        m_scattering_processes.push_back(std::move(process));
    }

    // Store ScatteringProcess::Executor(s).
#ifdef AMREX_USE_GPU
    amrex::Gpu::HostVector<ScatteringProcess::Executor> h_scattering_processes_exe;
    for (auto const& p : m_scattering_processes) {
        h_scattering_processes_exe.push_back(p.executor());
    }
    m_scattering_processes_exe.resize(h_scattering_processes_exe.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_scattering_processes_exe.begin(),
                          h_scattering_processes_exe.end(), m_scattering_processes_exe.begin());
    amrex::Gpu::streamSynchronize();
#else
    for (auto const& p : m_scattering_processes) {
        m_scattering_processes_exe.push_back(p.executor());
    }
#endif

    // Link executor to appropriate ScatteringProcess executors
    m_exe.m_scattering_processes_data = m_scattering_processes_exe.data();
    m_exe.m_process_count = static_cast<int>(m_scattering_processes_exe.size());
    m_exe.m_isSameSpecies = m_isSameSpecies;
}
