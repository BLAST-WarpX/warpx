/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "HybridPICModel.H"

#include "Fields.H"
#include "Particles/MultiParticleContainer.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Utility.H>
#include <AMReX_VisMF.H>

#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace
{
    using History = std::vector<std::pair<std::string, amrex::MultiFab *>>;

    std::string DepositionContract ()
    {
        std::ostringstream output;
        output << WarpX::nox << ' ' << WarpX::noy << ' ' << WarpX::noz << ' '
               << static_cast<int>(WarpX::current_deposition_algo) << ' ' << WarpX::use_filter
               << ' ' << WarpX::use_kspace_filter << ' ' << WarpX::use_filter_compensation << ' '
               << WarpX::do_single_precision_comms;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            output << ' ' << WarpX::filter_npass_each_dir[d];
        }
        output << std::setprecision(std::numeric_limits<amrex::ParticleReal>::max_digits10);
        auto const &particles = WarpX::GetInstance().GetPartContainer();
        for (auto const &name : particles.GetSpeciesNames()) {
            auto const &species = particles.GetParticleContainerFromName(name);
            output << ' ' << std::quoted(name) << ' ' << species.getCharge() << ' '
                   << species.getMass() << ' ' << species.do_not_deposit;
        }
        return output.str();
    }

    History HistoryFields (HybridPICModel const &model)
    {
        auto &simulation = WarpX::GetInstance();
        auto &fields = simulation.m_fields;
        using warpx::fields::FieldType;
        History result{{"rho", fields.get(FieldType::rho_fp, 0)}};
        for (int d = 0; d < 3; ++d) {
            result.emplace_back(
                "current_" + std::to_string(d),
                fields.get(FieldType::current_fp, ablastr::fields::Direction{d}, 0));
        }
        if (model.m_need_per_species_fields) {
            auto const &particles = simulation.GetPartContainer();
            for (auto const &species : particles.GetSpeciesNames()) {
                if (particles.GetParticleContainerFromName(species).getCharge() == 0) {
                    continue;
                }
                auto const name = "rho_fp_" + species;
                result.emplace_back(name, fields.get(name, 0));
            }
            for (int material = 0; material < model.electronThermodynamicsNumMaterials();
                 ++material) {
                auto const name =
                    "ni_charge_fp_" + model.electronThermodynamicsMaterialSpeciesName(material);
                result.emplace_back(name, fields.get(name, 0));
            }
            result.emplace_back("rho_species_sum", fields.get("hybrid_rho_species_sum_fp", 0));
        }
        return result;
    }

    bool Exists (std::string const &path)
    {
        int exists = 0;
        if (amrex::ParallelDescriptor::IOProcessor()) {
            exists = amrex::FileExists(path);
        }
        amrex::ParallelDescriptor::Bcast(&exists, 1,
                                         amrex::ParallelDescriptor::IOProcessorNumber());
        return exists != 0;
    }
} // namespace

void HybridPICModel::WriteMomentHistory (std::string const &directory) const
{
    auto const &simulation = WarpX::GetInstance();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(simulation.finestLevel() == 0,
                                     "Hybrid moment history requires a single level.");
    auto const fields = HistoryFields(*this);
    if (m_moment_history_valid) {
        for (auto const &[name, field] : fields) {
            amrex::VisMF::Write(*field, std::string(directory).append("/HybridMomentHistory_").append(name));
        }
    }
    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::ofstream output(directory + "/HybridMomentHistory.txt");
        output << "hybrid_moments_v1 " << m_moment_history_valid << ' ' << simulation.getistep(0)
               << ' ' << fields.size() << '\n';
        output << std::quoted(DepositionContract()) << '\n';
        for (auto const &[name, field] : fields) {
            amrex::ignore_unused(field);
            output << name << '\n';
        }
        output.flush();
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(output.good(),
                                         "Could not checkpoint hybrid moment history.");
    }
}

void HybridPICModel::ReadMomentHistory (std::string const &directory)
{
    m_moment_history_valid = false;
    m_restored_moment_history_pending = false;
    auto const fields = HistoryFields(*this);
    auto const manifest = directory + "/HybridMomentHistory.txt";
    if (!Exists(manifest)) {
        for (auto const &[name, field] : fields) {
            amrex::ignore_unused(field);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                !Exists(std::string(directory).append("/HybridMomentHistory_").append(name).append("_H")),
                "Hybrid moment history has data but no manifest.");
        }
        return; // Legacy checkpoints reconstruct all deposits at bootstrap.
    }
    amrex::Vector<char> buffer;
    amrex::ParallelDescriptor::ReadAndBcastFile(manifest, buffer);
    std::istringstream input(std::string(buffer.data()));
    std::string version;
    int valid = -1, step = -1;
    std::size_t count = 0;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (input >> version >> valid >> step >> count) && version == "hybrid_moments_v1" &&
            (valid == 0 || valid == 1) && step == WarpX::GetInstance().getistep(0) &&
            count == fields.size(),
        "Invalid or incompatible hybrid moment history manifest.");
    std::string contract;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE((input >> std::quoted(contract)) &&
                                         contract == DepositionContract(),
                                     "Hybrid moment history deposition contract changed.");
    for (auto const &[name, field] : fields) {
        amrex::ignore_unused(field);
        std::string stored;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE((input >> stored) && stored == name,
                                         "Hybrid moment history species or field layout changed.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            Exists(std::string(directory).append("/HybridMomentHistory_").append(name).append("_H")) == (valid == 1),
            "Incomplete or inconsistent hybrid moment history.");
    }
    input >> std::ws;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(input.eof(), "Trailing hybrid moment history metadata.");
    if (valid == 0) {
        return;
    } // A checkpoint before the first native bootstrap.
    for (auto const &[name, field] : fields) {
        amrex::VisMF::Read(*field, std::string(directory).append("/HybridMomentHistory_").append(name));
    }
    m_moment_history_valid = true;
    m_restored_moment_history_pending = true;
}
