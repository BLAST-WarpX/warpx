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
        if (m_electron_heat_conduction) {
            std::ofstream conduction(directory + "/HybridElectronConduction.txt");
            conduction << "ideal_isotropic_lagged_harmonic_v1 "
                       << std::quoted(m_electron_conductivity_expression) << ' '
                       << std::setprecision(std::numeric_limits<amrex::Real>::max_digits10)
                       << m_electron_conduction_flux_limiter << '\n';
            conduction.flush();
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(conduction.good(),
                "Could not checkpoint electron heat-conduction model.");
        }
#if defined(WARPX_DIM_RZ)
        if (m_conservative_pressure_work) {
            std::ofstream pressure(directory + "/HybridRZPressureWork.txt");
            pressure << "rz_centered_pressure_work_v1 "
                     << simulation.verboncoeurAxisCorrection() << ' '
                     << simulation.Geom(0).isPeriodic(1) << '\n';
            pressure.flush();
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(pressure.good(),
                "Could not checkpoint RZ pressure-work model.");
        }
#endif
        if (m_resolved_qei_support) {
            std::ofstream support(directory + "/HybridQeiSupport.txt");
            support << "resolved_pairwise_v3 " << m_resolved_qei_seed << ' '
                    << m_resolved_qei_counter << '\n';
            support.flush();
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(support.good(),
                "Could not checkpoint resolved Qei support.");
        }
        if (m_fv_transport_internal_energy && m_electron_thermodynamics.executor().isIdealGas()) {
            std::ofstream transport(directory + "/HybridIdealElectronTransport.txt");
            transport << (m_fv_transport_implicit ? "ideal_finite_volume_implicit_v1\n"
                                                : "ideal_finite_volume_v1\n");
            transport.flush();
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(transport.good(),
                "Could not checkpoint ideal finite-volume electron transport.");
        }
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
    auto const conduction_manifest = directory + "/HybridElectronConduction.txt";
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(Exists(conduction_manifest) == m_electron_heat_conduction,
        "Restart must preserve the electron heat-conduction model and its manifest.");
    if (m_electron_heat_conduction) {
        amrex::Vector<char> buffer;
        amrex::ParallelDescriptor::ReadAndBcastFile(conduction_manifest, buffer);
        std::istringstream conduction(std::string(buffer.data()));
        std::string version, expression, trailing;
        amrex::Real limiter = -1;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE((conduction >> version >> std::quoted(expression) >> limiter)
            && version == "ideal_isotropic_lagged_harmonic_v1"
            && expression == m_electron_conductivity_expression
            && limiter == m_electron_conduction_flux_limiter && !(conduction >> trailing),
            "Invalid or changed electron heat-conduction checkpoint contract.");
    }
#if defined(WARPX_DIM_RZ)
    auto const pressure_manifest = directory + "/HybridRZPressureWork.txt";
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(Exists(pressure_manifest) == m_conservative_pressure_work,
        "Restart must preserve the RZ pressure-work model and its manifest.");
    if (m_conservative_pressure_work) {
        amrex::Vector<char> buffer;
        amrex::ParallelDescriptor::ReadAndBcastFile(pressure_manifest, buffer);
        std::istringstream pressure(std::string(buffer.data()));
        std::string version, trailing;
        int axis = -1, periodic_z = -1;
        auto const& simulation = WarpX::GetInstance();
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE((pressure >> version >> axis >> periodic_z)
            && version == "rz_centered_pressure_work_v1"
            && axis == static_cast<int>(simulation.verboncoeurAxisCorrection())
            && periodic_z == static_cast<int>(simulation.Geom(0).isPeriodic(1))
            && !(pressure >> trailing),
            "Invalid or changed RZ pressure-work checkpoint contract.");
    }
#endif
    auto const support_manifest = directory + "/HybridQeiSupport.txt";
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(Exists(support_manifest) == m_resolved_qei_support,
        "Restart must preserve the Qei thermal support model and its manifest.");
    if (m_resolved_qei_support) {
        amrex::Vector<char> buffer;
        amrex::ParallelDescriptor::ReadAndBcastFile(support_manifest, buffer);
        std::istringstream support(std::string(buffer.data()));
        std::string version, trailing;
        int stored_seed = 0;
        std::string counter_token;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE((support >> version >> stored_seed >> counter_token)
            && version == "resolved_pairwise_v3" && stored_seed == m_resolved_qei_seed
            && counter_token.find_first_not_of("0123456789") == std::string::npos
            && !(support >> trailing),
            "Invalid resolved Qei support checkpoint manifest.");
        std::istringstream counter_stream(counter_token);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(counter_stream >> m_resolved_qei_counter,
            "Invalid resolved Qei support checkpoint counter.");
    }
    auto const transport_manifest = directory + "/HybridIdealElectronTransport.txt";
    bool const ideal_fv = m_fv_transport_internal_energy
        && m_electron_thermodynamics.executor().isIdealGas();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(Exists(transport_manifest) == ideal_fv,
        "Restart must preserve the ideal electron transport model and its manifest.");
    if (ideal_fv) {
        amrex::Vector<char> transport_buffer;
        amrex::ParallelDescriptor::ReadAndBcastFile(transport_manifest, transport_buffer);
        std::istringstream transport(std::string(transport_buffer.data()));
        std::string version, trailing;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE((transport >> version)
            && version == (m_fv_transport_implicit ? "ideal_finite_volume_implicit_v1"
                                                 : "ideal_finite_volume_v1")
            && !(transport >> trailing),
            "Invalid ideal electron transport checkpoint manifest.");
    }
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
