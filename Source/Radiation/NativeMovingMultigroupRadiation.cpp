/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "RadiationTransport.H"

#include "CoupledGroupSource.H"
#include "EmbeddedBoundary/Enabled.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Fields.H"
#include "GroupMomentGeometry.H"
#include "HybridCaloricResidual.H"
#include "HybridRadiationCoefficients.H"
#include "ImplicitMomentTransport.H"
#include "MovingMultigroupRuntime.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/PhotonParticleContainer.H"
#include "ResolvedMaterialOpacity.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Reduce.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>

using warpx::fields::FieldType;
using namespace amrex::literals;
using namespace warpx::radiation;

namespace {
constexpr char manifest_name[] = "/RadiationMovingMultigroup_data.txt";

std::string
ParserIdentity (amrex::Parser const& parser) {
    amrex::Parser const raw(parser.expr());
    std::set<std::string> const variables{"x",  "y", "z", "t", "photon_energy",
                                          "ne", "Te"};
    std::ostringstream identity;
    identity << std::setprecision(
                    std::numeric_limits<amrex::Real>::max_digits10)
             << std::quoted(parser.expr());
    for (auto const& symbol : raw.symbols()) {
        if (variables.contains(symbol)) {
            continue;
        }
        auto const constant = utils::parser::makeParser(symbol, {});
        identity << ' ' << std::quoted(symbol) << ' '
                 << constant.compileHost<0>()();
    }
    return identity.str();
}

#if defined(WARPX_DIM_RZ)
// Actual stored Cartesian native u/carry at the actual r/theta, not the
// requested cell impulse. Components are (KE+work carry,c*Pz,c*Lz,
// uncancelled c*Pz operands,uncancelled c*Lz operands,KE/work magnitude).
amrex::GpuArray<amrex::Real, 6>
StoredRZMaterialInventory (WarpXParticleContainer::Base const& material, amrex::Real rest_mass,
                           int groups)
{
    amrex::GpuArray<int, 36> components{};
    auto const& names = material.GetRealSoANames();
    int const paths = groups + 1;
    for (int path = 0; path < paths; ++path) {
        auto const label =
            path == groups ? std::string("streaming") : "diffusion_" + std::to_string(path);
        for (int d = 0; d < 4; ++d) {
            std::array<std::string, 4> const suffix{"ux", "uy", "uz", "work"};
            auto const found = std::find(names.begin(), names.end(),
                                         "radiation_impulse_" + label + "_" + suffix[d]);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                found != names.end(),
                "Moving RZ group material is missing its persisted carry participant.");
            components[4 * path + d] = static_cast<int>(found - names.begin());
        }
    }
    amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
                     amrex::ReduceOpSum, amrex::ReduceOpSum>
        ops;
    amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real>
        reduced(ops);
    for (auto const& [tile_key, tile] : material.GetParticles(0)) {
        amrex::ignore_unused(tile_key);
        auto const data = tile.getConstParticleTileData();
        amrex::GpuArray<amrex::ParticleReal const*, 36> carry{};
        for (int d = 0; d < 4 * paths; ++d) {
            carry[d] = tile.GetStructOfArrays().GetRealData(components[d]).dataPtr();
        }
        ops.eval(
            tile.numParticles(), reduced,
            [=] AMREX_GPU_DEVICE(long p) -> amrex::GpuTuple<amrex::Real, amrex::Real, amrex::Real,
                                                            amrex::Real, amrex::Real, amrex::Real> {
                auto const mass = rest_mass * data.m_rdata[PIdx::w][p];
                auto const ux = data.m_rdata[PIdx::ux][p], uy = data.m_rdata[PIdx::uy][p],
                           uz = data.m_rdata[PIdx::uz][p];
                auto const u2 = ux * ux + uy * uy + uz * uz;
                auto const ke = mass * u2 / (std::sqrt(1 + u2 / (PhysConst::c * PhysConst::c)) + 1);
                auto const radius = data.m_rdata[PIdx::r][p];
                auto const cosine = std::cos(data.m_rdata[PIdx::theta][p]),
                           sine = std::sin(data.m_rdata[PIdx::theta][p]);
                amrex::Real cx = 0, cy = 0, cz = 0, work = 0;
                auto sx = std::abs(ux), sy = std::abs(uy), sz = std::abs(uz), sw = std::abs(ke);
                for (int path = 0; path < paths; ++path) {
                    auto const x = carry[4 * path][p];
                    auto const y = carry[4 * path + 1][p];
                    auto const z = carry[4 * path + 2][p];
                    auto const w = carry[4 * path + 3][p];
                    cx += x;
                    cy += y;
                    cz += z;
                    work += w;
                    sx += std::abs(x);
                    sy += std::abs(y);
                    sz += std::abs(z);
                    sw += mass * std::abs(w);
                }
                return {ke + mass * work,
                        mass * PhysConst::c * (uz + cz),
                        mass * PhysConst::c * radius * ((uy + cy) * cosine - (ux + cx) * sine),
                        mass * PhysConst::c * sz,
                        mass * PhysConst::c * radius *
                            (sy * std::abs(cosine) + sx * std::abs(sine)),
                        sw};
            });
    }
    auto const value = reduced.value();
    amrex::GpuArray<amrex::Real, 6> result{amrex::get<0>(value), amrex::get<1>(value),
                                           amrex::get<2>(value), amrex::get<3>(value),
                                           amrex::get<4>(value), amrex::get<5>(value)};
    amrex::ParallelDescriptor::ReduceRealSum(result.data(), 6);
    return result;
}

amrex::Real
RZAngularMoment (amrex::MultiFab const& radiation, int component, amrex::Geometry const& geometry)
{
    GroupMomentMetric const metric(geometry);
    amrex::ReduceOps<amrex::ReduceOpSum> ops;
    amrex::ReduceData<amrex::Real> reduced(ops);
    for (amrex::MFIter it(radiation); it.isValid(); ++it) {
        auto const q = radiation.const_array(it);
        ops.eval(it.validbox(), reduced,
                 [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<amrex::Real> {
                     return {metric.Angular(i).mean_radius * q(i, j, k, component)};
                 });
    }
    auto result = amrex::get<0>(reduced.value());
    amrex::ParallelDescriptor::ReduceRealSum(result);
    return result;
}

bool
AuditPrivateRZGroupInventory (MultiParticleContainer& particles,
                              std::vector<std::string> const& species,
                              ParticleImpulseMaterial const& candidate,
                              amrex::MultiFab const& old_radiation,
                              amrex::MultiFab const& new_radiation, amrex::MultiFab const& heat,
                              amrex::Geometry const& geometry, int groups, amrex::Real tolerance,
                              ImplicitGroupMomentTransportResult const& spatial)
{
    amrex::GpuArray<amrex::Real, 6> old{}, next{};
    for (auto const& name : species) {
        auto const& live = particles.GetParticleContainerFromName(name);
        auto const* trial = candidate.CandidateSpecies(name);
        bool present = trial != nullptr;
        amrex::ParallelDescriptor::ReduceBoolAnd(present);
        if (!present) {
            return false;
        }
        auto const a = StoredRZMaterialInventory(live, live.getMass(), groups);
        auto const b = StoredRZMaterialInventory(*trial, live.getMass(), groups);
        for (int d = 0; d < 6; ++d) {
            old[d] += a[d];
            next[d] += b[d];
        }
    }
    amrex::GpuArray<amrex::Real, 3> residual{
        next[0] - old[0] + heat.sum(0) + spatial.boundary_exchange[0],
        next[1] - old[1] + spatial.boundary_exchange[3] - spatial.geometric_exchange[3],
        next[2] - old[2]}; // P1 mirror has zero theta wall current; no radial invariant.
    amrex::Real radiation_scale = 0;
    for (int group = 0; group < groups; ++group) {
        residual[0] += new_radiation.sum(4 * group) - old_radiation.sum(4 * group);
        residual[1] += new_radiation.sum(4 * group + 3) - old_radiation.sum(4 * group + 3);
        residual[2] += RZAngularMoment(new_radiation, 4 * group + 2, geometry) -
                       RZAngularMoment(old_radiation, 4 * group + 2, geometry);
        radiation_scale += old_radiation.norm1(4 * group) + new_radiation.norm1(4 * group);
    }
    auto const eps = 128 * std::numeric_limits<amrex::Real>::epsilon();
    amrex::GpuArray<amrex::Real, 3> bound{
        tolerance * radiation_scale + eps * (radiation_scale + heat.norm1(0) + old[5] + next[5]),
        tolerance * radiation_scale + eps * (radiation_scale + old[3] + next[3]),
        tolerance * radiation_scale * geometry.ProbHi(0) +
            eps * (radiation_scale * geometry.ProbHi(0) + old[4] + next[4])};
    bool valid = true;
    for (int d = 0; d < 3; ++d) {
        valid = valid && std::isfinite(residual[d]) && std::isfinite(bound[d]) &&
                std::abs(residual[d]) <= bound[d];
    }
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    return valid;
}
#endif
} // namespace

void
RadiationTransport::ConfigureMovingMultigroupRuntime (MultiParticleContainer& particles)
{
    auto const& sim = WarpX::GetInstance();
    auto const& geometry = sim.Geom(0);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_hybrid_model,
                                     "moving_multigroup requires a native hybrid electron model.");
    auto const eos = m_hybrid_model->electronThermodynamicsExecutor();
    amrex::ParmParse pp("radiation_transport");
    std::string rz_boundary = "none";
    pp.query("moving_group_rz_boundary", rz_boundary);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(rz_boundary == "none" || rz_boundary == "bounded_p1_mirror",
                                     "moving_group_rz_boundary must be none or bounded_p1_mirror.");
    m_moving_groups.bounded_rz_p1_mirror = rz_boundary == "bounded_p1_mirror";
    if (!pp.contains("minimum_diffusion_optical_depth")) {
        m_minimum_diffusion_optical_depth = 8;
    }
    int max_level = 0;
    amrex::ParmParse("amr").query("max_level", max_level);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        std::numeric_limits<amrex::Real>::digits >= 53 &&
            std::numeric_limits<amrex::ParticleReal>::digits >= 53 && m_num_groups >= 1 &&
            m_num_groups <= 8 && m_enable_diffusion && m_enable_lte_exchange &&
            m_enable_momentum_coupling && m_particle_momentum_carry && couplesToHybridElectrons() &&
            m_lte_exchange_solver == LteExchangeSolver::ImplicitTemperature &&
            !m_enable_particle_conversion && !m_coherent_packet_transport &&
            !m_momentum_species.empty() && max_level == 0 && !EB::enabled() &&
            WarpX::do_moving_window == 0 && !particles.hasHybridIonization() &&
            !m_use_species_opacity && !m_use_material_opacity_table && !m_has_diffusion_bath &&
            !m_use_planck_table && !m_use_spectral_planck_table && !m_use_rosseland_table &&
            !m_use_spectral_rosseland_table &&
            (eos.isIdealGas() || eos.isFixedChargeLatentEnergy()) && eos.m_num_materials == 0 &&
            m_hybrid_model->m_solve_electron_energy_equation && m_lte_exchange_tolerance > 0 &&
            m_lte_exchange_tolerance <= 1.e-10_rt && m_minimum_diffusion_optical_depth >= 8 &&
            static_cast<std::size_t>(particles.nSpecies()) == m_momentum_species.size() + 1,
        "moving_multigroup requires DP native fixed-charge ideal/latent hybrid "
        "material, "
        "all ions plus empty photons, 1-8 groups, implicit "
        "LTE/diffusion/recoil, particle carry, "
        "level zero without EB/window, periodic Cartesian geometry and no "
        "species/material "
        "opacity backend or bath; this initial runtime requires parser/scalar "
        "opacity, "
        "source tolerance<=1.e-10 and cell depth>=8.");
#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    WARPX_ABORT_WITH_MESSAGE(
        "moving_multigroup native runtime is currently Cartesian only.");
#endif
#if defined(WARPX_DIM_RZ)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_moving_groups.bounded_rz_p1_mirror && m_rz_angular_transport &&
            !m_hybrid_model->m_pressure_only &&
            m_hybrid_model->transportsElectronInternalEnergyAtRawDensity() &&
            !m_hybrid_model->m_fv_transport_implicit &&
            !m_hybrid_model->m_conservative_pressure_work &&
            !m_hybrid_model->m_include_joule_heating &&
            WarpX::n_rz_azimuthal_modes == 1 && geometry.ProbLo(0) == 0 &&
            !geometry.isPeriodic(0) && geometry.isPeriodic(1) &&
            WarpX::field_boundary_lo[0] == FieldBoundaryType::None &&
            WarpX::field_boundary_hi[0] == FieldBoundaryType::PEC &&
            WarpX::particle_boundary_lo[0] == ParticleBoundaryType::None &&
            WarpX::particle_boundary_hi[0] == ParticleBoundaryType::Reflecting,
        "RZ moving_multigroup requires explicit moving_group_rz_boundary=bounded_p1_mirror, "
        "rz_angular_transport=1, finite-volume ideal/latent electrons with "
        "conservative_pressure_work=0, m0 axis/outer-PEC reflecting material and periodic z; "
        "generic pressure-work/induction conservation and kinetic half-space wall pressure "
        "are not claimed.");
#else
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_moving_groups.bounded_rz_p1_mirror,
        "moving_group_rz_boundary=bounded_p1_mirror requires RZ geometry.");
#endif
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
#if defined(WARPX_DIM_RZ)
        if (d == 0) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                m_diffusion_boundary_lo[d] == static_cast<int>(DiffusionBoundary::Reflecting) &&
                m_diffusion_boundary_hi[d] == static_cast<int>(DiffusionBoundary::Reflecting),
                "RZ moving_multigroup requires bounded reflecting optical faces.");
            continue;
        }
#endif
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            geometry.isPeriodic(d) &&
                WarpX::particle_boundary_lo[d] ==
                    ParticleBoundaryType::Periodic &&
                WarpX::particle_boundary_hi[d] ==
                    ParticleBoundaryType::Periodic &&
                m_diffusion_boundary_lo[d] ==
                    static_cast<int>(DiffusionBoundary::Reflecting) &&
                m_diffusion_boundary_hi[d] ==
                    static_cast<int>(DiffusionBoundary::Reflecting),
            "moving_multigroup requires periodic Cartesian material and "
            "radiation faces.");
    }
    for (auto const& name : m_momentum_species) {
        auto const& species = particles.GetParticleContainerFromName(name);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !species.HasEvolvingChargeState() && !species.do_not_deposit,
            "moving_multigroup requires deposited fixed-charge material.");
#if defined(WARPX_DIM_RZ)
        int do_not_gather = 0;
        amrex::ParmParse(name).query("do_not_gather", do_not_gather);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!species.doNotPush() && do_not_gather,
            "Bounded RZ moving_multigroup requires moving ions with do_not_gather=1; "
            "radiation realizes the native velocity/energy-carry kick and positions drift.");
#endif
        m_moving_groups.configured_material_identity.push_back(
            {static_cast<amrex::Real>(species.getMass()),
             static_cast<amrex::Real>(species.getCharge())});
    }
    utils::parser::queryWithParser(pp, "moving_group_relaxation",
                                   m_moving_groups.relaxation);
    utils::parser::queryWithParser(pp, "moving_group_linear_tolerance",
                                   m_moving_groups.linear_tolerance);
    pp.query("moving_group_max_linear_iterations",
             m_moving_groups.max_linear_iterations);
    pp.query("moving_group_verbosity", m_moving_groups.verbosity);
    std::vector<amrex::Real> ratio{0, 0, 0};
    utils::parser::queryArrWithParser(pp, "moving_group_initial_lab_flux_ratio",
                                      ratio);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(ratio.size() == 3,
                                     "moving_group_initial_lab_flux_ratio "
                                     "requires three LAB F/(cE) components.");
    amrex::Real norm = 0;
    for (int d = 0; d < 3; ++d) {
        m_moving_groups.initial_lab_flux_ratio[d] = ratio[d];
        norm += ratio[d] * ratio[d];
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        std::isfinite(norm) && norm <= 1._rt / 9 &&
            m_moving_groups.relaxation > 0 && m_moving_groups.relaxation <= 1 &&
            m_moving_groups.linear_tolerance > 0 &&
            m_moving_groups.linear_tolerance <= 1.e-13_rt &&
            m_moving_groups.max_linear_iterations > 0 &&
            m_moving_groups.verbosity >= 0,
        "Invalid moving multigroup initial dipole, relaxation or linear-solver "
        "controls.");
    std::ostringstream identity;
    identity << std::setprecision(
                    std::numeric_limits<amrex::Real>::max_digits10)
             << "lab_fixed_bins_exact_com_planck_dipole_ap_v1 "
                "caloric_unique_nodes_v1 "
             << (eos.isIdealGas() ? "ideal_gas" : "fixed_charge_latent_energy")
             << ' ' << eos.m_gamma << ' ' << eos.m_num_latent_transitions << ' '
             << m_minimum_electron_density << ' '
             << m_hybrid_model->electronDensityFloor() << ' '
             << m_hybrid_model->transportsElectronInternalEnergyAtRawDensity()
             << ' ';
#if defined(WARPX_DIM_RZ)
    identity << "rz_annular_lab_conjugate_p1_mirror_fv_caloric_v1 "
             << WarpX::n_rz_azimuthal_modes << ' ' << m_hybrid_model->m_pressure_only << ' '
             << m_hybrid_model->m_conservative_pressure_work << ' '
             << static_cast<int>(WarpX::grid_type) << ' '
             << WarpX::GetInstance().verboncoeurAxisCorrection() << ' ';
#endif
    for (int n = 0; n < eos.m_num_latent_transitions; ++n) {
        identity << eos.m_latent_transition_temperature[n] << ' '
                 << eos.m_latent_energy_per_electron[n] << ' '
                 << eos.m_latent_transition_sharpness[n] << ' ';
    }
    // Capture the requested species parameters before checkpoint particle
    // restoration. An old checkpoint must not overwrite an incompatible new
    // request and thereby hide an identity change.
    identity << std::quoted(m_photon_species) << ' '
             << m_momentum_species.size() << ' ';
    for (std::size_t n = 0; n < m_momentum_species.size(); ++n) {
        identity << std::quoted(m_momentum_species[n]) << ' '
                 << m_moving_groups.configured_material_identity[n][0] << ' '
                 << m_moving_groups.configured_material_identity[n][1] << ' ';
    }
    identity << m_num_groups << ' ';
    for (auto edge : m_energy_group_boundaries_h) {
        identity << edge << ' ';
    }
    for (auto energy : m_group_photon_energies_h) {
        identity << energy << ' ';
    }
    identity
        << std::quoted(ParserIdentity(m_planck_absorption_coefficient_parser))
        << ' '
        << std::quoted(ParserIdentity(m_rosseland_transport_coefficient_parser))
        << ' ' << m_lte_exchange_max_iterations << ' '
        << m_lte_exchange_tolerance << ' ' << m_moving_groups.relaxation << ' '
        << m_moving_groups.linear_tolerance << ' '
        << m_moving_groups.max_linear_iterations << ' '
        << m_minimum_diffusion_optical_depth << ' '
        << "beta<=0.01;absorbing-beta-chi-L<=0.01;lab-dipole<=1/3 "
        << AMREX_SPACEDIM << ' ' << geometry.Coord() << ' ' << WarpX::nox
        << ' ';
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        identity << geometry.Domain().smallEnd(d) << ' '
                 << geometry.Domain().bigEnd(d) << ' ' << geometry.ProbLo(d)
                 << ' ' << geometry.ProbHi(d) << ' ' << geometry.isPeriodic(d)
                 << ' ' << static_cast<int>(WarpX::field_boundary_lo[d]) << ' '
                 << static_cast<int>(WarpX::field_boundary_hi[d]) << ' '
                 << static_cast<int>(WarpX::particle_boundary_lo[d]) << ' '
                 << static_cast<int>(WarpX::particle_boundary_hi[d]) << ' '
                 << m_diffusion_boundary_lo[d] << ' '
                 << m_diffusion_boundary_hi[d] << ' ';
    }
    m_moving_groups.model_identity = identity.str();
    auto const& local = m_moving_groups.model_identity;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        local.size() <=
            static_cast<std::size_t>(std::numeric_limits<int>::max()),
        "Moving multigroup model identity exceeds the MPI count limit.");
    int size = static_cast<int>(local.size());
    auto const root = amrex::ParallelDescriptor::IOProcessorNumber();
    amrex::ParallelDescriptor::Bcast(&size, 1, root);
    std::string reference(static_cast<std::size_t>(size), '\0');
    if (amrex::ParallelDescriptor::IOProcessor()) {
        reference = local;
    }
    amrex::ParallelDescriptor::Bcast(reference.data(), size, root);
    bool same = reference == local;
    amrex::ParallelDescriptor::ReduceBoolAnd(same);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(same,
                                     "Moving multigroup physical model "
                                     "identity must agree across MPI ranks.");
    m_track_energy_balance = true;
}

void
RadiationTransport::AllocateMovingMultigroupRuntime (
    ablastr::fields::MultiFabRegister& fields, int lev,
    amrex::BoxArray const& boxes,
    amrex::DistributionMapping const& distribution) const {
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        lev == 0, "Moving multigroup runtime is level zero only.");
    auto const& energy =
        *fields.get(FieldType::radiation_diffusion_energy, lev);
    for (int group = 0; group < m_num_groups; ++group) {
        for (int direction = 0; direction < 3; ++direction) {
            auto const name = MovingGroupMomentumFieldName(group, direction);
            fields.alloc_init(name, lev, boxes, distribution, 1,
                              amrex::IntVect(1), 0._rt, true, true, true,
                              false);
            if (!m_is_restart) {
                auto& momentum = *fields.get(name, lev);
                amrex::MultiFab::Copy(momentum, energy, group, 0, 1, 0);
                momentum.mult(
                    m_moving_groups.initial_lab_flux_ratio[direction]);
                momentum.FillBoundary(
                    WarpX::GetInstance().Geom(lev).periodicity());
            }
        }
    }
}

void
RadiationTransport::AdvanceMovingMultigroupRuntime (
    MultiParticleContainer& particles,
    ablastr::fields::MultiFabRegister& fields, amrex::Real current_time,
    amrex::Real dt) const {
    auto& sim = WarpX::GetInstance();
    auto const& geometry = sim.Geom(0);
    auto& photons = particles.GetParticleContainerFromName(m_photon_species);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(photons.TotalNumberOfParticles() == 0,
                                     "moving_multigroup owns opaque moments "
                                     "and requires empty native photons.");
    auto& energy = *fields.get(FieldType::radiation_diffusion_energy, 0);
    auto& heat = *fields.get(FieldType::radiation_material_energy, 0);
#if defined(WARPX_DIM_RZ)
    amrex::MultiFab private_heat(heat.boxArray(), heat.DistributionMap(), 1, heat.nGrowVect());
    amrex::MultiFab::Copy(private_heat, heat, 0, 0, 1, heat.nGrowVect());
    auto& trial_heat = private_heat;
#else
    auto& trial_heat = heat;
#endif
    auto const& live_temperature =
        *fields.get(FieldType::hybrid_electron_temperature_fp, 0);
    auto const& live_density = *fields.get(FieldType::rho_fp, 0);
    auto const eos = m_hybrid_model->electronThermodynamicsExecutor();
    amrex::MultiFab radiation(energy.boxArray(), energy.DistributionMap(),
                              4 * m_num_groups, 1);
    radiation.setVal(0);
    for (int group = 0; group < m_num_groups; ++group) {
        amrex::MultiFab::Copy(radiation, energy, group, 4 * group, 1, 0);
        for (int direction = 0; direction < 3; ++direction) {
            amrex::MultiFab::Copy(
                radiation,
                *fields.get(MovingGroupMomentumFieldName(group, direction), 0),
                0, 4 * group + 1 + direction, 1, 0);
        }
    }
    radiation.FillBoundary(geometry.periodicity());
    amrex::MultiFab temperature(live_temperature.boxArray(),
                                live_temperature.DistributionMap(), 1,
                                live_temperature.nGrowVect());
    amrex::MultiFab density(live_density.boxArray(),
                            live_density.DistributionMap(),
                            live_density.nComp(), live_density.nGrowVect());
    amrex::MultiFab::Copy(temperature, live_temperature, 0, 0, 1,
                          temperature.nGrowVect());
    amrex::MultiFab::Copy(density, live_density, 0, 0, live_density.nComp(),
                          density.nGrowVect());
    temperature.FillBoundary(geometry.periodicity());
    density.FillBoundary(geometry.periodicity());
    auto const minimum = m_minimum_electron_density;
    auto const floor =
        m_hybrid_model->transportsElectronInternalEnergyAtRawDensity()
            ? 0._rt
            : m_hybrid_model->electronDensityFloor();
    detail::OpacityEvaluator<1> const planck{false,
                                             0,
                                             {},
                                             {},
                                             {},
                                             {},
                                             m_use_planck_table,
                                             m_use_spectral_planck_table,
                                             m_planck_table.executor(),
                                             m_spectral_planck_table.executor(),
                                             m_planck_absorption_coefficient};
    detail::OpacityEvaluator<1> const transport{
        false,
        0,
        {},
        {},
        {},
        {},
        m_use_rosseland_table,
        m_use_spectral_rosseland_table,
        m_rosseland_table.executor(),
        m_spectral_rosseland_table.executor(),
        m_rosseland_transport_coefficient};
    CoupledGroupSourceCallbacks callbacks;
    callbacks.coefficients =
        [&] (amrex::MultiFab const& trial, amrex::MultiFab& chi,
             amrex::MultiFab& absorption, amrex::MultiFab& cell_temperature,
             amrex::Real time) {
            amrex::MultiFab emission(chi.boxArray(), chi.DistributionMap(),
                                     m_num_groups, 0);
            detail::FillCoupledHybridCoefficients(
                density, trial, chi, absorption, emission, geometry, eos,
                minimum, floor, planck, transport, m_energy_groups, time, {});
            absorption.mult(1 / PhysConst::c);
            amrex::MultiFab cell_density(chi.boxArray(), chi.DistributionMap(),
                                         1, 0);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                detail::SnapshotResolvedMaterialOpacity(
                    density, trial, cell_density, cell_temperature, geometry,
                    eos, minimum, floor),
                "Moving group source could not sample native material/color "
                "temperature.");
        };
    callbacks.caloric.material_response = [&] (amrex::MultiFab& trial,
                                               amrex::MultiFab& source) {
        return m_hybrid_model->EvaluateElectronEnergySource(
            0, trial, source, minimum, nullptr, nullptr, nullptr, true);
    };
    {
        callbacks.caloric.material_residual =
            [&] (amrex::MultiFab const& old, amrex::MultiFab const& candidate,
                 amrex::MultiFab const& source) {
                return detail::CoupledCaloricResidual(
                    density, old, candidate, source, eos, minimum, floor, {});
            };
        callbacks.caloric.material_local_residual =
            [&] (amrex::MultiFab const& old, amrex::MultiFab const& candidate,
                 amrex::MultiFab const& source, amrex::MultiFab const& lower,
                 amrex::MultiFab const& upper, amrex::Real tolerance) {
                return detail::CoupledLocalCaloricResidual(
                    density, old, candidate, source, lower, upper, eos, minimum,
                    floor, {}, tolerance);
            };
    }
    CoupledGroupSourceOptions options;
    options.groups = m_energy_groups;
    options.max_iterations = m_lte_exchange_max_iterations;
    options.tolerance = m_lte_exchange_tolerance;
    options.energy_tolerance = m_lte_exchange_tolerance;
    options.relaxation = m_moving_groups.relaxation;
    options.verbose = m_moving_groups.verbosity > 0;
    options.spatial_transport = true;
    options.spatial.groups = m_energy_groups;
    options.spatial.linear_tolerance = m_moving_groups.linear_tolerance;
    options.spatial.max_linear_iterations =
        m_moving_groups.max_linear_iterations;
    options.spatial.minimum_optical_depth = m_minimum_diffusion_optical_depth;
#if defined(WARPX_DIM_RZ)
    options.assignment = ParticleImpulseAssignment::RZAngularConservative;
    options.spatial.reflecting_boundaries = m_moving_groups.bounded_rz_p1_mirror;
    options.defer_live_commit = true;
#endif
    auto const result = TryAdvanceCoupledGroupSource(
        particles, m_momentum_species, "diffusion_0", radiation, temperature,
        trial_heat, geometry, current_time, dt, options, callbacks);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        result.valid, "Moving multigroup spatial/heat/recoil stage rejected "
                      "before accepted publication: " +
                          result.reason);
#if defined(WARPX_DIM_RZ)
    MomentTransportLedger candidate_ledger = m_moment_transport_ledger;
    MomentTransportAccounting const accounting{
        result.spatial_trial->boundary_exchange, result.spatial_trial->geometric_exchange};
    // Audit actual stored private material and independent physical ownership
    // BEFORE any live field, carry, temperature or accounting publication.
    amrex::MultiFab old_radiation(radiation.boxArray(), radiation.DistributionMap(),
                                 radiation.nComp(), 0);
    for (int group = 0; group < m_num_groups; ++group) {
        amrex::MultiFab::Copy(old_radiation, energy, group, 4 * group, 1, 0);
        for (int d = 0; d < 3; ++d) {
            amrex::MultiFab::Copy(old_radiation,
                *fields.get(MovingGroupMomentumFieldName(group, d), 0), 0, 4 * group + d + 1, 1, 0);
        }
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(result.private_material &&
        candidate_ledger.TryAccumulate(accounting) &&
        AuditPrivateRZGroupInventory(particles, m_momentum_species, *result.private_material,
            old_radiation, radiation, trial_heat, geometry, m_num_groups,
            m_lte_exchange_tolerance, *result.spatial_trial),
        "RZ moving group actual private E/Pz/Lz/carry/wall inventory rejected before publication.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(result.private_material->Commit(),
        "RZ moving group accepted private native material publication rejected.");
    amrex::MultiFab::Copy(heat, trial_heat, 0, 0, 1, heat.nGrowVect());
    m_moment_transport_ledger = candidate_ledger;
#endif
    m_hybrid_model->CommitElectronTemperature(0, temperature);
    for (int group = 0; group < m_num_groups; ++group) {
        amrex::MultiFab::Copy(energy, radiation, 4 * group, group, 1, 0);
        for (int direction = 0; direction < 3; ++direction) {
            auto& q =
                *fields.get(MovingGroupMomentumFieldName(group, direction), 0);
            amrex::MultiFab::Copy(q, radiation, 4 * group + 1 + direction, 0, 1,
                                  0);
            q.FillBoundary(geometry.periodicity());
        }
    }
    energy.FillBoundary(geometry.periodicity());
    auto& impulse = *fields.get(FieldType::radiation_material_momentum, 0);
    auto& work = *fields.get(FieldType::radiation_material_kinetic_energy, 0);
    impulse.setVal(0);
    work.setVal(0);
    amrex::MultiFab::Copy(impulse, result.actual_material_impulse, 0, 0, 3, 0);
    amrex::MultiFab::Copy(work, result.actual_material_work, 0, 0, 1, 0);
    m_last_numerical_energy_residual = result.energy_inventory_defect;
    m_cumulative_numerical_energy_residual += result.energy_inventory_defect;
    m_last_material_carry_energy_change = result.energy_carry_change;
    m_last_boundary_energy_loss = 0;
    m_last_diffusion_boundary_energy_loss = 0;
    m_last_streaming_boundary_energy_loss = 0;
    m_last_boundary_energy_injection = 0;
    m_last_diffusion_boundary_momentum_loss = {0, 0, 0};
    m_last_streaming_boundary_momentum_loss = {0, 0, 0};
    m_last_particle_wall_momentum = {0, 0, 0};
    m_last_coupled_solve = {
        result.iterations,         1, 0, 0, result.caloric_residual,
        result.energy_inventory_defect};
    m_last_implicit_diffusion = {};
    if (result.spatial_trial) {
        m_last_implicit_diffusion.linear_iterations =
            result.spatial_trial->linear_iterations;
        m_last_implicit_diffusion.maximum_relative_residual =
            result.spatial_trial->equation_residual;
    }
    if (m_moving_groups.verbosity > 0) {
        amrex::Print() << "Moving multigroup accepted groups=" << m_num_groups
                       << " iterations=" << result.iterations
                       << " heat=" << heat.sum(0)
                       << " work=" << result.actual_kinetic_work
                       << " raw_energy=" << result.raw_energy_residual
                       << " raw_momentum=" << result.raw_momentum_residual
#if defined(WARPX_DIM_RZ)
                       << " raw_angular=" << result.raw_angular_momentum_residual
                       << " angular_balance=" << result.angular_balance_residual
                       << " radial_is_local_connection_balance=1"
#endif
                       << '\n';
    }
}

void
RadiationTransport::WriteMovingMultigroupCheckpoint (
    std::string const& directory) const {
    if (!m_use_moving_multigroup) {
        return;
    }
    std::ofstream manifest{directory + manifest_name};
    manifest << "moving_multigroup_lab_exact_v1\n"
             << std::quoted(m_moving_groups.model_identity) << '\n';
#if defined(WARPX_DIM_RZ)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_moment_transport_ledger.Write(manifest),
        "Could not write RZ moving multigroup wall/connection accounting.");
#endif
    manifest.flush();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        manifest.good(),
        "Could not write moving multigroup checkpoint identity.");
}

void
RadiationTransport::ReadMovingMultigroupCheckpoint (
    std::string const& directory) {
    if (!m_use_moving_multigroup) {
        return;
    }
    std::ifstream manifest{directory + manifest_name};
    std::string version, identity, trailing;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (manifest >> version >> std::quoted(identity)) &&
            version == "moving_multigroup_lab_exact_v1" &&
            identity == m_moving_groups.model_identity,
        "Moving multigroup restart changed its LAB/proper-bin model, opacity, "
        "material, "
        "mesh, boundaries or solver controls, or has incomplete metadata.");
#if defined(WARPX_DIM_RZ)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_moment_transport_ledger.Read(manifest),
        "Invalid RZ moving multigroup wall/connection checkpoint accounting.");
#endif
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!(manifest >> trailing),
        "Moving multigroup checkpoint has trailing model/accounting data.");
    auto const& fields = WarpX::GetInstance().m_fields;
    auto const& particles = WarpX::GetInstance().GetPartContainer();
    for (std::size_t n = 0; n < m_momentum_species.size(); ++n) {
        auto const& material =
            particles.GetParticleContainerFromName(m_momentum_species[n]);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            material.getMass() ==
                    m_moving_groups.configured_material_identity[n][0] &&
                material.getCharge() ==
                    m_moving_groups.configured_material_identity[n][1],
            "Moving multigroup restart changed checkpoint material mass or "
            "charge.");
    }
    auto const& energy = *fields.get(FieldType::radiation_diffusion_energy, 0);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        energy.is_finite() && energy.min(0) >= 0,
        "Moving multigroup restart requires finite nonnegative LAB energy.");
    for (int group = 0; group < m_num_groups; ++group) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            energy.min(group) >= 0, "Moving multigroup restart requires "
                                    "nonnegative energy in every LAB bin.");
        for (int direction = 0; direction < 3; ++direction) {
            auto const name = MovingGroupMomentumFieldName(group, direction);
            std::ifstream const header{directory + "/Level_0/" + name +
                                       "[level=0]_H"};
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                header.good() && fields.get(name, 0)->is_finite(),
                "Moving multigroup restart requires every finite persisted LAB "
                "momentum field.");
        }
    }
}
