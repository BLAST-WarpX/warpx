/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "RadiationTransport.H"

#include "AdaptivePacketDiffusionTransition.H"
#include "CartesianPhotonBoundary.H"
#include "CollidedDiffusionAdmission.H"
#include "EmbeddedBoundary/Enabled.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Fields.H"
#include "FluxRelaxedPackets.H"
#include "PacketDiffusionNativeInterval.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/PhotonParticleContainer.H"
#include "PhotonLogicalDrawKey.H"
#include "ResolvedDiffusionFront.H"
#include "ResolvedMaterialOpacity.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Reduce.H>
#include <AMReX_iMultiFab.H>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string>

using namespace amrex::literals;
using warpx::fields::FieldType;
using namespace warpx::radiation;

namespace {
// Free function keeps CUDA extended lambdas out of private member functions.
void
InitializeOrValidateResolvedFields (
    amrex::MultiFab& mask, amrex::MultiFab& opacity,
    amrex::MultiFab const& energy, amrex::Geometry const& geometry,
    amrex::ParserExecutor<3> mask_parser,
    ResolvedOpacityExecutor opacity_parser, bool restored = false,
    amrex::Real opacity_time = 0,
    amrex::MultiFab const* electron_density = nullptr,
    amrex::MultiFab const* electron_temperature = nullptr,
    bool defer_material_opacity = false, bool preserve_adaptive_mask = false,
    bool allow_uniform_mask = false,
    amrex::MultiFab const* validated_table_opacity = nullptr) {
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (!defer_material_opacity || (opacity_parser.material_dependent && !restored)) &&
        (!opacity_parser.material_dependent || defer_material_opacity ||
         (electron_density && electron_temperature)),
        "Resolved material opacity requires explicit frozen samples outside fresh allocation.");
    auto const lo = geometry.ProbLoArray(), dx = geometry.CellSizeArray();
    auto const origin = amrex::lbound(geometry.Domain());
    amrex::ReduceOps<amrex::ReduceOpMax> ops;
    amrex::ReduceData<int> data(ops);
    for (amrex::MFIter it(mask); it.isValid(); ++it) {
        auto const m = mask.array(it), chi = opacity.array(it);
        auto const e = energy.const_array(it);
        auto const ne = electron_density ? electron_density->const_array(it) :
                                          amrex::Array4<amrex::Real const>{};
        auto const te = electron_temperature ? electron_temperature->const_array(it) :
                                              amrex::Array4<amrex::Real const>{};
        auto const table_chi = validated_table_opacity ? validated_table_opacity->const_array(it) :
                                                       amrex::Array4<amrex::Real const>{};
        ops.eval(it.validbox(), data,
                 [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<int> {
                     amrex::Real x = 0, y = 0, z = 0;
#if defined(WARPX_DIM_3D)
                     x = lo[0] + (i - origin.x + 0.5_rt) * dx[0];
                     y = lo[1] + (j - origin.y + 0.5_rt) * dx[1];
                     z = lo[2] + (k - origin.z + 0.5_rt) * dx[2];
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                x = lo[0] + (i-origin.x+0.5_rt)*dx[0];
                z = lo[1] + (j-origin.y+0.5_rt)*dx[1];
#elif defined(WARPX_DIM_1D_Z)
                z = lo[0] + (i-origin.x+0.5_rt)*dx[0];
#endif
                     auto const expected_mask = mask_parser(x, y, z);
                     auto const density_value = ne ? ne(i,j,k) : 0;
                     auto const temperature_value = te ? te(i,j,k) : 0;
                     auto const expected_opacity = table_chi ? table_chi(i, j, k) :
                         defer_material_opacity ? 0 :
                         opacity_parser(x, y, z, opacity_time, density_value, temperature_value);
                     if (!restored) {
                         if (!preserve_adaptive_mask) {
                             m(i, j, k) = expected_mask;
                         }
                         chi(i, j, k) = expected_opacity;
                     }
                     // Permit only floating evaluation roundoff across CPU/GPU
                     // parsers, not a changed binding hidden behind the same text.
                     auto const opacity_roundoff =
                         64 * std::numeric_limits<amrex::Real>::epsilon() *
                         amrex::max(std::abs(chi(i, j, k)), std::abs(expected_opacity));
                     return {
                         !(std::isfinite(m(i, j, k)) &&
                           (m(i, j, k) == 0 || m(i, j, k) == 1) &&
                           std::isfinite(chi(i, j, k)) && chi(i, j, k) >= 0 &&
                           std::isfinite(e(i, j, k)) && e(i, j, k) >= 0 &&
                           (m(i, j, k) == 1 || e(i, j, k) == 0) &&
                           std::isfinite(density_value) && density_value >= 0 &&
                           std::isfinite(temperature_value) &&
                           temperature_value >= 0 &&
                           std::isfinite(expected_mask) &&
                           (preserve_adaptive_mask ||
                            expected_mask == m(i, j, k)) &&
                           std::isfinite(expected_opacity) &&
                           expected_opacity >= 0 &&
                           std::abs(expected_opacity - chi(i, j, k)) <=
                               opacity_roundoff)};
                 });
    }
    int invalid = amrex::get<0>(data.value());
    amrex::ParallelDescriptor::ReduceIntMax(invalid);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !invalid,
        restored
            ? "Resolved packet-FLD restart changed evaluated mask/opacity or has invalid E state."
            : "resolved_packet_fld requires a binary mask, finite nonnegative opacity/E, "
              "and zero initial diffusion energy in kinetic cells.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        allow_uniform_mask || (mask.min(0) == 0 && mask.max(0) == 1),
        "resolved_packet_fld requires a mixed domain containing both kinetic "
        "and diffusion cells.");
    mask.FillBoundary(geometry.periodicity());
    opacity.FillBoundary(geometry.periodicity());
}

void
CopyResolvedMask (amrex::MultiFab const& source, amrex::iMultiFab& mask)
{
    amrex::ReduceOps<amrex::ReduceOpMax> ops;
    amrex::ReduceData<int> data(ops);
    for (amrex::MFIter it(mask); it.isValid(); ++it) {
        auto const src = source.const_array(it);
        auto const dst = mask.array(it);
        ops.eval(it.validbox(), data,
                 [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<int> {
                     if (!std::isfinite(src(i, j, k)) || (src(i, j, k) != 0 && src(i, j, k) != 1)) {
                         return {1};
                     }
                     dst(i, j, k) = static_cast<int>(src(i, j, k));
                     return {0};
                 });
    }
    int invalid = amrex::get<0>(data.value());
    amrex::ParallelDescriptor::ReduceIntMax(invalid);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!invalid, "Invalid persistent resolved diffusion mask.");
}

void
AppendParserConstants (std::ostream& identity, amrex::Parser const& parser,
                       amrex::Vector<std::string> const& variables)
{
    amrex::Parser const raw(parser.expr());
    for (auto const& symbol : raw.symbols()) {
        if (std::find(variables.begin(), variables.end(), symbol) != variables.end()) {
            continue;
        }
        auto const constant = utils::parser::makeParser(symbol, {});
        identity << std::quoted(symbol) << ' ' << constant.compileHost<0>()() << ' ';
    }
}

std::string
ResolvedAdaptiveParserIdentity (amrex::Parser const& mask, amrex::Parser const& opacity)
{
    // Keep the v8 adaptive manifest representation byte-for-byte compatible.
    amrex::Vector<std::string> const variables{"x", "y", "z", "t", "ne", "Te"};
    std::ostringstream identity;
    identity << std::setprecision(std::numeric_limits<amrex::Real>::max_digits10);
    for (auto const* parser : {&mask, &opacity}) {
        identity << std::quoted(parser->expr()) << ' ';
        AppendParserConstants(identity, *parser, variables);
    }
    return identity.str();
}

void
MaskResolvedOpacity (amrex::MultiFab const& opacity,
                     amrex::iMultiFab const& mask,
                     amrex::MultiFab& diffusion_opacity) {
    for (amrex::MFIter it(diffusion_opacity); it.isValid(); ++it) {
        auto const chi = opacity.const_array(it);
        auto const output = diffusion_opacity.array(it);
        auto const active = mask.const_array(it);
        amrex::ParallelFor(
            it.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                output(i, j, k) = active(i, j, k) == 1 ? chi(i, j, k) : 0;
            });
    }
}

void
PublishResolvedMask (amrex::iMultiFab const& mask, amrex::MultiFab& field) {
    for (amrex::MFIter it(field); it.isValid(); ++it) {
        auto const source = mask.const_array(it);
        auto const target = field.array(it);
        amrex::ParallelFor(it.validbox(),
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                               target(i, j, k) = source(i, j, k);
                           });
    }
}

void
ConstrainAdaptiveMask (amrex::iMultiFab const& old, amrex::iMultiFab& desired,
                       amrex::MultiFab const& opacity,
                       amrex::Geometry const& geometry, amrex::Real dt,
                       NativePhotonPathBoundaries const& boundaries,
                       bool demotion_only) {
    auto const domain = geometry.Domain();
    auto const low = amrex::lbound(domain), high = amrex::ubound(domain);
    auto const escape_lo = boundaries.escape_lo,
               escape_hi = boundaries.escape_hi;
    for (amrex::MFIter it(desired); it.isValid(); ++it) {
        auto const previous = old.const_array(it);
        auto const next = desired.array(it);
        auto const chi = opacity.const_array(it);
        amrex::ParallelFor(it.validbox(), [=] AMREX_GPU_DEVICE(int i, int j,
                                                               int k) {
            int const position[3]{i, j, k};
            int const lo[3]{low.x, low.y, low.z}, hi[3]{high.x, high.y, high.z};
            bool kinetic = chi(i, j, k) * PhysConst::c * dt < 8;
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                kinetic = kinetic || (escape_lo[d] && position[d] == lo[d]) ||
                          (escape_hi[d] && position[d] == hi[d]);
            }
            if (kinetic || (demotion_only && previous(i, j, k) == 0)) {
                next(i, j, k) = 0;
            }
        });
    }
}
} // namespace

void
RadiationTransport::ConfigureResolvedPacketRuntime (MultiParticleContainer& particles)
{
    amrex::ParmParse pp("radiation_transport");
    auto const& geometry = WarpX::GetInstance().Geom(0);
    bool const initial_momentum_present = pp.contains("resolved_initial_momentum_ratio");
    std::vector<amrex::Real> initial_momentum{0, 0, 0};
    utils::parser::queryArrWithParser(pp, "resolved_initial_momentum_ratio", initial_momentum);
    bool valid_initial_momentum = initial_momentum.size() == 3;
    if (valid_initial_momentum)
    {
        for (auto const value : initial_momentum)
        {
            valid_initial_momentum = valid_initial_momentum && std::isfinite(value);
        }
    }
    amrex::ParallelDescriptor::ReduceBoolAnd(valid_initial_momentum);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(valid_initial_momentum,
        "resolved_initial_momentum_ratio requires three finite LAB Q/E components.");
    std::array<amrex::Real, 4> minimum_ratio{initial_momentum_present ? 1._rt : 0._rt,
        initial_momentum[0], initial_momentum[1], initial_momentum[2]};
    auto maximum_ratio = minimum_ratio;
    amrex::ParallelDescriptor::ReduceRealMin(minimum_ratio.data(), 4);
    amrex::ParallelDescriptor::ReduceRealMax(maximum_ratio.data(), 4);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(minimum_ratio == maximum_ratio &&
        std::hypot(initial_momentum[0], initial_momentum[1], initial_momentum[2]) <= 1,
        "resolved_initial_momentum_ratio must agree across MPI ranks and have norm <= 1.");
    m_resolved_packet.has_initial_momentum_ratio = initial_momentum_present;
    std::copy(initial_momentum.begin(), initial_momentum.end(),
              m_resolved_packet.initial_momentum_ratio.begin());
    int max_level = 0;
    amrex::ParmParse("amr").query("max_level", max_level);
    m_resolved_packet.thermal_exchange = m_enable_lte_exchange;
    if (m_resolved_packet.thermal_material_opacity_table)
    {
        ConfigureResolvedThermalMaterialOpacity();
    }
    int reflecting_z = 0;
    pp.query("resolved_thermal_reflecting_z", reflecting_z);
    int reflecting_min = reflecting_z, reflecting_max = reflecting_z;
    amrex::ParallelDescriptor::ReduceIntMin(reflecting_min);
    amrex::ParallelDescriptor::ReduceIntMax(reflecting_max);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(reflecting_min == reflecting_max &&
        (reflecting_z == 0 || reflecting_z == 1),
        "resolved_thermal_reflecting_z must be 0 or 1 and agree across MPI ranks.");
    m_resolved_packet.thermal_reflecting_z = reflecting_z == 1;
    int wall_witness = 0;
    pp.query("resolved_thermal_axial_wall_witness", wall_witness);
    int witness_min = wall_witness, witness_max = wall_witness;
    amrex::ParallelDescriptor::ReduceIntMin(witness_min);
    amrex::ParallelDescriptor::ReduceIntMax(witness_max);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(witness_min == witness_max &&
        (wall_witness == 0 || wall_witness == 1) &&
        (wall_witness == 0 || m_resolved_packet.thermal_reflecting_z),
        "resolved_thermal_axial_wall_witness is a default-OFF closed thermal diagnostic and must agree across MPI ranks.");
    m_resolved_packet.thermal_axial_wall_witness = wall_witness == 1;
    if (m_resolved_packet.thermal_reflecting_z)
    {
        int photon_weights = 0;
        pp.query("resolved_thermal_history_photon_weights", photon_weights);
        int weights_min = photon_weights, weights_max = photon_weights;
        amrex::ParallelDescriptor::ReduceIntMin(weights_min);
        amrex::ParallelDescriptor::ReduceIntMax(weights_max);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(weights_min == weights_max &&
            (photon_weights == 0 || photon_weights == 1) && usesResolvedThermalExchange(),
            "resolved_thermal_reflecting_z requires thermal exchange and agrees "
            "on its photon-weight selector across MPI ranks.");
        bool supported_weights = photon_weights == 0;
        if (photon_weights == 1)
        {
            int intervals = 0;
            pp.query("resolved_thermal_history_subintervals", intervals);
            int interval_min = intervals, interval_max = intervals;
            amrex::ParallelDescriptor::ReduceIntMin(interval_min);
            amrex::ParallelDescriptor::ReduceIntMax(interval_max);
#if defined(WARPX_DIM_RZ) && !defined(AMREX_USE_GPU)
            if (m_hybrid_model)
            {
                auto const eos = m_hybrid_model->electronThermodynamicsExecutor();
                supported_weights = interval_min == interval_max && intervals > 0 &&
                    intervals <= 32 && !usesAdaptivePacketRepresentation() &&
                    (eos.isIdealGas() || eos.isFixedChargeLatentEnergy()) &&
                    eos.m_num_materials == 0 &&
                    m_resolved_packet.thermal_history_photon_weights_original_certificate &&
                    m_resolved_packet.thermal_history_photon_weights_direction_method ==
                        warpx::radiation::PhotonWeightDirectionMethod::RankRevealingOriginalRows;
            }
#endif
            amrex::ParallelDescriptor::ReduceBoolAnd(supported_weights);
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(supported_weights,
            "resolved_thermal_reflecting_z requires thermal exchange and rejects "
            "resolved_thermal_history_photon_weights unless CPU built-in "
            "original-certificate fixed history is explicitly selected.");
    }
#if defined(WARPX_DIM_RZ) && !defined(AMREX_USE_GPU)
    if (m_hybrid_model && usesResolvedThermalExchange() &&
        m_hybrid_model->electronThermodynamicsExecutor().isSingularitySpiner())
    {
        // Reject incompatible CPU table combinations before reserved photon
        // columns, field snapshots, private histories or counter namespaces.
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !m_resolved_packet.thermal_history_photon_weights,
            "Table thermal history does not support photon-weight proposals; "
            "built-in ideal/latent electrons are required.");
        bool closed_table_history = false;
#if defined(WARPX_USE_SINGULARITY_EOS)
        if (m_resolved_packet.thermal_reflecting_z_table)
        {
            auto const eos = m_hybrid_model->electronThermodynamicsExecutor();
            int intervals = 0;
            bool adaptive = false;
            pp.query("resolved_thermal_history_subintervals", intervals);
            pp.query("resolved_adaptive_representation", adaptive);
            closed_table_history = m_resolved_packet.thermal_reflecting_z &&
                !geometry.isPeriodic(1) && intervals > 0 && intervals <= 32 && !adaptive &&
                eos.m_num_materials == 1 && !eos.m_use_resolved_material_selection &&
                eos.tableStorage() != nullptr &&
                m_hybrid_model->transportsElectronInternalEnergyAtRawDensity() &&
                m_momentum_species.size() == 1 &&
                m_hybrid_model->electronThermodynamicsMaterialSpeciesName(0) ==
                    m_momentum_species.front() &&
                WarpX::field_boundary_lo[1] == FieldBoundaryType::PEC &&
                WarpX::field_boundary_hi[1] == FieldBoundaryType::PEC &&
                WarpX::particle_boundary_lo[1] == ParticleBoundaryType::Reflecting &&
                WarpX::particle_boundary_hi[1] == ParticleBoundaryType::Reflecting;
            if (closed_table_history)
            {
                closed_table_history = !particles.GetParticleContainerFromName(
                    m_momentum_species.front()).HasEvolvingChargeState();
            }
            amrex::ParallelDescriptor::ReduceBoolAnd(closed_table_history);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(closed_table_history,
                "resolved_thermal_reflecting_z_table requires CPU fixed-mask table material "
                "history and actual PEC/reflecting z before photon schema allocation.");
        }
#endif
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            (geometry.isPeriodic(1) && !m_resolved_packet.thermal_reflecting_z) ||
                closed_table_history,
            "Table thermal history requires periodic z; table+closed is unsupported.");
    }
#endif
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        std::numeric_limits<amrex::Real>::digits >= 53 &&
            std::numeric_limits<amrex::ParticleReal>::digits >= 53 && m_num_groups == 1 &&
            m_enable_diffusion && !m_enable_particle_conversion && !m_coherent_packet_transport &&
            m_enable_momentum_coupling && m_particle_momentum_carry && couplesToHybridElectrons() &&
            max_level == 0 && !EB::enabled() && WarpX::do_moving_window == 0 &&
            !m_use_species_opacity &&
            (!m_use_material_opacity_table || m_resolved_packet.thermal_material_opacity_table) &&
            !m_use_planck_table &&
            !m_use_spectral_planck_table && !m_use_rosseland_table &&
            !m_use_spectral_rosseland_table && !m_has_diffusion_bath &&
            !particles.hasHybridIonization(),
        "resolved_packet_fld requires gray double-precision coherent scattering, native hybrid "
        "fixed-charge material, every ion in momentum_species plus photons, particle carry, "
        "level zero, "
        "legacy conversion, tables, bath, EB or moving window.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_momentum_species.empty() &&
            static_cast<std::size_t>(particles.nSpecies()) == m_momentum_species.size()+1,
        "resolved_packet_fld requires every ion in momentum_species plus photons.");
    auto& photons = particles.GetParticleContainerFromName(m_photon_species);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        photons.doNotPush() && !photons.DoResampling() && !photons.DoQED(),
        "resolved_packet_fld must exclusively own photon pushing and births.");
    for (auto const& name : m_momentum_species) {
        auto const& material = particles.GetParticleContainerFromName(name);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !material.HasEvolvingChargeState(),
            "resolved_packet_fld requires fixed material charge states.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !material.do_not_deposit,
            "resolved_packet_fld hybrid material must deposit its charge and current.");
        m_resolved_packet.configured_material_identity.push_back(
            {static_cast<amrex::Real>(material.getMass()),
             static_cast<amrex::Real>(material.getCharge())});
    }
#if defined(WARPX_DIM_RZ)
    bool const closed_z = WarpX::particle_boundary_lo[1] == ParticleBoundaryType::Reflecting &&
        WarpX::particle_boundary_hi[1] == ParticleBoundaryType::Reflecting &&
        WarpX::field_boundary_lo[1] == FieldBoundaryType::PEC &&
        WarpX::field_boundary_hi[1] == FieldBoundaryType::PEC;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_absorb_nonperiodic_photons && !m_has_explicit_photon_faces &&
        m_rz_angular_transport && geometry.ProbLo(0) == 0 && !geometry.isPeriodic(0) &&
            (geometry.isPeriodic(1) || closed_z) && WarpX::n_rz_azimuthal_modes == 1 &&
            WarpX::field_boundary_hi[0] == FieldBoundaryType::PEC,
        "resolved_packet_fld RZ runtime requires an axis-containing mode-zero disk, "
        "stationary reflecting/PEC radial wall, periodic or reflecting/PEC z and angular carry.");
#elif defined(WARPX_DIM_1D_Z) || defined(WARPX_DIM_XZ) || defined(WARPX_DIM_3D)
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        bool const reflecting = WarpX::particle_boundary_lo[d] == ParticleBoundaryType::Reflecting &&
            WarpX::particle_boundary_hi[d] == ParticleBoundaryType::Reflecting &&
            WarpX::field_boundary_lo[d] == FieldBoundaryType::PEC &&
            WarpX::field_boundary_hi[d] == FieldBoundaryType::PEC;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(geometry.isPeriodic(d) || reflecting,
            "resolved_packet_fld Cartesian faces must be periodic or stationary reflecting/PEC.");
        for (int const face : {m_photon_boundary_lo[d], m_photon_boundary_hi[d]}) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(geometry.isPeriodic(d) ||
                face == static_cast<int>(PhotonFacePolicy::Reflecting) ||
                face == static_cast<int>(PhotonFacePolicy::Absorbing),
                "Resolved Cartesian photon faces must reflect or escape on nonperiodic axes.");
        }
    }
#else
    WARPX_ABORT_WITH_MESSAGE("resolved_packet_fld supports Cartesian 1D/XZ/3D and RZ only.");
#endif
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            geometry.isPeriodic(d) ||
                (m_diffusion_boundary_lo[d] == static_cast<int>(DiffusionBoundary::Reflecting) &&
                 m_diffusion_boundary_hi[d] == static_cast<int>(DiffusionBoundary::Reflecting)),
            "resolved_packet_fld requires reflecting diffusion settings on nonperiodic faces.");
    }
    for (auto const* key :
         {"rosseland_transport_coefficient", "rosseland_transport_coefficient(x,y,z,t,ne,Te)",
          "rosseland_transport_coefficient(x,y,z,t,photon_energy,ne,Te)",
          "rosseland_transport_coefficient_table_file",
          "rosseland_transport_coefficient_spectral_table_file"}) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !pp.contains(key), "resolved_packet_fld uses resolved_transport_coefficient(x,y,z), "
                               "not legacy opacity inputs.");
    }
    utils::parser::Store_parserString(pp, "resolved_diffusion_region(x,y,z)",
                                      m_resolved_packet.mask_expression);
    bool const temporal = pp.contains("resolved_transport_coefficient(x,y,z,t)");
    m_resolved_packet.material_dependent_opacity =
        pp.contains("resolved_transport_coefficient(x,y,z,t,ne,Te)") ||
        m_resolved_packet.thermal_material_opacity_table;
    bool const material = m_resolved_packet.material_dependent_opacity;
    m_resolved_packet.time_dependent_opacity = temporal || material;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        static_cast<int>(pp.contains("resolved_transport_coefficient(x,y,z)")) +
            static_cast<int>(temporal) +
            static_cast<int>(pp.contains("resolved_transport_coefficient(x,y,z,t,ne,Te)")) ==
                (m_resolved_packet.thermal_material_opacity_table ? 0 : 1),
        "Specify exactly one resolved transport coefficient form.");
    if (material && !m_resolved_packet.thermal_material_opacity_table) {
        auto const eos = m_hybrid_model->electronThermodynamicsExecutor();
        bool table_history = false;
#if defined(WARPX_DIM_RZ) && !defined(AMREX_USE_GPU)
        int history_subintervals = 0;
        pp.query("resolved_thermal_history_subintervals", history_subintervals);
        table_history = usesResolvedThermalExchange() && history_subintervals > 0 &&
            history_subintervals <= 32 &&
            (geometry.isPeriodic(1) || m_resolved_packet.thermal_reflecting_z_table) &&
            eos.isSingularitySpiner() && eos.m_num_materials == 1 &&
            !eos.m_use_resolved_material_selection &&
            m_hybrid_model->transportsElectronInternalEnergyAtRawDensity();
        // ConfigureResolvedThermalRuntime below completes the fixed-charge,
        // matching-species, reciprocal, L0/m0/fixed-mask admission. This seam
        // only admits its parser; it does not sample or publish a table state.
#endif
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_hybrid_model->m_solve_electron_energy_equation &&
            (table_history || ((eos.isIdealGas() || eos.isFixedChargeLatentEnergy()) &&
                               eos.m_num_materials == 0)),
            "Resolved material opacity requires evolved builtin electron thermodynamics "
            "without a composition adapter, or qualified CPU periodic-z table thermal history.");
    }
    if (m_resolved_packet.thermal_material_opacity_table)
    {
        // Allocation defers this nonphysical placeholder. All physical source,
        // endpoint and restart coefficients use the validated table dispatcher.
        m_resolved_packet.opacity_expression = "0";
    }
    else
    {
        utils::parser::Store_parserString(pp, material ?
            "resolved_transport_coefficient(x,y,z,t,ne,Te)" : temporal ?
            "resolved_transport_coefficient(x,y,z,t)" : "resolved_transport_coefficient(x,y,z)",
                                          m_resolved_packet.opacity_expression);
    }
    m_resolved_packet.mask_parser =
        utils::parser::makeParser(m_resolved_packet.mask_expression, {"x", "y", "z"});
    amrex::Vector<std::string> opacity_variables{"x", "y", "z"};
    if (m_resolved_packet.time_dependent_opacity) { opacity_variables.push_back("t"); }
    if (material) {
        opacity_variables.push_back("ne");
        opacity_variables.push_back("Te");
    }
    m_resolved_packet.opacity_parser =
        utils::parser::makeParser(m_resolved_packet.opacity_expression, opacity_variables);
    pp.query("resolved_packet_substeps", m_resolved_packet.packet_substeps);
    pp.query("resolved_moving_diffusion", m_resolved_packet.moving_diffusion);
    pp.query("resolved_kinetic_empty_cells", m_resolved_packet.kinetic_empty_cells);
    pp.query("resolved_adaptive_representation",
             m_resolved_packet.adaptive_representation);
    int minimum_adaptive =
        static_cast<int>(m_resolved_packet.adaptive_representation);
    int maximum_adaptive = minimum_adaptive;
    amrex::ParallelDescriptor::ReduceIntMin(minimum_adaptive);
    amrex::ParallelDescriptor::ReduceIntMax(maximum_adaptive);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        minimum_adaptive == maximum_adaptive,
        "Adaptive packet representation mode must agree across ranks.");
    if (!m_resolved_packet.adaptive_representation) {
        std::ostringstream bindings;
        bindings << std::setprecision(std::numeric_limits<double>::max_digits10);
        AppendParserConstants(bindings, m_resolved_packet.mask_parser, {"x", "y", "z"});
        AppendParserConstants(bindings, m_resolved_packet.opacity_parser, opacity_variables);
        m_resolved_packet.parser_identity = bindings.str();
    }
    if (m_resolved_packet.adaptive_representation) {
#if defined(WARPX_DIM_RZ)
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !m_enable_lte_exchange || m_resolved_packet.moving_diffusion,
            "RZ adaptive thermal transport requires the native moving "
            "reciprocal "
            "chronological thermal coordinator.");
#endif
        utils::parser::queryWithParser(
            pp, "resolved_demote_optical_depth",
            m_resolved_packet.adaptive_policy.demote_depth);
        utils::parser::queryWithParser(
            pp, "resolved_promote_optical_depth",
            m_resolved_packet.adaptive_policy.promote_depth);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            std::isfinite(m_resolved_packet.adaptive_policy.demote_depth) &&
                m_resolved_packet.adaptive_policy.demote_depth >= 8 &&
                std::isfinite(
                    m_resolved_packet.adaptive_policy.promote_depth) &&
                m_resolved_packet.adaptive_policy.promote_depth >
                    m_resolved_packet.adaptive_policy.demote_depth,
            "Adaptive representation requires promote_depth > demote_depth >= "
            "8.");
        m_resolved_packet.parser_identity =
            ResolvedAdaptiveParserIdentity(m_resolved_packet.mask_parser,
                                           m_resolved_packet.opacity_parser);
        std::ostringstream controls;
        controls << std::setprecision(
                        std::numeric_limits<amrex::Real>::max_digits10)
                 << m_resolved_packet.adaptive_policy.demote_depth << ' '
                 << m_resolved_packet.adaptive_policy.promote_depth << ' '
                 << m_resolved_packet.parser_identity;
        auto const local = controls.str();
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            local.size() <=
                static_cast<std::size_t>(std::numeric_limits<int>::max()),
            "Adaptive packet identity exceeds the MPI count limit.");
        int size = static_cast<int>(local.size());
        amrex::ParallelDescriptor::Bcast(
            &size, 1, amrex::ParallelDescriptor::IOProcessorNumber());
        std::string reference = local;
        reference.resize(static_cast<std::size_t>(size));
        amrex::ParallelDescriptor::Bcast(
            reference.data(), size,
            amrex::ParallelDescriptor::IOProcessorNumber());
        bool same = reference == local;
        amrex::ParallelDescriptor::ReduceBoolAnd(same);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            same, "Adaptive packet hysteresis and parser identities must agree "
                  "across ranks.");
    }
    pp.query("resolved_packet_subsegment_limit", m_resolved_packet.packet_subsegment_limit);
    pp.query("resolved_max_iterations", m_resolved_packet.max_iterations);
    pp.query("resolved_verbosity", m_resolved_packet.verbosity);
    utils::parser::queryWithParser(pp, "resolved_photon_energy", m_resolved_packet.photon_energy);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_resolved_packet.packet_subsegment_limit > 0 && m_resolved_packet.max_iterations > 0 &&
            m_resolved_packet.verbosity >= 0 && std::isfinite(m_resolved_packet.photon_energy) &&
            m_resolved_packet.photon_energy > 0,
        "Invalid resolved packet runtime budget or photon energy.");
#ifdef WARPX_USE_MATERIAL_OPACITY_HDF5
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_resolved_packet.thermal_material_opacity_table ||
        (m_registered_material_opacity_tables[0].representativeEnergies().size() == 1 &&
         m_registered_material_opacity_tables[0].representativeEnergies()[0] ==
             m_resolved_packet.photon_energy),
        "Registered thermal table opacity requires the actual matching gray photon energy.");
#endif
    for (auto const* name : {ResolvedDrawKeyLoName, ResolvedDrawKeyHiName}) {
        auto const& names = photons.GetRealSoANames();
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            std::find(names.begin(), names.end(), name) == names.end(),
            "Resolved photon draw-key column is reserved by the runtime.");
        photons.AddRealComp(name, /*comm=*/1);
    }
    amrex::ignore_unused(RegisterCollidedDiffusionAdmissionState(photons));
    if (usesResolvedThermalExchange())
    {
        ConfigureResolvedThermalRuntime();
    }
    m_track_energy_balance = true;
}

void
RadiationTransport::AllocateResolvedPacketRuntime (ablastr::fields::MultiFabRegister& fields,
                                                   int lev, amrex::BoxArray const& ba,
                                                   amrex::DistributionMapping const& dm) const
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(lev == 0, "Resolved packet runtime is level zero only.");
    for (auto const* name :
         {ResolvedMaskFieldName, ResolvedOpacityFieldName, ResolvedAdmissionFieldName}) {
        fields.alloc_init(name, lev, ba, dm, 1, amrex::IntVect(1), 0.0_rt, true, true, true, false);
    }
    for (int d = 0; d < 3; ++d) {
        fields.alloc_init(ResolvedMomentumFieldName(d), lev, ba, dm, 1, amrex::IntVect(1), 0.0_rt,
                          true, true, true, false);
    }
    if (usesResolvedMaterialOpacity() || usesResolvedThermalExchange())
    {
        for (auto const* name : {ResolvedElectronDensityFieldName,
                                ResolvedElectronTemperatureFieldName}) {
            fields.alloc_init(name, lev, ba, dm, 1, amrex::IntVect(1), 0.0_rt, true, true, true, false);
        }
    }
    if (usesAdaptivePacketRepresentation()) {
        fields.alloc_init(ResolvedRepresentationCarryFieldName, lev, ba, dm, 4,
                          amrex::IntVect(1), 0.0_rt, true, true, true, false);
    }
    if (usesResolvedThermalExchange())
    {
        fields.alloc_init(ResolvedAbsorptionFieldName, lev, ba, dm, 1, amrex::IntVect(1), 0._rt,
                          true, true, true, false);
    }
    if (!m_is_restart) {
        if (m_resolved_packet.time_dependent_opacity) {
            m_resolved_packet.opacity_time = WarpX::GetInstance().gett_new(lev);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(std::isfinite(m_resolved_packet.opacity_time),
                "Resolved opacity initialization requires a finite time.");
        }
        InitializeOrValidateResolvedFields(
            *fields.get(ResolvedMaskFieldName, lev), *fields.get(ResolvedOpacityFieldName, lev),
            *fields.get(FieldType::radiation_diffusion_energy, lev), WarpX::GetInstance().Geom(lev),
            m_resolved_packet.mask_parser.compile<3>(), m_resolved_packet.opacityExecutor(), false,
            m_resolved_packet.opacity_time, nullptr, nullptr, usesResolvedMaterialOpacity(), false,
            usesAdaptivePacketRepresentation() || usesResolvedThermalExchange());
        if (m_resolved_packet.has_initial_momentum_ratio &&
            !m_resolved_packet.initial_momentum_initialized)
        {
            // Seed only fresh valid-cell LAB inventory. K energy is already
            // required to be exactly zero. Checkpoint restoration owns Q on restart.
            auto const& energy = *fields.get(FieldType::radiation_diffusion_energy, lev);
            for (int d = 0; d < 3; ++d)
            {
                auto& q = *fields.get(ResolvedMomentumFieldName(d), lev);
                amrex::MultiFab::Copy(q, energy, 0, 0, 1, 0);
                q.mult(m_resolved_packet.initial_momentum_ratio[d], 0, 1, 0);
                q.FillBoundary(WarpX::GetInstance().Geom(lev).periodicity());
            }
            m_resolved_packet.initial_momentum_initialized = true;
        }
    }
}

void
RadiationTransport::ValidateResolvedPacketRestartFields () const
{
    auto& simulation = WarpX::GetInstance();
    auto& fields = simulation.m_fields;
    amrex::MultiFab table_scattering, table_absorption;
    if (m_resolved_packet.thermal_material_opacity_table)
    {
        auto const& stored = *fields.get(ResolvedOpacityFieldName, 0);
        table_scattering.define(stored.boxArray(), stored.DistributionMap(), 1, 1);
        table_absorption.define(stored.boxArray(), stored.DistributionMap(), 1, 1);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(FillResolvedThermalOpacityFromSamples(
            table_scattering, table_absorption, *fields.get(ResolvedElectronDensityFieldName, 0),
            *fields.get(ResolvedElectronTemperatureFieldName, 0), simulation.Geom(0),
            m_resolved_packet.opacity_time),
            "Registered thermal table restart has invalid immutable coefficient samples.");
    }
    InitializeOrValidateResolvedFields(
        *fields.get(ResolvedMaskFieldName, 0), *fields.get(ResolvedOpacityFieldName, 0),
        *fields.get(FieldType::radiation_diffusion_energy, 0), simulation.Geom(0),
        m_resolved_packet.mask_parser.compile<3>(), m_resolved_packet.opacityExecutor(), true,
        m_resolved_packet.opacity_time,
        usesResolvedMaterialOpacity() ? fields.get(ResolvedElectronDensityFieldName, 0) : nullptr,
        usesResolvedMaterialOpacity() ? fields.get(ResolvedElectronTemperatureFieldName, 0)
                                      : nullptr,
        false, usesAdaptivePacketRepresentation(),
        usesAdaptivePacketRepresentation() || usesResolvedThermalExchange(),
        m_resolved_packet.thermal_material_opacity_table ? &table_scattering : nullptr);
}

std::string
RadiationTransport::ResolvedMaterialOpacityIdentity () const
{
    auto const eos = m_hybrid_model->electronThermodynamicsExecutor();
    std::ostringstream result;
    if (m_resolved_packet.thermal_material_opacity_table)
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_resolved_packet.configured_material_identity.size() == 1,
            "Registered thermal table sampling identity requires one configured native carrier.");
        // Checkpoint reading precedes HybridPICModel::InitData, which binds
        // the EOS mass/charge conversion. Use the requested native identity
        // captured before particle headers can overwrite it, in both epochs.
        auto const& configured = m_resolved_packet.configured_material_identity.front();
        result << std::setprecision(std::numeric_limits<amrex::Real>::max_digits10)
            << "single_selected_table_cell_gather_v2_configured_native_mass_charge_raw_ne_Cv_weighted_Te "
            << configured[0] / PhysConst::q_e << ' ' << configured[0] / configured[1] << ' '
            << eos.m_atomic_mass[0] << ' ' << eos.m_atomic_number[0] << ' '
            << m_minimum_electron_density << ' ' << eos.minimumTemperature() << ' '
            << eos.maximumTemperature();
        return result.str();
    }
    result << std::setprecision(std::numeric_limits<amrex::Real>::max_digits10)
           << "builtin_cell_gather_v1 "
           << (eos.isIdealGas() ? "ideal_gas" : "fixed_charge_latent_energy") << ' '
           << eos.m_gamma << ' ' << m_minimum_electron_density << ' '
           << m_hybrid_model->electronDensityFloor() << ' '
           << m_hybrid_model->transportsElectronInternalEnergyAtRawDensity() << ' '
           << eos.m_num_latent_transitions;
    for (int n = 0; n < eos.m_num_latent_transitions; ++n) {
        result << ' ' << eos.m_latent_transition_temperature[n] << ' '
               << eos.m_latent_energy_per_electron[n] << ' ' << eos.m_latent_transition_sharpness[n];
    }
    return result.str();
}

void
RadiationTransport::InitializeResolvedMaterialOpacity (
    ablastr::fields::MultiFabRegister& fields) const
{
    if (usesResolvedThermalExchange())
    {
        InitializeResolvedThermalOpacity(fields);
        return;
    }
    if (!usesResolvedMaterialOpacity()) { return; }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_is_restart && !m_resolved_packet.material_opacity_initialized,
        "Resolved material opacity fresh initialization must occur exactly once.");
    auto const& simulation = WarpX::GetInstance();
    auto& ne = *fields.get(ResolvedElectronDensityFieldName, 0);
    auto& te = *fields.get(ResolvedElectronTemperatureFieldName, 0);
    auto const floor = m_hybrid_model->transportsElectronInternalEnergyAtRawDensity() ?
        0.0_rt : m_hybrid_model->electronDensityFloor();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(detail::SnapshotResolvedMaterialOpacity(
        *fields.get(FieldType::rho_fp, 0), *fields.get(FieldType::hybrid_electron_temperature_fp, 0),
        ne, te, simulation.Geom(0), m_hybrid_model->electronThermodynamicsExecutor(),
        m_minimum_electron_density, floor), "Invalid initial resolved material-opacity state.");
    m_resolved_packet.opacity_time = simulation.gett_new(0);
    InitializeOrValidateResolvedFields(
        *fields.get(ResolvedMaskFieldName, 0),
        *fields.get(ResolvedOpacityFieldName, 0),
        *fields.get(FieldType::radiation_diffusion_energy, 0),
        simulation.Geom(0), m_resolved_packet.mask_parser.compile<3>(),
        m_resolved_packet.opacityExecutor(), false,
        m_resolved_packet.opacity_time, &ne, &te, false, false,
        usesAdaptivePacketRepresentation());
    m_resolved_packet.material_opacity_initialized = true;
}

void
RadiationTransport::AdvanceResolvedPacketRuntime (MultiParticleContainer& particles,
                                                  ablastr::fields::MultiFabRegister& fields,
                                                  amrex::Real current_time, amrex::Real dt) const
{
#if defined(WARPX_DIM_RZ)
    if (usesAdaptivePacketRepresentation() && !usesResolvedThermalExchange())
    {
        AdvanceResolvedRZAdaptiveRadiation(particles, fields, current_time, dt);
        return;
    }
#endif
    if (usesResolvedThermalExchange())
    {
        AdvanceResolvedThermalRuntime(particles, fields, current_time, dt);
        return;
    }
    auto const& geometry = WarpX::GetInstance().Geom(0);
    auto& photons = particles.GetParticleContainerFromName(m_photon_species);
    auto& energy = *fields.get(FieldType::radiation_diffusion_energy, 0);
    auto& mask_field = *fields.get(ResolvedMaskFieldName, 0);
    auto& opacity_state = *fields.get(ResolvedOpacityFieldName, 0);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_resolved_packet.interval <
                                         std::numeric_limits<std::uint64_t>::max(),
                                     "Resolved packet interval counter overflow.");
    amrex::iMultiFab mask(energy.boxArray(), energy.DistributionMap(), 1, 1);
    mask.setVal(0);
    CopyResolvedMask(mask_field, mask);
    mask.FillBoundary(geometry.periodicity());
    amrex::MultiFab opacity(energy.boxArray(), energy.DistributionMap(), 1, 1);
    opacity.setVal(0);
    amrex::MultiFab::Copy(opacity, opacity_state, 0, 0, 1, 0);
    std::unique_ptr<amrex::MultiFab> ne, te;
    if (usesResolvedMaterialOpacity()) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_resolved_packet.material_opacity_initialized,
            "Resolved material opacity requires initialized native material state.");
        ne = std::make_unique<amrex::MultiFab>(energy.boxArray(), energy.DistributionMap(), 1, 1);
        te = std::make_unique<amrex::MultiFab>(energy.boxArray(), energy.DistributionMap(), 1, 1);
        ne->setVal(0);
        te->setVal(0);
        auto const floor = m_hybrid_model->transportsElectronInternalEnergyAtRawDensity() ?
            0.0_rt : m_hybrid_model->electronDensityFloor();
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(detail::SnapshotResolvedMaterialOpacity(
            *fields.get(FieldType::rho_fp, 0),
            *fields.get(FieldType::hybrid_electron_temperature_fp, 0), *ne, *te, geometry,
            m_hybrid_model->electronThermodynamicsExecutor(), m_minimum_electron_density, floor),
            "Resolved opacity requires finite admissible native material samples.");
    }
    amrex::Real const opacity_time = m_resolved_packet.time_dependent_opacity ?
        current_time + dt : 0;
    if (m_resolved_packet.time_dependent_opacity) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(std::isfinite(current_time) &&
            std::isfinite(opacity_time) && opacity_time > current_time,
            "Resolved opacity refresh requires a finite advancing endpoint time.");
        // Keep every accepted field untouched if coefficient evaluation or the
        // subsequent coupled interval rejects. The geometric mask is fixed;
        // refreshing opacity does not authorize deleting an existing D state.
        amrex::MultiFab mask_trial(mask_field.boxArray(), mask_field.DistributionMap(), 1, 1);
        amrex::MultiFab::Copy(mask_trial, mask_field, 0, 0, 1, 1);
        InitializeOrValidateResolvedFields(
            mask_trial, opacity, energy, geometry,
            m_resolved_packet.mask_parser.compile<3>(),
            m_resolved_packet.opacityExecutor(), false, opacity_time, ne.get(),
            te.get(), false, usesAdaptivePacketRepresentation(),
            usesAdaptivePacketRepresentation());
    }
    opacity.FillBoundary(geometry.periodicity());
    // The packet rate covers the whole domain. The scalar operator's K cells
    // are inactive and require exactly zero opacity; active D rates agree.
    amrex::MultiFab diffusion_opacity(energy.boxArray(), energy.DistributionMap(), 1, 1);
    diffusion_opacity.setVal(0);
    amrex::MultiFab::Copy(diffusion_opacity, opacity, 0, 0, 1, 0);
    MaskResolvedOpacity(opacity, mask, diffusion_opacity);
    diffusion_opacity.FillBoundary(geometry.periodicity());
    amrex::MultiFab q(energy.boxArray(), energy.DistributionMap(), 3, 1);
    q.setVal(0);
    for (int d = 0; d < 3; ++d) {
        amrex::MultiFab::Copy(q, *fields.get(ResolvedMomentumFieldName(d), 0), 0, d, 1, 0);
    }
    q.FillBoundary(geometry.periodicity());
    PacketDiffusionNativeIntervalOptions options;
    options.resolved_collided_source = true;
    options.quasi_static_angular_traction = true;
    options.retain_constitutive_momentum = true;
    options.moving_diffusion = m_resolved_packet.moving_diffusion;
    options.kinetic_empty_cells = m_resolved_packet.kinetic_empty_cells;
    options.transactional_packet_substeps = m_resolved_packet.packet_substeps;
    options.packet_subsegment_limit = m_resolved_packet.packet_subsegment_limit;
    options.photon_energy = m_resolved_packet.photon_energy;
    options.kinetic_scattering_opacity = &opacity;
    options.material.max_iterations = m_resolved_packet.max_iterations;
    options.diffusion.verbosity = m_resolved_packet.verbosity;
    options.collisions.seed = m_particle_conversion_seed;
    options.collisions.interval = m_resolved_packet.interval;
    options.logical_birth_keys = LogicalPhotonBirthKeyContext{
        m_particle_conversion_seed, m_resolved_packet.interval};
#if defined(WARPX_DIM_RZ)
    options.material.assignment = ParticleImpulseAssignment::RZAngularConservative;
#else
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        options.photon_boundaries.escape_lo[d] =
            m_photon_boundary_lo[d] == static_cast<int>(PhotonFacePolicy::Absorbing);
        options.photon_boundaries.escape_hi[d] =
            m_photon_boundary_hi[d] == static_cast<int>(PhotonFacePolicy::Absorbing);
    }
#endif
    std::unique_ptr<PhotonPacketTransaction> adaptive_photons;
    std::unique_ptr<amrex::MultiFab> adaptive_energy;
    amrex::MultiFab adaptive_carry;
    amrex::iMultiFab desired_mask;
    amrex::GpuArray<amrex::Real, 4> initial_packet_inventory{};
    amrex::GpuArray<amrex::Real, 4> initial_field_inventory{};
    amrex::Long demoted_cells = 0, promoted_cells = 0, deferred_cells = 0;
    CollidedDiffusionAdmissionComponents census;
    if (usesAdaptivePacketRepresentation()) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            detail::PhotonInventory(photons, initial_packet_inventory),
            "Adaptive representation requires a valid initial native photon "
            "inventory.");
        auto const& old_carry =
            *fields.get(ResolvedRepresentationCarryFieldName, 0);
        initial_field_inventory[0] = energy.sum(0) + old_carry.sum(0);
        for (int d = 0; d < 3; ++d) {
            initial_field_inventory[d + 1] = q.sum(d) + old_carry.sum(d + 1);
        }
        for (int d = 0; d < 3; ++d) {
            census.initial_u[d] = photons.GetRealCompIndex(
                std::string("radiation_diffusion_census_u") + "xyz"[d] + "0");
        }
        adaptive_photons = std::make_unique<PhotonPacketTransaction>(photons);
        auto const snapshot = SnapshotCollidedDiffusionAdmission(
            adaptive_photons->Particles(), census);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            snapshot.valid,
            "Adaptive representation could not snapshot the physical census.");
        desired_mask.define(mask.boxArray(), mask.DistributionMap(), 1, 1);
        desired_mask.setVal(0);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            TryProposeAdaptiveDiffusionMask(mask, opacity, geometry,
                                            desired_mask,
                                            m_resolved_packet.adaptive_policy),
            "Adaptive representation rejected its opacity/mask proposal.");
        ConstrainAdaptiveMask(mask, desired_mask, opacity, geometry, dt,
                              options.photon_boundaries, false);
        // Demotion must precede physical streaming. Promotion instead needs
        // the fresh collision census produced by this interval's actual replay.
        amrex::iMultiFab early_mask(mask.boxArray(), mask.DistributionMap(), 1,
                                    1);
        early_mask.setVal(0);
        amrex::iMultiFab::Copy(early_mask, desired_mask, 0, 0, 1, 0);
        ConstrainAdaptiveMask(mask, early_mask, opacity, geometry, dt,
                              options.photon_boundaries, true);
        auto converted = TryConvertAdaptivePacketDiffusionRepresentation(
            *adaptive_photons, energy, q, mask, early_mask, old_carry, geometry,
            census, *options.logical_birth_keys, options.photon_energy,
            options.admission);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            converted.valid,
            "Adaptive demotion rejected before transport: " + converted.reason);
        adaptive_energy =
            std::make_unique<amrex::MultiFab>(std::move(converted.energy));
        q = std::move(converted.momentum);
        mask = std::move(converted.mask);
        adaptive_carry = std::move(converted.representation_carry);
        demoted_cells = converted.demoted_cells;
        mask.FillBoundary(geometry.periodicity());
        q.FillBoundary(geometry.periodicity());
        MaskResolvedOpacity(opacity, mask, diffusion_opacity);
        diffusion_opacity.FillBoundary(geometry.periodicity());
        options.private_photon_snapshot = adaptive_photons.get();
        options.defer_live_commit = true;
    }
    auto result = TryAdvancePacketDiffusionNativeInterval(
        particles, photons, m_momentum_species, "diffusion_0",
        adaptive_energy ? *adaptive_energy : energy, diffusion_opacity, mask,
        geometry, dt, nullptr, options, &q);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        result.valid,
        "Resolved packet/FLD interval rejected before radiation-stage commit: " + result.reason);
    if (usesAdaptivePacketRepresentation()) {
        AMREX_ALWAYS_ASSERT(result.private_photons && result.private_material);
        PhotonPacketTransaction promotion(photons, *result.private_photons);
        auto converted = TryConvertAdaptivePacketDiffusionRepresentation(
            promotion, result.private_radiation_energy,
            result.private_diffusion_momentum, mask, desired_mask,
            adaptive_carry, geometry, census, *options.logical_birth_keys,
            options.photon_energy, options.admission);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            converted.valid,
            "Adaptive promotion rejected before radiation-stage commit: " +
                converted.reason);
        MaskResolvedOpacity(opacity, converted.mask, diffusion_opacity);
        diffusion_opacity.FillBoundary(geometry.periodicity());
        bool admissible = true;
        if (converted.promoted_cells > 0) {
            amrex::MultiFab zero_impulse(energy.boxArray(),
                                         energy.DistributionMap(), 3, 0);
            zero_impulse.setVal(0);
            ParticleImpulseTransaction frame;
            admissible =
                frame.Stage(*result.private_material, "diffusion_0",
                            zero_impulse, true, options.material.assignment);
            if (admissible) {
                admissible = AdmissibleResolvedDiffusionCensus(
                    converted.energy, converted.momentum, diffusion_opacity,
                    converted.mask, frame.WorkVelocity(), geometry, dt,
                    options.diffusion, options.moving_diffusion);
            }
        }
        if (admissible) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                result.private_photons->AdoptCandidate(promotion),
                "Adaptive promotion lost its original private photon "
                "provenance.");
            result.private_radiation_energy = std::move(converted.energy);
            result.private_diffusion_momentum = std::move(converted.momentum);
            mask = std::move(converted.mask);
            adaptive_carry = std::move(converted.representation_carry);
            promoted_cells = converted.promoted_cells;
            deferred_cells = converted.deferred_cells;
        } else {
            // Keep the actual physical photons. A failed scalar predictor is
            // not authority to alter the source, relax its bounds, or lose E/q.
            deferred_cells =
                converted.promoted_cells + converted.deferred_cells;
        }
        amrex::GpuArray<amrex::Real, 4> final_packets{};
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            detail::PhotonInventory(result.private_photons->Particles(),
                                    final_packets),
            "Invalid private endpoint photons in adaptive combined ledger.");
        amrex::GpuArray<amrex::Real, 4> final_fields{};
        final_fields[0] =
            result.private_radiation_energy.sum(0) + adaptive_carry.sum(0);
        for (int d = 0; d < 3; ++d) {
            final_fields[d + 1] = result.private_diffusion_momentum.sum(d) +
                                  adaptive_carry.sum(d + 1);
        }
        auto const initial_energy =
            initial_packet_inventory[0] + initial_field_inventory[0];
        auto const final_energy = final_packets[0] + final_fields[0];
        auto const scale = std::abs(initial_energy) + std::abs(final_energy) +
                           std::abs(result.actual_material_work) +
                           std::abs(result.material_energy_carry_change) +
                           result.escaped_packet_inventory[0];
        auto const energy_residual = final_energy - initial_energy +
                                     result.actual_material_work +
                                     result.material_energy_carry_change +
                                     result.escaped_packet_inventory[0];
        bool balanced = std::isfinite(scale) &&
                        std::isfinite(energy_residual) &&
                        std::abs(energy_residual) <=
                            options.energy_relative_tolerance * scale;
        for (int d = 0; d < 3; ++d) {
            auto const residual =
                (final_packets[d + 1] - initial_packet_inventory[d + 1] +
                 final_fields[d + 1] - initial_field_inventory[d + 1] +
                 result.escaped_packet_inventory[d + 1]) /
                    PhysConst::c +
                result.actual_material_impulse_field.sum(d) +
                result.material_momentum_carry_change_field.sum(d) +
                result.packet_wall_momentum[d] +
                result.radiation_wall_momentum[d];
            balanced = balanced && std::isfinite(residual) &&
                       std::abs(residual) <= options.energy_relative_tolerance *
                                                 scale / PhysConst::c;
        }
        amrex::ParallelDescriptor::ReduceBoolAnd(balanced);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            balanced, "Adaptive packet/FLD/material/carry combined inventory "
                      "rejected before live commit.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            result.private_photons->ValidateLive(),
            "Live photons changed before adaptive joint commit.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            result.private_material->Commit(),
            "Live material changed before adaptive joint commit.");
        result.private_photons->Commit();
        amrex::MultiFab::Copy(energy, result.private_radiation_energy, 0, 0, 1,
                              0);
        energy.FillBoundary(geometry.periodicity());
        amrex::MultiFab::Copy(q, result.private_diffusion_momentum, 0, 0, 3, 0);
        q.FillBoundary(geometry.periodicity());
        PublishResolvedMask(mask, mask_field);
        mask_field.FillBoundary(geometry.periodicity());
        auto& accepted_carry =
            *fields.get(ResolvedRepresentationCarryFieldName, 0);
        amrex::MultiFab::Copy(accepted_carry, adaptive_carry, 0, 0, 4, 0);
        accepted_carry.FillBoundary(geometry.periodicity());
        result.raw_energy_residual = energy_residual;
    }
    // All remaining writes publish the already accepted interval. The outer PIC
    // macrostep is not a transactional object and is not retried by this wrapper.
    if (m_resolved_packet.time_dependent_opacity) {
        amrex::MultiFab::Copy(opacity_state, opacity, 0, 0, 1, opacity_state.nGrowVect());
        m_resolved_packet.opacity_time = opacity_time;
        if (usesResolvedMaterialOpacity()) {
            amrex::MultiFab::Copy(*fields.get(ResolvedElectronDensityFieldName, 0), *ne,
                0, 0, 1, ne->nGrowVect());
            amrex::MultiFab::Copy(*fields.get(ResolvedElectronTemperatureFieldName, 0), *te,
                0, 0, 1, te->nGrowVect());
        }
    }
    for (int d = 0; d < 3; ++d) {
        amrex::MultiFab::Copy(*fields.get(ResolvedMomentumFieldName(d), 0), q, d, 0, 1, 1);
    }
    if (result.admitted_four_inventory.ok()) {
        amrex::MultiFab::Add(*fields.get(ResolvedAdmissionFieldName, 0),
                             result.admitted_four_inventory, 0, 0, 1, 0);
    }
    fields.get(FieldType::radiation_material_energy, 0)->setVal(0);
    auto& impulse = *fields.get(FieldType::radiation_material_momentum, 0);
    auto& work = *fields.get(FieldType::radiation_material_kinetic_energy, 0);
    impulse.setVal(0);
    work.setVal(0);
    amrex::MultiFab::Copy(impulse, result.actual_material_impulse_field, 0, 0, 3, 0);
    amrex::MultiFab::Copy(work, result.actual_material_work_field, 0, 0, 1, 0);
    m_last_numerical_energy_residual = result.raw_energy_residual;
    m_cumulative_numerical_energy_residual += result.raw_energy_residual;
    m_last_material_carry_energy_change = result.material_energy_carry_change;
    m_last_diffusion_boundary_momentum_loss = result.radiation_wall_momentum;
    m_last_streaming_boundary_momentum_loss = result.packet_wall_momentum;
    for (int d = 0; d < 3; ++d) {
        m_last_streaming_boundary_momentum_loss[d] += result.escaped_packet_inventory[d+1]/
            PhysConst::c;
    }
    m_last_boundary_energy_loss = result.escaped_packet_inventory[0];
    m_cumulative_boundary_energy_loss += m_last_boundary_energy_loss;
    m_last_diffusion_boundary_energy_loss = 0;
    m_last_streaming_boundary_energy_loss = result.escaped_packet_inventory[0];
    m_last_boundary_energy_injection = 0;
    m_last_particle_wall_momentum = {0, 0, 0};
    m_last_coupled_solve = {result.work_iterations, 1, 0, 0, 0, result.raw_energy_residual};
    ++m_resolved_packet.interval;
    if (m_resolved_packet.verbosity > 0) {
        amrex::Print() << "Resolved packet/FLD accepted interval=" << m_resolved_packet.interval
                       << " collisions=" << result.collision_events
                       << " work=" << result.actual_material_work
                       << " raw_energy_residual=" << result.raw_energy_residual
                       << " kinetic_front_cells=" << result.kinetic_front_cells
                       << " deferred_admissions=" << result.deferred_front_admissions
                       << " linear_tolerance=" << result.linear_tolerance << '\n';
        if (usesAdaptivePacketRepresentation()) {
            amrex::Print() << "Adaptive representation demoted="
                           << demoted_cells << " promoted=" << promoted_cells
                           << " deferred=" << deferred_cells << '\n';
        }
    }
}
