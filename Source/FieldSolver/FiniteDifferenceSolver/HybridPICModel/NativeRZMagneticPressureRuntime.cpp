/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "HybridPICModel.H"

#include "EmbeddedBoundary/Enabled.H"
#include "Fields.H"
#include "NativeRZMagneticPressure.H"
#include "Particles/MultiParticleContainer.H"
#include "Radiation/RadiationTransport.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParmParse.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace amrex::literals;
using warpx::fields::FieldType;

namespace
{
struct SavedField
{
    amrex::MultiFab& target;
    amrex::MultiFab values;
    explicit SavedField (amrex::MultiFab& field)
        : target(field), values(field.boxArray(), field.DistributionMap(), field.nComp(),
                                field.nGrowVect())
    {
        amrex::MultiFab::Copy(values, target, 0, 0, field.nComp(), field.nGrowVect());
    }
    void restore () const
    {
        amrex::MultiFab::Copy(target, values, 0, 0, target.nComp(), target.nGrowVect());
    }
};
}

void
HybridPICModel::ValidateNativePressureChannel () const
{
    auto& sim = WarpX::GetInstance();
    auto const& geom = sim.Geom(0);
    auto const eos = electronThermodynamicsExecutor();
    bool const zero_eta = m_resistivity_parser->symbols().empty() && m_eta(1, 1, 1) == 0;
    bool const zero_eta_h = m_hyper_resistivity_parser->symbols().empty() && m_eta_h(1, 1) == 0;
    std::vector<std::string> collisions;
    amrex::ParmParse("collisions").queryarr("collision_names", collisions);
    bool const supported =
        m_pressure_only && sim.maxLevel() == 0 && sim.finestLevel() == 0 &&
        WarpX::n_rz_azimuthal_modes == 1 &&
        geom.ProbLo(0) == 0 && !geom.isPeriodic(0) && geom.isPeriodic(1) &&
        WarpX::field_boundary_lo[0] == FieldBoundaryType::None &&
        WarpX::field_boundary_hi[0] == FieldBoundaryType::PEC &&
        !EB::enabled() && !sim.DoPML() && !sim.getdo_moving_window() &&
        WarpX::gamma_boost == 1 && !WarpX::use_filter &&
        !WarpX::use_fdtd_nci_corr && !WarpX::do_single_precision_comms &&
        std::numeric_limits<amrex::Real>::digits >= 53 &&
        std::numeric_limits<amrex::ParticleReal>::digits >= 53 &&
        eos.isIdealGas() && std::isfinite(eos.m_gamma) && eos.m_gamma > 1 &&
        eos.m_num_latent_transitions == 0 &&
        !eos.m_use_resolved_material_selection && m_electron_thermodynamics.numMaterials() == 0 &&
        !m_solve_electron_energy_equation && !m_fv_transport_internal_energy &&
        !m_conservative_pressure_work && !m_electron_heat_conduction && !m_include_joule_heating &&
        !m_include_temperature_relaxation && !m_has_per_species_eta && !m_has_resistive_drag &&
        !m_add_external_fields && !m_has_external_current && zero_eta && zero_eta_h &&
        collisions.empty() && !sim.GetRadiationTransport().isEnabled();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(supported,
        "pressure_only requires double-precision ideal electrons on a single-level mode-0 "
        "axis/outer-PEC/periodic-z RZ mesh, frozen ions, and no legacy electron transport, "
        "sources, resistivity, external fields/current, radiation, collisions or filtering.");
    bool depositing = false;
    for (auto const& name : sim.GetPartContainer().GetSpeciesNames()) {
        auto const& species = sim.GetPartContainer().GetParticleContainerFromName(name);
        int do_not_gather = 0;
        amrex::ParmParse(name).query("do_not_gather", do_not_gather);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            species.doNotPush() && do_not_gather && !species.HasEvolvingChargeState() &&
            !species.doContinuousInjection() && !species.DoResampling(),
            "pressure_only requires every species to be frozen (do_not_push=1 and "
            "do_not_gather=1), without ionization, injection or resampling.");
        if (!species.do_not_deposit && species.getCharge() != 0) {
            depositing = true;
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                std::isfinite(species.getCharge()) && species.getCharge() > 0 &&
                std::isfinite(species.getMass()) && species.getMass() > 0,
                "pressure_only charge carriers must be depositing massive positive ions.");
        }
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(depositing,
        "pressure_only requires at least one depositing positive ion species.");
    auto const B = sim.m_fields.get_alldirs(FieldType::Bfield_fp, 0);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(B[0]->norm0(0, 0) == 0 && B[2]->norm0(0, 0) == 0,
        "pressure_only currently supports toroidal magnetic field only.");
}

void
HybridPICModel::InitializeNativePressureState ()
{
    auto& sim = WarpX::GetInstance();
    auto& U = *sim.m_fields.get("hybrid_native_pressure_energy_fp", 0);
    auto& charge = *sim.m_fields.get("hybrid_native_pressure_charge_fp", 0);
    auto& volume = *sim.m_fields.get("hybrid_native_pressure_volume_fp", 0);
    auto const& T = *sim.m_fields.get(FieldType::hybrid_electron_temperature_fp, 0);
    auto const& rho = *sim.m_fields.get(FieldType::rho_fp, 0);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_native_pressure_state_initialized &&
        charge.ixType().nodeCentered() && rho.is_finite(0, 1, 0) && rho.min(0) > 0,
        "pressure_only requires finite strictly positive native deposited charge, without floors.");
    amrex::MultiFab::Copy(charge, rho, 0, 0, 1, charge.nGrowVect());
    auto const eos = electronThermodynamicsExecutor();
    auto const& geom = sim.Geom(0);
    auto const dr = geom.CellSize(0), dz = geom.CellSize(1);
    int const axis = geom.Domain().smallEnd(0), outer = geom.Domain().bigEnd(0) + 1;
    for (amrex::MFIter it(U); it.isValid(); ++it) {
        auto const u = U.array(it), v = volume.array(it);
        auto const q = charge.const_array(it), t = T.const_array(it);
        amrex::ParallelFor(it.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            u(i, j, k) = eos.stateFromChargeDensityTemperature(q(i, j, k), t(i, j, k))
                             .internal_energy_density;
            v(i, j, k) = i == axis ? MathConst::pi * dr * dr * dz / 3
                                  : 2 * MathConst::pi * (i - axis) * dr * dr * dz *
                                        (i == outer ? .5_rt : 1._rt);
        });
    }
    volume.OverrideSync(geom.periodicity());
    volume.FillBoundary(geom.periodicity());
    m_native_pressure_state_initialized = true;
}

void
HybridPICModel::RefreshNativePressureState ()
{
    auto& sim = WarpX::GetInstance();
    auto& fields = sim.m_fields;
    auto& U = *fields.get("hybrid_native_pressure_energy_fp", 0);
    auto& T = *fields.get(FieldType::hybrid_electron_temperature_fp, 0);
    auto const& charge = *fields.get("hybrid_native_pressure_charge_fp", 0);
    auto const& volume = *fields.get("hybrid_native_pressure_volume_fp", 0);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_native_pressure_state_initialized &&
        charge.is_finite(0, 1, 0) && charge.min(0) > 0 &&
        volume.is_finite(0, 1, 0) && volume.min(0) > 0,
        "Invalid restored pressure_only charge or caloric volume.");
    warpx::hybrid::NativeRZMagneticPressureDiagnostics diagnostics;
    warpx::hybrid::NativeRZMagneticPressure bundle(
        sim, U, T, charge, volume, electronThermodynamicsExecutor(), diagnostics);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(bundle.refresh(*this),
        "Invalid pressure_only electron internal energy/temperature state.");
    auto& pressure = *fields.get(FieldType::hybrid_electron_pressure_fp, 0);
    amrex::MultiFab::Copy(pressure, bundle.pressure(), 0, 0, 1, amrex::IntVect(1));
    amrex::MultiFab::Copy(*fields.get(FieldType::rho_fp, 0), charge, 0, 0, 1, charge.nGrowVect());
    amrex::MultiFab::Copy(*fields.get(FieldType::hybrid_rho_fp_temp, 0), charge,
                         0, 0, 1, charge.nGrowVect());
    // Frozen particles can carry momentum, but it does not enter this channel.
    for (auto const type : {FieldType::current_fp, FieldType::hybrid_current_fp_temp}) {
        for (auto* field : fields.get_alldirs(type, 0)) {
            field->setVal(0);
        }
    }
}

bool
HybridPICModel::tryNativePressureFieldEvolve (
    amrex::Real const dt, int const step,
    warpx::hybrid::NativeRZMagneticPressureDiagnostics& diagnostics)
{
    diagnostics = {};
    if (!m_pressure_only || !m_native_pressure_state_initialized || !std::isfinite(dt) || dt <= 0) {
        return false;
    }
    auto& sim = WarpX::GetInstance();
    auto& fields = sim.m_fields;
    auto& U = *fields.get("hybrid_native_pressure_energy_fp", 0);
    auto& T = *fields.get(FieldType::hybrid_electron_temperature_fp, 0);
    auto const& charge = *fields.get("hybrid_native_pressure_charge_fp", 0);
    auto const& volume = *fields.get("hybrid_native_pressure_volume_fp", 0);
    std::vector<std::unique_ptr<SavedField>> saved;
    for (auto const type : {FieldType::Bfield_fp, FieldType::Efield_fp,
                           FieldType::hybrid_current_fp_plasma}) {
        for (auto* field : fields.get_alldirs(type, 0)) {
            saved.push_back(std::make_unique<SavedField>(*field));
        }
    }
    saved.push_back(std::make_unique<SavedField>(U));
    saved.push_back(std::make_unique<SavedField>(T));
    saved.push_back(std::make_unique<SavedField>(
        *fields.get(FieldType::hybrid_electron_pressure_fp, 0)));
    int const substeps = m_substeps;
    auto restore = [&] {
        for (auto const& field : saved) { field->restore(); }
        m_substeps = substeps;
    };
    try {
        for (int half = 0; half < 2; ++half) {
            warpx::hybrid::NativeRZMagneticPressureDiagnostics current;
            bool const accepted = tryBfieldEvolveNativePressure(U, T, charge, volume, dt / 2,
                                                               step, current);
            diagnostics.accepted += current.accepted;
            diagnostics.rejected += current.rejected;
            diagnostics.rk4_restarts += current.rk4_restarts;
            diagnostics.max_ampere_current = std::max(diagnostics.max_ampere_current,
                                                      current.max_ampere_current);
            if (!accepted) {
                restore();
                return false;
            }
        }
        RefreshNativePressureState();
    } catch (...) {
        restore();
        throw;
    }
    return true;
}
