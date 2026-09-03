/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_TRANSPORT_H_
#define WARPX_RADIATION_TRANSPORT_H_

#include "EnergyGroups.H"
#ifdef WARPX_USE_MATERIAL_OPACITY_HDF5
#   include "MaterialOpacityTable.H"
#   include "Utils/MaterialRegistry.H"
#endif
#include "OpacityTable.H"
#include "RadiationTransport_fwd.H"
#include "Utils/MaterialRegistry_fwd.H"

#include <AMReX_Array.H>
#include <AMReX_BaseFwd.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_Parser.H>
#include <AMReX_REAL.H>

#include <array>
#include <string>
#include <vector>

namespace ablastr::fields
{
    struct MultiFabRegister;
}

class MultiParticleContainer;
class HybridPICModel;

/** Hybrid streaming and multigroup transport of an existing WarpX photon species.
 *
 * This first transport layer uses continuous absorption: photon macroparticle
 * weights are attenuated along c*dt and the removed energy is accumulated in
 * a cell-centered material-energy ledger. Material models consume that ledger
 * through solver-specific adapters, keeping transport independent of the
 * hybrid-electron or kinetic-electron representation.
 */
class RadiationTransport
{
public:
    enum class MaterialCoupling
    {
        None,
        HybridElectrons,
        KineticElectrons
    };

    enum class LteExchangeSolver
    {
        FrozenTemperature,
        ImplicitTemperature
    };

    /** Face condition for flux-limited radiation diffusion. */
    enum class DiffusionBoundary
    {
        Reflecting,
        Vacuum,
        Marshak
    };

    RadiationTransport (MultiParticleContainer& particles,
                        HybridPICModel const* hybrid_model,
                        warpx::materials::MaterialRegistry const*
                            material_registry = nullptr);

    /** Allocate per-level transport fields. */
    void AllocateLevelMFs (
        ablastr::fields::MultiFabRegister& fields,
        int lev,
        amrex::BoxArray const& ba,
        amrex::DistributionMapping const& dm) const;

    /** Attenuate photons and accumulate absorbed material energy [J/cell]. */
    void Advance (
        MultiParticleContainer& particles,
        ablastr::fields::MultiFabRegister& fields,
        amrex::Real current_time,
        amrex::Real dt) const;

    [[nodiscard]] bool isEnabled () const noexcept { return m_enabled; }

    [[nodiscard]] bool couplesToHybridElectrons () const noexcept
    {
        return m_material_coupling == MaterialCoupling::HybridElectrons;
    }

    [[nodiscard]] bool couplesToKineticElectrons () const noexcept
    {
        return m_material_coupling == MaterialCoupling::KineticElectrons;
    }

    [[nodiscard]] int numEnergyGroups () const noexcept { return m_num_groups; }

    /** Whether implicit LTE supplies a nonlinear cell-to-node EOS remap. */
    [[nodiscard]] bool usesNonlinearHybridLteRemap () const noexcept
    {
        return m_use_nonlinear_hybrid_lte_remap;
    }

    /** Preserve live cumulative boundary accounting across checkpoint/restart. */
    void WriteCheckpointData (std::string const& dir) const;
    void ReadCheckpointData (std::string const& dir);

    /** Electron-density threshold used to identify material during coupling. */
    [[nodiscard]] amrex::Real minimumElectronDensity () const noexcept
    {
        return m_minimum_electron_density;
    }

    /** Radiation energy lost through particle or diffusion boundaries last step. */
    [[nodiscard]] amrex::Real lastBoundaryEnergyLoss () const noexcept
    {
        return m_last_boundary_energy_loss;
    }

    /** Outward diffusion face-flux energy removed during the last step. */
    [[nodiscard]] amrex::Real lastDiffusionBoundaryEnergyLoss () const noexcept
    {
        return m_last_diffusion_boundary_energy_loss;
    }

    /** Streaming-packet energy carried through absorbing boundaries last step. */
    [[nodiscard]] amrex::Real lastStreamingBoundaryEnergyLoss () const noexcept
    {
        return m_last_streaming_boundary_energy_loss;
    }

    /** Radiation energy lost through boundaries since initialization. */
    [[nodiscard]] amrex::Real cumulativeBoundaryEnergyLoss () const noexcept
    {
        return m_cumulative_boundary_energy_loss;
    }

    /** Signed numerical-energy residual introduced by accepted updates last step. */
    [[nodiscard]] amrex::Real lastNumericalEnergyResidual () const noexcept
    {
        return m_last_numerical_energy_residual;
    }

    /** Signed numerical-energy residual since initialization. */
    [[nodiscard]] amrex::Real cumulativeNumericalEnergyResidual () const noexcept
    {
        return m_cumulative_numerical_energy_residual;
    }

    /** Record an already-MPI-global signed joule residual from a material-EOS
     * realization adapter; add it to both current and cumulative ledgers after
     * Advance. */
    void RecordMaterialEnergyRealizationResidual (amrex::Real residual) const;

    /** Signed diffusion-boundary momentum carried out in one coordinate component. */
    [[nodiscard]] amrex::Real lastDiffusionBoundaryMomentumLoss (
        int component) const noexcept
    {
        return m_last_diffusion_boundary_momentum_loss[component];
    }

    /** Signed streaming-boundary momentum in one coordinate component. */
    [[nodiscard]] amrex::Real lastStreamingBoundaryMomentumLoss (
        int component) const noexcept
    {
        return m_last_streaming_boundary_momentum_loss[component];
    }

private:
    static constexpr int max_opacity_species = 8;
#ifdef WARPX_USE_MATERIAL_OPACITY_HDF5
    static constexpr int max_opacity_materials =
        warpx::materials::MaterialRegistry::max_materials;
#endif

    bool m_enabled = false;
    bool m_enable_lte_exchange = false;
    bool m_enable_diffusion = false;
    bool m_enable_particle_conversion = false;
    bool m_enable_momentum_coupling = false;
    bool m_require_cell_interface_exact_streaming = false;
    bool m_use_nonlinear_hybrid_lte_remap = false;
    bool m_track_energy_balance = false;
    int m_num_groups = 1;
    amrex::Real m_path_cell_fraction = 0.5;
    amrex::Real m_diffusion_cfl = 0.1;
    amrex::Real m_minimum_diffusion_optical_depth = 1.0;
    amrex::Real m_emission_photon_energy = 0.0;
    amrex::Real m_particle_conversion_energy_threshold = 0.0;
    int m_max_transport_substeps = 10000;
    int m_max_diffusion_substeps = 10000;
    int m_lte_exchange_max_iterations = 64;
    amrex::Real m_lte_exchange_tolerance = 1.0e-10;
    std::string m_photon_species;
    std::string m_electron_species;
    std::vector<std::string> m_momentum_species;
    std::vector<std::string> m_opacity_species;
    bool m_use_species_opacity = false;
    bool m_use_material_opacity_table = false;
#ifdef WARPX_USE_MATERIAL_OPACITY_HDF5
    bool m_use_registered_material_opacity_tables = false;
#endif
    bool m_use_absorption_table = false;
    bool m_use_planck_table = false;
    bool m_use_spectral_planck_table = false;
    bool m_use_rosseland_table = false;
    bool m_use_spectral_rosseland_table = false;
    warpx::radiation::OpacityTable3D m_absorption_table;
    warpx::radiation::OpacityTable2D m_planck_table;
    warpx::radiation::OpacityTable3D m_spectral_planck_table;
    warpx::radiation::OpacityTable2D m_rosseland_table;
    warpx::radiation::OpacityTable3D m_spectral_rosseland_table;
    amrex::Parser m_absorption_coefficient_parser;
    amrex::ParserExecutor<7> m_absorption_coefficient;
    amrex::Parser m_planck_absorption_coefficient_parser;
    amrex::ParserExecutor<7> m_planck_absorption_coefficient;
    amrex::Parser m_rosseland_transport_coefficient_parser;
    amrex::ParserExecutor<7> m_rosseland_transport_coefficient;
    std::vector<amrex::Parser> m_species_absorption_coefficient_parsers;
    std::vector<amrex::Parser> m_species_planck_absorption_coefficient_parsers;
    std::vector<amrex::Parser> m_species_rosseland_transport_coefficient_parsers;
    std::array<warpx::radiation::OpacityTable3D, max_opacity_species>
        m_species_absorption_tables;
    std::array<warpx::radiation::OpacityTable2D, max_opacity_species>
        m_species_planck_tables;
    std::array<warpx::radiation::OpacityTable3D, max_opacity_species>
        m_species_spectral_planck_tables;
    std::array<warpx::radiation::OpacityTable2D, max_opacity_species>
        m_species_rosseland_tables;
    std::array<warpx::radiation::OpacityTable3D, max_opacity_species>
        m_species_spectral_rosseland_tables;
    amrex::GpuArray<int, max_opacity_species>
        m_species_absorption_backends{};
    amrex::GpuArray<int, max_opacity_species>
        m_species_planck_backends{};
    amrex::GpuArray<int, max_opacity_species>
        m_species_rosseland_backends{};
    amrex::GpuArray<amrex::ParserExecutor<8>, max_opacity_species>
        m_species_absorption_coefficients;
    amrex::GpuArray<amrex::ParserExecutor<8>, max_opacity_species>
        m_species_planck_absorption_coefficients;
    amrex::GpuArray<amrex::ParserExecutor<8>, max_opacity_species>
        m_species_rosseland_transport_coefficients;
    amrex::GpuArray<amrex::Real, max_opacity_species>
        m_opacity_species_masses{};
    amrex::GpuArray<
        warpx::radiation::OpacityTable3DExecutor, max_opacity_species>
        m_species_absorption_table_executors;
    amrex::GpuArray<
        warpx::radiation::OpacityTable2DExecutor, max_opacity_species>
        m_species_planck_table_executors;
    amrex::GpuArray<
        warpx::radiation::OpacityTable3DExecutor, max_opacity_species>
        m_species_spectral_planck_table_executors;
    amrex::GpuArray<
        warpx::radiation::OpacityTable2DExecutor, max_opacity_species>
        m_species_rosseland_table_executors;
    amrex::GpuArray<
        warpx::radiation::OpacityTable3DExecutor, max_opacity_species>
        m_species_spectral_rosseland_table_executors;
#ifdef WARPX_USE_MATERIAL_OPACITY_HDF5
    /** One pre-mixed, fixed-composition material table. Its executor is used
     * directly and is never included in the additive SpeciesOpacity path. */
    warpx::radiation::MaterialOpacityTable m_material_opacity_table;
    /** Registry-ordered pure-material tables. A resolved cell selects exactly
     * one table; their Rosseland means are never additively mixed. */
    std::array<warpx::radiation::MaterialOpacityTable, max_opacity_materials>
        m_registered_material_opacity_tables;
    amrex::GpuArray<int, max_opacity_species>
        m_opacity_species_material_indices{};
    warpx::materials::MaterialRegistry::ResolvedCellSelector
        m_registered_material_selector;
#endif
    amrex::Gpu::HostVector<amrex::Real> m_energy_group_boundaries_h;
    amrex::Gpu::HostVector<amrex::Real> m_group_photon_energies_h;
#ifdef AMREX_USE_GPU
    amrex::Gpu::DeviceVector<amrex::Real> m_energy_group_boundaries_d;
    amrex::Gpu::DeviceVector<amrex::Real> m_group_photon_energies_d;
#endif
    warpx::radiation::EnergyGroupsExecutor m_energy_groups;
    amrex::Real m_minimum_electron_density = 0.0;
    /** Non-owning pointer. WarpX constructs this model first and destroys it
     * last. */
    HybridPICModel const* m_hybrid_model = nullptr;
    mutable amrex::Real m_last_boundary_energy_loss = 0.0;
    mutable amrex::Real m_last_diffusion_boundary_energy_loss = 0.0;
    mutable amrex::Real m_last_streaming_boundary_energy_loss = 0.0;
    mutable amrex::Real m_cumulative_boundary_energy_loss = 0.0;
    mutable amrex::Real m_last_numerical_energy_residual = 0.0;
    mutable amrex::Real m_cumulative_numerical_energy_residual = 0.0;
    mutable amrex::GpuArray<amrex::Real, 3>
        m_last_diffusion_boundary_momentum_loss{0.0, 0.0, 0.0};
    mutable amrex::GpuArray<amrex::Real, 3>
        m_last_streaming_boundary_momentum_loss{0.0, 0.0, 0.0};
    amrex::GpuArray<int, 3> m_diffusion_boundary_lo{0, 0, 0};
    amrex::GpuArray<int, 3> m_diffusion_boundary_hi{0, 0, 0};
    MaterialCoupling m_material_coupling = MaterialCoupling::None;
    LteExchangeSolver m_lte_exchange_solver =
        LteExchangeSolver::FrozenTemperature;
};

#endif // WARPX_RADIATION_TRANSPORT_H_
