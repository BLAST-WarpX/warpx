/* Copyright 2019 Andrew Myers, Axel Huebl, David Grote
 * Maxence Thevenet, Remi Lehe, Weiqun Zhang
 *
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PLASMA_INJECTOR_H_
#define WARPX_PLASMA_INJECTOR_H_

#include "InjectorDensity.H"
#include "InjectorFlux.H"
#include "InjectorMomentum.H"
#include "TemperatureProperties.H"
#include "VelocityProperties.H"
#include "Particles/SpeciesPhysicalProperties.H"

#include "InjectorPosition_fwd.H"

#include <AMReX_Dim3.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include <AMReX_ParmParse.H>

#include <AMReX_BaseFwd.H>

#ifdef WARPX_USE_OPENPMD
#   include <openPMD/openPMD.hpp>
#endif

#include <limits>
#include <memory>
#include <string>

///
/// The PlasmaInjector class parses and stores information about the plasma
/// type used in the particle container. This information is used to create the
/// particles on initialization and whenever the window moves.
///
class PlasmaInjector
{

public:

    /** Default constructor*/
    PlasmaInjector () = default;

    PlasmaInjector (const std::string& name, const amrex::Geometry& geom,
                    const std::string& src_name="");

    // Default move and copy operations
    PlasmaInjector(const PlasmaInjector&) = delete;
    PlasmaInjector& operator=(const PlasmaInjector&) = delete;
    PlasmaInjector(PlasmaInjector&&) = default;
    PlasmaInjector& operator=(PlasmaInjector&&) = default;

    ~PlasmaInjector ();

    // bool: whether the point (x, y, z) is inside the plasma region
    [[nodiscard]] bool insideBounds (
        amrex::Real x, amrex::Real y, amrex::Real z) const noexcept;

    // bool: whether the region defined by lo and hi overlaps with the plasma region
    [[nodiscard]] bool overlapsWith (
        const amrex::XDim3& lo, const amrex::XDim3& hi) const noexcept;

    int num_particles_per_cell;
    amrex::Real num_particles_per_cell_real;

    amrex::Vector<int> num_particles_per_cell_each_dim;

    // gamma * beta
    [[nodiscard]] amrex::XDim3 getMomentum (
        amrex::Real x, amrex::Real y, amrex::Real z) const noexcept;

    bool queryCharge (amrex::ParticleReal& a_charge) const;
    bool queryMass (amrex::ParticleReal& a_mass) const;

    // bool: whether the initial injection of particles should be done
    // This routine is called during initialization of the plasma.
    [[nodiscard]] bool doInjection () const noexcept { return h_inj_pos != nullptr;}

    // bool: whether the flux injection of particles should be done.
    [[nodiscard]] bool doFluxInjection () const noexcept { return h_flux_pos != nullptr;}

    bool add_single_particle = false;
    amrex::Vector<amrex::ParticleReal> single_particle_pos;
    amrex::Vector<amrex::ParticleReal> single_particle_u;
    amrex::ParticleReal single_particle_weight;

    bool add_multiple_particles = false;
    amrex::Vector<amrex::ParticleReal> multiple_particles_pos_x;
    amrex::Vector<amrex::ParticleReal> multiple_particles_pos_y;
    amrex::Vector<amrex::ParticleReal> multiple_particles_pos_z;
    amrex::Vector<amrex::ParticleReal> multiple_particles_ux;
    amrex::Vector<amrex::ParticleReal> multiple_particles_uy;
    amrex::Vector<amrex::ParticleReal> multiple_particles_uz;
    amrex::Vector<amrex::ParticleReal> multiple_particles_weight;

    bool gaussian_beam = false;
    amrex::Real x_m;
    amrex::Real y_m;
    amrex::Real z_m;
    amrex::Real x_rms;
    amrex::Real y_rms;
    amrex::Real z_rms;
    amrex::Real x_cut = std::numeric_limits<amrex::Real>::max();
    amrex::Real y_cut = std::numeric_limits<amrex::Real>::max();
    amrex::Real z_cut = std::numeric_limits<amrex::Real>::max();
    amrex::Real q_tot = 0.0;
    long npart;
    int do_symmetrize = 0;
    int symmetrization_order = 4;
    bool do_focusing = false;
    amrex::Real focal_distance;

    bool external_file = false; //! initialize from an openPMD file
    amrex::Real z_shift = 0.0; //! additional z offset for particle positions
#ifdef WARPX_USE_OPENPMD
    //! openPMD::Series to load from in external_file injection
    std::unique_ptr<openPMD::Series> m_openpmd_input_series;
#endif

    amrex::Real surface_flux_pos; // surface location
    amrex::Real flux_tmin = -1.; // Time after which we start injecting particles
    amrex::Real flux_tmax = -1.; // Time after which we stop injecting particles
    // Flux normal axis represents the direction in which to emit particles
    // When compiled in Cartesian geometry, 0 = x, 1 = y, 2 = z
    // When compiled in cylindrical geometry, 0 = radial, 1 = azimuthal, 2 = z
    int flux_normal_axis;
    int flux_direction; // -1 for left, +1 for right

    bool m_inject_from_eb = false; // whether to inject from the embedded boundary

    bool radially_weighted = true;

    std::string str_flux_function;

    amrex::Real xmin, xmax;
    amrex::Real ymin, ymax;
    amrex::Real zmin, zmax;
    amrex::Real density_min = std::numeric_limits<amrex::Real>::epsilon();
    amrex::Real density_max = std::numeric_limits<amrex::Real>::max();

    [[nodiscard]] InjectorPosition* getInjectorPosition () const;
    [[nodiscard]] InjectorPosition* getInjectorFluxPosition () const;
    [[nodiscard]] InjectorDensity*  getInjectorDensity () const;

    [[nodiscard]] InjectorFlux*  getInjectorFlux () const;
    [[nodiscard]] InjectorMomentum* getInjectorMomentumDevice () const;
    [[nodiscard]] InjectorMomentum* getInjectorMomentumHost () const;

protected:

    bool mass_from_source = false;
    bool charge_from_source = false;
    amrex::ParticleReal mass, charge;

    PhysicalSpecies physical_species = PhysicalSpecies::unspecified;

    amrex::Real flux;

    int species_id;
    std::string species_name;
    std::string source_name;

    std::unique_ptr<InjectorPosition> h_inj_pos;
    InjectorPosition* d_inj_pos = nullptr;

    std::unique_ptr<InjectorPosition> h_flux_pos;
    InjectorPosition* d_flux_pos = nullptr;

    std::unique_ptr<InjectorDensity,InjectorDensityDeleter> h_inj_rho;
    InjectorDensity* d_inj_rho = nullptr;
    std::unique_ptr<amrex::Parser> density_parser;

    std::unique_ptr<InjectorFlux,InjectorFluxDeleter> h_inj_flux;
    InjectorFlux* d_inj_flux = nullptr;
    std::unique_ptr<amrex::Parser> flux_parser;

    std::unique_ptr<InjectorMomentum,InjectorMomentumDeleter> h_inj_mom;
    InjectorMomentum* d_inj_mom = nullptr;
    std::unique_ptr<amrex::Parser> ux_parser;
    std::unique_ptr<amrex::Parser> uy_parser;
    std::unique_ptr<amrex::Parser> uz_parser;
    std::unique_ptr<amrex::Parser> ux_th_parser;
    std::unique_ptr<amrex::Parser> uy_th_parser;
    std::unique_ptr<amrex::Parser> uz_th_parser;

    // Keep a pointer to TemperatureProperties to ensure the lifetime of the
    // contained Parser
    std::unique_ptr<TemperatureProperties> h_mom_temp;
    std::unique_ptr<VelocityProperties> h_mom_vel;

    void setupSingleParticle (amrex::ParmParse const& pp_species);
    void setupMultipleParticles (amrex::ParmParse const& pp_species);
    void setupGaussianBeam (amrex::ParmParse const& pp_species);
    void setupNRandomPerCell (amrex::ParmParse const& pp_species);
    void setupNFluxPerCell (amrex::ParmParse const& pp_species);
    void setupNuniformPerCell (amrex::ParmParse const& pp_species);
    void setupExternalFile (amrex::ParmParse const& pp_species);

    void parseFlux (amrex::ParmParse const& pp_species);
};

#endif //WARPX_PLASMA_INJECTOR_H_
