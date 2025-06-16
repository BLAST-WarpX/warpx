#ifndef WARPX_SURFACE_PHYSICS_BASE_H_
#define WARPX_SURFACE_PHYSICS_BASE_H_

#ifdef WARPX_SURFACE_PHYSICS

#include "Particles/MultiParticleContainer_fwd.H"
#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX.H>
#include <AMReX_IntVect.H>
#include <AMReX_Real.H>
#include <AMReX_Vector.H>
#include <AMReX_iMultiFab.H>

class SurfacePhysicsBase
{
public:
    SurfacePhysicsBase ();

    ~SurfacePhysicsBase() = default;

    void ReadInputs ();
    void InitData ();
    void initializeMapping ();
    void AllocAndInitInfluxBndVectors();
    void AllocAndInitOutfluxBndVectors();

    void countParticlesFromEmbeddedBoundaries(
        MultiParticleContainer& mypc, ablastr::fields::MultiLevelScalarField const& distance_to_eb);

    void nullifyInfluxParticleCounter ();
    void nullifyInfluxParticleCounter (int isp);
    void nullifyOutfluxParticleCounter ();
    void nullifyOutfluxParticleCounter (int isp);

    std::unique_ptr< amrex::iMultiFab > ivect_map;

    amrex::Vector<amrex::IntVect> surf_ijk;
    amrex::Vector<amrex::Real> surf_normal_x;
    amrex::Vector<amrex::Real> surf_normal_y;
    amrex::Vector<amrex::Real> surf_normal_z;
    int num_influx_species;
    int num_outflux_species;
    amrex::Vector< amrex::Vector<amrex::Real> > bnd_influx;
    amrex::Vector< amrex::Vector<amrex::Real> > bnd_outflux;
    amrex::Vector< amrex::Vector< amrex::Real > > num_in_particles;
    amrex::Vector< amrex::Vector< amrex::Real > > num_out_particles;


private:
    struct Reaction {
        std::string equation;
        amrex::Vector<std::string> reactants;
        amrex::Vector<std::string> products;
        double P_energy0;
        double P0;
        double E_ref;
        double E_th;
        double exp;
    };
    amrex::Vector<std::string> tokenize_reaction (const std::string& input);

    std::map<std::string, std::string> gas_species; //stores gas_species with the symbol
    std::map<std::string, std::string> surface_species; //stores surface_species with the symbol
    std::vector<Reaction> reactions;

};


#endif // ifdef(WARPX_SURFACE_PHYSICS)

#endif
