#ifndef WARPX_SURFACE_PHYSICS_BASE_H_
#define WARPX_SURFACE_PHYSICS_BASE_H_

#ifdef WARPX_SURFACE_PHYSICS

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

    void InitData ();
    void initializeMapping ();
    void AllocAndInitInfluxBndVectors();
    void AllocAndInitOutfluxBndVectors();

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
    amrex::Vector< amrex::Vector<int> > num_in_particles;
    amrex::Vector< amrex::Vector<int> > num_out_particles;

    

};


#endif // ifdef(WARPX_SURFACE_PHYSICS)

#endif
