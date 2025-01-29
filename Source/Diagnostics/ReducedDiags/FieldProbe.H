/* Copyright 2021 Lorenzo Giacomel, Neil Zaim, Yinjian Zhao
 *                Elisa Rheaume, Axel Huebl
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_DIAGNOSTICS_REDUCEDDIAGS_FIELDPROBE_H_
#define WARPX_DIAGNOSTICS_REDUCEDDIAGS_FIELDPROBE_H_

#include "ReducedDiags.H"
#include "FieldProbeParticleContainer.H"

#include <AMReX.H>
#include <AMReX_Vector.H>

#include <unordered_map>
#include <string>
#include <vector>

/**
 * This enumeration is used for assigning structural geometry levels (point vs line vs plane)
 */
enum struct DetectorGeometry
{
    Point = 0,
    Line,
    Plane
};

/**
 *  This class mainly contains a function that computes the value of each component
 * of the EM field at a given point
 */
class FieldProbe : public ReducedDiags
{
public:

    /**
     * constructor
     * @param[in] rd_name reduced diags names
     */
    FieldProbe (const std::string& rd_name);

    /**
     * This function assins test/data particles to constructed environemnt
     */
    void InitData () final;

    /** Redistribute parallel data structures during load balance
     */
    void LoadBalance () final;

    /**
     * This function computes the value of Ex, Ey, Ez, Bx, By, Bz and at a given point
     *
     * @param[in] step current time step
     */
    void ComputeDiags (int step) final;

    /*
     * Define constants used throughout FieldProbe
     */

    //! noutputs is 11 for particle id + (x, y, z, Ex, Ey, Ez, Bx, By, Bz, S)
    static const int noutputs = 11;

private:
    amrex::Real x_probe = 0.;
    amrex::Real y_probe = 0.;
    amrex::Real x1_probe = 0.;
    amrex::Real y1_probe = 0.;
    amrex::Real target_normal_x = 0.;
    amrex::Real target_normal_y = 1.;
    amrex::Real target_normal_z = 0.;
    amrex::Real target_up_x = 0.;
    amrex::Real target_up_y = 0.;
    amrex::Real target_up_z = 1.;
    amrex::Real z_probe, z1_probe;
    amrex::Real detector_radius;

    //! counts number of particles for all MPI ranks
    long m_valid_particles {0};

    //! remember the last time @see ComputeDiags was called to count the number of steps in between (for non-integrated detectors)
    int m_last_compute_step = 0;

    //! determines geometry of detector point distribution
    DetectorGeometry m_probe_geometry = DetectorGeometry::Point;

    //! determines number of particles places for non-point geometries
    int m_resolution = 0;

    //! Empty vector for to which data is pushed
    amrex::Vector<amrex::Real> m_data;

    //! Empty array to be used by IOProcessor node to store and output data
    amrex::Vector<amrex::Real> m_data_out;

    //! this is the particle container in which probe particles are stored
    FieldProbeParticleContainer m_probe;

    //! if true, integrate values over time instead of probing instantaneous values
    bool m_field_probe_integrate = false;

    //! particle shape used for field gather
    int interp_order = 1;

    //! Judges whether to follow a moving window
    bool do_moving_window_FP = false;

    /**
     * Built-in function in ReducedDiags to write out test data
     */
    void WriteToFile (int step) const override;

    /** Check if the probe is in the simulation domain boundary
     */
    bool ProbeInDomain () const;

    /**
     * Simple utility function to normalize the components of a "vector"
     */
    void normalize(amrex::Real &AMREX_RESTRICT x, amrex::Real &AMREX_RESTRICT y,
                   amrex::Real &AMREX_RESTRICT z){
        const amrex::Real mag = std::sqrt(x*x + y*y + z*z);
        x /= mag;
        y /= mag;
        z /= mag;
    }
};

#endif // WARPX_DIAGNOSTICS_REDUCEDDIAGS_FIELDPROBE_H_
