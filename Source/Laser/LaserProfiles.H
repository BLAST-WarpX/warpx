/* Copyright 2019-2020 Luca Fedeli, Maxence Thevenet
 * Ilian Kara-Mostefa
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_LaserProfiles_H_
#define WARPX_LaserProfiles_H_

#include <AMReX_Gpu.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Parser.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>

#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "Utils/WarpX_Complex.H"

namespace WarpXLaserProfiles {

/** Common laser profile parameters
 *
 * Parameters for each laser profile as shared among all laser profile classes.
 */
struct CommonLaserParameters
{
    amrex::Real wavelength; //! central wavelength
    amrex::Real e_max;  //! maximum electric field at peak
    amrex::Vector<amrex::Real> p_X;// ! Polarization
    amrex::Vector<amrex::Real> nvec; //! Normal of the plane of the antenna
};


/** Abstract interface for laser profile classes
 *
 * Each new laser profile should inherit this interface and implement three
 * methods: init, update and fill_amplitude (described below).
 *
 * The declaration of a LaserProfile class should be placed in this file,
 * while the implementation of the methods should be in a dedicated file in
 * LaserProfilesImpl folder. LaserProfile classes should appear in
 * laser_profiles_dictionary to be used by LaserParticleContainer.
 */
class ILaserProfile
{
public:
    /** Initialize Laser Profile
     *
     * Reads the section of the inputfile relative to the laser beam
     * (e.g. laser_name.profile_t_peak, laser_name.profile_duration...)
     * and the "my_constants" section. It also receives some common
     * laser profile parameters. It uses these data to initialize the
     * member variables of the laser profile class.
     *
     * @param[in] ppl should be amrex::ParmParse(laser_name)
     * @param[in] params common laser profile parameters
     */
    virtual void
    init (
        const amrex::ParmParse& ppl,
        CommonLaserParameters params) = 0;

    /** Update Laser Profile
     *
     * Some laser profiles might need to perform an "update" operation per
     * time step.
     *
     * @param[in] t Current physical time in the simulation (seconds)
     */
    virtual void
    update (
        amrex::Real t) = 0;

    /** Fill Electric Field Amplitude for each particle of the antenna.
     *
     * Xp, Yp and amplitude must be arrays with the same length
     *
     * @param[in] np number of antenna particles
     * @param[in] Xp X coordinate of the particles of the antenna
     * @param[in] Yp Y coordinate of the particles of the antenna
     * @param[in] t time (seconds)
     * @param[out] amplitude of the electric field (V/m)
     */
    virtual void
    fill_amplitude (
        int np,
        amrex::Real const * AMREX_RESTRICT Xp,
        amrex::Real const * AMREX_RESTRICT Yp,
        amrex::Real t,
        amrex::Real* AMREX_RESTRICT amplitude) const = 0;

    ILaserProfile () = default;
    virtual ~ILaserProfile() = default;

    ILaserProfile ( ILaserProfile const &)             = default;
    ILaserProfile& operator= ( ILaserProfile const & ) = default;
    ILaserProfile ( ILaserProfile&& )                  = default;
    ILaserProfile& operator= ( ILaserProfile&& )       = default;
};

/**
 * Gaussian laser profile
 */
class GaussianLaserProfile : public ILaserProfile
{

public:
    void
    init (
        const amrex::ParmParse& ppl,
        CommonLaserParameters params) final;

    //No update needed
    void
    update (amrex::Real /*t */) final {}

    void
    fill_amplitude (
        int np,
        amrex::Real const * AMREX_RESTRICT Xp,
        amrex::Real const * AMREX_RESTRICT Yp,
        amrex::Real t,
        amrex::Real * AMREX_RESTRICT amplitude) const final;

private:
    struct {
        amrex::Real waist          = std::numeric_limits<amrex::Real>::quiet_NaN();
        amrex::Real duration       = std::numeric_limits<amrex::Real>::quiet_NaN();
        amrex::Real t_peak         = std::numeric_limits<amrex::Real>::quiet_NaN();
        amrex::Real focal_distance = std::numeric_limits<amrex::Real>::quiet_NaN();
        amrex::Real zeta           = 0;
        amrex::Real beta           = 0;
        amrex::Real phi2           = 0;
        amrex::Real phi0           = 0;

        amrex::Vector<amrex::Real> stc_direction; //! Direction of the spatio-temporal couplings
        amrex::Real theta_stc; //! Angle between polarization (p_X) and direction of spatiotemporal coupling (stc_direction)
    } m_params;

    CommonLaserParameters m_common_params;
};

/**
 * Laser profile defined by the used with an analytical expression
 */
class FieldFunctionLaserProfile : public ILaserProfile
{

public:
    void
    init (
        const amrex::ParmParse& ppl,
        CommonLaserParameters params) final;

    //No update needed
    void
    update (amrex::Real /*t */) final {}

    void
    fill_amplitude (
        int np,
        amrex::Real const * AMREX_RESTRICT Xp,
        amrex::Real const * AMREX_RESTRICT Yp,
        amrex::Real t,
        amrex::Real * AMREX_RESTRICT amplitude) const final;

private:
    struct{
        std::string field_function;
    } m_params;

    amrex::Parser m_parser;
};

/**
 * Laser profile read from a file (lasy or binary)
 * The binary file must contain:
 * - 3 unsigned integers (4 bytes): nt (points along t), nx (points along x) and ny (points along y)
 * - nt*nx*ny doubles (8 bytes) in row major order : field amplitude
 */
class FromFileLaserProfile : public ILaserProfile
{

public:
    void
    init (
        const amrex::ParmParse& ppl,
        CommonLaserParameters params) final;

    /** \brief Reads new field data chunk from file if needed
    *
    * @param[in] t simulation time (seconds)
    */
    void
    update (amrex::Real t) final;

    /** \brief compute field amplitude at particles' position for a laser beam
    * loaded from an E(x,y,t) file.
    *
    * Both Xp and Yp are given in laser plane coordinate.
    * For each particle with position Xp and Yp, this routine computes the
    * amplitude of the laser electric field, stored in array amplitude.
    *
    * \param np: number of laser particles
    * \param Xp: pointer to first component of positions of laser particles
    * \param Yp: pointer to second component of positions of laser particles
    * \param t: Current physical time
    * \param amplitude: pointer to array of field amplitude.
    */
    void
    fill_amplitude (
        int np,
        amrex::Real const * AMREX_RESTRICT Xp,
        amrex::Real const * AMREX_RESTRICT Yp,
        amrex::Real t,
        amrex::Real * AMREX_RESTRICT amplitude) const final;

    /** \brief Function to fill the amplitude in case of a uniform grid and for the lasy format in 3D Cartesian.
    * This function cannot be private due to restrictions related to
    * the use of extended __device__ lambda
    *
    * \param idx_t_left index of the last time coordinate < t
    * \param np: number of laser particles
    * \param Xp: pointer to first component of positions of laser particles
    * \param Yp: pointer to second component of positions of laser particles
    * \param t: Current physical time
    * \param amplitude: pointer to array of field amplitude.
    */
    void internal_fill_amplitude_uniform_cartesian(
        int idx_t_left,
        int np,
        amrex::Real const * AMREX_RESTRICT Xp,
        amrex::Real const * AMREX_RESTRICT Yp,
        amrex::Real t,
        amrex::Real * AMREX_RESTRICT amplitude) const;

    /** \brief Function to fill the amplitude in case of a uniform grid and for the lasy format in RZ geometry.
    * This function cannot be private due to restrictions related to
    * the use of extended __device__ lambda
    *
    * \param idx_t_left index of the last time coordinate < t
    * \param np: number of laser particles
    * \param Xp: pointer to first component of positions of laser particles
    * \param Yp: pointer to second component of positions of laser particles
    * \param t: Current physical time
    * \param amplitude: pointer to array of field amplitude.
    */
    void internal_fill_amplitude_uniform_cylindrical(
        int idx_t_left,
        int np,
        amrex::Real const * AMREX_RESTRICT Xp,
        amrex::Real const * AMREX_RESTRICT Yp,
        amrex::Real t,
        amrex::Real * AMREX_RESTRICT amplitude) const;



    /** \brief Function to fill the amplitude in case of a uniform grid and for the binary format.
    * This function cannot be private due to restrictions related to
    * the use of extended __device__ lambda
    *
    * \param idx_t_left index of the last time coordinate < t
    * \param np: number of laser particles
    * \param Xp: pointer to first component of positions of laser particles
    * \param Yp: pointer to second component of positions of laser particles
    * \param t: Current physical time
    * \param amplitude: pointer to array of field amplitude.
    */
        void internal_fill_amplitude_uniform_binary(
        int idx_t_left,
        int np,
        amrex::Real const * AMREX_RESTRICT Xp,
        amrex::Real const * AMREX_RESTRICT Yp,
        amrex::Real t,
        amrex::Real * AMREX_RESTRICT amplitude) const;

private:
    /** \brief parse a field file in the HDF5 'lasy' format
    * \param lasy_file_name: name of the lasy file to parse
    */
    void parse_lasy_file(const std::string& lasy_file_name);

    /** \brief parse a field file in the binary 'binary' format (whose details are given below).
    *
    * A 'binary' file should be a binary file with the following format:
    * -np, number of timesteps (uint32_t, must be >=2)
    * -nx, number of points along x (uint32_t, must be >=2)
    * -ny, number of points along y (uint32_t, must be 1 for 2D simulations and >=2 for 3D simulations)
    * -timesteps (double[2])
    * -x_coords (double[2])
    * -y_coords (double[1] if 2D, double[2] if 3D
    * -field_data (double[nt * nx * ny], with nt being the slowest coordinate).
    * The spatiotemporal grid must be rectangular and uniform.
    * \param binary_file_name: name of the binary file to parse
    */
    void parse_binary_file(const std::string& binary_file_name);

    /** \brief Finds left and right time indices corresponding to time t
    *
    *
    * \param t: simulation time
    */
    [[nodiscard]] std::pair<int,int> find_left_right_time_indices(amrex::Real t) const;

    /** \brief Load field data within the temporal range [t_begin, t_end]
    *
    * Must be called after having parsed a lasy data file with the 'parse_lasy_file' function.
    *
    * \param t_begin: left limit of the timestep range to read
    * \param t_end: right limit of the timestep range to read (t_end is not read)
    */
    void read_data_t_chunk(int t_begin, int t_end);

        /** \brief Load field data within the temporal range [t_begin, t_end]
    *
    * Must be called after having parsed a binary data file with the 'parse_binary_file' function.
    *
    * \param t_begin: left limit of the timestep range to read
    * \param t_end: right limit of the timestep range to read (t_end is not read)
    */
    void read_binary_data_t_chunk(int t_begin, int t_end);

    /**
     * \brief m_params contains all the internal parameters
     * used by this laser profile
     */
    struct{

        /** Name of the binary file containing the data */
        std::string binary_file_name;
        /** Name of the lasy file containing the data */
        std::string lasy_file_name;
        /** true if the file is in the lasy format, false if it is in the binary format */
        bool file_in_lasy_format;
        /** lasy file geometry ("cartesian" for 3D cartesian or "thetaMode" for RZ) */
        int file_in_cartesian_geom;
        /** Dimensions of E_binary_data or E_lasy_data. nt, nx must be >=2.
         * If DIM=3, ny must be >=2 as well.
         * If DIM=2, ny must be 1 */
        int nt, nx, ny;
        /** Dimensions of E_lasy_data in RZ */
        int nr;
        /** Number of azimuthal components (2 per mode, 1 for mode 0) */
        int n_rz_azimuthal_components;
        /** Start time*/
        amrex::Real t_min;
        /** Stop time*/
        amrex::Real t_max;
        /** min of x coordinates*/
        amrex::Real x_min;
        /** max of x coordinates*/
        amrex::Real x_max;
        /** min of y coordinates*/
        amrex::Real y_min;
        /** max of y coordinates*/
        amrex::Real y_max;
        amrex::Real r_min;
        amrex::Real r_max;
        /** Size of the timestep range to load */
        int time_chunk_size;
        /** Index of the first timestep in memory */
        int first_time_index;
        /** Index of the last timestep in memory */
        int last_time_index;
        /** lasy field data */
        amrex::Gpu::DeviceVector<Complex> E_lasy_data;
        /** binary field data */
        amrex::Gpu::DeviceVector<amrex::Real> E_binary_data;
        /** This parameter is subtracted to simulation time before interpolating field data in file (either lasy or binary).
        *   If t_delay > 0, the laser is delayed, otherwise it is anticipated. */
        amrex::Real t_delay = amrex::Real(0.0);

    } m_params;

    CommonLaserParameters m_common_params;
};

/**
 * Maps laser profile names to lambdas returing unique pointers
 * to the corresponding laser profile objects.
 */
const
std::map<
std::string,
std::function<std::unique_ptr<ILaserProfile>()>
>
laser_profiles_dictionary =
{
    {"gaussian",
        [] () {return std::make_unique<GaussianLaserProfile>();} },
    {"parse_field_function",
        [] () {return std::make_unique<FieldFunctionLaserProfile>();} },
    {"from_file",
        [] () {return std::make_unique<FromFileLaserProfile>();} }
};

} //WarpXLaserProfiles

#endif //WARPX_LaserProfiles_H_
