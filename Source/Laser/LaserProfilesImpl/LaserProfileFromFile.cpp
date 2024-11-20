/* Copyright 2019-2020 Luca Fedeli
 * Ilian Kara-Mostefa
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "Laser/LaserProfiles.H"

#include "Utils/Algorithms/LinearInterpolation.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpX_Complex.H"
#include "Utils/WarpXConst.H"

#include <ablastr/warn_manager/WarnManager.H>

#include <AMReX.H>
#include <AMReX_Algorithm.H>
#include <AMReX_Config.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_PODVector.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <iostream>
#include <memory>

#ifdef WARPX_USE_OPENPMD
#   include <openPMD/openPMD.hpp>
    namespace io = openPMD;
#endif

using namespace amrex;

void
WarpXLaserProfiles::FromFileLaserProfile::init (
    const amrex::ParmParse& ppl,
    CommonLaserParameters params)
{
    if (!std::numeric_limits< double >::is_iec559)
    {
        ablastr::warn_manager::WMRecordWarning("Laser",
            "(Double does not comply with IEEE 754: bad"
                "things will happen parsing the X, Y and T profiles for the laser!)",
        ablastr::warn_manager::WarnPriority::high);
    }
    // Parse the lasy or binary file
    ppl.query("lasy_file_name", m_params.lasy_file_name);
    ppl.query("binary_file_name", m_params.binary_file_name);
    const std::string lasy_file_name = m_params.lasy_file_name;
    const std::string binary_file_name = m_params.binary_file_name;
    m_params.file_in_lasy_format = false;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE (
        lasy_file_name.empty() != binary_file_name.empty(),
        "Exactly one of 'binary_file_name' and 'lasy_file_name' has to be specified");
    if (!lasy_file_name.empty()) {
#ifdef WARPX_USE_OPENPMD
        m_params.file_in_lasy_format = true;
        parse_lasy_file(lasy_file_name);
#else
        WARPX_ABORT_WITH_MESSAGE("WarpX has to be compiled with the option openPMD=ON to read a lasy file");
        amrex::ignore_unused(ppl, params);
#endif
    } else{
        parse_binary_file(binary_file_name);
        std::stringstream warnMsg;
            warnMsg << "Laser profile from a binary file will soon be replaced by 'lasy' files reading. " ;
            ablastr::warn_manager::WMRecordWarning("AddPlasmaFromFile",
               warnMsg.str(), ablastr::warn_manager::WarnPriority::low);
    }

    //Set time_chunk_size
    m_params.time_chunk_size = m_params.nt;
    int temp = 1;
    if(utils::parser::queryWithParser(ppl ,"time_chunk_size", temp)){
        m_params.time_chunk_size = min(
        temp, m_params.time_chunk_size);
    }
    if(m_params.time_chunk_size < 2){
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_params.time_chunk_size >= 2,
        "Error! time_chunk_size must be >= 2!");
    }
    //Reads the (optional) delay
    utils::parser::queryWithParser(ppl, "delay", m_params.t_delay);

    //Read first time chunk
    if (m_params.file_in_lasy_format){
        read_data_t_chunk(0, m_params.time_chunk_size);
    } else{
        read_binary_data_t_chunk(0, m_params.time_chunk_size);
    }
    //Copy common params
    m_common_params = params;
}

void
WarpXLaserProfiles::FromFileLaserProfile::update (amrex::Real t)
{
    t += m_params.t_min - m_params.t_delay;
    if(t >= m_params.t_max) {
        return;
    }
    const auto idx_times = find_left_right_time_indices(t);
    const auto idx_t_left = idx_times.first;
    const auto idx_t_right = idx_times.second;
    //Load data chunk if needed
    if(idx_t_right >  m_params.last_time_index){
        if (m_params.file_in_lasy_format){
            read_data_t_chunk(idx_t_left, idx_t_left+m_params.time_chunk_size);
        } else{
            read_binary_data_t_chunk(idx_t_left, idx_t_left+m_params.time_chunk_size);
        }
    }
}

void
WarpXLaserProfiles::FromFileLaserProfile::fill_amplitude (
    const int np,
    Real const * AMREX_RESTRICT const Xp, Real const * AMREX_RESTRICT const Yp,
    Real t, Real * AMREX_RESTRICT const amplitude) const
{
    t += m_params.t_min - m_params.t_delay;
    //Amplitude is 0 if time is out of range
    if(t < m_params.t_min ||  t > m_params.t_max){
        amrex::ParallelFor(np,
            [=] AMREX_GPU_DEVICE (int i) {
                amplitude[i] = 0.0_rt;});
        return;
    }
    //Find left time index
    const auto idx_t_left = find_left_right_time_indices(t).first;
    if(idx_t_left <  m_params.first_time_index){
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(idx_t_left >=  m_params.first_time_index,
        "Something bad has happened with the simulation time");
    }
    if (m_params.file_in_lasy_format){
        if (m_params.file_in_cartesian_geom==1){
            internal_fill_amplitude_uniform_cartesian(idx_t_left, np, Xp, Yp, t, amplitude);
        } else {
            internal_fill_amplitude_uniform_cylindrical(idx_t_left, np, Xp, Yp, t, amplitude);
        }
    } else{
        internal_fill_amplitude_uniform_binary(idx_t_left, np, Xp, Yp, t, amplitude);
    }
}

void
WarpXLaserProfiles::FromFileLaserProfile::parse_lasy_file(const std::string& lasy_file_name)
{
#ifdef WARPX_USE_OPENPMD
    if(ParallelDescriptor::IOProcessor()){
        auto series = io::Series(lasy_file_name, io::Access::READ_ONLY);
        auto i = series.iterations[0];
        auto E = i.meshes["laserEnvelope"];

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(E.getAttribute("dataOrder").get<std::string>() == "C",
                                         "Reading from files with non-C dataOrder is not implemented");

        auto fileGeom = E.getAttribute("geometry").get<std::string>();
        auto E_laser = E[io::RecordComponent::SCALAR];
        auto extent = E_laser.getExtent();
        // Extract grid offset and grid spacing
        std::vector<double> offset = E.gridGlobalOffset();
        std::vector<double> position = E_laser.position<double>();
        std::vector<double> spacing = E.gridSpacing<double>();
        if (fileGeom=="thetaMode") {
            //Dimensions of lasy file data: {m,t,r}
            amrex::Print() << Utils::TextMsg::Info( "Found lasy file in RZ geometry" );
            m_params.file_in_cartesian_geom = 0;
            m_params.n_rz_azimuthal_components = static_cast<int>(extent[0]);
            m_params.nt = static_cast<int>(extent[1]);
            m_params.nr = static_cast<int>(extent[2]);
            if(m_params.nt <= 1) { WARPX_ABORT_WITH_MESSAGE("nt in lasy file must be >=2"); }
            if(m_params.nr <= 1) { WARPX_ABORT_WITH_MESSAGE("nr in lasy file must be >=2"); }
            // Calculate the min and max of the grid
            m_params.t_min = static_cast<amrex::Real>(offset[0] + position[0]*spacing[0]);
            m_params.t_max = static_cast<amrex::Real>(m_params.t_min + (m_params.nt-1)*spacing[0]);
            m_params.r_min = static_cast<amrex::Real>(offset[1] + position[1]*spacing[1]);
            m_params.r_max = static_cast<amrex::Real>(m_params.r_min + (m_params.nr-1)*spacing[1]);
        } else if (fileGeom=="cartesian"){
            //Dimensions of lasy file data: {t,y,x}
            amrex::Print() << Utils::TextMsg::Info( "Found lasy file in 3D cartesian geometry");
            m_params.file_in_cartesian_geom = 1;
            m_params.nt = static_cast<int>(extent[0]);
            m_params.ny = static_cast<int>(extent[1]);
            m_params.nx = static_cast<int>(extent[2]);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_params.nt > 1, "nt in lasy file must be >=2");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_params.nx > 1, "nx in lasy file must be >=2");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_params.ny > 1, "ny in lasy file must be >=2 in 3D");
            // Calculate the min and max of the grid
            m_params.t_min = static_cast<amrex::Real>(offset[0] + position[0]*spacing[0]);
            m_params.t_max = static_cast<amrex::Real>(m_params.t_min + (m_params.nt-1)*spacing[0]);
            m_params.y_min = static_cast<amrex::Real>(offset[1] + position[1]*spacing[1]);
            m_params.y_max = static_cast<amrex::Real>(m_params.y_min + (m_params.ny-1)*spacing[1]);
            m_params.x_min = static_cast<amrex::Real>(offset[2] + position[2]*spacing[2]);
            m_params.x_max = static_cast<amrex::Real>(m_params.x_min + (m_params.nx-1)*spacing[2]);
        } else{
            WARPX_ABORT_WITH_MESSAGE("The lasy file's geometry has to be in either RZ or 3D cartesian coordinates");
        }
    }

    //Broadcast parameters
    ParallelDescriptor::Bcast(&m_params.file_in_cartesian_geom, 1, ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::Bcast(&m_params.nt, 1, ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::Bcast(&m_params.nx, 1, ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::Bcast(&m_params.ny, 1, ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::Bcast(&m_params.nr, 1, ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::Bcast(&m_params.n_rz_azimuthal_components, 1, ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::Bcast(&m_params.t_min, 1, ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::Bcast(&m_params.t_max, 1, ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::Bcast(&m_params.x_min, 1, ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::Bcast(&m_params.x_max, 1, ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::Bcast(&m_params.y_min, 1, ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::Bcast(&m_params.y_max, 1, ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::Bcast(&m_params.r_min, 1, ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::Bcast(&m_params.r_max, 1, ParallelDescriptor::IOProcessorNumber());
#else
    amrex::ignore_unused(lasy_file_name);
#endif
}

void
WarpXLaserProfiles::FromFileLaserProfile::parse_binary_file (const std::string& binary_file_name)
{
    if(ParallelDescriptor::IOProcessor()){
        std::ifstream inp(binary_file_name, std::ios::binary);
        if(!inp) { WARPX_ABORT_WITH_MESSAGE("Failed to open binary file"); }
        inp.exceptions(std::ios_base::failbit | std::ios_base::badbit);
        //Uniform grid flag
        char flag;
        inp.read(&flag, 1);
        if(!inp) { WARPX_ABORT_WITH_MESSAGE("Failed to read grid type from binary file"); }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(flag, "Binary files with non uniform grid are no longer supported");
        //Grid points along t, x and y
        inp.read(reinterpret_cast<char*>(&m_params.nt), sizeof(uint32_t));
        inp.read(reinterpret_cast<char*>(&m_params.nx), sizeof(uint32_t));
        inp.read(reinterpret_cast<char*>(&m_params.ny), sizeof(uint32_t));
        if(!inp) { WARPX_ABORT_WITH_MESSAGE("Failed to read sizes from binary file"); }
        if(m_params.nt <= 1) { WARPX_ABORT_WITH_MESSAGE("nt in binary file must be >=2"); }
        if(m_params.nx <= 1) { WARPX_ABORT_WITH_MESSAGE("nx in binary file must be >=2"); }
#if (defined(WARPX_DIM_3D) || (defined WARPX_DIM_RZ))
        if(m_params.ny <= 1) { WARPX_ABORT_WITH_MESSAGE("ny in binary file must be >=2 in 3D"); }
#elif defined(WARPX_DIM_XZ)
        if(m_params.ny != 1) { WARPX_ABORT_WITH_MESSAGE("ny in binary file must be 1 in 2D"); }
#endif
        //Coordinates
        Vector<double> dbuf_t, dbuf_x, dbuf_y;
        dbuf_t.resize(2);
        dbuf_x.resize(2);
#if (defined(WARPX_DIM_3D) || (defined WARPX_DIM_RZ))
        dbuf_y.resize(2);
#elif defined(WARPX_DIM_XZ)
        dbuf_y.resize(1);
#endif
        inp.read(reinterpret_cast<char*>(dbuf_t.dataPtr()),
            static_cast<std::streamsize>(dbuf_t.size()*sizeof(double)));
        inp.read(reinterpret_cast<char*>(dbuf_x.dataPtr()),
            static_cast<std::streamsize>(dbuf_x.size()*sizeof(double)));
        inp.read(reinterpret_cast<char*>(dbuf_y.dataPtr()),
            static_cast<std::streamsize>(dbuf_y.size()*sizeof(double)));
        if(!inp) { WARPX_ABORT_WITH_MESSAGE("Failed to read coords from binary file"); }

        m_params.t_min = static_cast<amrex::Real>(dbuf_t[0]);
        m_params.t_max = static_cast<amrex::Real>(dbuf_t[1]);
        m_params.x_min = static_cast<amrex::Real>(dbuf_x[0]);
        m_params.x_max = static_cast<amrex::Real>(dbuf_x[1]);
#if (defined(WARPX_DIM_3D) || (defined WARPX_DIM_RZ))
        m_params.y_min = static_cast<amrex::Real>(dbuf_y[0]);
        m_params.y_max = static_cast<amrex::Real>(dbuf_y[1]);
#endif
    }

    //Broadcast parameters
    ParallelDescriptor::Bcast(&m_params.nt, 1, ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::Bcast(&m_params.nx, 1, ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::Bcast(&m_params.ny, 1, ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::Bcast(&m_params.t_min, 1, ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::Bcast(&m_params.t_max, 1, ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::Bcast(&m_params.x_min, 1, ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::Bcast(&m_params.x_max, 1, ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::Bcast(&m_params.y_min, 1, ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::Bcast(&m_params.y_max, 1, ParallelDescriptor::IOProcessorNumber());
}

std::pair<int,int>
WarpXLaserProfiles::FromFileLaserProfile::find_left_right_time_indices(amrex::Real t) const
{
    const auto t_min = m_params.t_min;
    const auto t_max = m_params.t_max;
    const auto temp_idx_t_right = static_cast<int>(std::ceil( (m_params.nt-1)*(t-t_min)/(t_max-t_min)));
    const int idx_t_right = max(min(temp_idx_t_right, m_params.nt-1),1);
    return std::make_pair(idx_t_right-1, idx_t_right);
}

void
WarpXLaserProfiles::FromFileLaserProfile::read_data_t_chunk (int t_begin, int t_end)
{
#ifdef WARPX_USE_OPENPMD
    //Indices of the first and last timestep to read
    auto const i_first = static_cast<long unsigned int>(max(0, t_begin));
    auto const i_last = static_cast<long unsigned int>(min(t_end-1, m_params.nt-1));
    amrex::Print() << Utils::TextMsg::Info(
        "Reading [" + std::to_string(i_first) + ", " + std::to_string(i_last) +
            "] data chunk from " + m_params.lasy_file_name);
    const auto data_size =
        (m_params.file_in_cartesian_geom==0)?
        (m_params.n_rz_azimuthal_components*(i_last-i_first+1)*m_params.nr) :
        (i_last-i_first+1)*m_params.nx*m_params.ny;
    m_params.E_lasy_data.resize(data_size);
    Vector<Complex> h_E_lasy_data(m_params.E_lasy_data.size());
    if(ParallelDescriptor::IOProcessor()){
        auto series = io::Series(m_params.lasy_file_name, io::Access::READ_ONLY);
        auto i = series.iterations[0];
        auto E = i.meshes["laserEnvelope"];
        auto E_laser = E[io::RecordComponent::SCALAR];
        openPMD:: Extent full_extent = E_laser.getExtent();
        if (m_params.file_in_cartesian_geom==0) {
            const openPMD::Extent read_extent = { full_extent[0], (i_last - i_first + 1), full_extent[2]};
            auto r_data = E_laser.loadChunk< std::complex<double> >(io::Offset{ 0, i_first,  0}, read_extent);
            const auto read_size = (i_last - i_first + 1)*m_params.nr;
            series.flush();
            for (int m=0; m<m_params.n_rz_azimuthal_components; m++){
                for (auto j=0u; j<read_size; j++) {
                    h_E_lasy_data[j+m*read_size] = Complex{
                        static_cast<amrex::Real>(r_data.get()[j+m*read_size].real()),
                        static_cast<amrex::Real>(r_data.get()[j+m*read_size].imag())};
                }
            }
        } else{
            const openPMD::Extent read_extent = {(i_last - i_first + 1), full_extent[1], full_extent[2]};
            auto x_data = E_laser.loadChunk< std::complex<double> >(io::Offset{i_first, 0, 0}, read_extent);
            const auto read_size = (i_last - i_first + 1)*m_params.nx*m_params.ny;
            series.flush();
            for (auto j=0u; j<read_size; j++) {
                h_E_lasy_data[j] = Complex{
                    static_cast<amrex::Real>(x_data.get()[j].real()),
                    static_cast<amrex::Real>(x_data.get()[j].imag())};
            }
        }
    }
    //Broadcast E_lasy_data
    ParallelDescriptor::Bcast(h_E_lasy_data.dataPtr(),
        h_E_lasy_data.size(), ParallelDescriptor::IOProcessorNumber());
    Gpu::copyAsync(Gpu::hostToDevice,h_E_lasy_data.begin(),h_E_lasy_data.end(),m_params.E_lasy_data.begin());
    Gpu::synchronize();
    //Update first and last indices
    m_params.first_time_index = static_cast<int>(i_first);
    m_params.last_time_index = static_cast<int>(i_last);
#else
    amrex::ignore_unused(t_begin, t_end);
#endif
}

void
WarpXLaserProfiles::FromFileLaserProfile::read_binary_data_t_chunk (int t_begin, int t_end)
{
    amrex::Print() << Utils::TextMsg::Info(
        "Reading [" + std::to_string(t_begin) + ", " + std::to_string(t_end) +
            "] data chunk from " + m_params.binary_file_name);

    //Indices of the first and last timestep to read
    auto i_first = max(0, t_begin);
    auto i_last = min(t_end-1, m_params.nt-1);
    const int data_size = (i_last-i_first+1)*m_params.nx*m_params.ny;
    m_params.E_binary_data.resize(data_size);
    Vector<Real> h_E_binary_data(m_params.E_binary_data.size());
    if(ParallelDescriptor::IOProcessor()){
        //Read data chunk
        std::ifstream inp(m_params.binary_file_name, std::ios::binary);
        if(!inp) { WARPX_ABORT_WITH_MESSAGE("Failed to open binary file"); }
        inp.exceptions(std::ios_base::failbit | std::ios_base::badbit);
#if (defined(WARPX_DIM_3D))
        auto skip_amount = 1 +
        3*sizeof(uint32_t) +
        2*sizeof(double) +
        2*sizeof(double) +
        2*sizeof(double) +
        sizeof(double)*t_begin*m_params.nx*m_params.ny;
#else
        auto skip_amount = 1 +
        3*sizeof(uint32_t) +
        2*sizeof(double) +
        2*sizeof(double) +
        1*sizeof(double) +
        sizeof(double)*t_begin*m_params.nx*m_params.ny;
#endif
        inp.seekg(static_cast<std::streamoff>(skip_amount));
        if(!inp) { WARPX_ABORT_WITH_MESSAGE("Failed to read field data from binary file"); }
        const int read_size = (i_last - i_first + 1)*
            m_params.nx*m_params.ny;
        Vector<double> buf_e(read_size);
        inp.read(reinterpret_cast<char*>(buf_e.dataPtr()), static_cast<std::streamsize>(read_size*sizeof(double)));
        if(!inp) { WARPX_ABORT_WITH_MESSAGE("Failed to read field data from binary file"); }
        std::transform(buf_e.begin(), buf_e.end(), h_E_binary_data.begin(),
            [](auto x) {return static_cast<amrex::Real>(x);} );
    }

    //Broadcast E_binary_data
    ParallelDescriptor::Bcast(h_E_binary_data.dataPtr(),
        h_E_binary_data.size(), ParallelDescriptor::IOProcessorNumber());

    Gpu::copyAsync(Gpu::hostToDevice,h_E_binary_data.begin(),h_E_binary_data.end(),m_params.E_binary_data.begin());
    Gpu::synchronize();

    //Update first and last indices
    m_params.first_time_index = static_cast<int>(i_first);
    m_params.last_time_index = static_cast<int>(i_last);
}

void
WarpXLaserProfiles::FromFileLaserProfile::internal_fill_amplitude_uniform_cartesian (
    const int idx_t_left,
    const int np,
    Real const * AMREX_RESTRICT const Xp, Real const * AMREX_RESTRICT const Yp,
    Real t, Real * AMREX_RESTRICT const amplitude) const
{
    // Copy member variables to tmp copies
    // and get pointers to underlying data for GPU.
    const amrex::Real omega_t = 2._rt*MathConst::pi*PhysConst::c*t/m_common_params.wavelength;
    const Complex exp_omega_t = Complex{ std::cos(-omega_t), std::sin(-omega_t) };
    const auto tmp_x_min = m_params.x_min;
    const auto tmp_x_max = m_params.x_max;
    const auto tmp_y_min = m_params.y_min;
    const auto tmp_y_max = m_params.y_max;
    const auto tmp_nx = m_params.nx;
    const auto tmp_ny = m_params.ny;
    const auto *const p_E_lasy_data = m_params.E_lasy_data.dataPtr();
    const auto tmp_idx_first_time = m_params.first_time_index;
    const int idx_t_right = idx_t_left+1;
    const auto t_left = idx_t_left*
        (m_params.t_max-m_params.t_min)/(m_params.nt-1) +
        m_params.t_min;
    const auto t_right = idx_t_right*
        (m_params.t_max-m_params.t_min)/(m_params.nt-1) +
        m_params.t_min;
    // Loop through the macroparticle to calculate the proper amplitude
    amrex::ParallelFor(
    np,
    [=] AMREX_GPU_DEVICE (int i) {
        //Amplitude is zero if we are out of bounds
        if (Xp[i] <= tmp_x_min || Xp[i] >= tmp_x_max){
            amplitude[i] = 0.0_rt;
            return;
        }
        if (Yp[i] <= tmp_y_min || Yp[i] >= tmp_y_max){
            amplitude[i] = 0.0_rt;
            return;
        }
        //Find indices and coordinates along x
        const int temp_idx_x_right = static_cast<int>(
            std::ceil((tmp_nx-1)*(Xp[i]- tmp_x_min)/(tmp_x_max-tmp_x_min)));
        const int idx_x_right =
            max(min(temp_idx_x_right,tmp_nx-1),static_cast<int>(1));
        const int idx_x_left = idx_x_right - 1;
        const auto x_0 =
            idx_x_left*(tmp_x_max-tmp_x_min)/(tmp_nx-1) + tmp_x_min;
        const auto x_1 =
            idx_x_right*(tmp_x_max-tmp_x_min)/(tmp_nx-1) + tmp_x_min;
        //Find indices and coordinates along y
        const int temp_idx_y_right = static_cast<int>(
            std::ceil((tmp_ny-1)*(Yp[i]- tmp_y_min)/(tmp_y_max-tmp_y_min)));
        const int idx_y_right =
            max(min(temp_idx_y_right,tmp_ny-1),static_cast<int>(1));
        const int idx_y_left = idx_y_right - 1;
        const auto y_0 =
            idx_y_left*(tmp_y_max-tmp_y_min)/(tmp_ny-1) + tmp_y_min;
        const auto y_1 =
            idx_y_right*(tmp_y_max-tmp_y_min)/(tmp_ny-1) + tmp_y_min;
        //Interpolate amplitude
        const auto idx = [=](int i_interp, int j_interp, int k_interp){
            return
                (i_interp-tmp_idx_first_time)*tmp_nx*tmp_ny+
                j_interp*tmp_nx + k_interp;
        };
        const Complex val = utils::algorithms::trilinear_interp(
            t_left, t_right,
            x_0, x_1,
            y_0, y_1,
            p_E_lasy_data[idx(idx_t_left, idx_y_left, idx_x_left)],
            p_E_lasy_data[idx(idx_t_left, idx_y_right, idx_x_left)],
            p_E_lasy_data[idx(idx_t_left, idx_y_left, idx_x_right)],
            p_E_lasy_data[idx(idx_t_left, idx_y_right, idx_x_right)],
            p_E_lasy_data[idx(idx_t_right, idx_y_left, idx_x_left)],
            p_E_lasy_data[idx(idx_t_right, idx_y_right, idx_x_left)],
            p_E_lasy_data[idx(idx_t_right, idx_y_left, idx_x_right)],
            p_E_lasy_data[idx(idx_t_right, idx_y_right, idx_x_right)],
            t, Xp[i], Yp[i]);
            // The interpolated amplitude was only the envelope.
            // Here we add the laser oscillations.
            amplitude[i] = (val*exp_omega_t).real();
        }
    );
}

void
WarpXLaserProfiles::FromFileLaserProfile::internal_fill_amplitude_uniform_cylindrical (
    const int idx_t_left,
    const int np,
    Real const * AMREX_RESTRICT const Xp, Real const * AMREX_RESTRICT const Yp,
    Real t, Real * AMREX_RESTRICT const amplitude) const
{
    // Copy member variables to tmp copies
    // and get pointers to underlying data for GPU.
    const amrex::Real omega_t = 2._rt*MathConst::pi*PhysConst::c*t/m_common_params.wavelength;
    const Complex exp_omega_t = Complex{ std::cos(-omega_t), std::sin(-omega_t) };
    const auto tmp_r_min = m_params.r_min;
    const auto tmp_r_max = m_params.r_max;
    const auto tmp_nr = m_params.nr;
    const auto tmp_n_rz_azimuthal_components = m_params.n_rz_azimuthal_components;
    const auto *const p_E_lasy_data = m_params.E_lasy_data.dataPtr();
    const auto tmp_idx_first_time = m_params.first_time_index;
    const int idx_t_right = idx_t_left+1;
    const auto t_left = idx_t_left*
        (m_params.t_max-m_params.t_min)/(m_params.nt-1) +
        m_params.t_min;
    const auto t_right = idx_t_right*
        (m_params.t_max-m_params.t_min)/(m_params.nt-1) +
        m_params.t_min;
    const auto current_time_chunk_size = m_params.last_time_index - m_params.first_time_index +1;

    // Loop through the macroparticle to calculate the proper amplitude
    amrex::ParallelFor(
    np,
    [=] AMREX_GPU_DEVICE (int i) {
        auto Rp_i = std::sqrt(Xp[i] * Xp[i] + Yp[i] * Yp[i]);
        //Amplitude is zero if we are out of bounds
        if (Rp_i <= tmp_r_min || Rp_i >= tmp_r_max){
            amplitude[i] = 0.0_rt;
            return;
        }
        //Find indices and coordinates along x
        const int temp_idx_r_right = static_cast<int>(
            std::ceil((tmp_nr-1)*(Rp_i- tmp_r_min)/(tmp_r_max-tmp_r_min)));
        const int idx_r_right =
            max(min(temp_idx_r_right,tmp_nr-1),static_cast<int>(1));
        const int idx_r_left = idx_r_right - 1;
        const auto r_0 =
            idx_r_left*(tmp_r_max-tmp_r_min)/(tmp_nr-1) + tmp_r_min;
        const auto r_1 =
            idx_r_right*(tmp_r_max-tmp_r_min)/(tmp_nr-1) + tmp_r_min;

        const auto idx = [=](int im, int i_interp, int j_interp){
            return
                    im*current_time_chunk_size*tmp_nr+(i_interp-tmp_idx_first_time)*tmp_nr+
                    j_interp;
        };
        amrex::Real costheta;
        amrex::Real sintheta;
        if (Rp_i > 0.) {
            costheta = Xp[i]/Rp_i;
            sintheta = Yp[i]/Rp_i;
        } else {
            costheta = 1._rt;
            sintheta = 0._rt;
        }
        Complex val = 0;
        Complex fact = Complex{costheta, sintheta};

        // azimuthal mode 0
        val += utils::algorithms::bilinear_interp(
            t_left, t_right,
            r_0, r_1,
            p_E_lasy_data[idx(0, idx_t_left, idx_r_left)],
            p_E_lasy_data[idx(0, idx_t_left, idx_r_right)],
            p_E_lasy_data[idx(0, idx_t_right, idx_r_left)],
            p_E_lasy_data[idx(0, idx_t_right, idx_r_right)],
            t, Rp_i);

        // higher modes
        for (int m=1 ; m <= tmp_n_rz_azimuthal_components/2; m++) {
            val += utils::algorithms::bilinear_interp(
                t_left, t_right,
                r_0, r_1,
                p_E_lasy_data[idx(2*m-1, idx_t_left, idx_r_left)],
                p_E_lasy_data[idx(2*m-1, idx_t_left, idx_r_right)],
                p_E_lasy_data[idx(2*m-1, idx_t_right, idx_r_left)],
                p_E_lasy_data[idx(2*m-1, idx_t_right, idx_r_right)],
                t, Rp_i)*(fact.real()) +
                utils::algorithms::bilinear_interp(
                t_left, t_right,
                r_0, r_1,
                p_E_lasy_data[idx(2*m, idx_t_left, idx_r_left)],
                p_E_lasy_data[idx(2*m, idx_t_left, idx_r_right)],
                p_E_lasy_data[idx(2*m, idx_t_right, idx_r_left)],
                p_E_lasy_data[idx(2*m, idx_t_right, idx_r_right)],
                t, Rp_i)*(fact.imag()) ;
            fact = fact*Complex{costheta, sintheta};
        }
        amplitude[i] = (val*exp_omega_t).real();
        }
    );
}

void
WarpXLaserProfiles::FromFileLaserProfile::internal_fill_amplitude_uniform_binary (
    const int idx_t_left,
    const int np,
    Real const * AMREX_RESTRICT const Xp, Real const * AMREX_RESTRICT const Yp,
    Real t, Real * AMREX_RESTRICT const amplitude) const
{
    // Copy member variables to tmp copies
    // and get pointers to underlying data for GPU.
    const auto tmp_e_max = m_common_params.e_max;
    const auto tmp_x_min = m_params.x_min;
    const auto tmp_x_max = m_params.x_max;
#if (defined(WARPX_DIM_3D) || (defined WARPX_DIM_RZ))
    const auto tmp_y_min = m_params.y_min;
    const auto tmp_y_max = m_params.y_max;
    const auto tmp_ny = m_params.ny;
#endif
    const auto tmp_nx = m_params.nx;
    const auto *const p_E_binary_data = m_params.E_binary_data.dataPtr();
    const auto tmp_idx_first_time = m_params.first_time_index;
    const int idx_t_right = idx_t_left+1;
    const auto t_left = idx_t_left*
        (m_params.t_max-m_params.t_min)/(m_params.nt-1) +
        m_params.t_min;
    const auto t_right = idx_t_right*
        (m_params.t_max-m_params.t_min)/(m_params.nt-1) +
        m_params.t_min;

#if (defined WARPX_DIM_1D_Z)
    WARPX_ABORT_WITH_MESSAGE("WarpXLaserProfiles::FromFileLaserProfile Not implemented for 1D");
#endif

    // Loop through the macroparticle to calculate the proper amplitude
    amrex::ParallelFor(
    np,
    [=] AMREX_GPU_DEVICE (int i) {
        //Amplitude is zero if we are out of bounds
        if (Xp[i] <= tmp_x_min || Xp[i] >= tmp_x_max){
            amplitude[i] = 0.0_rt;
            return;
        }
#if (defined(WARPX_DIM_3D) || (defined WARPX_DIM_RZ))
        if (Yp[i] <= tmp_y_min || Yp[i] >= tmp_y_max){
            amplitude[i] = 0.0_rt;
            return;
        }
#endif
        //Find indices and coordinates along x
        const int temp_idx_x_right = static_cast<int>(
            std::ceil((tmp_nx-1)*(Xp[i]- tmp_x_min)/(tmp_x_max-tmp_x_min)));
        const int idx_x_right =
            max(min(temp_idx_x_right,tmp_nx-1),static_cast<int>(1));
        const int idx_x_left = idx_x_right - 1;
        const auto x_0 =
            idx_x_left*(tmp_x_max-tmp_x_min)/(tmp_nx-1) + tmp_x_min;
        const auto x_1 =
            idx_x_right*(tmp_x_max-tmp_x_min)/(tmp_nx-1) + tmp_x_min;

#if (defined(WARPX_DIM_3D) || (defined WARPX_DIM_RZ))
        //Find indices and coordinates along y
        const int temp_idx_y_right = static_cast<int>(
            std::ceil((tmp_ny-1)*(Yp[i]- tmp_y_min)/(tmp_y_max-tmp_y_min)));
        const int idx_y_right =
            max(min(temp_idx_y_right,tmp_ny-1),static_cast<int>(1));
        const int idx_y_left = idx_y_right - 1;
        const auto y_0 =
            idx_y_left*(tmp_y_max-tmp_y_min)/(tmp_ny-1) + tmp_y_min;
        const auto y_1 =
            idx_y_right*(tmp_y_max-tmp_y_min)/(tmp_ny-1) + tmp_y_min;
        //Interpolate amplitude
        const auto idx = [=](int i_interp, int j_interp, int k_interp){
            return
                (i_interp-tmp_idx_first_time)*tmp_nx*tmp_ny+
                j_interp*tmp_ny + k_interp;
        };
        amplitude[i] = utils::algorithms::trilinear_interp(
            t_left, t_right,
            x_0, x_1,
            y_0, y_1,
            p_E_binary_data[idx(idx_t_left, idx_x_left, idx_y_left)],
            p_E_binary_data[idx(idx_t_left, idx_x_left, idx_y_right)],
            p_E_binary_data[idx(idx_t_left, idx_x_right, idx_y_left)],
            p_E_binary_data[idx(idx_t_left, idx_x_right, idx_y_right)],
            p_E_binary_data[idx(idx_t_right, idx_x_left, idx_y_left)],
            p_E_binary_data[idx(idx_t_right, idx_x_left, idx_y_right)],
            p_E_binary_data[idx(idx_t_right, idx_x_right, idx_y_left)],
            p_E_binary_data[idx(idx_t_right, idx_x_right, idx_y_right)],
            t, Xp[i], Yp[i])*tmp_e_max;

#elif defined(WARPX_DIM_XZ)
        //Interpolate amplitude
        const auto idx = [=](int i_interp, int j_interp){
            return (i_interp-tmp_idx_first_time) * tmp_nx + j_interp;
        };
        amplitude[i] = utils::algorithms::bilinear_interp(
            t_left, t_right,
            x_0, x_1,
            p_E_binary_data[idx(idx_t_left, idx_x_left)],
            p_E_binary_data[idx(idx_t_left, idx_x_right)],
            p_E_binary_data[idx(idx_t_right, idx_x_left)],
            p_E_binary_data[idx(idx_t_right, idx_x_right)],
            t, Xp[i])*tmp_e_max;
        amrex::ignore_unused(Yp);
#else
        // TODO: implement WARPX_DIM_1D_Z
        amrex::ignore_unused(x_0, x_1, tmp_e_max, p_E_binary_data, tmp_idx_first_time,
                             t_left, t_right, Xp, Yp, t, idx_x_left);
#endif
        }
    );
}
