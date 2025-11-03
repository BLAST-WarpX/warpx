/* Copyright 2023 Luca Fedeli
 *
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_EXTERNAL_FIELD_H_
#define WARPX_EXTERNAL_FIELD_H_

#include "ExternalField_fwd.H"

#include <ablastr/math/LinearInterpolation.H>

#include <AMReX_Array.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Parser.H>
#include <AMReX_RealVect.H>
#include <AMReX_TableData.H>

#include <memory>
#include <string>

enum class ExternalFieldType
{
    default_zero,
    constant,
    parse_ext_grid_function,
    read_from_file,
    load_from_python
};

/**
 * \brief Struct to store data related to external electromagnetic fields
 * (flags, field values, and field parsers)
 */
struct ExternalFieldParams
{

    /**
    * \brief The constructor reads and stores the parameters related to the external fields.
    * "pp_warpx" must point at the "warpx" parameter group in the inputfile.
    */
    ExternalFieldParams(const amrex::ParmParse& pp_warpx);

    //! Initial electric field on the grid
    amrex::GpuArray<amrex::Real,3> E_external_grid = {0,0,0};
    //! Initial magnetic field on the grid
    amrex::GpuArray<amrex::Real,3> B_external_grid = {0,0,0};

    //! Initialization type for external magnetic field on the grid
    ExternalFieldType B_ext_grid_type = ExternalFieldType::default_zero;
    //! Initialization type for external electric field on the grid
    ExternalFieldType E_ext_grid_type = ExternalFieldType::default_zero;

    //! User-defined parser to initialize x-component of the magnetic field on the grid
    std::unique_ptr<amrex::Parser> Bxfield_parser;
    //! User-defined parser to initialize y-component of the magnetic field on the grid
    std::unique_ptr<amrex::Parser> Byfield_parser;
    //! User-defined parser to initialize z-component of the magnetic field on the grid
    std::unique_ptr<amrex::Parser> Bzfield_parser;
    //! User-defined parser to initialize x-component of the electric field on the grid
    std::unique_ptr<amrex::Parser> Exfield_parser;
    //! User-defined parser to initialize y-component of the electric field on the grid
    std::unique_ptr<amrex::Parser> Eyfield_parser;
    //! User-defined parser to initialize z-component of the electric field on the grid
    std::unique_ptr<amrex::Parser> Ezfield_parser;

     //! Path of the file where external fields are stored
    std::string external_fields_path;
};

struct ExternalFieldView
{
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real operator() (amrex::RealVect const& pos) const
    {
#if defined(AMREX_USE_GPU)
        AMREX_IF_ON_HOST((amrex::Abort("For GPU builds, ExternalFieldView only works on device. We could make it work on host if necessary.");))
#endif

        // Get index of the external field array
        AMREX_D_TERM(int const i0 = std::floor( (pos[0]-offset[0])/dx[0] );,
                     int const i1 = std::floor( (pos[1]-offset[1])/dx[1] );,
                     int const i2 = std::floor( (pos[2]-offset[2])/dx[2] ));

        // Get coordinates of external grid point
        AMREX_D_TERM(amrex::Real const xx0 = offset[0] + i0 * dx[0];,
                     amrex::Real const xx1 = offset[1] + i1 * dx[1];,
                     amrex::Real const xx2 = offset[2] + i2 * dx[2]);

#if (AMREX_SPACEDIM == 1)

        if (i0 < table.begin || i0 >= table.end-1) {
            return 0;
        } else {
            return static_cast<amrex::Real>(
                ablastr::math::linear_interp<double>(
                    xx0, xx0+dx[0], table(i0), table(i0+1), pos[0]));
        }

#elif (AMREX_SPACEDIM == 2)

        if (i0 < table.begin[0] || i0 >= table.end[0]-1 ||
            i1 < table.begin[1] || i1 >= table.end[1]-1) {
            return 0;
        } else {
            return static_cast<amrex::Real>(
                ablastr::math::bilinear_interp<double>(
                    xx0, xx0+dx[0], xx1, xx1+dx[1],
                    table(i0  ,i1  ),
                    table(i0  ,i1+1),
                    table(i0+1,i1  ),
                    table(i0+1,i1+1),
                    pos[0], pos[1]));
        }

#elif (AMREX_SPACEDIM == 3)

        if (i0 < table.begin[0] || i0 >= table.end[0]-1 ||
            i1 < table.begin[1] || i1 >= table.end[1]-1 ||
            i2 < table.begin[2] || i2 >= table.end[2]-1) {
            return 0;
        } else {
            return static_cast<amrex::Real>(
                ablastr::math::trilinear_interp<double>(
                    xx0, xx0+dx[0], xx1, xx1+dx[1], xx2, xx2+dx[2],
                    table(i0  ,i1  ,i2  ),
                    table(i0  ,i1  ,i2+1),
                    table(i0  ,i1+1,i2  ),
                    table(i0  ,i1+1,i2+1),
                    table(i0+1,i1  ,i2  ),
                    table(i0+1,i1  ,i2+1),
                    table(i0+1,i1+1,i2  ),
                    table(i0+1,i1+1,i2+1),
                    pos[0], pos[1], pos[2]));
        }
#endif
    }

    amrex::RealVect dx;
    amrex::RealVect offset;
#if (AMREX_SPACEDIM == 1)
    amrex::Table1D<double> table;
#elif (AMREX_SPACEDIM == 2)
    amrex::Table2D<double,amrex::Order::C> table;
#elif (AMREX_SPACEDIM == 3)
    amrex::Table3D<double,amrex::Order::C> table;
#endif
};

class ExternalFieldReader
{
public:
    ExternalFieldReader (std::string const& read_fields_from_path,
                         std::string const& F_name, std::string const& F_component);

    ExternalFieldView const& getView () const { return m_external_field_view; }

private:
    std::shared_ptr<double> m_FC_data_cpu;
    amrex::Gpu::DeviceVector<double> m_FC_data_gpu;
    ExternalFieldView m_external_field_view;
};

#endif //WARPX_EXTERNAL_FIELD_H_
