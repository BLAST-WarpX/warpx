/* Copyright 2019 Axel Huebl, Maxence Thevenet, Weiqun Zhang
 * Michael Rowan
 *
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_INJECTOR_DENSITY_H_
#define WARPX_INJECTOR_DENSITY_H_

#include "Initialization/ExternalField.H"

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DataAllocator.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Parser.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <functional>
#include <string>

// struct whose getDensity returns constant density.
struct InjectorDensityConstant
{
    InjectorDensityConstant (amrex::Real a_rho) noexcept : m_rho(a_rho) {}

    [[nodiscard]]
    AMREX_GPU_HOST_DEVICE
    amrex::Real
    getDensity (amrex::Real, amrex::Real, amrex::Real) const noexcept
    {
        return m_rho;
    }

private:
    amrex::Real m_rho;
};

// struct whose getDensity returns local density computed from parser.
struct InjectorDensityParser
{
    InjectorDensityParser (amrex::ParserExecutor<3> const& a_parser) noexcept
        : m_parser(a_parser) {}

    [[nodiscard]]
    AMREX_GPU_HOST_DEVICE
    amrex::Real
    getDensity (amrex::Real x, amrex::Real y, amrex::Real z) const noexcept
    {
        return m_parser(x,y,z);
    }

    amrex::ParserExecutor<3> m_parser;
};

// struct whose getDensity returns density from file.
struct InjectorDensityFromFile
{
    InjectorDensityFromFile (std::string const& a_file_name, amrex::Geometry const& a_geom, bool a_distributed);

    void clear ();

    // This function needs to be called before intializing the density
    // profile, if the openPMD data are distributed. When the openPMD data
    // are loaded in the constructor of this class, we do not know the
    // paricle containers' BoxArray and DistributionMapping yet.
    void prepare (amrex::BoxArray const& grids,
                  amrex::DistributionMapping const& dmap,
                  amrex::IntVect const& ngrow,
                  std::function<amrex::Real(amrex::Real)> const& get_zlab);

    // This function needs to be called before we do continuous injection of
    // particles to inform the injector to load the data we need.
    void prepare (amrex::RealBox const& pbox, int moving_dir, int moving_sign,
                  std::function<amrex::Real(amrex::Real)> const& get_zlab);

    // For distributed data, this function prepares the injector for loading
    // data in this Box. Here `li` is the local index for the Box in a
    // particle container.
    void prepare (int li);

    [[nodiscard]] bool distributed () const;

    [[nodiscard]]
    AMREX_GPU_HOST_DEVICE
    amrex::Real
    getDensity (amrex::Real x, amrex::Real y, amrex::Real z) const noexcept
    {
#if (AMREX_SPACEDIM < 3)
        amrex::ignore_unused(x,y,z);
#endif
#if (AMREX_SPACEDIM == 1)
        return m_external_field_view(amrex::RealVect{z});
#elif defined(WARPX_DIM_RZ)
        return m_external_field_view(amrex::RealVect{std::sqrt(x*x+y*y),z});
#elif defined(WARPX_DIM_XZ)
        return m_external_field_view(amrex::RealVect{x,z});
#else
        return m_external_field_view(amrex::RealVect{x,y,z});
#endif
    }

private:
    ExternalFieldView m_external_field_view;
    ExternalFieldReader* m_external_field_reader = nullptr;
};

// Base struct for density injector.
// InjectorDensity contains a union (called Object) that holds any one
// instance of:
// - InjectorDensityConstant  : to generate constant density;
// - InjectorDensityParser    : to generate density from parser;
// - InjectorDensityFromFile  : to generate density from file;
// The choice is made at runtime, depending in the constructor called.
// This mimics virtual functions.
struct InjectorDensity
{
    // This constructor stores a InjectorDensityConstant in union object.
    InjectorDensity (InjectorDensityConstant* t, amrex::Real a_rho)
        : type(Type::constant),
          object(t,a_rho)
    { }

    // This constructor stores a InjectorDensityParser in union object.
    InjectorDensity (InjectorDensityParser* t, amrex::ParserExecutor<3> const& a_parser)
        : type(Type::parser),
          object(t,a_parser)
    { }

    // This constructor stores a InjectorDensityFromFile in union object.
    InjectorDensity (InjectorDensityFromFile* t, std::string const& a_file_name,
                     amrex::Geometry const& a_geom, bool a_distributed)
        : type(Type::fromfile),
          object(t,a_file_name,a_geom,a_distributed)
    { }

    // Explicitly prevent the compiler from generating copy constructors
    // and copy assignment operators.
    InjectorDensity (InjectorDensity const&) = delete;
    InjectorDensity (InjectorDensity&&) = delete;
    void operator= (InjectorDensity const&) = delete;
    void operator= (InjectorDensity &&) = delete;

    // Default destructor
    ~InjectorDensity () = default;

    void clear ();

    void prepare (amrex::BoxArray const& grids,
                  amrex::DistributionMapping const& dmap,
                  amrex::IntVect const& ngrow,
                  std::function<amrex::Real(amrex::Real)> const& get_zlab);

    void prepare (amrex::RealBox const& pbox, int moving_dir, int moving_sign,
                  std::function<amrex::Real(amrex::Real)> const& get_zlab);

    void prepare (int li, InjectorDensity** inj_rho);

    [[nodiscard]] bool needPreparation () const;

    [[nodiscard]] bool distributed () const;

    // call getDensity from the object stored in the union
    // (the union is called Object, and the instance is called object).
    [[nodiscard]]
    AMREX_GPU_HOST_DEVICE
    amrex::Real
    getDensity (amrex::Real x, amrex::Real y, amrex::Real z) const noexcept
    {
        switch (type)
        {
        case Type::parser:
        {
            return object.parser.getDensity(x,y,z);
        }
        case Type::constant:
        {
            return object.constant.getDensity(x,y,z);
        }
        case Type::fromfile:
        {
            return object.fromfile.getDensity(x,y,z);
        }
        default:
        {
            amrex::Abort("InjectorDensity: unknown type");
            return 0.0;
        }
        }
    }

private:
    enum struct Type { constant, parser, fromfile };
    Type type;

    // An instance of union Object constructs and stores any one of
    // the objects declared (constant or parser or predefined).
    union Object {
        Object (InjectorDensityConstant*, amrex::Real a_rho) noexcept
            : constant(a_rho) {}
        Object (InjectorDensityParser*, amrex::ParserExecutor<3> const& a_parser) noexcept
            : parser(a_parser) {}
        Object (InjectorDensityFromFile*, std::string const& a_file_name,
                amrex::Geometry const& a_geom, bool a_distributed)
            : fromfile(a_file_name, a_geom, a_distributed) {}
        InjectorDensityConstant   constant;
        InjectorDensityParser     parser;
        InjectorDensityFromFile   fromfile;
    };
    Object object;
#if defined(AMREX_USE_OMP) && !defined(AMREX_USE_GPU)
    std::unique_ptr<void,amrex::DataDeleter> inj_rho_data;
    amrex::Vector<InjectorDensity*> inj_rho_omp;
#endif
};

// In order for InjectorDensity to be trivially copyable, its destructor
// must be trivial.  So we have to rely on a custom deleter for unique_ptr.
struct InjectorDensityDeleter {
    void operator () (InjectorDensity* p) const {
        if (p) {
            p->clear();
            delete p;
        }
    }
};

#endif //WARPX_INJECTOR_DENSITY_H_
