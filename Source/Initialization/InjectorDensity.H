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

#include "Utils/WarpXConst.H"

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_Parser.H>
#include <AMReX_REAL.H>

#include <cmath>
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

// struct whose getDensity returns local density computed from predefined profile.
struct InjectorDensityPredefined
{
    InjectorDensityPredefined (std::string const& a_species_name) noexcept;

    void clear ();

    [[nodiscard]]
    AMREX_GPU_HOST_DEVICE
    amrex::Real
    getDensity (amrex::Real x, amrex::Real y, amrex::Real z) const noexcept
    {
        // Choices for profile are:
        // - parabolic_channel
        switch (profile)
        {
        case Profile::parabolic_channel:
        {
            // These are cast as double to ensure sufficient precision in the
            // initialized profile density profile; without these, single and
            // double precision versions of the executable can show disagreement
            // From testing, it seems that (for at least some setups), it is only
            // necessary to case n0 as double to get good agreement between single
            // and double precision, but just in case, all are cast as double.
            const double z_start   = p[0];
            const double ramp_up   = p[1];
            const double plateau   = p[2];
            const double ramp_down = p[3];
            const double rc        = p[4];
            const double n0        = p[5];
            const double kp = PhysConst::q_e/PhysConst::c
                *std::sqrt( n0/(PhysConst::m_e*PhysConst::ep0) );
            double n;

            // Longitudinal profile, normalized to 1
            if        ((z-z_start)>=0               and
                       (z-z_start)<ramp_up ) {
                n = 0.5*(1.-std::cos(MathConst::pi*(z-z_start)/ramp_up));
            } else if ((z-z_start)>=ramp_up         and
                       (z-z_start)< ramp_up+plateau ) {
                n = 1.;
            } else if ((z-z_start)>=ramp_up+plateau and
                       (z-z_start)< ramp_up+plateau+ramp_down) {
                n = 0.5*(1.+std::cos(MathConst::pi*((z-z_start)-ramp_up-plateau)/ramp_down));
            } else {
                n = 0.;
            }
            // Multiply by transverse profile, and physical density
            n *= n0*(1.+4.*(x*x+y*y)/(kp*kp*rc*rc*rc*rc));
            return static_cast<amrex::Real>(n);
        }
        default:
            amrex::Abort("InjectorDensityPredefined: how did we get here?");
            return amrex::Real(0.0);
        }
    }

private:
    enum struct Profile { null, parabolic_channel };
    Profile profile{Profile::null};
    amrex::GpuArray<amrex::Real,6> p;
};

// Base struct for density injector.
// InjectorDensity contains a union (called Object) that holds any one
// instance of:
// - InjectorDensityConstant  : to generate constant density;
// - InjectorDensityParser    : to generate density from parser;
// - InjectorDensityPredefined: to generate density from predefined profile;
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

    // This constructor stores a InjectorDensityPredefined in union object.
    InjectorDensity (InjectorDensityPredefined* t, std::string const& a_species_name)
        : type(Type::predefined),
          object(t,a_species_name)
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
        case Type::predefined:
        {
            return object.predefined.getDensity(x,y,z);
        }
        default:
        {
            amrex::Abort("InjectorDensity: unknown type");
            return 0.0;
        }
        }
    }

private:
    enum struct Type { constant, predefined, parser };
    Type type;

    // An instance of union Object constructs and stores any one of
    // the objects declared (constant or parser or predefined).
    union Object {
        Object (InjectorDensityConstant*, amrex::Real a_rho) noexcept
            : constant(a_rho) {}
        Object (InjectorDensityParser*, amrex::ParserExecutor<3> const& a_parser) noexcept
            : parser(a_parser) {}
        Object (InjectorDensityPredefined*, std::string const& a_species_name) noexcept
            : predefined(a_species_name) {}
        InjectorDensityConstant   constant;
        InjectorDensityParser     parser;
        InjectorDensityPredefined predefined;
    };
    Object object;
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
