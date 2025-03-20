/* Copyright 2023 Remi Lehe
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_INJECTOR_FLUX_H_
#define WARPX_INJECTOR_FLUX_H_

#include "Utils/WarpXConst.H"

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_Parser.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <string>

// struct whose getFlux returns constant flux.
struct InjectorFluxConstant
{
    InjectorFluxConstant (amrex::Real a_flux) noexcept : m_flux(a_flux) {}

    [[nodiscard]]
    AMREX_GPU_HOST_DEVICE
    amrex::Real
    getFlux (amrex::Real, amrex::Real, amrex::Real, amrex::Real) const noexcept
    {
        return m_flux;
    }

private:
    amrex::Real m_flux;
};

// struct whose getFlux returns local flux computed from parser.
struct InjectorFluxParser
{
    InjectorFluxParser (amrex::ParserExecutor<4> const& a_parser) noexcept
        : m_parser(a_parser) {}

    [[nodiscard]]
    AMREX_GPU_HOST_DEVICE
    amrex::Real
    getFlux (amrex::Real x, amrex::Real y, amrex::Real z, amrex::Real t) const noexcept
    {
        return m_parser(x,y,z,t);
    }

    amrex::ParserExecutor<4> m_parser;
};

// Base struct for flux injector.
// InjectorFlux contains a union (called Object) that holds any one
// instance of:
// - InjectorFluxConstant  : to generate constant flux;
// - InjectorFluxParser    : to generate flux from parser;
// The choice is made at runtime, depending in the constructor called.
// This mimics virtual functions.
struct InjectorFlux
{
    // This constructor stores a InjectorFluxConstant in union object.
    InjectorFlux (InjectorFluxConstant* t, amrex::Real a_flux)
        : type(Type::constant),
          object(t,a_flux)
    { }

    // This constructor stores a InjectorFluxParser in union object.
    InjectorFlux (InjectorFluxParser* t, amrex::ParserExecutor<4> const& a_parser)
        : type(Type::parser),
          object(t,a_parser)
    { }

    // Explicitly prevent the compiler from generating copy constructors
    // and copy assignment operators.
    InjectorFlux (InjectorFlux const&) = delete;
    InjectorFlux (InjectorFlux&&) = delete;
    void operator= (InjectorFlux const&) = delete;
    void operator= (InjectorFlux &&) = delete;

    // Default destructor
    ~InjectorFlux () = default;

    void clear ()
    {
        switch (type)
        {
        case Type::constant:
        case Type::parser:
        {
            break;
        }
        }
    }

    // call getFlux from the object stored in the union
    // (the union is called Object, and the instance is called object).
    [[nodiscard]]
    AMREX_GPU_HOST_DEVICE
    amrex::Real
    getFlux (amrex::Real x, amrex::Real y, amrex::Real z, amrex::Real t) const noexcept
    {
        switch (type)
        {
        case Type::parser:
        {
            return object.parser.getFlux(x,y,z,t);
        }
        case Type::constant:
        {
            return object.constant.getFlux(x,y,z,t);
        }
        default:
        {
            amrex::Abort("InjectorFlux: unknown type");
            return 0.0;
        }
        }
    }

private:
    enum struct Type { constant, parser };
    Type type;

    // An instance of union Object constructs and stores any one of
    // the objects declared (constant or parser).
    union Object {
        Object (InjectorFluxConstant*, amrex::Real a_flux) noexcept
            : constant(a_flux) {}
        Object (InjectorFluxParser*, amrex::ParserExecutor<4> const& a_parser) noexcept
            : parser(a_parser) {}
        InjectorFluxConstant   constant;
        InjectorFluxParser     parser;
    };
    Object object;
};

// In order for InjectorFlux to be trivially copyable, its destructor
// must be trivial.  So we have to rely on a custom deleter for unique_ptr.
struct InjectorFluxDeleter {
    void operator () (InjectorFlux* p) const {
        if (p) {
            p->clear();
            delete p;
        }
    }
};

#endif //WARPX_INJECTOR_FLUX_H_
