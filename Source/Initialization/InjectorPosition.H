/* Copyright 2019 Axel Huebl, David Grote, Maxence Thevenet
 * Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_INJECTOR_POSITION_H_
#define WARPX_INJECTOR_POSITION_H_

#include "InjectorPosition_fwd.H"

#include <AMReX_Dim3.H>
#include <AMReX_Gpu.H>
#include <AMReX_Utility.H>

// struct whose getPositionUnitBox returns x, y and z for a particle with
// random distribution inside a unit cell.
struct InjectorPositionRandom
{
    [[nodiscard]]
    AMREX_GPU_HOST_DEVICE
    amrex::XDim3
    getPositionUnitBox (int /*i_part*/, amrex::IntVect const /*ref_fac*/,
                        amrex::RandomEngine const& engine) const noexcept
    {
        return amrex::XDim3{amrex::Random(engine), amrex::Random(engine), amrex::Random(engine)};
    }
};

// struct whose getPositionUnitBox returns x, y and z for a particle with
// random distribution on a plane inside a unit cell.
struct InjectorPositionRandomPlane
{
    InjectorPositionRandomPlane (int const& a_dir) noexcept : dir(a_dir) {}

    [[nodiscard]]
    AMREX_GPU_HOST_DEVICE
    amrex::XDim3
    getPositionUnitBox (int /*i_part*/, amrex::IntVect const /*ref_fac*/,
                        amrex::RandomEngine const& engine) const noexcept
    {
        using namespace amrex::literals;
#if ((defined WARPX_DIM_3D) || (defined WARPX_DIM_RZ))
        // In RZ, the 3 components of the `XDim3` vector below correspond to r, theta, z respectively
        if (dir == 0)  { return amrex::XDim3{0._rt, amrex::Random(engine), amrex::Random(engine)}; }
        if (dir == 1)  { return amrex::XDim3{amrex::Random(engine), 0._rt, amrex::Random(engine)}; }
        else           { return amrex::XDim3{amrex::Random(engine), amrex::Random(engine), 0._rt}; }
#elif (defined(WARPX_DIM_XZ))
        // In 2D, the 2 first components of the `XDim3` vector below correspond to x and z
        if (dir == 0) { return amrex::XDim3{0._rt, amrex::Random(engine), 0._rt}; }
        if (dir == 1) { return amrex::XDim3{amrex::Random(engine), amrex::Random(engine), 0._rt}; }
        else          { return amrex::XDim3{amrex::Random(engine), 0._rt, 0._rt }; }
#elif (defined(WARPX_DIM_1D_Z))
        // In 2D, the first components of the `XDim3` vector below correspond to z
        if (dir == 0) { return amrex::XDim3{amrex::Random(engine), 0._rt, 0._rt}; }
        if (dir == 1) { return amrex::XDim3{amrex::Random(engine), 0._rt, 0._rt}; }
        else          { return amrex::XDim3{0._rt, 0._rt, 0._rt}; }
#endif
    }
private:
    int dir;
};

// struct whose getPositionUnitBox returns x, y and z for a particle with
// regular distribution inside a unit cell.
struct InjectorPositionRegular
{
    InjectorPositionRegular (amrex::Dim3 const& a_ppc) noexcept : ppc(a_ppc) {}

    // i_part: particle number within the cell, required to evenly space
    // particles within the cell.
    // ref_fac: the number of particles evenly-spaced within a cell
    // is a_ppc*(ref_fac**AMREX_SPACEDIM).
    [[nodiscard]]
    AMREX_GPU_HOST_DEVICE
    amrex::XDim3
    getPositionUnitBox (int const i_part, amrex::IntVect const ref_fac,
                        amrex::RandomEngine const&) const noexcept
    {
        using namespace amrex;

#if (defined WARPX_DIM_3D)
        int const nx = ref_fac[0]*ppc.x;
        int const ny = ref_fac[1]*ppc.y;
        int const nz = ref_fac[2]*ppc.z;
#elif (defined WARPX_DIM_RZ)
        int const nx = ref_fac[0]*ppc.x;
        int const ny = ref_fac[1]*ppc.y;
        int const nz = ppc.z; // Number of particles in theta ; no refinement
#elif (defined WARPX_DIM_XZ)
        int const nx = ref_fac[0]*ppc.x;
        int const ny = ref_fac[1]*ppc.y;
        int const nz = 1;
#elif (defined WARPX_DIM_1D_Z)
        int const nx = ref_fac[0]*ppc.x;
        int const ny = 1;
        int const nz = 1;
#endif
        int const ix_part = i_part / (ny*nz);  // written this way backward compatibility
        int const iz_part = (i_part-ix_part*(ny*nz)) / ny;
        int const iy_part = (i_part-ix_part*(ny*nz)) - ny*iz_part;
        return XDim3{
            (0.5_rt + ix_part) / nx,
            (0.5_rt + iy_part) / ny,
            (0.5_rt + iz_part) / nz
        };
    }
private:
    amrex::Dim3 ppc;
};

// Base struct for position injector.
// InjectorPosition contains a union (called Object) that holds any one
// instance of:
// - InjectorPositionRandom : to generate random distribution;
// - InjectorPositionRegular: to generate regular distribution.
// The choice is made at runtime, depending in the constructor called.
// This mimics virtual functions.
struct InjectorPosition
{
    // This constructor stores a InjectorPositionRandom in union object.
    InjectorPosition (InjectorPositionRandom* t,
                      amrex::Real a_xmin, amrex::Real a_xmax,
                      amrex::Real a_ymin, amrex::Real a_ymax,
                      amrex::Real a_zmin, amrex::Real a_zmax)
        : type(Type::random),
          object(t),
          xmin(a_xmin), xmax(a_xmax),
          ymin(a_ymin), ymax(a_ymax),
          zmin(a_zmin), zmax(a_zmax)
    { }

    // This constructor stores a InjectorPositionRandomPlane in union object.
    InjectorPosition (InjectorPositionRandomPlane* t,
                      amrex::Real a_xmin, amrex::Real a_xmax,
                      amrex::Real a_ymin, amrex::Real a_ymax,
                      amrex::Real a_zmin, amrex::Real a_zmax,
                      int const& a_dir)
        : type(Type::randomplane),
          object(t, a_dir),
          xmin(a_xmin), xmax(a_xmax),
          ymin(a_ymin), ymax(a_ymax),
          zmin(a_zmin), zmax(a_zmax)
    { }

    // This constructor stores a InjectorPositionRegular in union object.
    InjectorPosition (InjectorPositionRegular* t,
                      amrex::Real a_xmin, amrex::Real a_xmax,
                      amrex::Real a_ymin, amrex::Real a_ymax,
                      amrex::Real a_zmin, amrex::Real a_zmax,
                      amrex::Dim3 const& a_ppc)
        : type(Type::regular),
          object(t, a_ppc),
          xmin(a_xmin), xmax(a_xmax),
          ymin(a_ymin), ymax(a_ymax),
          zmin(a_zmin), zmax(a_zmax)
    { }

    ~InjectorPosition () = default;

    // Explicitly prevent the compiler from generating copy constructors
    // and copy assignment operators.
    InjectorPosition (InjectorPosition const&) = delete;
    InjectorPosition (InjectorPosition&&)      = delete;
    void operator= (InjectorPosition const&)   = delete;
    void operator= (InjectorPosition &&)       = delete;

    // call getPositionUnitBox from the object stored in the union
    // (the union is called Object, and the instance is called object).
    [[nodiscard]]
    AMREX_GPU_HOST_DEVICE
    amrex::XDim3
    getPositionUnitBox (int const i_part, amrex::IntVect const ref_fac,
                        amrex::RandomEngine const& engine) const noexcept
    {
        switch (type)
        {
        case Type::regular:
        {
            return object.regular.getPositionUnitBox(i_part, ref_fac, engine);
        }
        case Type::randomplane:
        {
            return object.randomplane.getPositionUnitBox(i_part, ref_fac, engine);
        }
        default:
        {
            return object.random.getPositionUnitBox(i_part, ref_fac, engine);
        }
        };
    }

    /* \brief Flags whether the point (x, y, z) is inside the plasma region
     *        or on the lower boundary
     * \param x, y, z the point to check
     * \returns bool flag
     */
    [[nodiscard]]
    AMREX_GPU_HOST_DEVICE
    bool
    insideBounds (amrex::Real x, amrex::Real y, amrex::Real z) const noexcept
    {
        return (x < xmax and x >= xmin and
                y < ymax and y >= ymin and
                z < zmax and z >= zmin);
    }

    /* \brief Flags whether the point (x, y, z) is inside the plasma region
     *        or on the lower or upper boundary
     * \param x, y, z the point to check
     * \returns bool flag
     */
    [[nodiscard]]
    AMREX_GPU_HOST_DEVICE
    bool
    insideBoundsInclusive (amrex::Real x, amrex::Real y, amrex::Real z) const noexcept
    {
        return (x <= xmax and x >= xmin and
                y <= ymax and y >= ymin and
                z <= zmax and z >= zmin);
    }

    // bool: whether the region defined by lo and hi overlaps with the plasma region
    [[nodiscard]]
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    bool
    overlapsWith (const amrex::XDim3& lo, const amrex::XDim3& hi) const noexcept
    {
        return (     (xmin <= hi.x) && (xmax >= lo.x)
                  && (ymin <= hi.y) && (ymax >= lo.y)
                  && (zmin <= hi.z) && (zmax >= lo.z) );
    }

private:
    enum struct Type { random, randomplane, regular };
    Type type;

    // An instance of union Object constructs and stores any one of
    // the objects declared (random or regular).
    union Object {
        Object (InjectorPositionRandom*) noexcept : random() {}
        Object (InjectorPositionRandomPlane*, int const& a_dir) noexcept
            : randomplane(a_dir) {}
        Object (InjectorPositionRegular*, amrex::Dim3 const& a_ppc) noexcept
            : regular(a_ppc) {}
        InjectorPositionRandom random;
        InjectorPositionRandomPlane randomplane;
        InjectorPositionRegular regular;
    };
    Object object;

    amrex::Real xmin, xmax;
    amrex::Real ymin, ymax;
    amrex::Real zmin, zmax;
};

#endif //WARPX_INJECTOR_POSITION_H_
