/* Copyright 2026 Axel Huebl
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_UTILS_WARPXDIM_H_
#define WARPX_UTILS_WARPXDIM_H_

#include <AMReX.H>

#include <string>

namespace warpx
{
    /** Dimensionality of the simulation (geometry.dims)
     *
     * Runtime counterpart of the compile-time WARPX_DIM_* macros. During the
     * transition to runtime-selectable dimensionality, kernels are templated
     * on this enum and instantiate to exactly the code the macros used to
     * select.
     */
    enum class Dim : int
    {
        Z = 0,      //!< geometry.dims = 1          (WARPX_DIM_1D_Z)
        XZ,         //!< geometry.dims = 2          (WARPX_DIM_XZ)
        XYZ,        //!< geometry.dims = 3          (WARPX_DIM_3D)
        RZ,         //!< geometry.dims = RZ         (WARPX_DIM_RZ)
        RCylinder,  //!< geometry.dims = RCYLINDER  (WARPX_DIM_RCYLINDER)
        RSphere,    //!< geometry.dims = RSPHERE    (WARPX_DIM_RSPHERE)
        Runtime     //!< sentinel: "selected at runtime", never a valid kernel instantiation
    };

    /** Dimensionality selected at compile time via the WARPX_DIM_* macros
     *
     * In per-dimension builds, kernel templates default to this value, so
     * existing call sites compile unchanged. In unified (runtime-dimension)
     * builds this is the Dim::Runtime sentinel and kernels must receive their
     * dimension explicitly (see WarpXDimDispatch.H).
     */
#if   defined(WARPX_DIM_1D_Z)
    inline constexpr Dim CompileDim = Dim::Z;
#elif defined(WARPX_DIM_XZ)
    inline constexpr Dim CompileDim = Dim::XZ;
#elif defined(WARPX_DIM_3D)
    inline constexpr Dim CompileDim = Dim::XYZ;
#elif defined(WARPX_DIM_RZ)
    inline constexpr Dim CompileDim = Dim::RZ;
#elif defined(WARPX_DIM_RCYLINDER)
    inline constexpr Dim CompileDim = Dim::RCylinder;
#elif defined(WARPX_DIM_RSPHERE)
    inline constexpr Dim CompileDim = Dim::RSphere;
#elif defined(WARPX_DIM_RUNTIME)
    inline constexpr Dim CompileDim = Dim::Runtime;
#else
#   error WarpX dimension is not correctly set!
#endif

    /** Traits replacing the compile-time WARPX_DIM_* conditions 1:1
     *
     * "x"/"y"/"z" are logical axes: for the radial geometries (RZ, RCYLINDER,
     * RSPHERE), the radial coordinate r takes the place of the x axis, as in
     * the existing per-dimension code.
     * @{
     */
    //! the simulated grid has an x (or r) axis: !WARPX_DIM_1D_Z
    template <Dim D> inline constexpr bool has_x = (D != Dim::Z);
    //! the simulated grid has a y axis: WARPX_DIM_3D
    template <Dim D> inline constexpr bool has_y = (D == Dim::XYZ);
    //! the simulated grid has a z axis: !(WARPX_DIM_RCYLINDER || WARPX_DIM_RSPHERE)
    template <Dim D> inline constexpr bool has_z = (D != Dim::RCylinder && D != Dim::RSphere);
    //! radial geometry: WARPX_DIM_RZ || WARPX_DIM_RCYLINDER || WARPX_DIM_RSPHERE
    template <Dim D> inline constexpr bool is_radial = (D == Dim::RZ || D == Dim::RCylinder || D == Dim::RSphere);
    //! the particle pusher updates the (3D Cartesian) x position: !WARPX_DIM_1D_Z
    template <Dim D> inline constexpr bool pushes_x_3d = has_x<D>;
    //! the particle pusher updates the (3D Cartesian) y position:
    //! particles are pushed in 3D even in the radial geometries
    template <Dim D> inline constexpr bool pushes_y_3d = (D == Dim::XYZ) || is_radial<D>;
    //! the particle pusher updates the (3D Cartesian) z position: !WARPX_DIM_RCYLINDER
    template <Dim D> inline constexpr bool pushes_z_3d = (D != Dim::RCylinder);
    /** @} */

    /** Convert a geometry.dims input value to the corresponding Dim
     *
     * @param dims value of the geometry.dims input parameter ("1", "2", "3",
     *             "RZ", "RCYLINDER" or "RSPHERE")
     * @return the corresponding Dim; aborts on unknown values
     */
    inline Dim
    dim_from_geometry_dims (std::string const& dims)
    {
        if (dims == "1")         { return Dim::Z; }
        if (dims == "2")         { return Dim::XZ; }
        if (dims == "3")         { return Dim::XYZ; }
        if (dims == "RZ")        { return Dim::RZ; }
        if (dims == "RCYLINDER") { return Dim::RCylinder; }
        if (dims == "RSPHERE")   { return Dim::RSphere; }
        amrex::Abort("geometry.dims must be one of 1, 2, 3, RZ, RCYLINDER or RSPHERE, but is '"
                     + dims + "'");
        return Dim::Runtime; // unreachable
    }

    /** Convert a Dim to the corresponding geometry.dims input value */
    inline std::string
    geometry_dims_name (Dim const dim)
    {
        switch (dim)
        {
            case Dim::Z:         return "1";
            case Dim::XZ:        return "2";
            case Dim::XYZ:       return "3";
            case Dim::RZ:        return "RZ";
            case Dim::RCylinder: return "RCYLINDER";
            case Dim::RSphere:   return "RSPHERE";
            default:             return "Runtime";
        }
    }
} // namespace warpx

#endif // WARPX_UTILS_WARPXDIM_H_
