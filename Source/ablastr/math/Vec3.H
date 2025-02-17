/* Copyright 2025 Luca Fedeli
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

 #ifndef ABLASTR_MATH_VEC3_H_
 #define ABLASTR_MATH_VEC3_H_

#include <AMReX_Array.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>

#include <cmath>

namespace ablastr::math
{
    //This file contains functions to perform operations on 3-vectors
    // (norm, scalar multiplication, vector and cross product...)
    template <typename RealType>
    using vec3 = amrex::GpuArray<RealType, 3>;

    /**
    * This function returns the squared norm of a 3-vector
    *
    * @tparam RealType the floating point type to be used
    * @param[in] the vector for which the squared norm has to be calculated
    * @return the squared norm
    */
    template <typename RealType>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    RealType norm_square(const vec3<RealType>& vec) noexcept
    {
        return vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2];
    }

    /**
    * This function returns the norm of a 3-vector
    *
    * @tparam RealType the floating point type to be used
    * @param[in] the vector for which the norm has to be calculated
    * @return the norm
    */
    template <typename RealType>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    RealType norm(const vec3<RealType>& vec) noexcept
    {
        return std::sqrt(norm_square(vec));
    }

    /**
    * This function returns the dot product between two 3-vectors
    *
    * @tparam RealType the floating point type to be used
    * @param[in] va the first vector
    * @param[in] vb the second vector
    * @return the dot product between va and vb
    */
    template <typename RealType>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    RealType dot(const vec3<RealType>& va, const vec3<RealType>& vb) noexcept
    {
        return va[0]*vb[0] + va[1]*vb[1] + va[2]*vb[2];
    }

    /**
    * This function returns the cross product between two 3-vectors
    *
    * @tparam RealType the floating point type to be used
    * @param[in] va the first vector
    * @param[in] vb the second vector
    * @return the cross product between va and vb
    */
    template <typename RealType>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    vec3<RealType> cross(const vec3<RealType>& va, const vec3<RealType>& vb) noexcept
    {
        return {
            va[1]*vb[2] - va[2]*vb[1],
            va[2]*vb[0] - va[0]*vb[2],
            va[0]*vb[1] - va[1]*vb[0]};
    }

    /**
    * This function overloads the * operator for a vector and a scalar
    *
    * @tparam RealType the floating point type to be used
    * @param[in] v the vector
    * @param[in] s the scalar
    * @return the scalar product between v and s
    */
    template <typename RealType>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    vec3<RealType> operator*(const vec3<RealType>& v, const RealType s) noexcept
    {
        return {v[0]*s, v[1]*s, v[2]*s};
    }

    /**
    * This function overloads the * operator for a scalar and a vector
    *
    * @tparam RealType the floating point type to be used
    * @param[in] s the scalar
    * @param[in] v the vector
    * @return the scalar product between s and v
    */
    template <typename RealType>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    vec3<RealType> operator*(const RealType s, const vec3<RealType>& v) noexcept
    {
        return {s*v[0], s*v[1], s*v[2]};
    }

    /**
    * This function overloads the / operator for a vector and a scalar
    *
    * @tparam RealType the floating point type to be used
    * @param[in] v the vector
    * @param[in] s the scalar
    * @return v/s
    */
    template <typename RealType>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    vec3<RealType> operator/(const vec3<RealType>& v, const RealType s) noexcept
    {
       return {v[0]/s, v[1]/s, v[2]/s};
    }

    /**
    * This function overloads the + operator for two vectors
    *
    * @tparam RealType the floating point type to be used
    * @param[in] va the first vector
    * @param[in] vb the second vector
    * @return va+vb
    */
    template <typename RealType>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    vec3<RealType> operator+(const vec3<RealType>& va, const vec3<RealType>& vb) noexcept
    {
       return {va[0] + vb[0], va[1] + vb[1], va[2] + vb[2]};
    }

    /**
    * This function overloads the - operator for two vectors
    *
    * @tparam RealType the floating point type to be used
    * @param[in] va the first vector
    * @param[in] vb the second vector
    * @return va-vb
    */
    template <typename RealType>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    vec3<RealType> operator-(const vec3<RealType>& va, const vec3<RealType>& vb) noexcept
    {
        return {va[0] - vb[0], va[1] - vb[1], va[2] - vb[2]};
    }
}

#endif //ABLASTR_MATH_VEC3_H_
