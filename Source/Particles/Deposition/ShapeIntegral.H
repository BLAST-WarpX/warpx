/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_DEPOSITION_SHAPE_INTEGRAL_H_
#define WARPX_DEPOSITION_SHAPE_INTEGRAL_H_

#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>

namespace warpx::deposition
{
    /** Integral of old-minus-new shape through a face. Both shapes have
     * unit total weight. Summing from the nearer support edge avoids erasing
     * a small right-hand tail by cancellation of the interior weights.
     * The two forms differ only by floating-point partition-of-unity error.
     * This does not project charge or relax a continuity residual check.
     *
     * \tparam size Padded shape support length.
     * \param old_shape Shape at the old particle endpoint.
     * \param new_shape Shape at the new particle endpoint.
     * \param face Upper node index included in the left integral.
     * \return Integrated old-minus-new shape through the requested face.
     */
    template <int size>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    double shapeIntegral (double const (&old_shape)[size],
                          double const (&new_shape)[size], int const face) noexcept
    {
        double value = 0.0;
        if (face < (size-1)/2) {
            for (int index = 0; index <= face; ++index) {
                value += old_shape[index] - new_shape[index];
            }
        } else {
            for (int index = size-1; index > face; --index) {
                value += new_shape[index] - old_shape[index];
            }
        }
        return value;
    }
}
#endif
