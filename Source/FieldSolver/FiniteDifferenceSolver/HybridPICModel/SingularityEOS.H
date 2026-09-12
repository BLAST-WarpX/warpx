/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_SINGULARITYEOS_H_
#define WARPX_SINGULARITYEOS_H_

#include <type_traits>

#ifdef WARPX_USE_SINGULARITY_EOS

#   if defined(_PORTABILITY_HPP_)
#       error "WarpX Singularity-EOS adapter must be included before ports-of-call portability.hpp"
#   endif
#   if defined(Real)
#       error "WarpX Singularity-EOS adapter requires the Real macro to be undefined"
#   endif

// This is WarpX's sole permitted gateway to Singularity-EOS and ports-of-call;
// native vendor headers must not precede it or be mixed with adapted definitions
// across translation units. ports-of-call declares a global Real typedef, so
// contain it while importing Singularity-EOS to avoid colliding with
// AMReX's amrex::Real.
#   define Real WarpxSingularityEosReal
#   include <singularity-eos/eos/eos_spiner_rho_temp.hpp>
#   include <singularity-eos/eos/eos_spiner_common.hpp>
#   include <singularity-utils/fast-math/logs.hpp>
#   include <singularity-utils/sp5/singularity_eos_sp5.hpp>
#   include <singularity-utils/spiner_params.hpp>
#   undef Real

static_assert(std::is_same_v<WarpxSingularityEosReal, double>);

#   if !defined(PORTABILITY_STRATEGY_NONE)
#       error "WarpX table EOS currently requires host-only Singularity-EOS"
#   endif

#endif // WARPX_USE_SINGULARITY_EOS

#endif // WARPX_SINGULARITYEOS_H_
