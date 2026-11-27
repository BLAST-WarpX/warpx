/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_KINETICDIFFUSIONCOLLAR_H_
#define WARPX_RADIATION_KINETICDIFFUSIONCOLLAR_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace warpx::radiation
{
/** 1D host safety bound on the number of cells. Not a physics parameter. */
inline constexpr std::size_t KineticDiffusionCollarMaxCells1D = 1'048'576;

/** Fixed 1D mask labels. This is geometric eligibility only. */
enum class KineticDiffusionCollarClass
{
    Kinetic,
    RetainedKineticCollar,
    BulkDiffusion
};

/** How optical_distance is defined on a valid result.
 *
 * NoKineticInterface is the legitimate all-diffusion sentinel (every distance
 * is +infinity). It is not inferred from infinity alone and is not a failure.
 * DistanceToKineticInterface is a finite walk to a kinetic/diffusion face.
 */
enum class KineticDiffusionCollarDistanceKind
{
    DistanceToKineticInterface,
    NoKineticInterface
};

struct KineticDiffusionCollar1DResult
{
    bool valid = false;
    KineticDiffusionCollarDistanceKind optical_distance_kind =
        KineticDiffusionCollarDistanceKind::DistanceToKineticInterface;
    std::vector<KineticDiffusionCollarClass> classifications;
    std::vector<double> optical_distance;
};

/** Classify a motionless 1D kinetic/diffusion mask by collar optical depth.
 *
 * This only defines conservative geometric eligibility of fixed 1D labels. It
 * does not convert photons, apply kicks, run transport, activate FLD, certify
 * isotropy, or claim a one-scatter closure. There is no hidden threshold:
 * collar_optical_depth is the caller's explicit positive value.
 *
 * For a desired diffusion cell, optical_distance is the integrated normal
 * optical depth from the nearest kinetic/diffusion face to the nearest face of
 * that cell, so the whole cell stays behind the retained kinetic optical
 * layer. The cell is BulkDiffusion iff that minimum is at least
 * collar_optical_depth; otherwise it is RetainedKineticCollar. Kinetic cells
 * have distance zero. Nonperiodic reflecting physical ends do not seed a
 * collar. Periodic wrapping joins runs across the domain ends.
 *
 * A domain with no kinetic face is all BulkDiffusion, with +infinity distances
 * and optical_distance_kind NoKineticInterface. That sentinel is valid output.
 * A domain with only kinetic cells is all Kinetic.
 *
 * Failed results have valid == false and empty classification/distance vectors.
 */
[[nodiscard]] inline KineticDiffusionCollar1DResult
TryClassifyKineticDiffusionCollar1D (
    std::vector<int> const& desired_diffusion,
    std::vector<double> const& opacity,
    double const cell_width,
    double const collar_optical_depth,
    bool const periodic)
{
    KineticDiffusionCollar1DResult failed{};
    if (desired_diffusion.size() != opacity.size())
    {
        return failed;
    }
    std::size_t const n = desired_diffusion.size();
    if (n == 0 || n > KineticDiffusionCollarMaxCells1D)
    {
        return failed;
    }
    if (!std::isfinite(cell_width) || !(cell_width > 0.0))
    {
        return failed;
    }
    if (!std::isfinite(collar_optical_depth) || !(collar_optical_depth > 0.0))
    {
        return failed;
    }

    std::vector<double> tau(n);
    bool any_kinetic = false;
    bool any_diffusion = false;
    for (std::size_t i = 0; i < n; ++i)
    {
        int const mask = desired_diffusion[i];
        if (mask != 0 && mask != 1)
        {
            return failed;
        }
        double const chi = opacity[i];
        if (!std::isfinite(chi) || chi < 0.0)
        {
            return failed;
        }
        double const cell_tau = chi * cell_width;
        if (!std::isfinite(cell_tau) || cell_tau < 0.0)
        {
            return failed;
        }
        if (mask == 1)
        {
            any_diffusion = true;
            if (!(chi > 0.0) || !(cell_tau > 0.0))
            {
                return failed;
            }
        }
        else
        {
            any_kinetic = true;
        }
        tau[i] = cell_tau;
    }

    KineticDiffusionCollar1DResult result;
    result.valid = true;
    if (!any_kinetic)
    {
        // No K/D face exists. +infinity is the no-interface sentinel, not an
        // overflow or a rejected input.
        result.optical_distance_kind =
            KineticDiffusionCollarDistanceKind::NoKineticInterface;
        result.classifications.assign(n, KineticDiffusionCollarClass::BulkDiffusion);
        result.optical_distance.assign(n, std::numeric_limits<double>::infinity());
        return result;
    }
    result.optical_distance_kind =
        KineticDiffusionCollarDistanceKind::DistanceToKineticInterface;
    if (!any_diffusion)
    {
        result.classifications.assign(n, KineticDiffusionCollarClass::Kinetic);
        result.optical_distance.assign(n, 0.0);
        return result;
    }

    std::vector<double> distance(n, std::numeric_limits<double>::infinity());
    auto const accumulate_from = [&] (std::size_t i, double& acc, bool& seen_k)
    {
        if (desired_diffusion[i] == 0)
        {
            seen_k = true;
            acc = 0.0;
            return true;
        }
        if (!seen_k)
        {
            return true;
        }
        distance[i] = std::min(distance[i], acc);
        acc += tau[i];
        return std::isfinite(acc);
    };

    if (periodic)
    {
        std::size_t seed_k = 0;
        for (std::size_t i = 0; i < n; ++i)
        {
            if (desired_diffusion[i] == 0)
            {
                seed_k = i;
                break;
            }
        }
        double acc = 0.0;
        bool seen_k = true;
        std::size_t i = seed_k;
        for (std::size_t step = 0; step < n; ++step)
        {
            i = (i + 1U == n) ? 0U : (i + 1U);
            if (!accumulate_from(i, acc, seen_k))
            {
                return failed;
            }
        }
        acc = 0.0;
        seen_k = true;
        i = seed_k;
        for (std::size_t step = 0; step < n; ++step)
        {
            i = (i == 0U) ? (n - 1U) : (i - 1U);
            if (!accumulate_from(i, acc, seen_k))
            {
                return failed;
            }
        }
    }
    else
    {
        double acc = 0.0;
        bool seen_k = false;
        for (std::size_t i = 0; i < n; ++i)
        {
            if (!accumulate_from(i, acc, seen_k))
            {
                return failed;
            }
        }
        acc = 0.0;
        seen_k = false;
        for (std::size_t i = n; i-- > 0U;)
        {
            if (!accumulate_from(i, acc, seen_k))
            {
                return failed;
            }
        }
    }

    result.classifications.resize(n);
    result.optical_distance.resize(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        if (desired_diffusion[i] == 0)
        {
            result.classifications[i] = KineticDiffusionCollarClass::Kinetic;
            result.optical_distance[i] = 0.0;
            continue;
        }
        double const d = distance[i];
        if (!std::isfinite(d) || d < 0.0)
        {
            return failed;
        }
        result.optical_distance[i] = d;
        result.classifications[i] = (d >= collar_optical_depth)
            ? KineticDiffusionCollarClass::BulkDiffusion
            : KineticDiffusionCollarClass::RetainedKineticCollar;
    }
    return result;
}
} // namespace warpx::radiation

#endif
