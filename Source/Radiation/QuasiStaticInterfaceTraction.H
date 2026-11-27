/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_RADIATION_QUASI_STATIC_INTERFACE_TRACTION_H_
#define WARPX_RADIATION_QUASI_STATIC_INTERFACE_TRACTION_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace warpx::radiation
{

//! Classification of a control-volume face in the reduced quasi-static model.
enum class QuasiStaticInterfaceFaceKind
{
    RayBoundary,
    IsotropicInterior
};

//! One geometric face of a complete 1D spatial control volume.
struct QuasiStaticInterfaceFace
{
    //! Stable geometric identifier; both faces supplied to one call must differ.
    std::uint64_t face_id = 0;

    //! Outward-normal sign along z; the unit outward normal is side*ez.
    int side = 0;

    //! Selects the exclusive contribution used for this face.
    QuasiStaticInterfaceFaceKind kind = QuasiStaticInterfaceFaceKind::RayBoundary;

    //! Integrated incoming photon inventory [E,cpx,cpy,cpz] (no dt or area factor).
    std::array<double, 4> ray_incoming = {0.0, 0.0, 0.0, 0.0};

    //! Integrated outgoing photon inventory [E,cpx,cpy,cpz] (no dt or area factor).
    std::array<double, 4> ray_outgoing = {0.0, 0.0, 0.0, 0.0};

    //! Scalar isotropic radiation pressure; used only for IsotropicInterior.
    double interior_pressure = 0.0;

    //! Geometric face area.
    double face_area = 0.0;
};

//! Why a traction evaluation was accepted or rejected.
enum class QuasiStaticInterfaceStatus
{
    Valid,
    InvalidFaceCount,
    DuplicateFaceId,
    DuplicateSide,
    InvalidSide,
    InvalidFaceKind,
    NonfiniteTimeStep,
    NonpositiveTimeStep,
    NonfiniteSpeedOfLight,
    NonpositiveSpeedOfLight,
    NonfiniteFaceArea,
    NonpositiveFaceArea,
    NonfinitePressure,
    NegativePressure,
    NonfiniteRayInventory,
    NegativeRayEnergy,
    PhotonConeViolation,
    IncomingOrientationViolation,
    OutgoingOrientationViolation,
    NonfiniteTraction
};

//! Integrated boundary terms and the resulting material impulse.
struct QuasiStaticInterfaceTraction
{
    bool valid = false;
    QuasiStaticInterfaceStatus status = QuasiStaticInterfaceStatus::InvalidFaceCount;
    std::array<double, 3> material_impulse = {0.0, 0.0, 0.0};
    std::array<double, 3> outer_ray_traction = {0.0, 0.0, 0.0};
    std::array<double, 3> inner_pressure_traction = {0.0, 0.0, 0.0};
};

namespace detail
{

[[nodiscard]] constexpr double RoundoffScale (double a, double b)
{
    return 8.0*std::numeric_limits<double>::epsilon()*(a < b ? b : a);
}

[[nodiscard]] inline bool FinitePhotonInventory (
    std::array<double, 4> const& photon, QuasiStaticInterfaceStatus& status)
{
    for (double const component : photon)
    {
        if (!std::isfinite(component))
        {
            status = QuasiStaticInterfaceStatus::NonfiniteRayInventory;
            return false;
        }
    }
    if (photon[0] < 0.0)
    {
        status = QuasiStaticInterfaceStatus::NegativeRayEnergy;
        return false;
    }

    double const momentum_norm = std::hypot(photon[1], photon[2], photon[3]);
    if (!std::isfinite(momentum_norm) ||
        momentum_norm - photon[0] > RoundoffScale(momentum_norm, photon[0]))
    {
        status = QuasiStaticInterfaceStatus::PhotonConeViolation;
        return false;
    }
    return true;
}

} // namespace detail

//! Compute the bounded quasi-static material impulse for exactly two z faces.
/*!
 * The ray inventories are already integrated over time and area. A RayBoundary
 * contributes (outgoing_q-incoming_q)/c. An IsotropicInterior face contributes
 * dt*area*P*side*ez. Kind selection is exclusive. The reported material impulse
 * is minus the sum of those boundary terms once.
 *
 * Inputs are rejected rather than clamped. Photon-cone and orientation checks
 * allow only an operand-scaled 8*double-epsilon roundoff margin.
 */
[[nodiscard]] inline QuasiStaticInterfaceTraction ComputeQuasiStaticInterfaceTraction (
    std::vector<QuasiStaticInterfaceFace> const& faces, double const dt, double const c)
{
    QuasiStaticInterfaceTraction result;

    if (!std::isfinite(dt))
    {
        result.status = QuasiStaticInterfaceStatus::NonfiniteTimeStep;
        return result;
    }
    if (dt <= 0.0)
    {
        result.status = QuasiStaticInterfaceStatus::NonpositiveTimeStep;
        return result;
    }
    if (!std::isfinite(c))
    {
        result.status = QuasiStaticInterfaceStatus::NonfiniteSpeedOfLight;
        return result;
    }
    if (c <= 0.0)
    {
        result.status = QuasiStaticInterfaceStatus::NonpositiveSpeedOfLight;
        return result;
    }
    if (faces.size() != 2)
    {
        result.status = QuasiStaticInterfaceStatus::InvalidFaceCount;
        return result;
    }

    for (QuasiStaticInterfaceFace const& face : faces)
    {
        if (face.kind != QuasiStaticInterfaceFaceKind::RayBoundary &&
            face.kind != QuasiStaticInterfaceFaceKind::IsotropicInterior)
        {
            result.status = QuasiStaticInterfaceStatus::InvalidFaceKind;
            return result;
        }
        if (face.side != -1 && face.side != 1)
        {
            result.status = QuasiStaticInterfaceStatus::InvalidSide;
            return result;
        }
        if (!std::isfinite(face.face_area) || face.face_area <= 0.0)
        {
            result.status = std::isfinite(face.face_area)
                ? QuasiStaticInterfaceStatus::NonpositiveFaceArea
                : QuasiStaticInterfaceStatus::NonfiniteFaceArea;
            return result;
        }
        if (!std::isfinite(face.interior_pressure))
        {
            result.status = QuasiStaticInterfaceStatus::NonfinitePressure;
            return result;
        }
        if (face.interior_pressure < 0.0)
        {
            result.status = QuasiStaticInterfaceStatus::NegativePressure;
            return result;
        }
        if (!detail::FinitePhotonInventory(face.ray_incoming, result.status) ||
            !detail::FinitePhotonInventory(face.ray_outgoing, result.status))
        {
            return result;
        }

        double const incoming_normal_momentum = static_cast<double>(face.side)
            *face.ray_incoming[3];
        double const outgoing_normal_momentum = static_cast<double>(face.side)
            *face.ray_outgoing[3];
        if (incoming_normal_momentum > detail::RoundoffScale(
                std::abs(incoming_normal_momentum), face.ray_incoming[0]))
        {
            result.status = QuasiStaticInterfaceStatus::IncomingOrientationViolation;
            return result;
        }
        if (outgoing_normal_momentum < -detail::RoundoffScale(
                std::abs(outgoing_normal_momentum), face.ray_outgoing[0]))
        {
            result.status = QuasiStaticInterfaceStatus::OutgoingOrientationViolation;
            return result;
        }
    }

    if (faces[0].face_id == faces[1].face_id)
    {
        result.status = QuasiStaticInterfaceStatus::DuplicateFaceId;
        return result;
    }
    if (faces[0].side == faces[1].side)
    {
        result.status = QuasiStaticInterfaceStatus::DuplicateSide;
        return result;
    }

    for (QuasiStaticInterfaceFace const& face : faces)
    {
        if (face.kind == QuasiStaticInterfaceFaceKind::RayBoundary)
        {
            for (std::size_t d = 0; d < 3; ++d)
            {
                result.outer_ray_traction[d] +=
                    (face.ray_outgoing[d + 1] - face.ray_incoming[d + 1])/c;
            }
        }
        else
        {
            result.inner_pressure_traction[2] +=
                dt*face.face_area*face.interior_pressure*static_cast<double>(face.side);
        }
    }

    for (std::size_t d = 0; d < 3; ++d)
    {
        result.material_impulse[d] =
            -(result.outer_ray_traction[d] + result.inner_pressure_traction[d]);
        if (!std::isfinite(result.material_impulse[d]) ||
            !std::isfinite(result.outer_ray_traction[d]) ||
            !std::isfinite(result.inner_pressure_traction[d]))
        {
            QuasiStaticInterfaceTraction rejected;
            rejected.status = QuasiStaticInterfaceStatus::NonfiniteTraction;
            return rejected;
        }
    }
    result.valid = true;
    result.status = QuasiStaticInterfaceStatus::Valid;
    return result;
}

} // namespace warpx::radiation

#endif // WARPX_RADIATION_QUASI_STATIC_INTERFACE_TRACTION_H_
