/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_MOVINGGROUPINTERACTION_H_
#define WARPX_RADIATION_MOVINGGROUPINTERACTION_H_

#include "NearLteGroupSpectrum.H"

namespace warpx::radiation
{
/** Physical material interaction rate in an explicitly declared frame.
 * energy is q [J/m^3/s]; c_force is c*f [J/m^3/s], with f [N/m^3].
 * These are signed sources, not radiation moments or opacity averages. */
struct GroupInteractionRate
{
    amrex::Real energy = 0;
    amrex::GpuArray<amrex::Real, 3> c_force{};
};

/** Independent one-sided edge moments nu*(E_nu,F_nu), in the supplied frame.
 * below belongs to the lower-frequency bin; above to the higher bin.
 * Both outer edges are zero. Neither E_nu nor F_nu is forced to share a
 * global Planck shape; the caller owns its spectral reconstruction. */
struct GroupOneSidedMomentEdge
{
    GroupSpectralMoments below;
    GroupSpectralMoments above;
};

/** One-sided edge rates nu*(q_nu,c*f_nu), not unweighted moment edges. */
struct ComovingGroupInteractionEdge
{
    GroupInteractionRate below;
    GroupInteractionRate above;
};

struct GroupInteractionTotals
{
    GroupInteractionRate comoving;
    GroupInteractionRate laboratory;
    // Actual laboratory source projection q_lab-v.f_lab. Its difference
    // from rest heat is O(beta^2); it is never silently reconciled.
    amrex::Real laboratory_energy_minus_work = 0;
    int reconstruction_iterations = 0;
    amrex::Real reconstruction_relative_residual = 0;
};

namespace group_interaction_detail
{
AMREX_GPU_HOST_DEVICE inline bool Finite (GroupInteractionRate const &rate)
{
    if (!amrex::Math::isfinite(rate.energy))
    {
        return false;
    }
    for (auto value : rate.c_force)
    {
        if (!amrex::Math::isfinite(value))
        {
            return false;
        }
    }
    return true;
}
AMREX_GPU_HOST_DEVICE inline bool Zero (GroupInteractionRate const &rate)
{
    return rate.energy == 0 && rate.c_force[0] == 0 && rate.c_force[1] == 0 && rate.c_force[2] == 0;
}
AMREX_GPU_HOST_DEVICE inline GroupInteractionRate RestRate (GroupSpectralMoments const &moments,
                                                            amrex::Real equilibrium,
                                                            amrex::Real absorption,
                                                            amrex::Real transport)
{
    GroupInteractionRate result;
    result.energy = PhysConst::c * absorption * (moments.energy - equilibrium);
    for (int d = 0; d < 3; ++d)
    {
        result.c_force[d] = transport * moments.flux[d];
    }
    return result;
}

AMREX_GPU_HOST_DEVICE inline ComovingGroupInteractionEdge
MomentEdge (GroupOneSidedMomentEdge const &edge)
{
    ComovingGroupInteractionEdge result;
    result.below.energy = edge.below.energy;
    result.above.energy = edge.above.energy;
    for (int d = 0; d < 3; ++d)
    {
        result.below.c_force[d] = edge.below.flux[d] / PhysConst::c;
        result.above.c_force[d] = edge.above.flux[d] / PhysConst::c;
    }
    return result;
}

/** Exact hemispherical angular integrals of the first-order Doppler edge
 * crossing. For n.v>0 the edge is crossed from below, otherwise from above.
 * The source angular closure is [q_nu+3*n.(c*f_nu)]/(4*pi). */
AMREX_GPU_HOST_DEVICE inline GroupInteractionRate
Crossing (ComovingGroupInteractionEdge const &edge, amrex::GpuArray<amrex::Real, 3> const &beta,
          amrex::Real speed)
{
    GroupInteractionRate result;
    result.energy = speed * (edge.below.energy - edge.above.energy) / 4;
    amrex::Real parallel_difference = 0;
    for (int d = 0; d < 3; ++d)
    {
        result.energy += beta[d] * (edge.below.c_force[d] + edge.above.c_force[d]) / 2;
        parallel_difference += beta[d] * (edge.below.c_force[d] - edge.above.c_force[d]);
    }
    for (int d = 0; d < 3; ++d)
    {
        result.c_force[d] = beta[d] * (edge.below.energy + edge.above.energy) / 6;
        if (speed != 0)
        {
            result.c_force[d] += 3.0_rt / 16 *
                                 (speed * (edge.below.c_force[d] - edge.above.c_force[d]) +
                                  beta[d] * parallel_difference / speed);
        }
    }
    return result;
}
AMREX_GPU_HOST_DEVICE inline GroupInteractionRate
Transform (GroupInteractionRate const &rest, ComovingGroupInteractionEdge const &lower,
           ComovingGroupInteractionEdge const &upper, amrex::GpuArray<amrex::Real, 3> const &beta,
           amrex::Real speed)
{
    auto const low = Crossing(lower, beta, speed), high = Crossing(upper, beta, speed);
    auto lab = rest;
    lab.energy += low.energy - high.energy;
    for (int d = 0; d < 3; ++d)
    {
        lab.energy += beta[d] * rest.c_force[d];
        lab.c_force[d] += beta[d] * rest.energy + low.c_force[d] - high.c_force[d];
    }
    return lab;
}
} // namespace group_interaction_detail

/** Reconstruct a positive Planck-weighted angular dipole separately in every
 * supplied-frame bin. E_g and F_g have independent amplitudes in each bin:
 * E_nu=E_g*p_nu/fraction_g, F_nu=F_g*p_nu/fraction_g within that bin.
 * This permits non-Planck global E_nu and F_nu and retains jumps at edges.
 * The color temperature specifies only this declared within-bin shape;
 * it is not a claim that E_g=B_g(T_material). Unresolved Planck tail fractions
 * with nonzero input moments reject. Outputs stay unchanged on failure. */
AMREX_GPU_HOST_DEVICE inline bool
TryReconstructPlanckBinDipoleEdges (GroupSpectralMoments const *moments,
                                    EnergyGroupsExecutor const &groups,
                                    amrex::Real color_temperature, GroupOneSidedMomentEdge *edges)
{
    using namespace spectral_detail;
    if (!moments || !edges || !GroupsValid(groups) || !(color_temperature >= 0) ||
        !amrex::Math::isfinite(color_temperature))
    {
        return false;
    }
    for (int g = 0; g < groups.m_num_groups; ++g)
    {
        auto const fraction = Fraction(groups, g, color_temperature);
        if (!MomentValid(moments[g], 1.0_rt / 3) || !(fraction >= 0) ||
            !amrex::Math::isfinite(fraction) || (fraction == 0 && moments[g].energy != 0))
        {
            return false;
        }
        for (int side = 0; side < 2; ++side)
        {
            int const e = g + side;
            if (e == 0 || e == groups.m_num_groups || fraction == 0)
            {
                continue;
            }
            auto const weight =
                EdgeFraction(groups.m_boundaries[e - 1], color_temperature) / fraction;
            GroupSpectralMoments edge;
            edge.energy = weight * moments[g].energy;
            for (int d = 0; d < 3; ++d)
            {
                edge.flux[d] = weight * moments[g].flux[d];
            }
            if (!MomentValid(edge, 1.0_rt / 3))
            {
                return false;
            }
        }
    }
    edges[0] = {};
    edges[groups.m_num_groups] = {};
    for (int e = 1; e < groups.m_num_groups; ++e)
    {
        auto const density = EdgeFraction(groups.m_boundaries[e - 1], color_temperature);
        for (int side = 0; side < 2; ++side)
        {
            int const g = side == 0 ? e - 1 : e;
            auto const fraction = Fraction(groups, g, color_temperature);
            auto const weight = fraction == 0 ? 0 : density / fraction;
            auto &edge = side == 0 ? edges[e].below : edges[e].above;
            edge.energy = weight * moments[g].energy;
            for (int d = 0; d < 3; ++d)
            {
                edge.flux[d] = weight * moments[g].flux[d];
            }
        }
    }
    return true;
}

namespace group_interaction_detail
{
AMREX_GPU_HOST_DEVICE inline GroupSpectralMoments
TransformMoment (GroupSpectralMoments const &input, GroupOneSidedMomentEdge const &lower,
                 GroupOneSidedMomentEdge const &upper, amrex::GpuArray<amrex::Real, 3> const &beta,
                 amrex::Real speed)
{
    auto const low = Crossing(MomentEdge(lower), beta, speed);
    auto const high = Crossing(MomentEdge(upper), beta, speed);
    auto output = input;
    output.energy += low.energy - high.energy;
    for (int d = 0; d < 3; ++d)
    {
        output.energy += 2 * beta[d] * input.flux[d] / PhysConst::c;
        output.flux[d] +=
            PhysConst::c * (4 * beta[d] * input.energy / 3 + low.c_force[d] - high.c_force[d]);
    }
    return output;
}
} // namespace group_interaction_detail

/** First-order LAB/COMOVING moment conversion for an explicitly declared
 * isotropic-pressure dipole spectrum with one-sided edges. The bulk terms
 * transform a stress-energy tensor, while shared angular-upwind crossings
 * handle the frequency bins. Continuous edges reduce to the smooth-spectrum
 * conversion in NearLteGroupSpectrum.H; discontinuities add |beta| terms.
 * Changing sign is the inverse conversion to retained order, not an exact
 * inverse of an angular truncation. All outputs stay unchanged on failure. */
AMREX_GPU_HOST_DEVICE inline bool
TryTransformOneSidedGroupMoments (GroupSpectralMoments const *input,
                                  GroupOneSidedMomentEdge const *edges, int groups,
                                  amrex::GpuArray<amrex::Real, 3> const &velocity,
                                  GroupFrameTransform direction, GroupSpectralMoments *output)
{
    using namespace spectral_detail;
    using namespace group_interaction_detail;
    if (!input || !edges || !output || groups <= 0 || !VelocityValid(velocity) ||
        (direction != GroupFrameTransform::ComovingToLab &&
         direction != GroupFrameTransform::LabToComoving))
    {
        return false;
    }
    for (int e = 0; e <= groups; ++e)
    {
        if (!MomentValid(edges[e].below, 1.0_rt / 3) || !MomentValid(edges[e].above, 1.0_rt / 3))
        {
            return false;
        }
    }
    if (edges[0].below.energy != 0 || edges[0].above.energy != 0 ||
        edges[groups].below.energy != 0 || edges[groups].above.energy != 0)
    {
        return false;
    }
    auto const sign = direction == GroupFrameTransform::ComovingToLab ? 1.0_rt : -1.0_rt;
    amrex::GpuArray<amrex::Real, 3> beta{};
    for (int d = 0; d < 3; ++d)
    {
        beta[d] = sign * velocity[d] / PhysConst::c;
    }
    auto const speed = Norm(beta);
    for (int g = 0; g < groups; ++g)
    {
        auto const candidate = TransformMoment(input[g], edges[g], edges[g + 1], beta, speed);
        if (!MomentValid(input[g], 1.0_rt / 3) || !MomentValid(candidate, 1))
        {
            return false;
        }
    }
    for (int g = 0; g < groups; ++g)
    {
        output[g] = TransformMoment(input[g], edges[g], edges[g + 1], beta, speed);
    }
    return true;
}

/** Assemble rest-frame LTE absorption and coherent isotropic scattering.
 * State is COMOVING, P_nu=E_nu/3. Each bin has a constant proper chi_a/chi_t
 * (inverse metres), but the two limits at a boundary retain their own bin's
 * opacity and moment reconstruction. Rest q_g=c*chi_a,g*(E0,g-B_g),
 * c*f_g=chi_t,g*F0,g. This handles non-Planck flux spectra supplied by the
 * caller. The declared positive dipole requires |F0|<=c*E0/3 in every bin
 * and at every one-sided edge. No near-LTE energy assumption is imposed by
 * this local source evaluation. Every output stays unchanged on failure. */
AMREX_GPU_HOST_DEVICE inline bool TryEvaluateComovingGroupInteraction (
    GroupSpectralMoments const *comoving, GroupOneSidedMomentEdge const *edges,
    EnergyGroupsExecutor const &groups, amrex::Real const *transport, amrex::Real const *absorption,
    amrex::Real material_temperature, GroupInteractionRate *rates,
    ComovingGroupInteractionEdge *rate_edges)
{
    using namespace spectral_detail;
    using namespace group_interaction_detail;
    auto const equilibrium = Blackbody(material_temperature);
    if (!comoving || !edges || !transport || !absorption || !rates || !rate_edges ||
        !GroupsValid(groups) || !(material_temperature >= 0) ||
        !amrex::Math::isfinite(material_temperature) || !amrex::Math::isfinite(equilibrium))
    {
        return false;
    }
    for (int e = 0; e <= groups.m_num_groups; ++e)
    {
        if (!MomentValid(edges[e].below, 1.0_rt / 3) || !MomentValid(edges[e].above, 1.0_rt / 3))
        {
            return false;
        }
    }
    if (edges[0].below.energy != 0 || edges[0].above.energy != 0 ||
        edges[groups.m_num_groups].below.energy != 0 ||
        edges[groups.m_num_groups].above.energy != 0)
    {
        return false;
    }
    for (int g = 0; g < groups.m_num_groups; ++g)
    {
        if (!(transport[g] >= 0) || !amrex::Math::isfinite(transport[g]) ||
            !(absorption[g] >= 0 && absorption[g] <= transport[g]) ||
            !MomentValid(comoving[g], 1.0_rt / 3))
        {
            return false;
        }
        auto const rest =
            RestRate(comoving[g], equilibrium * Fraction(groups, g, material_temperature),
                     absorption[g], transport[g]);
        if (!Finite(rest))
        {
            return false;
        }
        for (int side = 0; side < 2; ++side)
        {
            int const e = g + side;
            auto const moment = side == 0 ? edges[e].above : edges[e].below;
            auto const blackbody_edge =
                e == 0 || e == groups.m_num_groups
                    ? 0
                    : equilibrium * EdgeFraction(groups.m_boundaries[e - 1], material_temperature);
            if (!Finite(RestRate(moment, blackbody_edge, absorption[g], transport[g])))
            {
                return false;
            }
        }
    }
    for (int g = 0; g < groups.m_num_groups; ++g)
    {
        rates[g] = RestRate(comoving[g], equilibrium * Fraction(groups, g, material_temperature),
                            absorption[g], transport[g]);
    }
    rate_edges[0] = {};
    rate_edges[groups.m_num_groups] = {};
    for (int e = 1; e < groups.m_num_groups; ++e)
    {
        auto const blackbody_edge =
            equilibrium * EdgeFraction(groups.m_boundaries[e - 1], material_temperature);
        rate_edges[e].below =
            RestRate(edges[e].below, blackbody_edge, absorption[e - 1], transport[e - 1]);
        rate_edges[e].above = RestRate(edges[e].above, blackbody_edge, absorption[e], transport[e]);
    }
    return true;
}

/** Convert the physical COMOVING source spectrum to LABORATORY finite bins.
 * Source intensity transforms as S_lab=D^2*S0, rather than I_lab=D^3*I0.
 * Closed shared crossings telescope. Bolometrically q_lab=q0+v.f0 and
 * f_lab=f0+v*q0/c^2, with an O(beta^2) Lorentz remainder. At discontinuous
 * edges angular upwinding adds terms proportional to |beta|; they must not
 * be discarded or replaced by a common opacity. Requires beta<=.01.
 * This is a source rate, not a spatial transport or material-work solve. */
AMREX_GPU_HOST_DEVICE inline bool
TryTransformComovingGroupInteraction (GroupInteractionRate const *rest,
                                      ComovingGroupInteractionEdge const *edges, int groups,
                                      amrex::GpuArray<amrex::Real, 3> const &velocity,
                                      GroupInteractionRate *lab, GroupInteractionTotals &totals)
{
    using namespace group_interaction_detail;
    if (!rest || !edges || !lab || groups <= 0 || !spectral_detail::VelocityValid(velocity))
    {
        return false;
    }
    for (int e = 0; e <= groups; ++e)
    {
        if (!Finite(edges[e].below) || !Finite(edges[e].above))
        {
            return false;
        }
    }
    if (!Zero(edges[0].below) || !Zero(edges[0].above) || !Zero(edges[groups].below) ||
        !Zero(edges[groups].above))
    {
        return false;
    }
    amrex::GpuArray<amrex::Real, 3> beta{};
    for (int d = 0; d < 3; ++d)
    {
        beta[d] = velocity[d] / PhysConst::c;
    }
    auto const speed = spectral_detail::Norm(beta);
    GroupInteractionTotals result;
    for (int g = 0; g < groups; ++g)
    {
        auto const candidate = Transform(rest[g], edges[g], edges[g + 1], beta, speed);
        if (!Finite(rest[g]) || !Finite(candidate))
        {
            return false;
        }
        result.comoving.energy += rest[g].energy;
        result.laboratory.energy += candidate.energy;
        for (int d = 0; d < 3; ++d)
        {
            result.comoving.c_force[d] += rest[g].c_force[d];
            result.laboratory.c_force[d] += candidate.c_force[d];
        }
    }
    result.laboratory_energy_minus_work = result.laboratory.energy;
    for (int d = 0; d < 3; ++d)
    {
        result.laboratory_energy_minus_work -= beta[d] * result.laboratory.c_force[d];
    }
    if (!Finite(result.comoving) || !Finite(result.laboratory) ||
        !amrex::Math::isfinite(result.laboratory_energy_minus_work))
    {
        return false;
    }
    for (int g = 0; g < groups; ++g)
    {
        lab[g] = Transform(rest[g], edges[g], edges[g + 1], beta, speed);
    }
    totals = result;
    return true;
}

/** Infer COMOVING Planck-bin dipole amplitudes from stored LAB moments by
 * inverting the FORWARD declared spectral moment map. A separate inverse
 * lab-bin dipole reconstruction would lose the angular boundary layer of
 * discontinuous spectra. This bounded fixed-point inversion preserves the
 * declared comoving within-bin shape and its independent energy/flux amplitudes.
 * Every bin's forward residual must be <=64 epsilon times its own energy
 * scale; no other bin supplies a rounding allowance. Failed inference changes
 * neither output amplitudes nor diagnostics. Narrow/ill-conditioned bins,
 * inadmissible dipoles, and iteration exhaustion reject. */
template <int MaximumGroups = 16>
AMREX_GPU_HOST_DEVICE inline bool TryInferComovingPlanckBinMoments (
    GroupSpectralMoments const *lab, EnergyGroupsExecutor const &groups,
    amrex::Real color_temperature, amrex::GpuArray<amrex::Real, 3> const &velocity,
    GroupSpectralMoments *comoving, int &iterations, amrex::Real &relative_residual)
{
    using namespace spectral_detail;
    if (!lab || !comoving || !GroupsValid(groups) || groups.m_num_groups > MaximumGroups ||
        !VelocityValid(velocity))
    {
        return false;
    }
    amrex::GpuArray<GroupSpectralMoments, MaximumGroups> rest{}, forward{};
    amrex::GpuArray<GroupOneSidedMomentEdge, MaximumGroups + 1> edges{};
    for (int g = 0; g < groups.m_num_groups; ++g)
    {
        rest[g] = lab[g];
    }
    for (int iteration = 0; iteration < 64; ++iteration)
    {
        if (!TryReconstructPlanckBinDipoleEdges(rest.data(), groups, color_temperature,
                                                edges.data()) ||
            !TryTransformOneSidedGroupMoments(rest.data(), edges.data(), groups.m_num_groups,
                                              velocity, GroupFrameTransform::ComovingToLab,
                                              forward.data()))
        {
            return false;
        }
        amrex::Real residual = 0;
        for (int g = 0; g < groups.m_num_groups; ++g)
        {
            auto const scale = amrex::max(lab[g].energy, rest[g].energy);
            auto local = std::abs(lab[g].energy - forward[g].energy);
            for (int d = 0; d < 3; ++d)
            {
                local =
                    amrex::max(local, std::abs(lab[g].flux[d] - forward[g].flux[d]) / PhysConst::c);
            }
            if (scale == 0 && local != 0)
            {
                return false;
            }
            residual = amrex::max(residual, scale == 0 ? 0 : local / scale);
        }
        if (residual <= 64 * std::numeric_limits<amrex::Real>::epsilon())
        {
            for (int g = 0; g < groups.m_num_groups; ++g)
            {
                comoving[g] = rest[g];
            }
            iterations = iteration + 1;
            relative_residual = residual;
            return true;
        }
        for (int g = 0; g < groups.m_num_groups; ++g)
        {
            rest[g].energy += lab[g].energy - forward[g].energy;
            for (int d = 0; d < 3; ++d)
            {
                rest[g].flux[d] += lab[g].flux[d] - forward[g].flux[d];
            }
        }
    }
    return false;
}

/** Declared LAB-state source prototype. Storage is laboratory E_g/F_g, but
 * the spectral closure is a positive COMOVING Planck-bin dipole with independent
 * amplitudes inferred by its forward moment map. Proper opacity is constant
 * within those comoving bins and one-sided sources use that bin's own opacity.
 * Material temperature separately sets LTE emission. The source and frame map
 * are first order with an O(beta^2) physical remainder. Stiff near-LTE source
 * accuracy needs its own runtime acceptance; frame reconstruction convergence
 * does not establish that physical source accuracy. Group count is bounded by
 * MaximumGroups. No native thermodynamic iteration, work, or schema is changed. */
template <int MaximumGroups = 16>
AMREX_GPU_HOST_DEVICE inline bool TryEvaluateLabPlanckBinInteraction (
    GroupSpectralMoments const *lab, EnergyGroupsExecutor const &groups,
    amrex::Real color_temperature, amrex::Real material_temperature,
    amrex::GpuArray<amrex::Real, 3> const &velocity, amrex::Real const *transport,
    amrex::Real const *absorption, GroupInteractionRate *lab_rates, GroupInteractionTotals &totals)
{
    if (groups.m_num_groups <= 0 || groups.m_num_groups > MaximumGroups)
    {
        return false;
    }
    amrex::GpuArray<GroupOneSidedMomentEdge, MaximumGroups + 1> rest_edges{};
    amrex::GpuArray<GroupSpectralMoments, MaximumGroups> rest{};
    amrex::GpuArray<GroupInteractionRate, MaximumGroups> rest_rates{};
    amrex::GpuArray<ComovingGroupInteractionEdge, MaximumGroups + 1> rest_rate_edges{};
    int iterations = 0;
    amrex::Real residual = 0;
    auto const valid =
        TryInferComovingPlanckBinMoments<MaximumGroups>(lab, groups, color_temperature, velocity,
                                                        rest.data(), iterations, residual) &&
        TryReconstructPlanckBinDipoleEdges(rest.data(), groups, color_temperature,
                                           rest_edges.data()) &&
        TryEvaluateComovingGroupInteraction(rest.data(), rest_edges.data(), groups, transport,
                                            absorption, material_temperature, rest_rates.data(),
                                            rest_rate_edges.data()) &&
        TryTransformComovingGroupInteraction(rest_rates.data(), rest_rate_edges.data(),
                                             groups.m_num_groups, velocity, lab_rates, totals);
    if (valid)
    {
        totals.reconstruction_iterations = iterations;
        totals.reconstruction_relative_residual = residual;
    }
    return valid;
}

/** Explicit local LABORATORY moment source step. Positive source transfers
 * energy and momentum to material: E_new=E_old-dt*q_lab,
 * F_new=F_old-dt*c*(c*f_lab). Transfers retain dt*(q,c*f), so a tiny source
 * is not reconstructed from a subtraction of large radiation states.
 * The caller multiplies densities by native V and couples caloric heat and
 * finite-mass particle work consistently. This operation alone does not
 * advance PIC, perform diffusion, or authorize native runtime activation.
 * Negative or unrealizable radiation candidates reject without clipping. */
AMREX_GPU_HOST_DEVICE inline bool TryApplyExplicitLabGroupInteraction (
    GroupSpectralMoments const *old_lab, GroupInteractionRate const *lab_rates, int groups,
    amrex::Real dt, GroupSpectralMoments *candidate, GroupInteractionRate *material_transfer)
{
    using namespace spectral_detail;
    using namespace group_interaction_detail;
    if (!old_lab || !lab_rates || !candidate || !material_transfer || groups <= 0 || !(dt >= 0) ||
        !amrex::Math::isfinite(dt))
    {
        return false;
    }
    for (int g = 0; g < groups; ++g)
    {
        auto next = old_lab[g];
        next.energy -= dt * lab_rates[g].energy;
        for (int d = 0; d < 3; ++d)
        {
            next.flux[d] -= dt * PhysConst::c * lab_rates[g].c_force[d];
        }
        auto transfer = lab_rates[g];
        transfer.energy *= dt;
        for (auto &value : transfer.c_force)
        {
            value *= dt;
        }
        if (!MomentValid(old_lab[g], 1) || !Finite(lab_rates[g]) || !MomentValid(next, 1) ||
            !Finite(transfer))
        {
            return false;
        }
    }
    for (int g = 0; g < groups; ++g)
    {
        candidate[g] = old_lab[g];
        candidate[g].energy -= dt * lab_rates[g].energy;
        material_transfer[g].energy = dt * lab_rates[g].energy;
        for (int d = 0; d < 3; ++d)
        {
            candidate[g].flux[d] -= dt * PhysConst::c * lab_rates[g].c_force[d];
            material_transfer[g].c_force[d] = dt * lab_rates[g].c_force[d];
        }
    }
    return true;
}
} // namespace warpx::radiation
#endif
