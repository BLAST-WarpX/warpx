/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_NEARLTEGROUPSPECTRUM_H_
#define WARPX_RADIATION_NEARLTEGROUPSPECTRUM_H_

#include "EnergyGroups.H"
#include "Utils/WarpXConst.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_Math.H>

#include <cmath>
#include <limits>

namespace warpx::radiation
{
using namespace amrex::literals;

/** A declared continuous spectral/angular reconstruction, not arbitrary
 * multigroup moment closure. In its stated input frame,
 * I_nu(n) = c/(4*pi) p_nu(T) [E + 3*n.F/c], integral p_nu dnu = 1.
 * E_nu and F_nu share the Planck spectral shape. Nonnegative intensity requires
 * |F|<=c*E/3. Frequency-dependent transport opacity generally does not produce
 * this F_nu shape, so native moving multigroup transport must not consume this
 * prerequisite as though it reconstructed arbitrary group fluxes. */
struct PlanckDipoleSpectrum
{
    amrex::Real temperature = 0;
    amrex::Real energy_density = 0;
    amrex::GpuArray<amrex::Real, 3> energy_current{};
};

/** Group integrals have E [J/m^3], F [W/m^2]. At a spectral edge these fields
 * instead hold nu*E_nu and nu*F_nu in the same respective units. Separating the
 * edge data from the reconstruction permits a future conservative closure. */
struct GroupSpectralMoments
{
    amrex::Real energy = 0;
    amrex::GpuArray<amrex::Real, 3> flux{};
};

enum class GroupFrameTransform
{
    LabToComoving,
    ComovingToLab
};

namespace spectral_detail
{
constexpr amrex::Real radiation_constant = 7.565733250280007e-16_rt;

AMREX_GPU_HOST_DEVICE inline amrex::Real Norm (amrex::GpuArray<amrex::Real, 3> const &vector)
{
    return std::sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]);
}
AMREX_GPU_HOST_DEVICE inline amrex::Real Blackbody (amrex::Real temperature)
{
    auto const t2 = temperature * temperature;
    return radiation_constant * t2 * t2;
}
AMREX_GPU_HOST_DEVICE inline bool GroupsValid (EnergyGroupsExecutor const &groups)
{
    if (groups.m_num_groups <= 0 || (groups.m_num_groups > 1 && !groups.m_boundaries))
    {
        return false;
    }
    amrex::Real previous = 0;
    for (int g = 0; g < groups.m_num_groups - 1; ++g)
    {
        auto const edge = groups.m_boundaries[g];
        if (!(edge > previous) || !amrex::Math::isfinite(edge))
        {
            return false;
        }
        previous = edge;
    }
    return true;
}
AMREX_GPU_HOST_DEVICE inline amrex::Real Fraction (EnergyGroupsExecutor const &groups, int g,
                                                   amrex::Real temperature)
{
    return temperature == 0 ? 0 : groups.planckFraction(g, PhysConst::kb * temperature);
}
AMREX_GPU_HOST_DEVICE inline amrex::Real EdgeFraction (amrex::Real photon_energy,
                                                       amrex::Real temperature)
{
    if (temperature == 0)
    {
        return 0;
    }
    auto const x = photon_energy / (PhysConst::kb * temperature);
    if (x == 0)
    {
        return 0;
    }
    if (x > 800)
    {
        return 0;
    }
    if (x < 0.5_rt)
    {
        return x * x * x * x / std::expm1(x) / planck_integral;
    }
    return std::exp(4 * std::log(x) - x) / (1 - std::exp(-x)) / planck_integral;
}
AMREX_GPU_HOST_DEVICE inline bool MomentValid (GroupSpectralMoments const &moment,
                                               amrex::Real flux_limit)
{
    auto const norm = Norm(moment.flux);
    return moment.energy >= 0 && amrex::Math::isfinite(moment.energy) &&
           amrex::Math::isfinite(norm) && amrex::Math::isfinite(PhysConst::c * moment.energy) &&
           norm <= flux_limit * PhysConst::c * moment.energy;
}
AMREX_GPU_HOST_DEVICE inline bool VelocityValid (amrex::GpuArray<amrex::Real, 3> const &velocity)
{
    auto const norm = Norm(velocity);
    return amrex::Math::isfinite(norm) && norm <= 0.01_rt * PhysConst::c;
}
AMREX_GPU_HOST_DEVICE inline bool EdgesValid (GroupSpectralMoments const *edges, int groups)
{
    if (!edges)
    {
        return false;
    }
    for (int edge = 0; edge <= groups; ++edge)
    {
        if (!MomentValid(edges[edge], 1.0_rt / 3))
        {
            return false;
        }
    }
    return edges[0].energy == 0 && edges[groups].energy == 0;
}
AMREX_GPU_HOST_DEVICE inline GroupSpectralMoments
Transform (GroupSpectralMoments const &input, GroupSpectralMoments const &lower,
           GroupSpectralMoments const &upper, amrex::GpuArray<amrex::Real, 3> const &velocity,
           amrex::Real sign)
{
    auto output = input;
    for (int d = 0; d < 3; ++d)
    {
        output.energy += sign * velocity[d] / (PhysConst::c * PhysConst::c) *
                         (2 * input.flux[d] - (upper.flux[d] - lower.flux[d]));
        output.flux[d] +=
            sign * velocity[d] / 3 * (4 * input.energy - (upper.energy - lower.energy));
    }
    return output;
}
} // namespace spectral_detail

/** Fill N group integrals and N+1 spectral edges, including the zero/infinite
 * tails. Outputs stay unchanged on failure. Near-LTE is an explicit input
 * domain, |E-aT^4|/max(E,aT^4)<=maximum_relative_lte_mismatch. */
AMREX_GPU_HOST_DEVICE inline bool
TryReconstructPlanckDipoleGroups (PlanckDipoleSpectrum const &spectrum,
                                  EnergyGroupsExecutor const &groups, GroupSpectralMoments *moments,
                                  GroupSpectralMoments *edges,
                                  amrex::Real maximum_relative_lte_mismatch = 1.e-3_rt)
{
    using namespace spectral_detail;
    GroupSpectralMoments const total{spectrum.energy_density, spectrum.energy_current};
    auto const blackbody = Blackbody(spectrum.temperature);
    auto const scale = amrex::max(total.energy, blackbody);
    if (!moments || !edges || !GroupsValid(groups) || !MomentValid(total, 1.0_rt / 3) ||
        !(spectrum.temperature >= 0) || !amrex::Math::isfinite(spectrum.temperature) ||
        !amrex::Math::isfinite(blackbody) || !(maximum_relative_lte_mismatch > 0) ||
        maximum_relative_lte_mismatch > 1 ||
        !amrex::Math::isfinite(maximum_relative_lte_mismatch) ||
        std::abs(total.energy - blackbody) > maximum_relative_lte_mismatch * scale ||
        (spectrum.temperature == 0 && total.energy != 0))
    {
        return false;
    }
    for (int g = 0; g < groups.m_num_groups; ++g)
    {
        auto const fraction = Fraction(groups, g, spectrum.temperature);
        if (!(fraction >= 0) || !amrex::Math::isfinite(fraction))
        {
            return false;
        }
    }
    for (int edge = 1; edge < groups.m_num_groups; ++edge)
    {
        auto const fraction = EdgeFraction(groups.m_boundaries[edge - 1], spectrum.temperature);
        if (!(fraction >= 0) || !amrex::Math::isfinite(fraction))
        {
            return false;
        }
    }
    for (int g = 0; g < groups.m_num_groups; ++g)
    {
        auto const fraction = Fraction(groups, g, spectrum.temperature);
        moments[g].energy = fraction * total.energy;
        for (int d = 0; d < 3; ++d)
        {
            moments[g].flux[d] = fraction * total.flux[d];
        }
    }
    edges[0] = {};
    edges[groups.m_num_groups] = {};
    for (int edge = 1; edge < groups.m_num_groups; ++edge)
    {
        auto const fraction = EdgeFraction(groups.m_boundaries[edge - 1], spectrum.temperature);
        edges[edge].energy = fraction * total.energy;
        for (int d = 0; d < 3; ++d)
        {
            edges[edge].flux[d] = fraction * total.flux[d];
        }
    }
    return true;
}

/** First-order frame conversion with finite-bin edges:
 * E_lab,g=E0,g+2*v.F0,g/c^2-v.[nu*F0_nu]_g/c^2,
 * F_lab,g=F0,g+4*v*E0,g/3-v*[nu*E0_nu]_g/3.
 * The inverse uses the negative sign. This is a moment conversion prerequisite,
 * not an evolution update. Input pressure must be isotropic at retained order.
 * The full angular transform has an O(beta^2) remainder. */
AMREX_GPU_HOST_DEVICE inline bool
TryTransformGroupMoments (GroupSpectralMoments const *input, GroupSpectralMoments const *edges,
                          int groups, amrex::GpuArray<amrex::Real, 3> const &velocity,
                          GroupFrameTransform direction, GroupSpectralMoments *output)
{
    using namespace spectral_detail;
    if (!input || !output || groups <= 0 || !EdgesValid(edges, groups) ||
        !VelocityValid(velocity) ||
        (direction != GroupFrameTransform::LabToComoving &&
         direction != GroupFrameTransform::ComovingToLab))
    {
        return false;
    }
    auto const sign = direction == GroupFrameTransform::ComovingToLab ? 1.0_rt : -1.0_rt;
    for (int g = 0; g < groups; ++g)
    {
        auto const transformed = Transform(input[g], edges[g], edges[g + 1], velocity, sign);
        if (!MomentValid(input[g], 1.0_rt / 3) || !MomentValid(transformed, 1))
        {
            return false;
        }
    }
    for (int g = 0; g < groups; ++g)
    {
        output[g] = Transform(input[g], edges[g], edges[g + 1], velocity, sign);
    }
    return true;
}

/** Physical rest-frame absorption heat [J/m^3/s] evaluated from lab groups.
 * Distinct chi_a-weighted spectral edges must remain in the sum: they do not
 * telescope. LTE emission uses the supplied material temperature. No native
 * source/work ledger is changed by this local prerequisite. */
AMREX_GPU_HOST_DEVICE inline bool TryEvaluateGroupAbsorptionHeating (
    GroupSpectralMoments const *lab, GroupSpectralMoments const *edges,
    EnergyGroupsExecutor const &groups, amrex::GpuArray<amrex::Real, 3> const &velocity,
    amrex::Real const *transport_opacity, amrex::Real const *absorption_opacity,
    amrex::Real material_temperature, amrex::Real *group_heat, amrex::Real &total_heat)
{
    using namespace spectral_detail;
    auto const equilibrium = Blackbody(material_temperature);
    if (!lab || !group_heat || !transport_opacity || !absorption_opacity || !GroupsValid(groups) ||
        !EdgesValid(edges, groups.m_num_groups) || !VelocityValid(velocity) ||
        !(material_temperature >= 0) || !amrex::Math::isfinite(material_temperature) ||
        !amrex::Math::isfinite(equilibrium))
    {
        return false;
    }
    amrex::Real sum = 0;
    for (int g = 0; g < groups.m_num_groups; ++g)
    {
        auto const chi_t = transport_opacity[g], chi_a = absorption_opacity[g];
        auto const rest = Transform(lab[g], edges[g], edges[g + 1], velocity, -1);
        auto const heat =
            chi_a == 0
                ? 0
                : PhysConst::c * chi_a *
                      (rest.energy - equilibrium * Fraction(groups, g, material_temperature));
        if (!(chi_t >= 0) || !amrex::Math::isfinite(chi_t) || !(chi_a >= 0 && chi_a <= chi_t) ||
            !MomentValid(lab[g], 1.0_rt / 3) || !MomentValid(rest, 1) ||
            !amrex::Math::isfinite(heat))
        {
            return false;
        }
        sum += heat;
    }
    if (!amrex::Math::isfinite(sum))
    {
        return false;
    }
    for (int g = 0; g < groups.m_num_groups; ++g)
    {
        auto const rest = Transform(lab[g], edges[g], edges[g + 1], velocity, -1);
        group_heat[g] =
            absorption_opacity[g] == 0
                ? 0
                : PhysConst::c * absorption_opacity[g] *
                      (rest.energy - equilibrium * Fraction(groups, g, material_temperature));
    }
    total_heat = sum;
    return true;
}

/** Explicit conservative group-compression transfer [J/cell]. Shared edge
 * flux Phi=-dt*V*div(v)*nu*E_nu/3 yields transfer_g=Phi_g-Phi_(g+1).
 * Compression transports energy to higher-frequency groups. Closed spectral
 * tails telescope. This is ONLY the frequency redistribution contribution;
 * total radiation pressure work and spatial advection are separate equations.
 * Invalid or negative group candidates reject all outputs without clipping. */
AMREX_GPU_HOST_DEVICE inline bool
TryGroupCompressionTransfer (GroupSpectralMoments const *old, GroupSpectralMoments const *edges,
                             int groups, amrex::Real velocity_divergence, amrex::Real dt,
                             amrex::Real volume, amrex::Real *transfer, amrex::Real *spectral_flux)
{
    using namespace spectral_detail;
    if (!old || !transfer || !spectral_flux || groups <= 0 || !EdgesValid(edges, groups) ||
        !(dt >= 0) || !amrex::Math::isfinite(dt) || !(volume > 0) ||
        !amrex::Math::isfinite(volume) || !amrex::Math::isfinite(velocity_divergence))
    {
        return false;
    }
    auto const coefficient = -dt * volume * velocity_divergence / 3;
    if (!amrex::Math::isfinite(coefficient))
    {
        return false;
    }
    for (int g = 0; g < groups; ++g)
    {
        auto const lo = coefficient * edges[g].energy, hi = coefficient * edges[g + 1].energy;
        auto const next = old[g].energy * volume + (lo - hi);
        if (!MomentValid(old[g], 1.0_rt / 3) || !(next >= 0) || !amrex::Math::isfinite(next) ||
            !amrex::Math::isfinite(lo) || !amrex::Math::isfinite(hi))
        {
            return false;
        }
    }
    for (int edge = 0; edge <= groups; ++edge)
    {
        spectral_flux[edge] = coefficient * edges[edge].energy;
    }
    for (int g = 0; g < groups; ++g)
    {
        transfer[g] = spectral_flux[g] - spectral_flux[g + 1];
    }
    return true;
}
} // namespace warpx::radiation
#endif
