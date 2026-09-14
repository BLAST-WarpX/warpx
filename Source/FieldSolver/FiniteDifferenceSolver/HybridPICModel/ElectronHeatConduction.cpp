/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "ElectronHeatConduction.H"

#include "ImplicitChargeEnergyTransport.H"
#include "NonlinearHeatConduction.H"
#include "QdsmcMetricTransport.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"

#include <ablastr/utils/Communication.H>

#include <AMReX_GpuContainers.H>
#include <AMReX_Math.H>

#include <cmath>
#include <limits>
#include <sstream>

namespace warpx::hybrid
{
using namespace amrex::literals;

namespace
{
void
fillInsulatingGhosts (amrex::MultiFab& field, amrex::Geometry const& geometry)
{
    ablastr::utils::communication::FillBoundary(field, field.nGrowVect(), false,
                                                geometry.periodicity(), true);
    if (geometry.isAllPeriodic()) { return; }
    auto const domain = amrex::convert(geometry.Domain(), field.ixType());
    auto const periodic = geometry.isPeriodicArray();
    for (amrex::MFIter mfi(field); mfi.isValid(); ++mfi)
    {
        auto const values = field.array(mfi);
        amrex::ParallelFor(mfi.fabbox(),
                           [=] AMREX_GPU_DEVICE(int i, int j, int k)
                           {
                               amrex::ignore_unused(j, k);
                               amrex::IntVect const point(AMREX_D_DECL(i, j, k));
                               auto image = point;
                               for (int d = 0; d < AMREX_SPACEDIM; ++d)
                               {
                                   if (!periodic[d] && point[d] < domain.smallEnd(d))
                                   {
                                       image[d] = 2 * domain.smallEnd(d) - point[d];
                                   }
                                   else if (!periodic[d] && point[d] > domain.bigEnd(d))
                                   {
                                       image[d] = 2 * domain.bigEnd(d) - point[d];
                                   }
                               }
                               if (image != point)
                               {
                                   values(point) = values(image);
                               }
                           });
    }
}

struct HeatGeometry
{
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx;
    amrex::GpuArray<int, AMREX_SPACEDIM> periodic;
    amrex::Dim3 lo, hi;
    amrex::Real axis_factor;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    width (int i, int j, int k, int d) const noexcept
    {
        amrex::GpuArray<int, 3> const point{i, j, k}, lower{lo.x, lo.y, lo.z},
            upper{hi.x, hi.y, hi.z};
        return dx[d] *
               ((!periodic[d] && (point[d] == lower[d] || point[d] == upper[d])) ? 0.5_rt : 1.0_rt);
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    volume (int i, int j, int k) const noexcept
    {
#if defined(WARPX_DIM_RZ)
        auto const radius = (i - lo.x) * dx[0];
        return cylindricalTransportNodeVolume(radius, dx[0], width(i, j, k, 1), axis_factor) *
               (i == hi.x ? 0.5_rt : 1.0_rt);
#else
        amrex::Real result = 1.0_rt;
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            result *= width(i, j, k, d);
        }
        return result;
#endif
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    faceArea (int i, int j, int k, int d, int side) const noexcept
    {
#if defined(WARPX_DIM_RZ)
        if (d == 0)
        {
            int const face_index = i + side;
            auto const radius = (static_cast<amrex::Real>(face_index - lo.x) - 0.5_rt) * dx[0];
            return cylindricalRadialFaceArea(radius, width(i, j, k, 1));
        }
#else
        amrex::ignore_unused(side);
#endif
        return volume(i, j, k) / width(i, j, k, d);
    }
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
faceConductivity (amrex::Real left, amrex::Real right, amrex::Real tl, amrex::Real tr,
                  amrex::Real nl, amrex::Real nr, amrex::Real spacing, amrex::Real limiter) noexcept
{
    if (left == 0.0_rt || right == 0.0_rt || nl == 0.0_rt || nr == 0.0_rt)
    {
        return 0.0_rt;
    }
    auto const low = amrex::min(left, right), high = amrex::max(left, right);
    auto const harmonic = low / (0.5_rt + 0.5_rt * (low / high));
    if (limiter == 0.0_rt || tl == tr)
    {
        return harmonic;
    }
    auto const mean_temperature = 0.5_rt * tl + 0.5_rt * tr;
    auto const thermal_energy = PhysConst::kb * mean_temperature;
    auto const saturation =
        limiter * amrex::min(nl, nr) * thermal_energy * std::sqrt(thermal_energy / PhysConst::m_e);
    if (!(saturation > 0.0_rt) || !amrex::Math::isfinite(saturation))
    {
        return std::numeric_limits<amrex::Real>::quiet_NaN();
    }
    return 1.0_rt / (1.0_rt / harmonic + std::abs(tr - tl) / (spacing * saturation));
}
} // namespace

void
computeElectronChargeMoments (
    amrex::MultiFab& mean_charge, amrex::MultiFab& effective_charge, amrex::MultiFab const& charge,
    std::vector<std::pair<amrex::MultiFab const*, amrex::Real>> const& species)
{
    auto const compatible = [&] (amrex::MultiFab const& field)
    {
        return field.nComp() == 1 && field.boxArray() == charge.boxArray() &&
               field.DistributionMap() == charge.DistributionMap();
    };
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        charge.nComp() == 1 && charge.ixType().nodeCentered() && compatible(mean_charge) &&
            compatible(effective_charge) && !species.empty() && &mean_charge != &effective_charge &&
            &mean_charge != &charge && &effective_charge != &charge,
        "Electron charge moments require matching nodal fields and charged species.");
    mean_charge.setVal(0.0_rt);
    effective_charge.setVal(0.0_rt);
    amrex::MultiFab species_sum(charge.boxArray(), charge.DistributionMap(), 1, 0);
    species_sum.setVal(0.0_rt);
    for (auto const& entry : species)
    {
        auto const* density = entry.first;
        auto const z = entry.second;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            density != nullptr && compatible(*density) && density != &mean_charge &&
                density != &effective_charge && std::isfinite(z) && z > 0.0_rt &&
                density->is_finite(0, 1, 0) && density->min(0) >= 0.0_rt,
            "Electron charge moments require finite nonnegative species density and positive Z.");
        for (amrex::MFIter mfi(mean_charge); mfi.isValid(); ++mfi)
        {
            auto const number = mean_charge.array(mfi);
            auto const squared = effective_charge.array(mfi);
            auto const sum = species_sum.array(mfi);
            auto const rho_s = density->const_array(mfi);
            amrex::ParallelFor(mfi.validbox(),
                               [=] AMREX_GPU_DEVICE(int i, int j, int k)
                               {
                                   number(i, j, k) += rho_s(i, j, k) / z;
                                   squared(i, j, k) += rho_s(i, j, k) * z;
                                   sum(i, j, k) += rho_s(i, j, k);
                               });
        }
    }
    for (amrex::MFIter mfi(mean_charge); mfi.isValid(); ++mfi)
    {
        auto const mean = mean_charge.array(mfi), effective = effective_charge.array(mfi);
        auto const rho = charge.const_array(mfi), sum = species_sum.const_array(mfi);
        amrex::ParallelFor(
            mfi.validbox(),
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                auto const total = rho(i, j, k);
                auto const scale = amrex::max(std::abs(total), std::abs(sum(i, j, k)));
                bool const consistent =
                    amrex::Math::isfinite(total) && total >= 0.0_rt &&
                    amrex::Math::isfinite(sum(i, j, k)) && amrex::Math::isfinite(mean(i, j, k)) &&
                    amrex::Math::isfinite(effective(i, j, k)) &&
                    std::abs(total - sum(i, j, k)) <=
                        512.0_rt * std::numeric_limits<amrex::Real>::epsilon() * scale;
                if (!consistent ||
                    (total > 0.0_rt && (mean(i, j, k) <= 0.0_rt || effective(i, j, k) <= 0.0_rt)))
                {
                    mean(i, j, k) = std::numeric_limits<amrex::Real>::quiet_NaN();
                    effective(i, j, k) = std::numeric_limits<amrex::Real>::quiet_NaN();
                }
                else if (total == 0.0_rt)
                {
                    mean(i, j, k) = 0.0_rt;
                    effective(i, j, k) = 0.0_rt;
                }
                else
                {
                    mean(i, j, k) = total / mean(i, j, k);
                    effective(i, j, k) /= total;
                }
            });
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        mean_charge.is_finite(0, 1, 0) && effective_charge.is_finite(0, 1, 0),
        "Electron charge moments require consistent total and species charge deposits.");
}

HeatConductionResult
advanceElectronHeatConduction (amrex::MultiFab& temperature, amrex::MultiFab& realized_energy,
                               amrex::MultiFab const& charge, amrex::Geometry const& geometry,
                               amrex::Real const gamma, amrex::ParserExecutor<2> const conductivity,
                               amrex::Real const flux_limiter, amrex::Real const dt,
                               int const maximum_substeps, amrex::Real const axis_volume_factor,
                               amrex::MultiFab const* const mean_charge,
                               amrex::MultiFab const* const effective_charge,
                               amrex::ParserExecutor<4> const composition_conductivity,
                               ElectronThermodynamicsExecutor const* thermodynamics,
                               amrex::MultiFab const* material_mass_density)
{
    auto const eos = thermodynamics ? *thermodynamics : ElectronThermodynamicsExecutor{};
    bool const nonlinear = thermodynamics && !eos.isIdealGas();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!eos.isSingularitySpiner()
        || (eos.m_num_materials == 1 && material_mass_density),
        "Table-EOS conduction requires one explicit material mass-density field.");
    if (material_mass_density) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(material_mass_density->nComp() == 1
            && material_mass_density->boxArray() == temperature.boxArray()
            && material_mass_density->DistributionMap() == temperature.DistributionMap()
            && material_mass_density->is_finite() && material_mass_density->min(0) >= 0,
            "Conduction material mass density must be finite, nonnegative and match temperature.");
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        std::isfinite(dt) && dt >= 0.0_rt && std::isfinite(gamma) && gamma > 1.0_rt &&
            std::isfinite(flux_limiter) && flux_limiter >= 0.0_rt && flux_limiter <= 1.0_rt &&
            maximum_substeps > 0 && temperature.nComp() == 1 && charge.nComp() == 1 &&
            temperature.ixType().nodeCentered() && charge.ixType() == temperature.ixType() &&
            charge.boxArray() == temperature.boxArray() &&
            charge.DistributionMap() == temperature.DistributionMap() &&
            realized_energy.boxArray() == temperature.boxArray() &&
            realized_energy.DistributionMap() == temperature.DistributionMap() &&
            realized_energy.nComp() == 1,
        "Electron heat conduction needs a compatible ideal nodal state and finite controls.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (mean_charge == nullptr) == (effective_charge == nullptr),
        "Composition-dependent conduction requires both Zbar and Zeff fields.");
    bool const use_composition = mean_charge != nullptr;
    if (use_composition)
    {
        for (auto const* field : {mean_charge, effective_charge})
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                field->nComp() == 1 && field->boxArray() == temperature.boxArray() &&
                    field->DistributionMap() == temperature.DistributionMap() &&
                    field->is_finite(0, 1, 0) && field->min(0) >= 0.0_rt,
                "Composition-dependent conductivity needs matching finite charge moments.");
        }
    }
#if defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    amrex::Abort("Electron heat conduction currently supports Cartesian and RZ geometries.");
#endif
#if defined(WARPX_DIM_RZ)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        geometry.ProbLo(0) == 0.0_rt &&
            (axis_volume_factor == 1.0_rt / 3.0_rt || axis_volume_factor == 1.0_rt / 4.0_rt),
        "RZ electron heat conduction requires an axis-containing domain and native axis metric.");
#endif
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            geometry.Domain().length(d) >= temperature.nGrowVect()[d] &&
                geometry.Domain().length(d) >= realized_energy.nGrowVect()[d],
            "Electron heat-conduction domain must span its insulating ghost support.");
    }
    HeatConductionResult result;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        temperature.is_finite(0, 1, 0) && temperature.min(0) >= 0.0_rt &&
            charge.is_finite(0, 1, 0) && charge.min(0) >= 0.0_rt,
        "Electron heat conduction requires finite nonnegative temperature and charge.");
    if (dt == 0.0_rt)
    {
        realized_energy.setVal(0.0_rt);
        return result;
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        use_composition ? static_cast<bool>(composition_conductivity)
                        : static_cast<bool>(conductivity),
        "Electron heat conduction requires a compiled parser for the selected signature.");
    auto const node_domain = amrex::surroundingNodes(geometry.Domain());
    HeatGeometry const metric{geometry.CellSizeArray(), geometry.isPeriodicArray(),
                              amrex::lbound(node_domain), amrex::ubound(node_domain),
                              axis_volume_factor};
    auto const& ba = temperature.boxArray();
    auto const& dm = temperature.DistributionMap();
    amrex::MultiFab candidate(ba, dm, 1, 1), density(ba, dm, 1, 1);
    amrex::MultiFab capacity(ba, dm, 1, 0), volume(ba, dm, 1, 0);
    amrex::MultiFab kappa(ba, dm, 1, 1), energy(ba, dm, 1, 0), old_energy(ba, dm, 1, 0);
    amrex::MultiFab incoming(ba, dm, 2 * AMREX_SPACEDIM, 0), outgoing(ba, dm, 1, 0);
    amrex::MultiFab rates(ba, dm, 2, 0), update(ba, dm, 1, 0);
    amrex::MultiFab::Copy(candidate, temperature, 0, 0, 1, 0);
    amrex::MultiFab::Copy(density, charge, 0, 0, 1, 0);
    ablastr::utils::communication::FillBoundary(density, density.nGrowVect(), false,
                                                geometry.periodicity(), true);
    for (amrex::MFIter mfi(capacity); mfi.isValid(); ++mfi)
    {
        auto const c = capacity.array(mfi), v = volume.array(mfi);
        auto const rho = density.const_array(mfi);
        amrex::ParallelFor(mfi.validbox(),
                           [=] AMREX_GPU_DEVICE(int i, int j, int k)
                           {
                               c(i, j, k) =
                                   rho(i, j, k) / PhysConst::q_e * PhysConst::kb / (gamma - 1.0_rt);
                               v(i, j, k) = metric.volume(i, j, k);
                           });
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        capacity.is_finite(0, 1, 0) && capacity.min(0) >= 0.0_rt && volume.is_finite(0, 1, 0) &&
            volume.min(0) > 0.0_rt,
        "Electron heat conduction received invalid charge or native node volumes.");

    amrex::Real remaining = dt;
    while (remaining > 0.0_rt)
    {
        if (nonlinear) {
            for (amrex::MFIter it(capacity); it.isValid(); ++it) {
                auto const t = candidate.const_array(it), rho = density.const_array(it);
                auto const m = material_mass_density ? material_mass_density->const_array(it)
                    : amrex::Array4<amrex::Real const>{};
                auto const cv = capacity.array(it), u = old_energy.array(it);
                amrex::ParallelFor(it.validbox(), [=] AMREX_GPU_DEVICE(int i,int j,int k) {
                    ElectronThermodynamicsExecutor::MaterialMassDensities materials{};
                    if (eos.isSingularitySpiner()) { materials[0] = m(i,j,k); }
                    auto const state = eos.stateFromMaterialMassDensitiesTemperature(
                        rho(i,j,k), materials, t(i,j,k));
                    bool const valid = std::isfinite(state.internal_energy_density)
                        && std::isfinite(state.heat_capacity_density)
                        && (rho(i,j,k) == 0 ? state.heat_capacity_density == 0
                                               && state.internal_energy_density == 0
                                           : state.heat_capacity_density > 0);
                    cv(i,j,k) = valid ? state.heat_capacity_density
                        : std::numeric_limits<amrex::Real>::quiet_NaN();
                    u(i,j,k) = state.internal_energy_density;
                });
            }
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(capacity.is_finite() && old_energy.is_finite(),
                "Nonlinear conduction received an invalid EOS state or nonpositive heat capacity.");
        }
        // Define physical as well as periodic ghosts before the nonlinear
        // solver validates its input. Native arena reuse can otherwise expose
        // uninitialized insulating ghosts even with valid interior temperatures.
        fillInsulatingGhosts(candidate, geometry);
        for (amrex::MFIter mfi(kappa); mfi.isValid(); ++mfi)
        {
            auto const out = kappa.array(mfi);
            auto const t = candidate.const_array(mfi), rho = density.const_array(mfi);
            auto const zbar = use_composition ? mean_charge->const_array(mfi)
                                              : amrex::Array4<amrex::Real const>{};
            auto const zeff = use_composition ? effective_charge->const_array(mfi)
                                              : amrex::Array4<amrex::Real const>{};
            amrex::ParallelFor(
                mfi.validbox(),
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    bool const valid = amrex::Math::isfinite(t(i, j, k)) && t(i, j, k) >= 0.0_rt &&
                                       (!use_composition || rho(i, j, k) == 0.0_rt ||
                                        (zbar(i, j, k) > 0.0_rt && zeff(i, j, k) > 0.0_rt));
                    auto const te = t(i, j, k) * PhysConst::kb / PhysConst::q_e;
                    out(i, j, k) =
                        !valid ? std::numeric_limits<amrex::Real>::quiet_NaN()
                               : (rho(i, j, k) > 0.0_rt
                                      ? (use_composition
                                             ? composition_conductivity(
                                                   rho(i, j, k), te, zbar(i, j, k), zeff(i, j, k))
                                             : conductivity(rho(i, j, k), te))
                                      : 0.0_rt);
                });
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            kappa.is_finite(0, 1, 0) && kappa.min(0) >= 0.0_rt,
            "Electron thermal conductivity and temperature must be finite and nonnegative.");
        ablastr::utils::communication::FillBoundary(kappa, kappa.nGrowVect(), false,
                                                    geometry.periodicity(), true);
        for (amrex::MFIter mfi(incoming); mfi.isValid(); ++mfi)
        {
            auto const in = incoming.array(mfi), out = outgoing.array(mfi), rate = rates.array(mfi);
            auto const old = old_energy.array(mfi);
            auto const c = capacity.const_array(mfi), v = volume.const_array(mfi);
            auto const t = candidate.const_array(mfi), rho = density.const_array(mfi);
            auto const conductivity_values = kappa.const_array(mfi);
            amrex::ParallelFor(
                mfi.validbox(),
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    amrex::GpuArray<int, 3> const p{i, j, k},
                        lo{metric.lo.x, metric.lo.y, metric.lo.z},
                        hi{metric.hi.x, metric.hi.y, metric.hi.z};
                    amrex::Real sum = 0.0_rt, flux = 0.0_rt;
                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    {
                        for (int side = 0; side < 2; ++side)
                        {
                            amrex::Real coefficient = 0.0_rt;
                            bool const boundary =
                                !metric.periodic[d] && p[d] == (side == 0 ? lo[d] : hi[d]);
                            if (!boundary && c(i, j, k) > 0.0_rt)
                            {
                                auto neighbor = p;
                                neighbor[d] += side == 0 ? -1 : 1;
                                int const ni = neighbor[0], nj = neighbor[1], nk = neighbor[2];
                                auto const face = faceConductivity(
                                    conductivity_values(i, j, k), conductivity_values(ni, nj, nk),
                                    t(i, j, k), t(ni, nj, nk), rho(i, j, k) / PhysConst::q_e,
                                    rho(ni, nj, nk) / PhysConst::q_e, metric.dx[d], flux_limiter);
                                coefficient = face * metric.faceArea(i, j, k, d, side) /
                                              metric.dx[d] / v(i, j, k);
                                flux += coefficient * std::abs(t(ni, nj, nk) - t(i, j, k));
                            }
                            in(i, j, k, 2 * d + side) = coefficient;
                            sum += coefficient;
                        }
                    }
                    out(i, j, k) = sum;
                    if (!nonlinear) { old(i, j, k) = c(i, j, k) * t(i, j, k); }
                    rate(i, j, k, 0) = c(i, j, k) > 0.0_rt ? sum / c(i, j, k) : 0.0_rt;
                    rate(i, j, k, 1) = flux;
                });
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            rates.is_finite(0, 2, 0),
            "Electron heat-conduction face coefficients overflowed or became invalid.");
        if (rates.norm0(1, 0) == 0.0_rt)
        {
            break;
        } // Exact zero-gradient/conductivity identity.
        auto const maximum_rate = rates.norm0(0, 0);
        auto step = amrex::min(remaining, 8.0_rt / maximum_rate);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            result.substeps < maximum_substeps && std::isfinite(step) && step > 0.0_rt &&
                remaining - step < remaining,
            "Electron heat conduction exhausted its substep budget; reduce dt or increase "
            "electron_conduction_max_substeps. Live temperature has not been modified.");
        incoming.mult(step, 0, incoming.nComp(), 0);
        outgoing.mult(step, 0, 1, 0);
        // Same audited M-matrix algebra as charge-energy remap, with Cv as
        // the capacity and T as the specific unknown. No change to that
        // solver's iteration, positivity, residual or energy-inventory gates.
        if (nonlinear) {
            bool accepted = false;
            NonlinearHeatSolveResult last_solve;
            for (int refinement = 0; refinement <= 20; ++refinement) {
                auto const solve = tryNonlinearHeatConduction(candidate, energy, old_energy,
                    density, outgoing, incoming, volume, geometry, eos, material_mass_density);
                result.iterations += solve.iterations;
                last_solve = solve;
                if (solve.valid) { accepted = true; break; }
                ++result.rejected_steps;
                step *= 0.5_rt;
                incoming.mult(0.5_rt, 0, incoming.nComp(), 0);
                outgoing.mult(0.5_rt, 0, 1, 0);
            }
            auto const failure_message = [&last_solve] () {
                std::ostringstream message;
                message.precision(8);
                message << std::scientific
                    << "Nonlinear conduction exhausted its retry budget; live temperature is unchanged. "
                    << "Failure (1=input,2=local solve,3=iterations,4=energy)="
                    << static_cast<int>(last_solve.failure)
                    << ", equation residual=" << last_solve.residual
                    << ", energy residual=" << last_solve.energy_residual;
                return message.str();
            };
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(accepted && remaining-step < remaining, failure_message());
        } else {
            result.iterations +=
                implicitChargeEnergyRemap(energy, old_energy, capacity, outgoing, incoming, volume,
                                          geometry, "Electron heat conduction");
        }
        if (!nonlinear) {
        for (amrex::MFIter mfi(candidate); mfi.isValid(); ++mfi)
        {
            auto const t = candidate.array(mfi);
            auto const c = capacity.const_array(mfi), u = energy.const_array(mfi);
            amrex::ParallelFor(mfi.validbox(),
                               [=] AMREX_GPU_DEVICE(int i, int j, int k)
                               {
                                   if (c(i, j, k) > 0.0_rt)
                                   {
                                       t(i, j, k) = u(i, j, k) / c(i, j, k);
                                   }
                               });
        }
        }
        remaining = amrex::max(0.0_rt, remaining - step);
        ++result.substeps;
    }
    // No active face flux: preserve every live temperature bit, including
    // physical ghosts. The diagnostic for this accepted interval is zero.
    if (result.substeps == 0)
    {
        realized_energy.setVal(0.0_rt);
        return result;
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        candidate.is_finite(0, 1, 0) && candidate.min(0) >= 0.0_rt,
        "Electron heat conduction produced an invalid candidate temperature.");
    for (amrex::MFIter mfi(update); mfi.isValid(); ++mfi)
    {
        auto const delta = update.array(mfi);
        auto const before = temperature.const_array(mfi), after = candidate.const_array(mfi);
        auto const c = capacity.const_array(mfi);
        auto const rho = density.const_array(mfi);
        auto const mass = material_mass_density ? material_mass_density->const_array(mfi)
            : amrex::Array4<amrex::Real const>{};
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            if (nonlinear) {
                ElectronThermodynamicsExecutor::MaterialMassDensities materials{};
                if (eos.isSingularitySpiner()) { materials[0] = mass(i,j,k); }
                auto const initial = eos.stateFromMaterialMassDensitiesTemperature(
                    rho(i,j,k), materials, before(i,j,k));
                auto const final = eos.stateFromMaterialMassDensitiesTemperature(
                    rho(i,j,k), materials, after(i,j,k));
                delta(i,j,k) = final.internal_energy_density-initial.internal_energy_density;
            } else {
                delta(i,j,k) = c(i,j,k)*(after(i,j,k)-before(i,j,k));
            }
        });
    }
    amrex::MultiFab::Copy(temperature, candidate, 0, 0, 1, 0);
    amrex::MultiFab::Copy(realized_energy, update, 0, 0, 1, 0);
    fillInsulatingGhosts(temperature, geometry);
    fillInsulatingGhosts(realized_energy, geometry);
    return result;
}
} // namespace warpx::hybrid
