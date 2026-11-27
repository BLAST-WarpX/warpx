/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "ResolvedGrayMaterialTableOpacity.H"

#include "Utils/MaterialRegistry.H"

#include <AMReX_BoxIterator.H>
#include <AMReX_Geometry.H>
#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace warpx::radiation::detail
{
namespace
{
using LD = long double;
using Box = GrayOpacityArithmeticBox;
inline LD
Down (LD x) {
    return std::nextafter(x, -std::numeric_limits<LD>::infinity());
}
inline LD
Up (LD x) {
    return std::nextafter(x, std::numeric_limits<LD>::infinity());
}
struct Arithmetic {
    Box
    Save (char const*, Box, Box, LD lo, LD hi) {
        return {lo, hi};
    }
    Box
    Point (LD x) {
        return Save("point", {}, {}, x, x);
    }
    Box
    LogBox (double x) {
        double lo = x, hi = x;
        for (int i = 0; i < 2; ++i) {
            lo = std::nextafter(lo, -std::numeric_limits<double>::infinity());
            hi = std::nextafter(hi, std::numeric_limits<double>::infinity());
        }
        return Save("logbox", Point(x), {}, lo, hi);
    }
    Box
    Add (Box a, Box b) {
        return Save("add", a, b, Down(a.lo + b.lo), Up(a.hi + b.hi));
    }
    Box
    Sub (Box a, Box b) {
        return Save("sub", a, b, Down(a.lo - b.hi), Up(a.hi - b.lo));
    }
    Box
    Mul (Box a, Box b) {
        std::array<LD, 4> const v{a.lo * b.lo, a.lo * b.hi, a.hi * b.lo,
                                  a.hi * b.hi};
        return Save("mul", a, b, Down(*std::min_element(v.begin(), v.end())),
                    Up(*std::max_element(v.begin(), v.end())));
    }
    Box
    Div (Box a, Box b) {
        if (!(b.lo > 0)) {
            return Save("div_invalid", a, b, NAN, NAN);
        }
        std::array<LD, 4> const v{a.lo / b.lo, a.lo / b.hi, a.hi / b.lo,
                                  a.hi / b.hi};
        return Save("div", a, b, Down(*std::min_element(v.begin(), v.end())),
                    Up(*std::max_element(v.begin(), v.end())));
    }
    Box
    Round64 (Box a) {
        double lo = static_cast<double>(a.lo), hi = static_cast<double>(a.hi);
        if (static_cast<LD>(lo) > a.lo) {
            lo = std::nextafter(lo, -INFINITY);
        }
        if (static_cast<LD>(hi) < a.hi) {
            hi = std::nextafter(hi, INFINITY);
        }
        // Contains unrounded real values AND every nearest binary64 result.
        // Consequently intermediate contraction/FMA is also enclosed.
        return Save("round64", a, {}, lo, hi);
    }
    Box
    Exp (Box a, bool floating) {
        LD lo = std::exp(a.lo), hi = std::exp(a.hi);
        Box b = Save("exp_ld", a, {}, Down(Down(lo)), Up(Up(hi)));
        if (!floating) {
            return b;
        }
        b = Round64(b);
        double dl = static_cast<double>(b.lo), dh = static_cast<double>(b.hi);
        for (int i = 0; i < 2; ++i) {
            dl = std::nextafter(dl, -INFINITY);
            dh = std::nextafter(dh, INFINITY);
        }
        return Save("libm64", b, {}, dl, dh);
    }
};
inline bool
Finite (Box a) {
    return std::isfinite(a.lo) && std::isfinite(a.hi) && a.lo <= a.hi;
}
inline bool
Contains (Box a, double x) {
    return a.lo <= x && x <= a.hi;
}

bool SameBits (double a, double b)
{
    return std::memcmp(&a, &b, sizeof(double)) == 0;
}

bool AllZeroScatteringLinear (MaterialOpacityTableExecutor const& table)
{
    if (table.m_interpolation_mode != static_cast<int>(MaterialOpacityInterpolation::ExactZeroLinear) ||
        table.m_num_groups != 1 || table.m_num_density < 1 || table.m_num_temperature < 1 ||
        !table.m_zero_mask || !table.m_log_planck_absorption || !table.m_log_planck_emission ||
        !table.m_log_rosseland_transport || !table.m_log_scattering) { return false; }
    auto const density = static_cast<std::size_t>(table.m_num_density);
    auto const temperature = static_cast<std::size_t>(table.m_num_temperature);
    if (temperature > std::numeric_limits<std::size_t>::max() / density) { return false; }
    for (std::size_t i = 0; i < density * temperature; ++i)
    {
        double const p = table.m_log_planck_absorption[i], e = table.m_log_planck_emission[i],
                     r = table.m_log_rosseland_transport[i], s = table.m_log_scattering[i];
        if (table.m_zero_mask[i] != ScatteringZero || s != 0 || std::signbit(s) ||
            !(p > 0) || !std::isnormal(p) || !(e > 0) || !std::isnormal(e) ||
            !(r > 0) || !std::isnormal(r) || !SameBits(p, e)) { return false; }
    }
    return true;
}

GrayOpacityArithmeticResult
BoundResponse (MaterialOpacityTableExecutor const& table, double rho, double temperature,
    ResolvedGrayMaterialOpacityAxisPolicy policy,
    std::array<std::vector<double> const*, 2> const& raw_axes,
    ResolvedGrayMaterialOpacityChannelPolicy channel_policy) {
    using namespace warpx::radiation;
    GrayOpacityArithmeticResult out;
    out.reason = "unsupported arithmetic/table scope";
#if defined(__FAST_MATH__)
    return out;
#else
    bool const linear = channel_policy ==
        ResolvedGrayMaterialOpacityChannelPolicy::ExactZeroLinearAllScatteringZero;
    if (!linear && channel_policy != ResolvedGrayMaterialOpacityChannelPolicy::PositiveLogMatched) {
        return out;
    }
    if (policy != ResolvedGrayMaterialOpacityAxisPolicy::StrictInterior &&
        policy != ResolvedGrayMaterialOpacityAxisPolicy::ExactRawNodes) {
        return out;
    }
    if (std::numeric_limits<amrex::Real>::digits != 53 ||
        !std::numeric_limits<double>::is_iec559 ||
        std::numeric_limits<double>::radix != 2 ||
        !std::numeric_limits<LD>::is_iec559 ||
        std::numeric_limits<LD>::radix != 2 ||
        std::numeric_limits<LD>::digits < 64 ||
        std::numeric_limits<LD>::min_exponent > -8192 ||
        std::numeric_limits<LD>::max_exponent < 8192 ||
        std::fegetround() != FE_TONEAREST ||
        table.m_interpolation_mode !=
            static_cast<int>(linear ? MaterialOpacityInterpolation::ExactZeroLinear
                                   : MaterialOpacityInterpolation::PositiveLog) ||
        table.m_num_groups != 1 || table.m_num_density < 1 ||
        table.m_num_temperature < 1 || !table.m_log_density ||
        !table.m_log_temperature || !table.m_log_planck_absorption ||
        !table.m_log_planck_emission || !table.m_log_rosseland_transport ||
        !table.m_log_scattering) {
        return out;
    }
    if (linear && !AllZeroScatteringLinear(table)) {
        out.reason = "unsupported schema-1.1 all-scattering-zero channels/masks";
        return out;
    }
    if (linear || policy == ResolvedGrayMaterialOpacityAxisPolicy::ExactRawNodes) {
        double const* axes[2]{table.m_log_density, table.m_log_temperature};
        int const counts[2]{table.m_num_density, table.m_num_temperature};
        for (int d = 0; d < 2; ++d) {
            if (!raw_axes[d] || raw_axes[d]->size() != static_cast<std::size_t>(counts[d])) {
                out.reason = "invalid loaded raw-axis identity";
                return out;
            }
            for (int i = 0; i < counts[d]; ++i) {
                double const raw = (*raw_axes[d])[i];
                if (!(raw > 0) || !std::isfinite(raw) || !std::isfinite(axes[d][i]) ||
                    (i && (!(raw > (*raw_axes[d])[i - 1]) ||
                           !(axes[d][i] > axes[d][i - 1]))) ||
                    !SameBits(std::log(raw), axes[d][i])) {
                    out.reason = "invalid loaded raw/log axis pairing";
                    return out;
                }
            }
        }
    }
    out.response =
        EvaluateMaterialOpacityTableResponse(table, rho, temperature, 0, 1);
    out.reason = "invalid authentic response";
    if (!out.response.valid || !(rho > 0) || !(temperature > 0)) {
        return out;
    }
    if (linear && (out.response.scattering_opacity != 0 ||
                   std::signbit(out.response.scattering_opacity))) { return out; }
    out.recovered_alpha = out.response.absorption_rate / PhysConst::c;
    out.valid = true;
    Arithmetic a;
    std::array<Box, 2> f_fraction;
    double const query[2]{std::log(rho), std::log(temperature)};
    double const raw_query[2]{rho, temperature};
    double const* axes[2]{table.m_log_density, table.m_log_temperature};
    int const counts[2]{table.m_num_density, table.m_num_temperature};
    for (int d = 0; d < 2; ++d) {
        for (int i = 0; i < counts[d]; ++i) {
            if (!std::isfinite(axes[d][i]) ||
                (i && !(axes[d][i] > axes[d][i - 1]))) {
                out.valid = false;
                out.reason = "malformed stored axis";
                return out;
            }
        }
        if (counts[d] == 1) {
            out.ideal_fraction[d] = f_fraction[d] = a.Point(0);
            out.gap_lower[d] = INFINITY;
            out.selected[d] = 0;
            if (policy == ResolvedGrayMaterialOpacityAxisPolicy::ExactRawNodes &&
                SameBits(raw_query[d], (*raw_axes[d])[0]) &&
                SameBits(query[d], axes[d][0])) {
                out.exact_axis_node[d] = 0;
            }
            continue;
        }
        int const index =
            query[d] == axes[d][counts[d] - 1]
                ? counts[d] - 2
                : amrex::bisect(axes[d], 0, counts[d] - 1, query[d]);
        out.selected[d] = index;
        Box const q = a.LogBox(query[d]), lo = a.LogBox(axes[d][index]),
                  hi = a.LogBox(axes[d][index + 1]);
        Box const gap = a.Sub(hi, lo);
        out.gap_lower[d] = gap.lo;
        int node = -1;
        if (policy == ResolvedGrayMaterialOpacityAxisPolicy::ExactRawNodes) {
            int const candidate = query[d] == axes[d][counts[d] - 1]
                ? counts[d] - 1 : index;
            if (SameBits(raw_query[d], (*raw_axes[d])[candidate]) &&
                SameBits(query[d], axes[d][candidate])) {
                node = candidate;
            }
        }
        if (node >= 0) {
            // The ideal logs are correlated ONLY by an original raw identity.
            // Internal/lower nodes use the right segment; the final node uses
            // the left segment. Do not replace the floating affine DAG below.
            out.exact_axis_node[d] = node;
            out.ideal_fraction[d] = a.Point(node == index ? 0 : 1);
        } else {
            int elo = 0, ehi = 0;
            std::frexp(std::abs(q.lo), &elo);
            std::frexp(std::abs(q.hi), &ehi);
            if ((q.lo <= 0 && q.hi >= 0) || elo != ehi) {
                out.reason = "unresolved query-log binade box";
                return out;
            }
            if (!(gap.lo > 0) || !(q.lo > lo.hi && q.hi < hi.lo)) {
                out.reason = "unresolved axis/stencil branch";
                return out;
            }
            out.ideal_fraction[d] = a.Div(a.Sub(q, lo), gap);
        }
        f_fraction[d] = a.Round64(
            a.Div(a.Round64(a.Sub(a.Point(query[d]), a.Point(axes[d][index]))),
                  a.Round64(a.Sub(a.Point(axes[d][index + 1]),
                                  a.Point(axes[d][index])))));
    }
    double const* values[3]{table.m_log_rosseland_transport,
                            table.m_log_planck_absorption,
                            table.m_log_scattering};
    for (int channel = 0; channel < 3; ++channel) {
        if (linear && channel == 2) {
            // Every authentic corner is canonical masked +0. All products,
            // sums and the executor's zero-preserving density scaling give +0.
            // No outward arithmetic on zero may manufacture an S radius.
            out.ideal[channel] = out.floating[channel] = a.Point(0);
            continue;
        }
        Box outputs[2];
        for (int floating = 0; floating < 2; ++floating) {
            auto const load = [&] (int i, int j) {
                double const v =
                    values[channel][i * table.m_num_temperature + j];
                return floating || linear ? a.Point(v) : a.LogBox(v);
            };
            auto const round = [&] (Box b) {
                return floating ? a.Round64(b) : b;
            };
            int const i = out.selected[0], j = out.selected[1];
            int const i1 = std::min(i + 1, table.m_num_density - 1),
                      j1 = std::min(j + 1, table.m_num_temperature - 1);
            Box const tx = floating ? f_fraction[0] : out.ideal_fraction[0];
            Box const ty = floating ? f_fraction[1] : out.ideal_fraction[1];
            Box const v00 = load(i, j), v10 = load(i1, j), v01 = load(i, j1),
                      v11 = load(i1, j1);
            if (linear) {
                Box const one_minus_tx = round(a.Sub(a.Point(1), tx));
                Box const one_minus_ty = round(a.Sub(a.Point(1), ty));
                Box const v0 = round(a.Add(round(a.Mul(one_minus_tx, v00)), round(a.Mul(tx, v10))));
                Box const v1 = round(a.Add(round(a.Mul(one_minus_tx, v01)), round(a.Mul(tx, v11))));
                Box const value = round(a.Add(round(a.Mul(one_minus_ty, v0)), round(a.Mul(ty, v1))));
                outputs[floating] = round(a.Mul(a.Point(rho), value));
            } else {
                Box const v0 =
                    round(a.Add(v00, round(a.Mul(tx, round(a.Sub(v10, v00))))));
                Box const v1 =
                    round(a.Add(v01, round(a.Mul(tx, round(a.Sub(v11, v01))))));
                Box const log_value =
                    round(a.Add(v0, round(a.Mul(ty, round(a.Sub(v1, v0))))));
                outputs[floating] =
                    round(a.Mul(a.Point(rho), a.Exp(log_value, floating)));
            }
        }
        out.ideal[channel] = outputs[0];
        out.floating[channel] = outputs[1];
    }
    out.floating[1] = a.Round64(
        a.Div(a.Round64(a.Mul(a.Point(PhysConst::c), out.floating[1])),
              a.Point(PhysConst::c)));
    std::array<double, 3> const actual{out.response.transport_opacity,
                                       out.recovered_alpha,
                                       out.response.scattering_opacity};
    bool normal_positive = true;
    for (int channel = 0; channel < 3; ++channel) {
        auto const i = out.ideal[channel], f = out.floating[channel];
        if (!Finite(i) || !Finite(f) || !Contains(f, actual[channel])) {
            out.valid = false;
            out.reason = "invalid authentic operation enclosure";
            return out;
        }
        if (linear && channel == 2) {
            out.radius[channel] = 0;
            continue;
        }
        normal_positive = normal_positive && i.lo > 0 &&
                          f.lo >= std::numeric_limits<double>::min();
        out.radius[channel] = Up(
            std::max(std::abs(Down(f.lo - i.hi)), std::abs(Up(f.hi - i.lo))));
        out.bound_upper = Up(out.bound_upper + out.radius[channel]);
    }
    Box const residual = a.Sub(a.Sub(a.Point(actual[0]), a.Point(actual[1])),
                               a.Point(actual[2]));
    out.residual_upper =
        Up(std::max(std::abs(residual.lo), std::abs(residual.hi)));
    auto const scale = *std::max_element(actual.begin(), actual.end());
    out.relative_bound_upper = scale > 0
        ? Up(out.bound_upper / scale) : std::numeric_limits<LD>::infinity();
    if (!normal_positive) {
        out.reason = "unresolved underflow/positive-normal scope";
        return out;
    }
    // PREDECLARED engineering resolution policy, not an error-bound assumption.
    // Never trim the derived bound; unresolved diagnostics retain it in full.
    if (out.relative_bound_upper > 0x1p-26L) {
        out.reason = "unresolved optional half-precision resolution policy";
        return out;
    }
    out.resolved = true;
    out.matched = out.residual_upper <= out.bound_upper;
    out.reason = out.matched ? "arithmetic-resolved channel consistency"
                             : "resolved physical channel mismatch";
    return out;
#endif
}
} // namespace

GrayOpacityArithmeticResult BoundPositiveLogGrayOpacityResponse (
    MaterialOpacityTableExecutor const& table, double rho, double temperature)
{
    return BoundResponse(table, rho, temperature,
        ResolvedGrayMaterialOpacityAxisPolicy::StrictInterior, {nullptr, nullptr},
        ResolvedGrayMaterialOpacityChannelPolicy::PositiveLogMatched);
}

GrayOpacityArithmeticResult BoundPositiveLogGrayOpacityResponse (
    MaterialOpacityTable const& table, double rho, double temperature,
    ResolvedGrayMaterialOpacityAxisPolicy policy)
{
#if defined(WARPX_USE_MATERIAL_OPACITY_HDF5)
    if (!table.initialized()) {
        GrayOpacityArithmeticResult out;
        out.reason = "uninitialized material-opacity owner";
        return out;
    }
    return BoundResponse(table.hostExecutor(), rho, temperature, policy,
        {&table.rawDensityAxis(), &table.rawTemperatureAxis()},
        ResolvedGrayMaterialOpacityChannelPolicy::PositiveLogMatched);
#else
    amrex::ignore_unused(table, rho, temperature, policy);
    GrayOpacityArithmeticResult out;
    out.reason = "unsupported loaded material-opacity owner without HDF5";
    return out;
#endif
}

GrayOpacityArithmeticResult BoundExactZeroScatteringGrayOpacityResponse (
    MaterialOpacityTable const& table, double rho, double temperature,
    ResolvedGrayMaterialOpacityAxisPolicy policy)
{
#if defined(WARPX_USE_MATERIAL_OPACITY_HDF5)
    if (!table.initialized() || table.groupEdges().size() != 2 ||
        table.groupEdges()[0] != 0 || table.groupEdges()[1] != INFINITY ||
        table.representativeEnergies().size() != 1) {
        GrayOpacityArithmeticResult out;
        out.reason = "uninitialized or incomplete one-gray material-opacity owner";
        return out;
    }
    return BoundResponse(table.hostExecutor(), rho, temperature, policy,
        {&table.rawDensityAxis(), &table.rawTemperatureAxis()},
        ResolvedGrayMaterialOpacityChannelPolicy::ExactZeroLinearAllScatteringZero);
#else
    amrex::ignore_unused(table, rho, temperature, policy);
    GrayOpacityArithmeticResult out;
    out.reason = "unsupported loaded material-opacity owner without HDF5";
    return out;
#endif
}

namespace
{
bool SameControls (ResolvedGrayMaterialTableOpacityOptions const& o)
{
    int low[3]{o.enabled, static_cast<int>(o.axis_policy), static_cast<int>(o.channel_policy)};
    int high[3]{low[0], low[1], low[2]};
    amrex::ParallelDescriptor::ReduceIntMin(low, 3);
    amrex::ParallelDescriptor::ReduceIntMax(high, 3);
    amrex::Real minimum[4]{o.photon_energy, o.mass_per_electron,
                          o.minimum_electron_density, o.thermodynamic_floor};
    amrex::Real maximum[4]{minimum[0], minimum[1], minimum[2], minimum[3]};
    amrex::ParallelDescriptor::ReduceRealMin(minimum, 4);
    amrex::ParallelDescriptor::ReduceRealMax(maximum, 4);
    return std::equal(low, low + 3, high) && std::equal(minimum, minimum + 4, maximum) &&
        std::all_of(minimum, minimum + 4, [] (amrex::Real x) { return std::isfinite(x); });
}

#if defined(WARPX_DIM_RZ) && !defined(AMREX_USE_GPU) && \
    defined(WARPX_USE_SINGULARITY_EOS) && defined(WARPX_USE_MATERIAL_OPACITY_HDF5)
bool Collective (bool value)
{
    amrex::ParallelDescriptor::ReduceBoolAnd(value);
    return value;
}

bool SameTable (MaterialOpacityTable const& table, ResolvedGrayMaterialOpacityAxisPolicy policy,
               ResolvedGrayMaterialOpacityChannelPolicy channel_policy)
{
    auto const e = table.hostExecutor();
    int low[4]{e.m_num_density, e.m_num_temperature, e.m_num_groups, e.m_interpolation_mode};
    int high[4]{low[0], low[1], low[2], low[3]};
    amrex::ParallelDescriptor::ReduceIntMin(low, 4);
    amrex::ParallelDescriptor::ReduceIntMax(high, 4);
    if (!std::equal(low, low + 4, high)) { return false; }
    auto const count = static_cast<std::size_t>(low[0]) * low[1];
    std::vector<amrex::Real> local;
    local.insert(local.end(), e.m_log_density, e.m_log_density + low[0]);
    local.insert(local.end(), e.m_log_temperature, e.m_log_temperature + low[1]);
    bool const linear = channel_policy ==
        ResolvedGrayMaterialOpacityChannelPolicy::ExactZeroLinearAllScatteringZero;
    if (linear || policy == ResolvedGrayMaterialOpacityAxisPolicy::ExactRawNodes)
    {
        auto const& density = table.rawDensityAxis();
        auto const& temperature = table.rawTemperatureAxis();
        if (!Collective(density.size() == static_cast<std::size_t>(low[0]) &&
                        temperature.size() == static_cast<std::size_t>(low[1]))) {
            return false;
        }
        local.insert(local.end(), density.begin(), density.end());
        local.insert(local.end(), temperature.begin(), temperature.end());
    }
    if (linear) {
        std::vector<std::uint8_t> masks(e.m_zero_mask, e.m_zero_mask + count), reference = masks;
        amrex::ParallelDescriptor::Bcast(reference.data(), reference.size(),
                                         amrex::ParallelDescriptor::IOProcessorNumber());
        if (!Collective(masks == reference)) { return false; }
    }
    for (auto const* channel : {e.m_log_planck_absorption, e.m_log_planck_emission,
                                e.m_log_rosseland_transport, e.m_log_scattering})
    {
        local.insert(local.end(), channel, channel + count);
    }
    local.push_back(table.meanAtomicMassKg());
    local.push_back(table.meanAtomicNumber());
    auto reference = local;
    amrex::ParallelDescriptor::Bcast(reference.data(), reference.size(),
                                     amrex::ParallelDescriptor::IOProcessorNumber());
    amrex::Long id_min = table.materialId(), id_max = id_min;
    amrex::ParallelDescriptor::ReduceLongMin(id_min);
    amrex::ParallelDescriptor::ReduceLongMax(id_max);
    std::string key = table.materialKey();
    int key_size = static_cast<int>(key.size());
    amrex::ParallelDescriptor::Bcast(&key_size, 1,
                                     amrex::ParallelDescriptor::IOProcessorNumber());
    std::string expected(key_size, '\0');
    if (amrex::ParallelDescriptor::IOProcessor()) { expected = key; }
    amrex::ParallelDescriptor::Bcast(expected.data(), key_size,
                                     amrex::ParallelDescriptor::IOProcessorNumber());
    bool const same_values = linear
        ? std::memcmp(local.data(), reference.data(), local.size() * sizeof(amrex::Real)) == 0
        : local == reference;
    return Collective(same_values && id_min == id_max && key == expected);
}
#endif
} // namespace

bool TryFillResolvedGrayMaterialTableOpacity (
    amrex::MultiFab& scattering, amrex::MultiFab& absorption,
    amrex::MultiFab const& density, amrex::MultiFab const& temperature,
    amrex::Geometry const& geometry, MaterialOpacityTable const& table,
    ResolvedGrayMaterialTableOpacityOptions const& options, std::string& reason,
    ResolvedGrayMaterialTableOpacityReport* report)
{
    reason = "unsupported private matched gray material-table opacity";
    if (!SameControls(options))
    {
        reason = "matched gray table opacity controls differ between ranks";
        return false;
    }
#if !defined(WARPX_DIM_RZ) || defined(AMREX_USE_GPU) || \
    !defined(WARPX_USE_SINGULARITY_EOS) || !defined(WARPX_USE_MATERIAL_OPACITY_HDF5)
    amrex::ignore_unused(scattering, absorption, density, temperature, geometry, table, report);
    return false;
#else
    auto const executor = table.hostExecutor();
    bool const positive_log = options.channel_policy ==
        ResolvedGrayMaterialOpacityChannelPolicy::PositiveLogMatched;
    bool const linear = options.channel_policy ==
        ResolvedGrayMaterialOpacityChannelPolicy::ExactZeroLinearAllScatteringZero;
    bool valid = &scattering != &absorption && &scattering != &density &&
        &scattering != &temperature && &absorption != &density &&
        &absorption != &temperature &&
        options.enabled && std::numeric_limits<amrex::Real>::digits == 53 &&
        (options.axis_policy == ResolvedGrayMaterialOpacityAxisPolicy::StrictInterior ||
         options.axis_policy == ResolvedGrayMaterialOpacityAxisPolicy::ExactRawNodes) &&
        options.photon_energy > 0 && options.mass_per_electron > 0 &&
        options.minimum_electron_density >= 0 && options.thermodynamic_floor == 0 &&
        geometry.ProbLo(0) == 0 && !geometry.isPeriodic(0) && geometry.isPeriodic(1) &&
        table.initialized() && executor.m_num_groups == 1 &&
        executor.m_num_density > 0 && executor.m_num_temperature > 0 &&
        executor.m_log_density && executor.m_log_temperature &&
        executor.m_log_planck_absorption && executor.m_log_planck_emission &&
        executor.m_log_rosseland_transport && executor.m_log_scattering &&
        ((positive_log && executor.m_interpolation_mode ==
            static_cast<int>(MaterialOpacityInterpolation::PositiveLog)) ||
         (linear && AllZeroScatteringLinear(executor))) &&
        table.groupEdges().size() == 2 && table.groupEdges()[0] == 0 &&
        table.groupEdges()[1] == std::numeric_limits<amrex::Real>::infinity() &&
        table.representativeEnergies().size() == 1 &&
        table.representativeEnergies()[0] == options.photon_energy &&
        density.nComp() == 1 && temperature.nComp() == 1 && scattering.nComp() == 1 &&
        absorption.nComp() == 1 && density.ixType().cellCentered() &&
        density.boxArray() == temperature.boxArray() && density.boxArray() == scattering.boxArray() &&
        density.boxArray() == absorption.boxArray() &&
        density.DistributionMap() == temperature.DistributionMap() &&
        density.DistributionMap() == scattering.DistributionMap() &&
        density.DistributionMap() == absorption.DistributionMap();
    if (!Collective(valid) || !SameTable(table, options.axis_policy, options.channel_policy)) { return false; }
    amrex::MultiFab candidate_s(scattering.boxArray(), scattering.DistributionMap(), 1,
                                scattering.nGrowVect());
    amrex::MultiFab candidate_a(absorption.boxArray(), absorption.DistributionMap(), 1,
                                absorption.nGrowVect());
    candidate_s.setVal(0); candidate_a.setVal(0);
    ResolvedGrayMaterialTableOpacityReport candidate_report;
    int failure = 0;
    for (amrex::MFIter it(density); it.isValid(); ++it)
    {
        auto const ne = density.const_array(it), te = temperature.const_array(it);
        auto const s = candidate_s.array(it), a = candidate_a.array(it);
        for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
        {
            auto const n = ne(cell()), t = te(cell());
            auto const rho = n * options.mass_per_electron;
            if (!(n > 0) || n < options.minimum_electron_density ||
                !std::isfinite(n) || !std::isfinite(rho) || !(rho > 0) ||
                !std::isfinite(t) || !(t > 0))
            {
                failure = std::max(failure, 1); continue;
            }
            auto const result = linear
                ? BoundExactZeroScatteringGrayOpacityResponse(table, rho, t, options.axis_policy)
                : BoundPositiveLogGrayOpacityResponse(table, rho, t, options.axis_policy);
            if (!result.valid) { failure = std::max(failure, 2); continue; }
            if (!result.resolved) { failure = std::max(failure, 3); continue; }
            if (!result.matched) { failure = std::max(failure, 4); continue; }
            s(cell()) = result.response.scattering_opacity;
            a(cell()) = result.recovered_alpha;
            candidate_report.maximum_channel_residual = std::max(
                candidate_report.maximum_channel_residual,
                static_cast<amrex::Real>(result.residual_upper));
            candidate_report.maximum_channel_bound = std::max(
                candidate_report.maximum_channel_bound,
                static_cast<amrex::Real>(result.bound_upper));
        }
    }
    amrex::ParallelDescriptor::ReduceIntMax(failure);
    if (failure)
    {
        constexpr char const* messages[]{"", "invalid matched gray material sample",
            "invalid or out-of-domain matched gray table response",
            "unresolved matched gray table arithmetic", "resolved gray table channel mismatch"};
        reason = messages[failure]; return false;
    }
    candidate_s.FillBoundary(geometry.periodicity());
    candidate_a.FillBoundary(geometry.periodicity());
    amrex::ParallelDescriptor::ReduceRealMax(candidate_report.maximum_channel_residual);
    amrex::ParallelDescriptor::ReduceRealMax(candidate_report.maximum_channel_bound);
    amrex::MultiFab::Copy(scattering, candidate_s, 0, 0, 1, scattering.nGrowVect());
    amrex::MultiFab::Copy(absorption, candidate_a, 0, 0, 1, absorption.nGrowVect());
    if (report) { *report = candidate_report; }
    reason.clear(); return true;
#endif
}

ResolvedGrayMaterialTableOpacitySnapshot TrySnapshotResolvedGrayMaterialTableOpacity (
    amrex::MultiFab const& raw_density, amrex::MultiFab const& nodal_temperature,
    ResolvedMaterialChargeFields const& unit_fields, ElectronThermodynamicsExecutor const& eos,
    amrex::Geometry const& geometry, MaterialOpacityTable const& table,
    ResolvedGrayMaterialTableOpacityOptions const& options)
{
    ResolvedGrayMaterialTableOpacitySnapshot out;
    out.reason = "unsupported actual single selected gray material-table snapshot";
    if (!SameControls(options))
    {
        out.reason = "matched gray table opacity controls differ between ranks";
        return out;
    }
#if !defined(WARPX_DIM_RZ) || defined(AMREX_USE_GPU) || \
    !defined(WARPX_USE_SINGULARITY_EOS) || !defined(WARPX_USE_MATERIAL_OPACITY_HDF5)
    amrex::ignore_unused(raw_density, nodal_temperature, unit_fields, eos, geometry, table);
    return out;
#else
    bool valid = options.enabled && eos.isSingularitySpiner() && eos.m_num_materials == 1 &&
        eos.m_use_resolved_material_selection && eos.tableStorage() &&
        options.thermodynamic_floor == 0 && options.mass_per_electron > 0 &&
        options.mass_per_electron == PhysConst::q_e * eos.m_mass_per_physical_charge[0] &&
        raw_density.ixType().nodeCentered() && raw_density.nComp() >= 1 &&
        nodal_temperature.boxArray() == raw_density.boxArray() && nodal_temperature.nComp() == 1 &&
        nodal_temperature.DistributionMap() == raw_density.DistributionMap() &&
        unit_fields[0] && unit_fields[0]->nComp() == 1 &&
        unit_fields[0]->boxArray() == raw_density.boxArray() &&
        unit_fields[0]->DistributionMap() == raw_density.DistributionMap() &&
        unit_fields[0]->nGrowVect().allGE(1);
    if (!Collective(valid)) { return out; }
    for (amrex::MFIter it(raw_density); it.isValid(); ++it)
    {
        auto const u = unit_fields[0]->const_array(it);
        for (amrex::BoxIterator node(it.validbox()); node.ok(); ++node)
        {
            ElectronThermodynamicsExecutor::MaterialMassDensities mass{};
            mass[0] = u(node()) * eos.m_mass_per_charge[0];
            valid = valid && std::isfinite(mass[0]) && mass[0] > 0 &&
                eos.resolvedMaterial(mass) == 0;
        }
    }
    if (!Collective(valid))
    {
        out.reason = "matched gray table snapshot contains a non-material0 corner";
        return out;
    }
    auto const cells = amrex::convert(raw_density.boxArray(), amrex::IntVect(0));
    ResolvedGrayMaterialTableOpacitySnapshot candidate;
    candidate.electron_density.define(cells, raw_density.DistributionMap(), 1, 1);
    candidate.electron_temperature.define(cells, raw_density.DistributionMap(), 1, 1);
    candidate.scattering.define(cells, raw_density.DistributionMap(), 1, 1);
    candidate.absorption.define(cells, raw_density.DistributionMap(), 1, 1);
    if (!SnapshotResolvedMaterialOpacity(raw_density, nodal_temperature,
            candidate.electron_density, candidate.electron_temperature, geometry, eos,
            options.minimum_electron_density, options.thermodynamic_floor, unit_fields))
    {
        out.reason = "invalid native four-corner registered material gather"; return out;
    }
    if (!TryFillResolvedGrayMaterialTableOpacity(candidate.scattering, candidate.absorption,
            candidate.electron_density, candidate.electron_temperature, geometry, table,
            options, out.reason, &candidate.report)) { return out; }
    candidate.valid = true;
    return candidate;
#endif
}
namespace
{
ResolvedGrayMaterialTableOpacityOptions IndependentControls (
    ResolvedIndependentGrayMaterialTableOpacityOptions const& options)
{
    ResolvedGrayMaterialTableOpacityOptions result;
    result.enabled = options.enabled;
    result.photon_energy = options.photon_energy;
    result.mass_per_electron = options.mass_per_electron;
    result.minimum_electron_density = options.minimum_electron_density;
    result.thermodynamic_floor = options.thermodynamic_floor;
    result.axis_policy = options.axis_policy;
    // This private encoding adapter does not call the old matched provider or
    // activate a physics enum. Invalid integer values remain invalid.
    result.channel_policy = static_cast<ResolvedGrayMaterialOpacityChannelPolicy>(
        static_cast<int>(options.encoding));
    return result;
}

bool IndependentPresence (MaterialOpacityTable const& table, bool report_present)
{
    int low[2]{table.initialized(), report_present}, high[2]{low[0], low[1]};
    amrex::ParallelDescriptor::ReduceIntMin(low, 2);
    amrex::ParallelDescriptor::ReduceIntMax(high, 2);
    return std::equal(low, low + 2, high) && low[0] == 1;
}

bool SameIndependentControls (ResolvedIndependentGrayMaterialTableOpacityOptions const& provided)
{
    auto const options = IndependentControls(provided);
    int low[4]{options.enabled, static_cast<int>(options.axis_policy),
               static_cast<int>(options.channel_policy), provided.allow_exact_vacuum};
    int high[4]{low[0], low[1], low[2], low[3]};
    amrex::ParallelDescriptor::ReduceIntMin(low, 4);
    amrex::ParallelDescriptor::ReduceIntMax(high, 4);
    std::array<amrex::Real, 4> local{options.photon_energy, options.mass_per_electron,
        options.minimum_electron_density, options.thermodynamic_floor};
    auto reference = local;
    amrex::ParallelDescriptor::Bcast(reference.data(), reference.size(),
                                     amrex::ParallelDescriptor::IOProcessorNumber());
    bool same = std::memcmp(local.data(), reference.data(), sizeof(local)) == 0 &&
        std::all_of(local.begin(), local.end(), [] (amrex::Real x) { return std::isfinite(x); });
    amrex::ParallelDescriptor::ReduceBoolAnd(same);
    return std::equal(low, low + 4, high) && same;
}

#if defined(WARPX_DIM_RZ) && !defined(AMREX_USE_GPU) && \
    defined(WARPX_USE_SINGULARITY_EOS) && defined(WARPX_USE_MATERIAL_OPACITY_HDF5)
amrex::Real IndependentReportedUpper (long double value)
{
    auto result = static_cast<amrex::Real>(value);
    if (static_cast<long double>(result) < value)
    {
        result = std::nextafter(result, std::numeric_limits<amrex::Real>::infinity());
    }
    return result;
}

bool SameIndependentTable (MaterialOpacityTable const& table,
                          ResolvedGrayMaterialOpacityChannelPolicy encoding)
{
    // Existing collective metadata/cardinality validation is reused without
    // changing it. The new seam additionally compares ALL original bits.
    if (!SameTable(table, ResolvedGrayMaterialOpacityAxisPolicy::ExactRawNodes, encoding))
    {
        return false;
    }
    auto const e = table.hostExecutor();
    auto const count = static_cast<std::size_t>(e.m_num_density) * e.m_num_temperature;
    int present_low = e.m_zero_mask != nullptr, present_high = present_low;
    amrex::ParallelDescriptor::ReduceIntMin(present_low);
    amrex::ParallelDescriptor::ReduceIntMax(present_high);
    if (present_low != present_high) { return false; }
    if (present_low)
    {
        std::vector<std::uint8_t> masks(e.m_zero_mask, e.m_zero_mask + count), reference = masks;
        amrex::ParallelDescriptor::Bcast(reference.data(), reference.size(),
                                         amrex::ParallelDescriptor::IOProcessorNumber());
        if (!Collective(masks == reference)) { return false; }
    }
    std::vector<amrex::Real> local;
    for (auto const* axis : {&table.rawDensityAxis(), &table.rawTemperatureAxis()})
    {
        local.insert(local.end(), axis->begin(), axis->end());
    }
    local.insert(local.end(), e.m_log_density, e.m_log_density + e.m_num_density);
    local.insert(local.end(), e.m_log_temperature, e.m_log_temperature + e.m_num_temperature);
    for (auto const* channel : {e.m_log_planck_absorption, e.m_log_planck_emission,
                                e.m_log_rosseland_transport, e.m_log_scattering})
    {
        local.insert(local.end(), channel, channel + count);
    }
    local.insert(local.end(), table.groupEdges().begin(), table.groupEdges().end());
    local.insert(local.end(), table.representativeEnergies().begin(), table.representativeEnergies().end());
    local.push_back(table.meanAtomicMassKg()); local.push_back(table.meanAtomicNumber());
    auto reference = local;
    amrex::ParallelDescriptor::Bcast(reference.data(), reference.size(),
                                     amrex::ParallelDescriptor::IOProcessorNumber());
    return Collective(std::memcmp(local.data(), reference.data(),
                                  local.size() * sizeof(amrex::Real)) == 0);
}

bool DisjointStorage (amrex::MultiFab const& left, amrex::MultiFab const& right)
{
    if (&left == &right) { return false; }
    using Range = std::pair<std::uintptr_t, std::uintptr_t>;
    auto ranges = [] (amrex::MultiFab const& field)
    {
        std::vector<Range> result;
        for (amrex::MFIter it(field); it.isValid(); ++it)
        {
            auto const begin = reinterpret_cast<std::uintptr_t>(field[it].dataPtr());
            auto const length = static_cast<std::uintptr_t>(field[it].box().numPts()) * sizeof(amrex::Real);
            result.emplace_back(begin, begin + length);
        }
        std::sort(result.begin(), result.end());
        return result;
    };
    // Check every local allocation, not only equal FAB indices. Caller layout
    // preflight already guarantees one component and valid allocated fields.
    auto const a = ranges(left), b = ranges(right);
    std::size_t i = 0, j = 0;
    while (i < a.size() && j < b.size())
    {
        if (a[i].first < b[j].second && b[j].first < a[i].second) { return false; }
        if (a[i].second <= b[j].first) { ++i; } else { ++j; }
    }
    return true;
}
#endif
} // namespace

bool TryFillResolvedIndependentGrayMaterialTableOpacity (
    ResolvedIndependentGrayMaterialOpacityFields& output,
    amrex::MultiFab const& density, amrex::MultiFab const& temperature,
    amrex::Geometry const& geometry, MaterialOpacityTable const& table,
    ResolvedIndependentGrayMaterialTableOpacityOptions const& options, std::string& reason,
    ResolvedIndependentGrayMaterialTableOpacityReport* report)
{
    reason = "unsupported independent gray material-table coefficient scope";
    [[maybe_unused]] auto const controls = IndependentControls(options);
    // Consensus precedes disabled/backend/empty-rank short circuits and any
    // hostExecutor access. An absent owner never triggers its assertion.
    bool const same_controls = SameIndependentControls(options);
    bool const same_presence = IndependentPresence(table, report != nullptr);
    if (!same_controls || !same_presence)
    {
        reason = "independent gray coefficient controls/owner/report presence differ between ranks";
        return false;
    }
#if !defined(WARPX_DIM_RZ) || defined(AMREX_USE_GPU) || \
    !defined(WARPX_USE_SINGULARITY_EOS) || !defined(WARPX_USE_MATERIAL_OPACITY_HDF5)
    amrex::ignore_unused(output, density, temperature, geometry);
    return false;
#else
    auto const executor = table.hostExecutor();
    bool const positive = options.encoding ==
        IndependentGrayMaterialOpacityEncoding::PositiveLogPositiveScattering;
    bool const linear = options.encoding ==
        IndependentGrayMaterialOpacityEncoding::ExactZeroLinearAllScatteringZero;
    bool valid = options.enabled && std::numeric_limits<amrex::Real>::digits == 53 &&
        (options.axis_policy == ResolvedGrayMaterialOpacityAxisPolicy::StrictInterior ||
         options.axis_policy == ResolvedGrayMaterialOpacityAxisPolicy::ExactRawNodes) &&
        options.photon_energy > 0 && options.mass_per_electron > 0 &&
        options.minimum_electron_density >= 0 && options.thermodynamic_floor == 0 &&
        geometry.ProbLo(0) == 0 && !geometry.isPeriodic(0) && geometry.isPeriodic(1) &&
        executor.m_num_groups == 1 && executor.m_num_density > 0 && executor.m_num_temperature > 0 &&
        executor.m_log_density && executor.m_log_temperature &&
        executor.m_log_planck_absorption && executor.m_log_planck_emission &&
        executor.m_log_rosseland_transport && executor.m_log_scattering &&
        ((positive && executor.m_interpolation_mode ==
            static_cast<int>(MaterialOpacityInterpolation::PositiveLog)) ||
         (linear && AllZeroScatteringLinear(executor))) &&
        table.groupEdges().size() == 2 && table.groupEdges()[0] == 0 &&
        table.groupEdges()[1] == std::numeric_limits<amrex::Real>::infinity() &&
        table.representativeEnergies().size() == 1 &&
        table.representativeEnergies()[0] == options.photon_energy &&
        density.nComp() == 1 && temperature.nComp() == 1 && density.ixType().cellCentered();
    std::array<amrex::MultiFab const*, 5> const fields{&density, &temperature,
        &output.rosseland_transport, &output.true_scattering, &output.proper_absorption};
    for (auto const* field : fields)
    {
        valid = valid && field->ok() && field->nComp() == 1 &&
            field->boxArray() == density.boxArray() &&
            field->DistributionMap() == density.DistributionMap();
    }
    if (!Collective(valid)) { return false; }
    for (std::size_t i = 2; i < fields.size(); ++i)
    {
        for (std::size_t j = 0; j < i; ++j)
        {
            valid = DisjointStorage(*fields[i], *fields[j]) && valid;
        }
    }
    if (!Collective(valid)) { reason = "independent gray coefficient output aliases an epoch field"; return false; }
    // Always compare original raw-axis identities, even when arithmetic uses
    // StrictInterior. This only selects consensus data, not an axis admission.
    if (!SameIndependentTable(table, controls.channel_policy))
    {
        reason = "independent gray loaded-owner raw/mask/channel data differ between ranks";
        return false;
    }
    ResolvedIndependentGrayMaterialOpacityFields candidate;
    candidate.rosseland_transport.define(density.boxArray(), density.DistributionMap(), 1,
                                         output.rosseland_transport.nGrowVect());
    candidate.true_scattering.define(density.boxArray(), density.DistributionMap(), 1,
                                     output.true_scattering.nGrowVect());
    candidate.proper_absorption.define(density.boxArray(), density.DistributionMap(), 1,
                                       output.proper_absorption.nGrowVect());
    candidate.rosseland_transport.setVal(0);
    candidate.true_scattering.setVal(0);
    candidate.proper_absorption.setVal(0);
    ResolvedIndependentGrayMaterialTableOpacityReport candidate_report;
    int failure = 0;
    for (amrex::MFIter it(density); it.isValid(); ++it)
    {
        auto const ne = density.const_array(it), te = temperature.const_array(it);
        auto const r = candidate.rosseland_transport.array(it);
        auto const s = candidate.true_scattering.array(it), p = candidate.proper_absorption.array(it);
        for (amrex::BoxIterator cell(it.validbox()); cell.ok(); ++cell)
        {
            auto const n = ne(cell()), t = te(cell()), rho = n * options.mass_per_electron;
            if (options.allow_exact_vacuum && n == 0 && t == 0)
            {
                // Candidate fields were initialized to canonical +0. Never
                // evaluate log(0), clamp a table axis or invent a material.
                continue;
            }
            if (!(n > 0) || n < options.minimum_electron_density || !std::isfinite(n) ||
                !std::isfinite(rho) || !(rho > 0) || !std::isfinite(t) || !(t > 0))
            {
                failure = std::max(failure, 1); continue;
            }
            auto const result = linear
                ? BoundExactZeroScatteringGrayOpacityResponse(table, rho, t, options.axis_policy)
                : BoundPositiveLogGrayOpacityResponse(table, rho, t, options.axis_policy);
            if (!result.valid) { failure = std::max(failure, 2); continue; }
            if (!result.resolved) { failure = std::max(failure, 3); continue; }
            // Deliberately NO matched predicate. The unchanged enclosure and
            // resolution tests certify coefficients, not D/K source physics.
            r(cell()) = result.response.transport_opacity;
            s(cell()) = result.response.scattering_opacity;
            p(cell()) = result.recovered_alpha;
            for (int channel = 0; channel < 3; ++channel)
            {
                candidate_report.maximum_channel_radius[channel] = std::max(
                    candidate_report.maximum_channel_radius[channel],
                    IndependentReportedUpper(result.radius[channel]));
            }
            candidate_report.maximum_relative_arithmetic_bound = std::max(
                candidate_report.maximum_relative_arithmetic_bound,
                IndependentReportedUpper(result.relative_bound_upper));
        }
    }
    amrex::ParallelDescriptor::ReduceIntMax(failure);
    if (failure)
    {
        constexpr char const* messages[]{"", "invalid independent gray material sample",
            "invalid or out-of-domain independent gray table response",
            "unresolved independent gray table arithmetic"};
        reason = messages[failure]; return false;
    }
    candidate.rosseland_transport.FillBoundary(geometry.periodicity());
    candidate.true_scattering.FillBoundary(geometry.periodicity());
    candidate.proper_absorption.FillBoundary(geometry.periodicity());
    amrex::ParallelDescriptor::ReduceRealMax(candidate_report.maximum_channel_radius.data(), 3);
    amrex::ParallelDescriptor::ReduceRealMax(candidate_report.maximum_relative_arithmetic_bound);
    amrex::MultiFab::Copy(output.rosseland_transport, candidate.rosseland_transport, 0, 0, 1,
                          output.rosseland_transport.nGrowVect());
    amrex::MultiFab::Copy(output.true_scattering, candidate.true_scattering, 0, 0, 1,
                          output.true_scattering.nGrowVect());
    amrex::MultiFab::Copy(output.proper_absorption, candidate.proper_absorption, 0, 0, 1,
                          output.proper_absorption.nGrowVect());
    if (report) { *report = candidate_report; }
    reason.clear(); return true;
#endif
}

ResolvedIndependentGrayMaterialTableOpacitySnapshot
TrySnapshotResolvedIndependentGrayMaterialTableOpacity (
    amrex::MultiFab const& raw_density, amrex::MultiFab const& nodal_temperature,
    ResolvedMaterialChargeFields const& unit_fields, ElectronThermodynamicsExecutor const& eos,
    amrex::Geometry const& geometry, MaterialOpacityTable const& table,
    ResolvedIndependentGrayMaterialTableOpacityOptions const& options)
{
    ResolvedIndependentGrayMaterialTableOpacitySnapshot out;
    out.reason = "unsupported actual single selected independent gray table snapshot";
    bool const same_controls = SameIndependentControls(options);
    bool const same_presence = IndependentPresence(table, true);
    if (!same_controls || !same_presence)
    {
        out.reason = "independent gray coefficient controls/owner/report presence differ between ranks";
        return out;
    }
#if !defined(WARPX_DIM_RZ) || defined(AMREX_USE_GPU) || \
    !defined(WARPX_USE_SINGULARITY_EOS) || !defined(WARPX_USE_MATERIAL_OPACITY_HDF5)
    amrex::ignore_unused(raw_density, nodal_temperature, unit_fields, eos, geometry);
    return out;
#else
    int low[4]{eos.m_num_materials, eos.m_use_resolved_material_selection,
               eos.tableStorage() != nullptr, unit_fields[0] != nullptr};
    int high[4]{low[0], low[1], low[2], low[3]};
    amrex::ParallelDescriptor::ReduceIntMin(low, 4);
    amrex::ParallelDescriptor::ReduceIntMax(high, 4);
    if (!std::equal(low, low + 4, high))
    {
        out.reason = "independent gray native selector/EOS/unit-field presence differ between ranks";
        return out;
    }
    bool valid = options.enabled && eos.isSingularitySpiner() && eos.m_num_materials == 1 &&
        eos.m_use_resolved_material_selection && eos.tableStorage() &&
        options.thermodynamic_floor == 0 && options.mass_per_electron > 0 &&
        options.mass_per_electron == PhysConst::q_e * eos.m_mass_per_physical_charge[0] &&
        raw_density.ixType().nodeCentered() && raw_density.nComp() >= 1 &&
        raw_density.nGrowVect().allGE(1) && nodal_temperature.nGrowVect().allGE(1) &&
        nodal_temperature.boxArray() == raw_density.boxArray() && nodal_temperature.nComp() == 1 &&
        nodal_temperature.DistributionMap() == raw_density.DistributionMap() &&
        unit_fields[0] && unit_fields[0]->nComp() == 1 &&
        unit_fields[0]->boxArray() == raw_density.boxArray() &&
        unit_fields[0]->DistributionMap() == raw_density.DistributionMap() &&
        unit_fields[0]->nGrowVect().allGE(1);
    if (!Collective(valid)) { return out; }
    for (amrex::MFIter it(raw_density); it.isValid(); ++it)
    {
        auto const u = unit_fields[0]->const_array(it);
        for (amrex::BoxIterator node(it.validbox()); node.ok(); ++node)
        {
            ElectronThermodynamicsExecutor::MaterialMassDensities mass{};
            mass[0] = u(node()) * eos.m_mass_per_charge[0];
            auto const raw = raw_density[it](node());
            bool const occupied = std::isfinite(mass[0]) && mass[0] > 0 &&
                eos.resolvedMaterial(mass) == 0 &&
                (!options.allow_exact_vacuum || (std::isfinite(raw) && raw > 0));
            bool const vacant = options.allow_exact_vacuum && raw == 0 && u(node()) == 0 &&
                mass[0] == 0 && eos.resolvedMaterial(mass) ==
                    static_cast<int>(warpx::materials::MaterialRegistry::ResolvedCell::Vacuum);
            valid = valid && (occupied || vacant);
        }
    }
    if (!Collective(valid))
    {
        out.reason = "independent gray table snapshot contains a non-material0 corner";
        return out;
    }
    auto const cells = amrex::convert(raw_density.boxArray(), amrex::IntVect(0));
    ResolvedIndependentGrayMaterialTableOpacitySnapshot candidate;
    candidate.electron_density.define(cells, raw_density.DistributionMap(), 1, 1);
    candidate.electron_temperature.define(cells, raw_density.DistributionMap(), 1, 1);
    candidate.coefficients.rosseland_transport.define(cells, raw_density.DistributionMap(), 1, 1);
    candidate.coefficients.true_scattering.define(cells, raw_density.DistributionMap(), 1, 1);
    candidate.coefficients.proper_absorption.define(cells, raw_density.DistributionMap(), 1, 1);
    auto const gather = options.allow_exact_vacuum
        ? SnapshotResolvedMaterialOpacityWithExactVacuum : SnapshotResolvedMaterialOpacity;
    if (!gather(raw_density, nodal_temperature,
            candidate.electron_density, candidate.electron_temperature, geometry, eos,
            options.minimum_electron_density, options.thermodynamic_floor, unit_fields))
    {
        out.reason = "invalid native four-corner independent material gather"; return out;
    }
    if (!TryFillResolvedIndependentGrayMaterialTableOpacity(candidate.coefficients,
            candidate.electron_density, candidate.electron_temperature, geometry, table,
            options, out.reason, &candidate.report)) { return out; }
    candidate.valid = true;
    return candidate;
#endif
}
} // namespace warpx::radiation::detail
