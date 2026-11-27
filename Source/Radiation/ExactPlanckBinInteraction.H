/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_EXACTPLANCKBININTERACTION_H_
#define WARPX_RADIATION_EXACTPLANCKBININTERACTION_H_

#include "MovingGroupInteraction.H"
#include "IsotropicDipoleFrameMatrices.H"

#include <type_traits>

namespace warpx::radiation
{
constexpr int exact_group_maximum = 8;
constexpr int exact_group_components = 4 * exact_group_maximum;
using ExactGroupVector = amrex::GpuArray<amrex::Real, exact_group_components>;

struct ExactPlanckBinSourceResult
{
    ExactGroupVector radiation{};
    // Extensive laboratory (energy,c*impulse), J, retained independently of
    // a subtraction between large radiation states.
    ExactGroupVector material_transfer{};
    // Each spectral/component arithmetic envelope uses its own uncancelled
    // angular source terms and rate-limited response to incoming-state ulps.
    ExactGroupVector transfer_roundoff{};
    ExactGroupVector comoving_delta{};
    amrex::Real physical_heat = 0;
    amrex::Real physical_heat_roundoff = 0;
    amrex::Real momentum_roundoff = 0;
    amrex::Real projection_residual = 0;
    amrex::Real projection_roundoff = 0;
    amrex::Real maximum_equation_residual = 0;
    bool valid = false;
};

namespace exact_group_detail
{
template <int MaximumGroups> using SizedVector = amrex::GpuArray<amrex::Real, 4 * MaximumGroups>;
template <int MaximumGroups>
using SizedMatrix = amrex::GpuArray<SizedVector<MaximumGroups>, 4 * MaximumGroups>;
template <int MaximumGroups>
using SizedPressureMatrix = amrex::GpuArray<SizedVector<MaximumGroups>, 6 * MaximumGroups>;
using Matrix = SizedMatrix<exact_group_maximum>;
// Symmetric LAB pressure rows per group: xx, yy, zz, xy, xz, yz.
using PressureMatrix = SizedPressureMatrix<exact_group_maximum>;
AMREX_GPU_HOST_DEVICE inline int PressureComponent (int i, int j)
{
    if (i == j)
    {
        return i;
    }
    auto const low = amrex::min(i, j), high = amrex::max(i, j);
    return low == 0 ? high + 2 : 5;
}
// Independent angular refinement tests use the 8-point rule; native source
// defaults to 16. Four azimuthal points exactly integrate its dipole/quadratic
// angular dependence when the polar axis is aligned with the velocity.
AMREX_GPU_HOST_DEVICE inline amrex::GpuArray<amrex::Real, 2> GaussianNodeWeight (int order, int p)
{
    constexpr amrex::Real nodes8[8] = {-0.96028985649753623168_rt, -0.79666647741362673959_rt,
                                       -0.52553240991632898582_rt, -0.18343464249564980494_rt,
                                       0.18343464249564980494_rt,  0.52553240991632898582_rt,
                                       0.79666647741362673959_rt,  0.96028985649753623168_rt};
    constexpr amrex::Real weights8[8] = {0.10122853629037625915_rt, 0.22238103445337447054_rt,
                                         0.31370664587788728734_rt, 0.36268378337836198297_rt,
                                         0.36268378337836198297_rt, 0.31370664587788728734_rt,
                                         0.22238103445337447054_rt, 0.10122853629037625915_rt};
    constexpr amrex::Real nodes16[16] = {
        -0.98940093499164993260_rt, -0.94457502307323257608_rt, -0.86563120238783174388_rt,
        -0.75540440835500303390_rt, -0.61787624440264374845_rt, -0.45801677765722738634_rt,
        -0.28160355077925891323_rt, -0.09501250983763744019_rt, 0.09501250983763744019_rt,
        0.28160355077925891323_rt,  0.45801677765722738634_rt,  0.61787624440264374845_rt,
        0.75540440835500303390_rt,  0.86563120238783174388_rt,  0.94457502307323257608_rt,
        0.98940093499164993260_rt};
    constexpr amrex::Real weights16[16] = {
        0.02715245941175409485_rt, 0.06225352393864789286_rt, 0.09515851168249278481_rt,
        0.12462897125553387205_rt, 0.14959598881657673208_rt, 0.16915651939500253819_rt,
        0.18260341504492358887_rt, 0.18945061045506849629_rt, 0.18945061045506849629_rt,
        0.18260341504492358887_rt, 0.16915651939500253819_rt, 0.14959598881657673208_rt,
        0.12462897125553387205_rt, 0.09515851168249278481_rt, 0.06225352393864789286_rt,
        0.02715245941175409485_rt};
    return order == 8 ? amrex::GpuArray<amrex::Real, 2>{nodes8[p], weights8[p]}
                      : amrex::GpuArray<amrex::Real, 2>{nodes16[p], weights16[p]};
}

AMREX_GPU_HOST_DEVICE inline amrex::Real
PlanckDensity (amrex::Real x)
{
    if (x == 0)
    {
        return 0;
    }
    return (x < 64 ? x * x * x / std::expm1(x)
                   : std::exp(3 * std::log(x) - x) / (1 - std::exp(-x))) /
           spectral_detail::planck_integral;
}

// Shared positive-panel kernel qualified by the independent centered-bath
// oracle. Width is supplied before scaling/rounding the upper endpoint.
AMREX_GPU_HOST_DEVICE inline amrex::Real
PositivePlanckInterval (amrex::Real low, amrex::Real width)
{
    if (!(width > 0))
    {
        return 0;
    }
    auto const high = low + width;
    if (low >= 64 && width > .25_rt)
    {
        return spectral_detail::Tail(low) - spectral_detail::Tail(high);
    }
    auto const finite_width = low >= 64 ? width : amrex::min(width, 64 - low);
    auto const panels = static_cast<int>(std::ceil(finite_width));
    auto const panel_width = finite_width / panels;
    auto const order = width <= .25_rt ? 8 : 16;
    amrex::Real result = 0, carry = 0;
    for (int p = 0; p < panels; ++p)
    {
        for (int node = 0; node < order; ++node)
        {
            auto const rule = GaussianNodeWeight(order, node);
            auto const x = low + panel_width * (p + (1 + rule[0]) / 2);
            auto const value = panel_width / 2 * rule[1] * PlanckDensity(x) - carry;
            auto const next = result + value;
            carry = (next - result) - value;
            result = next;
        }
    }
    if (high > 64 && low < 64)
    {
        result += spectral_detail::Tail(64) - spectral_detail::Tail(high);
    }
    return result;
}

AMREX_GPU_HOST_DEVICE inline amrex::Real Dot (amrex::GpuArray<amrex::Real, 3> const &a,
                                              amrex::GpuArray<amrex::Real, 3> const &b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
AMREX_GPU_HOST_DEVICE inline amrex::GpuArray<amrex::Real, 3>
Cross (amrex::GpuArray<amrex::Real, 3> const &a, amrex::GpuArray<amrex::Real, 3> const &b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
AMREX_GPU_HOST_DEVICE inline amrex::Real
RangeFraction (amrex::Real lower, amrex::Real upper, amrex::Real thermal,
               bool positive_intervals = false)
{
    using namespace spectral_detail;
    auto const low = lower / thermal, high = upper / thermal;
    if (positive_intervals)
    {
        auto const width = (upper - lower) / thermal;
        if (width <= .25_rt)
        {
            return PositivePlanckInterval(low, width);
        }
    }
    if (high < 0.5_rt)
    {
        return LowCdf(high) - LowCdf(low);
    }
    if (low < 0.5_rt)
    {
        return 1 - LowCdf(low) - Tail(high);
    }
    return Tail(low) - Tail(high);
}

/** Direct Lorentz integration of the declared COMOVING bin dipoles.
 * moment maps comoving extensive (E,F/c) to LAB bins. source maps the same
 * variables to dt*(q_lab,c*f_lab). I_lab=D^3 I0 gives D^2 after changing
 * solid angle/frequency; interaction S_lab=D^2 S0 gives D. Proper opacities
 * belong to their original comoving bins even when a lab bin receives more
 * than one. Every angular kink at a shifted spectral boundary is split. */
template <typename MatrixType, typename PressureType = PressureMatrix>
AMREX_GPU_HOST_DEVICE inline bool
Assemble (EnergyGroupsExecutor const& groups, amrex::GpuArray<amrex::Real, 3> const& beta,
          amrex::Real temperature, amrex::Real dt, amrex::Real const* transport,
          amrex::Real const* absorption, MatrixType& moment, MatrixType& source, int angular_order,
          std::type_identity_t<MatrixType>* source_operands = nullptr,
          PressureType* pressure = nullptr, bool analytic_azimuth = false,
          bool positive_intervals = false)
{
    using namespace spectral_detail;
    auto const n = groups.m_num_groups;
    auto const speed = std::sqrt(Dot(beta, beta));
    auto const thermal = PhysConst::kb * temperature;
    if (!transport || !absorption || !GroupsValid(groups) || n > exact_group_maximum ||
        n > static_cast<int>(moment.size()) / 4 ||
        (pressure && n > static_cast<int>(pressure->size()) / 6) || !(temperature > 0) ||
        !amrex::Math::isfinite(temperature) || !(thermal > 0) || !amrex::Math::isfinite(thermal) ||
        !(speed <= 0.01_rt) || !amrex::Math::isfinite(speed) || !(dt >= 0) ||
        !amrex::Math::isfinite(dt) || (angular_order != 8 && angular_order != 16))
    {
        return false;
    }
    amrex::GpuArray<amrex::Real, exact_group_maximum> fractions{}, rates_a{}, rates_t{};
    for (int g = 0; g < n; ++g)
    {
        fractions[g] = Fraction(groups, g, temperature);
        rates_a[g] = PhysConst::c * dt * absorption[g];
        rates_t[g] = PhysConst::c * dt * transport[g];
        if (!(absorption[g] >= 0 && absorption[g] <= transport[g]) ||
            !amrex::Math::isfinite(rates_a[g]) || !amrex::Math::isfinite(rates_t[g]) ||
            !(fractions[g] > 0) || !amrex::Math::isfinite(fractions[g]))
        {
            return false;
        }
    }
    if (n == 1)
    {
        StressEnergy m{}, s{};
        if (!BuildIsotropicDipoleFrameMatrices(beta, m, s))
        {
            return false;
        }
        for (int row = 0; row < 4; ++row)
        {
            for (int column = 0; column < 4; ++column)
            {
                moment[row][column] = m[row][column];
                source[row][column] = s[row][column] * (column == 0 ? rates_a[0] : rates_t[0]);
                if (source_operands)
                {
                    (*source_operands)[row][column] = std::abs(source[row][column]);
                }
            }
        }
        if (pressure)
        {
            for (int column = 0; column < 4; ++column)
            {
                StressEnergy rest{};
                if (column == 0)
                {
                    rest[0][0] = 1;
                    for (int d = 1; d < 4; ++d)
                    {
                        rest[d][d] = 1.0_rt / 3;
                    }
                }
                else
                {
                    rest[0][column] = rest[column][0] = 1;
                }
                auto const lab = BoostStressEnergy(rest, beta);
                for (int i = 0; i < 3; ++i)
                {
                    for (int j = i; j < 3; ++j)
                    {
                        (*pressure)[PressureComponent(i, j)][column] = lab[i + 1][j + 1];
                    }
                }
            }
        }
        return true;
    }
    if (speed == 0)
    {
        for (int g = 0; g < n; ++g)
        {
            for (int d = 0; d < 4; ++d)
            {
                moment[4 * g + d][4 * g + d] = 1;
                source[4 * g + d][4 * g + d] = d == 0 ? rates_a[g] : rates_t[g];
                if (source_operands)
                {
                    (*source_operands)[4 * g + d][4 * g + d] = source[4 * g + d][4 * g + d];
                }
            }
            if (pressure)
            {
                for (int d = 0; d < 3; ++d)
                {
                    (*pressure)[6 * g + d][4 * g] = 1.0_rt / 3;
                }
            }
        }
        return true;
    }
    auto const gamma = 1 / std::sqrt(1 - speed * speed);
    amrex::GpuArray<amrex::Real, 3> axis{};
    for (int d = 0; d < 3; ++d)
    {
        axis[d] = beta[d] / speed;
    }
    auto transverse =
        Cross(axis, std::abs(axis[0]) < 0.9_rt ? amrex::GpuArray<amrex::Real, 3>{1, 0, 0}
                                               : amrex::GpuArray<amrex::Real, 3>{0, 1, 0});
    auto const transverse_norm = std::sqrt(Dot(transverse, transverse));
    for (auto &value : transverse)
    {
        value /= transverse_norm;
    }
    auto const second = Cross(axis, transverse);
    constexpr int maximum_cuts = (exact_group_maximum - 1) * (exact_group_maximum - 1) + 3;
    amrex::GpuArray<amrex::Real, maximum_cuts> cuts{};
    int cut_count = 2;
    cuts[0] = -1;
    cuts[1] = 1;
    for (int input = 0; input < n - 1; ++input)
    {
        for (int output = 0; output < n - 1; ++output)
        {
            auto const cut =
                (groups.m_boundaries[output] / groups.m_boundaries[input] / gamma - 1) / speed;
            if (!(cut > -1 && cut < 1))
            {
                continue;
            }
            int index = 0;
            while (index < cut_count && cuts[index] < cut)
            {
                ++index;
            }
            if (index < cut_count && cuts[index] == cut)
            {
                continue;
            }
            for (int e = cut_count; e > index; --e)
            {
                cuts[e] = cuts[e - 1];
            }
            cuts[index] = cut;
            ++cut_count;
        }
    }
    auto const infinity = std::numeric_limits<amrex::Real>::infinity();
    for (int segment = 0; segment < cut_count - 1; ++segment)
    {
        auto const low_mu = cuts[segment], high_mu = cuts[segment + 1];
        for (int p = 0; p < angular_order; ++p)
        {
            auto const quadrature = GaussianNodeWeight(angular_order, p);
            auto const node = quadrature[0], gauss_weight = quadrature[1];
            auto const mu = (low_mu + high_mu) / 2 + (high_mu - low_mu) / 2 * node;
            auto const weight = (high_mu - low_mu) / 4 * gauss_weight / 4;
            if (analytic_azimuth)
            {
                // At fixed mu every declared moment/pressure integrand is
                // at most cubic in the transverse direction. Its azimuth
                // average is exactly <t_i>=<t_i t_j t_k>=0 and
                // <t_i t_j>=(delta_ij-axis_i axis_j)/2. Use that tensor
                // identity, not cancellation of four rounded sin/cos samples.
                auto const radius = std::sqrt(1 - mu * mu);
                auto const doppler = gamma * (1 + speed * mu);
                auto const parallel = (mu + speed) / (1 + speed * mu);
                auto const perpendicular = radius / doppler;
                for (int input = 0; input < n; ++input)
                {
                    auto const input_low = input == 0 ? 0 : groups.m_boundaries[input - 1];
                    auto const input_high = input + 1 == n ? infinity : groups.m_boundaries[input];
                    for (int output = 0; output < n; ++output)
                    {
                        auto const output_low =
                            output == 0 ? 0 : groups.m_boundaries[output - 1] / doppler;
                        auto const output_high =
                            output + 1 == n ? infinity : groups.m_boundaries[output] / doppler;
                        auto const low = amrex::max(input_low, output_low);
                        auto const high = amrex::min(input_high, output_high);
                        if (!(high > low))
                        {
                            continue;
                        }
                        auto const fraction =
                            RangeFraction(low, high, thermal, positive_intervals) /
                            fractions[input];
                        if (!(fraction >= 0) || !amrex::Math::isfinite(fraction))
                        {
                            return false;
                        }
                        auto const m_weight = 4 * weight * doppler * doppler * fraction;
                        auto const s_weight = 4 * weight * doppler * fraction;
                        for (int row = 0; row < 4; ++row)
                        {
                            for (int column = 0; column < 4; ++column)
                            {
                                amrex::Real first = 1, second_term = 0;
                                if (row == 0 && column != 0)
                                {
                                    first = 3 * mu * axis[column - 1];
                                }
                                else if (row != 0 && column == 0)
                                {
                                    first = parallel * axis[row - 1];
                                }
                                else if (row != 0 && column != 0)
                                {
                                    auto const product = axis[row - 1] * axis[column - 1];
                                    first = 3 * parallel * mu * product;
                                    second_term = 3 * perpendicular * radius / 2 *
                                                  ((row == column ? 1 : 0) - product);
                                }
                                auto const factor = first + second_term;
                                auto const rate = column == 0 ? rates_a[input] : rates_t[input];
                                moment[4 * output + row][4 * input + column] += m_weight * factor;
                                source[4 * output + row][4 * input + column] +=
                                    s_weight * factor * rate;
                                if (source_operands)
                                {
                                    (*source_operands)[4 * output + row][4 * input + column] +=
                                        s_weight * (std::abs(first) + std::abs(second_term)) * rate;
                                }
                            }
                        }
                        if (pressure)
                        {
                            for (int i = 0; i < 3; ++i)
                            {
                                for (int j = i; j < 3; ++j)
                                {
                                    auto const tensor = (i == j ? 1 : 0) - axis[i] * axis[j];
                                    auto const monopole =
                                        parallel * parallel * axis[i] * axis[j] +
                                        perpendicular * perpendicular * tensor / 2;
                                    auto const pressure_row = 6 * output + PressureComponent(i, j);
                                    (*pressure)[pressure_row][4 * input] += m_weight * monopole;
                                    for (int d = 0; d < 3; ++d)
                                    {
                                        auto const id = (i == d ? 1 : 0) - axis[i] * axis[d];
                                        auto const jd = (j == d ? 1 : 0) - axis[j] * axis[d];
                                        auto const dipole =
                                            3 * (mu * axis[d] * monopole +
                                                 parallel * perpendicular * radius / 2 *
                                                     (axis[i] * jd + axis[j] * id));
                                        (*pressure)[pressure_row][4 * input + d + 1] +=
                                            m_weight * dipole;
                                    }
                                }
                            }
                        }
                    }
                }
                continue;
            }
            for (int a = 0; a < 4; ++a)
            {
                auto const phi = 2 * MathConst::pi * (a + 0.5_rt) / 4;
                amrex::GpuArray<amrex::Real, 3> direction{}, out{};
                for (int d = 0; d < 3; ++d)
                {
                    direction[d] = mu * axis[d] +
                                   std::sqrt(1 - mu * mu) *
                                       (std::cos(phi) * transverse[d] + std::sin(phi) * second[d]);
                }
                auto const projection = Dot(beta, direction);
                auto const doppler = gamma * (1 + projection);
                for (int d = 0; d < 3; ++d)
                {
                    out[d] = (direction[d] +
                              (gamma + gamma * gamma / (gamma + 1) * projection) * beta[d]) /
                             doppler;
                }
                for (int input = 0; input < n; ++input)
                {
                    auto const input_low = input == 0 ? 0 : groups.m_boundaries[input - 1];
                    auto const input_high = input + 1 == n ? infinity : groups.m_boundaries[input];
                    for (int output = 0; output < n; ++output)
                    {
                        auto const output_low =
                            output == 0 ? 0 : groups.m_boundaries[output - 1] / doppler;
                        auto const output_high =
                            output + 1 == n ? infinity : groups.m_boundaries[output] / doppler;
                        auto const low = amrex::max(input_low, output_low);
                        auto const high = amrex::min(input_high, output_high);
                        if (!(high > low))
                        {
                            continue;
                        }
                        auto const fraction =
                            RangeFraction(low, high, thermal, positive_intervals) /
                            fractions[input];
                        if (!(fraction >= 0) || !amrex::Math::isfinite(fraction))
                        {
                            return false;
                        }
                        auto const m_weight = weight * doppler * doppler * fraction;
                        auto const s_weight = weight * doppler * fraction;
                        for (int row = 0; row < 4; ++row)
                        {
                            auto const output_factor = row == 0 ? 1 : out[row - 1];
                            for (int column = 0; column < 4; ++column)
                            {
                                auto const input_factor =
                                    column == 0 ? 1 : 3 * direction[column - 1];
                                moment[4 * output + row][4 * input + column] +=
                                    m_weight * output_factor * input_factor;
                                auto const interaction =
                                    s_weight * output_factor * input_factor *
                                    (column == 0 ? rates_a[input] : rates_t[input]);
                                source[4 * output + row][4 * input + column] += interaction;
                                if (source_operands)
                                {
                                    (*source_operands)[4 * output + row][4 * input + column] +=
                                        std::abs(interaction);
                                }
                            }
                        }
                        if (pressure)
                        {
                            for (int i = 0; i < 3; ++i)
                            {
                                for (int j = i; j < 3; ++j)
                                {
                                    for (int column = 0; column < 4; ++column)
                                    {
                                        auto const input_factor =
                                            column == 0 ? 1 : 3 * direction[column - 1];
                                        (*pressure)[6 * output + PressureComponent(i, j)]
                                                   [4 * input + column] +=
                                            m_weight * out[i] * out[j] * input_factor;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return true;
}

// The caller provides disposable LU workspace: do not copy another full
// matrix onto each GPU thread's stack. RHS/solution may keep maximum-size
// vectors because those are small and are shared with the public result.
template <typename MatrixType>
AMREX_GPU_HOST_DEVICE inline bool Solve (MatrixType &matrix, ExactGroupVector rhs, int size,
                                         ExactGroupVector &solution)
{
    for (int row = 0; row < size; ++row)
    {
        amrex::Real scale = 0;
        for (int column = 0; column < size; ++column)
        {
            scale = amrex::max(scale, std::abs(matrix[row][column]));
        }
        if (!(scale > 0) || !amrex::Math::isfinite(scale))
        {
            return false;
        }
        for (int column = 0; column < size; ++column)
        {
            matrix[row][column] /= scale;
        }
        rhs[row] /= scale;
    }
    for (int column = 0; column < size; ++column)
    {
        int pivot = column;
        for (int row = column + 1; row < size; ++row)
        {
            if (std::abs(matrix[row][column]) > std::abs(matrix[pivot][column]))
            {
                pivot = row;
            }
        }
        if (matrix[pivot][column] == 0 || !amrex::Math::isfinite(matrix[pivot][column]))
        {
            return false;
        }
        auto const saved_row = matrix[column];
        matrix[column] = matrix[pivot];
        matrix[pivot] = saved_row;
        auto const saved_rhs = rhs[column];
        rhs[column] = rhs[pivot];
        rhs[pivot] = saved_rhs;
        for (int row = column + 1; row < size; ++row)
        {
            auto const factor = matrix[row][column] / matrix[column][column];
            for (int next = column + 1; next < size; ++next)
            {
                matrix[row][next] -= factor * matrix[column][next];
            }
            rhs[row] -= factor * rhs[column];
        }
    }
    for (int row = size - 1; row >= 0; --row)
    {
        auto value = rhs[row];
        for (int column = row + 1; column < size; ++column)
        {
            value -= matrix[row][column] * solution[column];
        }
        solution[row] = value / matrix[row][row];
        if (!amrex::Math::isfinite(solution[row]))
        {
            return false;
        }
    }
    return true;
}
} // namespace exact_group_detail

/** Backward-Euler finite-bin LAB source with the explicit COMOVING Planck-bin
 * positive dipole reconstruction. State stores extensive (E_g,F_g/c), J.
 * Density/composition and proper opacities are frozen for this temperature/
 * finite-kick-secant trial. LTE emission and reconstruction share the supplied
 * positive cell temperature. Exact angular/frequency Lorentz integration
 * retains opacity/spectral jumps and all beta orders; no gray-per-bin source
 * or first-order frame transform is substituted. Solve centered at comoving
 * LTE to retain stiff/sub-ULP heat without subtracting rounded large states.
 * Inputs are immutable. Failure returns incoming radiation and zero transfers.
 * Native caloric and recoil iteration is supplied by CoupledGroupSource. */
template <int MaximumGroups = exact_group_maximum>
AMREX_GPU_HOST_DEVICE inline ExactPlanckBinSourceResult TryImplicitExactPlanckBinSource (
    ExactGroupVector const &old_lab, EnergyGroupsExecutor const &groups,
    amrex::GpuArray<amrex::Real, 3> const &beta, amrex::Real temperature, amrex::Real volume,
    amrex::Real dt, amrex::Real const *transport, amrex::Real const *absorption,
    int angular_order = 16)
{
    using namespace exact_group_detail;
    using namespace spectral_detail;
    static_assert(MaximumGroups >= 1 && MaximumGroups <= exact_group_maximum);
    ExactPlanckBinSourceResult rejected;
    rejected.radiation = old_lab;
    auto const n = groups.m_num_groups;
    if (!GroupsValid(groups) || n > MaximumGroups || !(volume > 0) ||
        !amrex::Math::isfinite(volume))
    {
        return rejected;
    }
    auto const blackbody = Blackbody(temperature) * volume;
    if (!amrex::Math::isfinite(blackbody))
    {
        return rejected;
    }
    for (int g = 0; g < n; ++g)
    {
        GroupSpectralMoments old;
        old.energy = old_lab[4 * g];
        for (int d = 0; d < 3; ++d)
        {
            old.flux[d] = PhysConst::c * old_lab[4 * g + d + 1];
        }
        if (!MomentValid(old, 1))
        {
            return rejected;
        }
    }
    SizedMatrix<MaximumGroups> moment{}, source{}, combined{}, source_operands{};
    if (!Assemble(groups, beta, temperature, dt, transport, absorption, moment, source,
                  angular_order, &source_operands))
    {
        return rejected;
    }
    ExactGroupVector equilibrium{}, baseline{}, rhs{}, delta{};
    for (int g = 0; g < n; ++g)
    {
        equilibrium[4 * g] = blackbody * Fraction(groups, g, temperature);
    }
    for (int row = 0; row < 4 * n; ++row)
    {
        for (int column = 0; column < 4 * n; ++column)
        {
            baseline[row] += moment[row][column] * equilibrium[column];
            combined[row][column] = moment[row][column] + source[row][column];
        }
        rhs[row] = old_lab[row] - baseline[row];
    }
    if (!Solve(combined, rhs, 4 * n, delta))
    {
        return rejected;
    }
    ExactPlanckBinSourceResult result;
    result.comoving_delta = delta;
    amrex::Real total_energy = 0;
    amrex::GpuArray<amrex::Real, 3> total_q{};
    auto const gamma = 1 / std::sqrt(1 - Dot(beta, beta));
    amrex::Real projection_operands = 0;
    for (int g = 0; g < n; ++g)
    {
        GroupSpectralMoments rest;
        rest.energy = equilibrium[4 * g] + delta[4 * g];
        for (int d = 0; d < 3; ++d)
        {
            rest.flux[d] = PhysConst::c * delta[4 * g + d + 1];
        }
        if (!MomentValid(rest, 1.0_rt / 3))
        {
            return rejected;
        }
        auto const rest_heat = PhysConst::c * dt * absorption[g] * delta[4 * g];
        result.physical_heat += rest_heat / gamma;
        projection_operands += std::abs(rest_heat / gamma);
        result.physical_heat_roundoff += 64 * std::numeric_limits<amrex::Real>::epsilon() *
                                         amrex::min(1.0_rt, PhysConst::c * dt * absorption[g]) *
                                         (std::abs(old_lab[4 * g]) + std::abs(equilibrium[4 * g]));
        result.momentum_roundoff += 64 * std::numeric_limits<amrex::Real>::epsilon() *
                                    amrex::min(1.0_rt, PhysConst::c * dt * transport[g]) *
                                    (std::abs(old_lab[4 * g]) + std::abs(equilibrium[4 * g]));
    }
    for (int row = 0; row < 4 * n; ++row)
    {
        auto value = baseline[row];
        amrex::Real transfer = 0, operands = std::abs(old_lab[row]) + std::abs(baseline[row]);
        amrex::Real transfer_operands = 0;
        for (int column = 0; column < 4 * n; ++column)
        {
            auto const m = moment[row][column] * delta[column];
            auto const s = source[row][column] * delta[column];
            value += m;
            transfer += s;
            operands += std::abs(m) + std::abs(s);
            auto const input_group = column / 4;
            auto const rate = PhysConst::c * dt *
                              (column % 4 == 0 ? absorption[input_group] : transport[input_group]);
            transfer_operands +=
                source_operands[row][column] *
                (std::abs(delta[column]) +
                 (std::abs(old_lab[4 * input_group]) + std::abs(equilibrium[4 * input_group])) /
                     (1 + rate));
            projection_operands += (row % 4 == 0 ? 1 : std::abs(beta[row % 4 - 1])) *
                                   source_operands[row][column] * std::abs(delta[column]);
        }
        if (!amrex::Math::isfinite(value) || !amrex::Math::isfinite(transfer) ||
            !amrex::Math::isfinite(operands))
        {
            return rejected;
        }
        auto const error = std::abs((value - old_lab[row]) + transfer);
        auto const bound = 128 * std::numeric_limits<amrex::Real>::epsilon() * operands;
        result.maximum_equation_residual =
            amrex::max(result.maximum_equation_residual, operands == 0 ? error : error / operands);
        if (error > bound)
        {
            return rejected;
        }
        result.radiation[row] = value;
        result.material_transfer[row] = transfer;
        result.transfer_roundoff[row] =
            128 * std::numeric_limits<amrex::Real>::epsilon() * transfer_operands;
        if (row % 4 == 0)
        {
            total_energy += transfer;
            projection_operands += std::abs(transfer);
        }
        else
        {
            total_q[row % 4 - 1] += transfer;
            projection_operands += std::abs(beta[row % 4 - 1] * transfer);
        }
    }
    for (int g = 0; g < n; ++g)
    {
        GroupSpectralMoments lab;
        lab.energy = result.radiation[4 * g];
        for (int d = 0; d < 3; ++d)
        {
            lab.flux[d] = PhysConst::c * result.radiation[4 * g + d + 1];
        }
        if (!MomentValid(lab, 1))
        {
            return rejected;
        }
    }
    result.projection_residual = total_energy - Dot(beta, total_q) - result.physical_heat;
    result.projection_roundoff =
        128 * std::numeric_limits<amrex::Real>::epsilon() * projection_operands;
    if (!amrex::Math::isfinite(result.physical_heat) ||
        !amrex::Math::isfinite(result.physical_heat_roundoff) ||
        !amrex::Math::isfinite(result.projection_roundoff) ||
        std::abs(result.projection_residual) > result.projection_roundoff)
    {
        return rejected;
    }
    result.valid = true;
    return result;
}
} // namespace warpx::radiation
#endif
