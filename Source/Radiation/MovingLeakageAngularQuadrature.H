/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_MOVINGLEAKAGEANGULARQUADRATURE_H_
#define WARPX_RADIATION_MOVINGLEAKAGEANGULARQUADRATURE_H_

#include <array>
#include <cmath>
#include <utility>

namespace warpx::radiation
{
struct MovingLeakageAngularQuadrature
{
    bool valid = false;
    std::array<std::array<long double, 3>, 8> direction{};
    std::array<long double, 8> fraction{};
};

/** Lab-fixed outward hemisphere quadrature for isotropic comoving intensity.
 * I_lab=D^4 I0, D=1/[gamma(1-beta.n_lab)]. Weight by lab cosine quadrature,
 * then normalize to the caller's authoritative LAB leakage energy. This is
 * angular redistribution, not an additional Doppler energy exchange.
 * All O(beta) angular moments are integrated exactly by this eight-ray rule.
 * Higher angular quadrature errors remain; this is not an exact Milne law.
 * physical_axis selects the Cartesian normal (0=x, 1=y, 2=z).
 * A fixed comoving cosine quadrature times D^2 mu_lab/mu0 would incorrectly
 * integrate 1/mu0 at first order, yielding a biased normal derivative.
 */
inline MovingLeakageAngularQuadrature
ComputeMovingLeakageAngularQuadrature (std::array<double, 3> const& beta, int sign,
                                      int physical_axis = 2)
{
    MovingLeakageAngularQuadrature result;
    if (physical_axis < 0 || physical_axis > 2 ||
        (sign != -1 && sign != 1) || !std::isfinite(beta[0]) ||
        !std::isfinite(beta[1]) || !std::isfinite(beta[2]) ||
        std::hypot(beta[0], beta[1], beta[2]) > .001) { return result; }
    constexpr long double sqrt6 = 2.4494897427831780981972840747058913919659474806567L;
    constexpr std::array<long double, 2> mus = {(6-sqrt6)/10, (6+sqrt6)/10};
    constexpr std::array<long double, 2> weights = {(9-sqrt6)/18, (9+sqrt6)/18};
    bool const at_rest = beta[0] == 0 && beta[1] == 0 && beta[2] == 0;
    long double normalization = 0;
    for (int node = 0; node < 2; ++node) {
        auto const transverse = std::sqrt(1-mus[node]*mus[node]);
        for (int azimuth = 0; azimuth < 4; ++azimuth) {
            int const index = 4*node+azimuth;
            auto& n = result.direction[index];
            n = {transverse*(azimuth == 0 ? 1.L : azimuth == 2 ? -1.L : 0.L),
                 transverse*(azimuth == 1 ? 1.L : azimuth == 3 ? -1.L : 0.L),
                 sign*mus[node]};
            if (physical_axis != 2) { std::swap(n[physical_axis], n[2]); }
            long double const denominator = 1-beta[0]*n[0]-beta[1]*n[1]-beta[2]*n[2];
            long double const squared = denominator*denominator;
            // Common gamma^-4 cancels exactly in the prescribed-energy normalization.
            result.fraction[index] = at_rest ? weights[node]/4 : weights[node]/(4*squared*squared);
            normalization += result.fraction[index];
        }
    }
    if (!(normalization > 0) || !std::isfinite(normalization)) { return {}; }
    if (!at_rest) {
        for (auto& weight : result.fraction) { weight /= normalization; }
    }
    result.valid = true;
    return result;
}
}
#endif
