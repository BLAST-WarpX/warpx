/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_TEST_FINITEMASSPHOTONREFERENCE_H_
#define WARPX_TEST_FINITEMASSPHOTONREFERENCE_H_

#include <array>
#include <cmath>
#include <limits>
#include <vector>

/** Independent host reference in units M=c=1. It uses aberration for direction
 * and the scalar Doppler invariant E(1-beta.n) for energy; it does not call the
 * production four-vector boost, collision kernel, or material work evaluator.
 */
namespace finite_mass_reference
{
    using Vector = std::array<long double, 3>;

    inline long double Dot (Vector const& a, Vector const& b)
    {
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
    }

    inline Vector Unit (Vector v)
    {
        auto const norm = std::sqrt(Dot(v, v));
        for (auto& x : v) { x /= norm; }
        return v;
    }

    struct Material {
        long double mass_fraction;
        Vector proper_velocity;
    };

    struct Photon {
        long double energy;
        Vector incoming_direction;
        Vector frame_outgoing_direction;
    };

    struct Solution {
        bool valid = false;
        int iterations = 0;
        Vector impulse{};
        Vector frame_velocity{};
        long double material_work = 0;
        std::vector<long double> outgoing_energy;
        std::vector<Vector> outgoing_direction;
    };

    inline Vector Secant (std::vector<Material> const& matter, Vector const& impulse)
    {
        Vector beta{};
        for (auto const& particle : matter) {
            auto next = particle.proper_velocity;
            for (int d = 0; d < 3; ++d) { next[d] += impulse[d]; }
            auto const denominator = std::sqrt(1 + Dot(particle.proper_velocity,
                                                       particle.proper_velocity))
                                   + std::sqrt(1 + Dot(next, next));
            for (int d = 0; d < 3; ++d) {
                beta[d] += particle.mass_fraction
                         * (particle.proper_velocity[d] + next[d]) / denominator;
            }
        }
        return beta;
    }

    inline long double Work (std::vector<Material> const& matter, Vector const& impulse)
    {
        long double work = 0;
        for (auto const& particle : matter) {
            auto next = particle.proper_velocity;
            long double numerator = 0;
            for (int d = 0; d < 3; ++d) {
                next[d] += impulse[d];
                numerator += (2*particle.proper_velocity[d] + impulse[d])*impulse[d];
            }
            auto const denominator = std::sqrt(1 + Dot(particle.proper_velocity,
                                                       particle.proper_velocity))
                                   + std::sqrt(1 + Dot(next, next));
            work += particle.mass_fraction * numerator / denominator;
        }
        return work;
    }

    inline Vector Replay (std::vector<Photon> const& photons, Vector const& beta,
                          Solution& result)
    {
        auto const gamma = 1/std::sqrt(1-Dot(beta, beta));
        Vector impulse{};
        result.outgoing_energy.resize(photons.size());
        result.outgoing_direction.resize(photons.size());
        for (std::size_t p = 0; p < photons.size(); ++p) {
            auto const& photon = photons[p];
            auto const projection = Dot(beta, photon.frame_outgoing_direction);
            auto& direction = result.outgoing_direction[p];
            for (int d = 0; d < 3; ++d) {
                direction[d] = (photon.frame_outgoing_direction[d]/gamma + beta[d]
                    + gamma/(gamma+1)*projection*beta[d])/(1+projection);
            }
            auto const energy = photon.energy*(1-Dot(beta, photon.incoming_direction))
                                             /(1-Dot(beta, direction));
            result.outgoing_energy[p] = energy;
            for (int d = 0; d < 3; ++d) {
                impulse[d] += photon.energy*photon.incoming_direction[d]
                            - energy*direction[d];
            }
        }
        return impulse;
    }

    inline Solution Solve (std::vector<Material> const& matter,
                           std::vector<Photon> const& photons)
    {
        Solution result;
        auto const tolerance = 128*std::numeric_limits<long double>::epsilon();
        for (int iteration = 0; iteration < 4096; ++iteration) {
            result.iterations = iteration+1;
            result.frame_velocity = Secant(matter, result.impulse);
            auto const candidate = Replay(photons, result.frame_velocity, result);
            long double residual = 0;
            long double scale = 0;
            for (int d = 0; d < 3; ++d) {
                residual += std::abs(candidate[d]-result.impulse[d]);
                scale += std::abs(candidate[d])+std::abs(result.impulse[d]);
            }
            if (residual <= tolerance*scale) {
                result.material_work = Work(matter, result.impulse);
                result.valid = true;
                return result;
            }
            for (int d = 0; d < 3; ++d) {
                result.impulse[d] = (result.impulse[d]+candidate[d])/2;
            }
        }
        return result;
    }

    /** Single cold stream mass-shell identity, evaluated at the measured lab
     * outgoing direction. Valid for one photon and identical initial velocities
     * (including many weighted native particles); not for a thermal mixture.
     */
    inline long double MassShellOutgoingEnergy (Vector const& initial_proper_velocity,
                                                Photon const& photon,
                                                Vector const& outgoing_direction)
    {
        auto const gamma = std::sqrt(1+Dot(initial_proper_velocity, initial_proper_velocity));
        return photon.energy*(gamma-Dot(initial_proper_velocity, photon.incoming_direction))
            /(gamma-Dot(initial_proper_velocity, outgoing_direction)
              + photon.energy*(1-Dot(photon.incoming_direction, outgoing_direction)));
    }
}

#endif
