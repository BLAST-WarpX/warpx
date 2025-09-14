/* Copyright 2023 Arianna Formenti, Jinseo Park
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef ELASTIC_BHABHA_INITIALIZE_MOMENTUM_H
#define ELASTIC_BHABHA_INITIALIZE_MOMENTUM_H

#include "Particles/WarpXParticleContainer.H"
#include "Utils/ParticleUtils.H"
#include "Utils/WarpXConst.H"

#include <AMReX_DenseBins.H>
#include <AMReX_Random.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <limits>

namespace {
    // Define shortcuts for frequently-used type names
    using SoaData_type = WarpXParticleContainer::ParticleTileType::ParticleTileDataType;
    using ParticleType = WarpXParticleContainer::ParticleType;
    using ParticleTileType = WarpXParticleContainer::ParticleTileType;
    using ParticleTileDataType = ParticleTileType::ParticleTileDataType;
    using ParticleBins = amrex::DenseBins<ParticleTileDataType>;
    using index_type = ParticleBins::index_type;

    /**
     * \brief This function computes the angular part of the elastic Bhabha Tree level cross section.
     * \param[in] theta_m scattering angle (in radians)
     * \param[out] sigma the angular part of the elastic Bhabha Tree level cross section (dimensionless)
     */
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    amrex::ParticleReal TreeLevelElasticBhabhaCrossSectionAngularPart (
        amrex::ParticleReal const theta_m)
    {

        // Elastic Bhabha Tree level cross section.
        amrex::ParticleReal numerator = (
            111._prt * std::cos(theta_m) +
            6._prt * std::cos(2._prt * theta_m) +
            std::cos(3 * theta_m) +
            96._prt / (sin(theta_m / 2._prt)*sin(theta_m / 2._prt)) +
            384._prt * std::log(std::sin(theta_m / 2._prt)) +
            10._prt
        );
        amrex::ParticleReal denominator = 1536._prt * MathConst::pi;
        amrex::ParticleReal sigma = numerator / denominator;
        return sigma;
    }

     /**
     * \brief This function calculates the derivative of the angular part of the elastic Bhabha Tree level cross section.
     * \param[in] theta_m
     * \param[out] sigma
     */

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    amrex::ParticleReal TreeLevelElasticBhabhaCrossSectionAngularPartDerivative (
        amrex::ParticleReal const theta_m)
    {

        // Elastic Bhabha Tree level differential cross section. (times 2 Pi Sin theta)
        amrex::ParticleReal numerator = (std::cos(2._prt*theta_m) + 7._prt) * (std::cos(2._prt*theta_m) + 7._prt) * std::sin(theta_m) * 2._prt * MathConst::pi;
        amrex::ParticleReal denominator = 4096._prt * MathConst::pi* MathConst::pi * (std::sin(0.5_prt*theta_m) * std::sin(0.5_prt*theta_m) * std::sin(0.5_prt*theta_m) * std::sin(0.5_prt*theta_m)) ;
        amrex::ParticleReal sigma = numerator / denominator;
        return sigma;
    }

    /**
     * \brief This function initializes the momentum of the product particles,
     * in a Bhabha scattering event.
     *
     * @param[in] soa1_in struct of array data of the first colliding species (photon)
     * @param[in] soa2_in struct of array data of the second colliding species (lepton)
     * @param[out] soa1_out struct of array data of the first product species (photon)
     * @param[out] soa2_out struct of array data of the second product species (lepton)
     * @param[in] idx1_in index of first colliding macroparticle (photon)
     * @param[in] idx2_in index of second colliding macroparticle (lepton)
     * @param[in] idx1_out_start index of first product macroparticle (photon)
     * @param[in] idx2_out_start index of second product macroparticle (lepton)
     * @param[in] engine the random engine
     */
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void ElasticBhabhaMollerInitializeMomentum (
                            const SoaData_type& soa1_in, const SoaData_type& soa2_in,
                            SoaData_type& soa1_out, SoaData_type& soa2_out,
                            const index_type& idx1_in, const index_type& idx2_in,
                            const index_type& idx1_out_start, const index_type& idx2_out_start,
                            const amrex::RandomEngine& engine)
    {
        using namespace amrex::literals;

        // Extract the momenta of the incident particles
        amrex::ParticleReal const u1x = soa1_in.m_rdata[PIdx::ux][idx1_in];
        amrex::ParticleReal const u1y = soa1_in.m_rdata[PIdx::uy][idx1_in];
        amrex::ParticleReal const u1z = soa1_in.m_rdata[PIdx::uz][idx1_in];
        amrex::ParticleReal const u2x = soa2_in.m_rdata[PIdx::ux][idx2_in];
        amrex::ParticleReal const u2y = soa2_in.m_rdata[PIdx::uy][idx2_in];
        amrex::ParticleReal const u2z = soa2_in.m_rdata[PIdx::uz][idx2_in];

        amrex::ParticleReal constexpr inv_c2 = 1._prt / ( PhysConst::c * PhysConst::c );

        // Compute Lorentz factor gamma
        amrex::ParticleReal const gb1sq = (u1x*u1x + u1y*u1y + u1z*u1z)*inv_c2;
        amrex::ParticleReal const gb2sq = (u2x*u2x + u2y*u2y + u2z*u2z)*inv_c2;
        amrex::ParticleReal const g1 = std::sqrt( 1._prt + gb1sq );
        amrex::ParticleReal const g2 = std::sqrt( 1._prt + gb2sq );

        amrex::ParticleReal const diffx = u1x-u2x;
        amrex::ParticleReal const diffy = u1y-u2y;
        amrex::ParticleReal const diffz = u1z-u2z;
        amrex::ParticleReal const diffm = std::sqrt((diffx*diffx+diffy*diffy+diffz*diffz)*inv_c2);
        amrex::ParticleReal const summm = std::sqrt(gb1sq) + std::sqrt(gb2sq);
        // If g = u1 - u2 = 0, do not collide.
        // Or if the relative difference is less than 1.0e-10.
        if ( diffm < std::numeric_limits<amrex::ParticleReal>::min() || diffm/summm < 1.0e-10 ) { return; }

        amrex::ParticleReal const m1 = PhysConst::m_e;
        amrex::ParticleReal const m2 = PhysConst::m_e;
        // Compute momenta
        amrex::ParticleReal const p1x = u1x * m1;
        amrex::ParticleReal const p1y = u1y * m1;
        amrex::ParticleReal const p1z = u1z * m1;
        amrex::ParticleReal const p2x = u2x * m2;
        amrex::ParticleReal const p2y = u2y * m2;
        amrex::ParticleReal const p2z = u2z * m2;

        // Compute center-of-momentum (COM) velocity and gamma
        amrex::ParticleReal const mass_g = m1 * g1 + m2 * g2;
        amrex::ParticleReal const vcx    = (p1x+p2x) / mass_g;
        amrex::ParticleReal const vcy    = (p1y+p2y) / mass_g;
        amrex::ParticleReal const vcz    = (p1z+p2z) / mass_g;
        amrex::ParticleReal const vcms   = vcx*vcx + vcy*vcy + vcz*vcz;
        amrex::ParticleReal const gc     = 1._prt / std::sqrt( 1._prt - vcms*inv_c2 );

        // Compute vc dot v1 and v2
        amrex::ParticleReal const vcDv1 = (vcx*u1x + vcy*u1y + vcz*u1z) / g1;
        amrex::ParticleReal const vcDv2 = (vcx*u2x + vcy*u2y + vcz*u2z) / g2;

        // Compute p1 star
        // lorentz_transform_factor = ( (gc-1.0)/vcms*vcDv1 - gc )*m1*g1
        // Rewrite by multiplying and dividing first term by (gc+1) to
        // avoid loss of precision when gc is close to 1
        amrex::ParticleReal const lorentz_transform_factor =
                ( gc*gc*vcDv1*inv_c2/(1._prt + gc) - gc )*m1*g1;
        amrex::ParticleReal const p1sx = p1x + vcx*lorentz_transform_factor;
        amrex::ParticleReal const p1sy = p1y + vcy*lorentz_transform_factor;
        amrex::ParticleReal const p1sz = p1z + vcz*lorentz_transform_factor;
        amrex::ParticleReal const p1sm = std::sqrt( p1sx*p1sx + p1sy*p1sy + p1sz*p1sz );

        // Compute gamma star
        amrex::ParticleReal const g1s = ( 1._prt - vcDv1*inv_c2 )*gc*g1;
        amrex::ParticleReal const g2s = ( 1._prt - vcDv2*inv_c2 )*gc*g2;

        // Compute s12
        amrex::ParticleReal s12 = 0;
        if (p1sm > std::numeric_limits<amrex::ParticleReal>::min()) {

            // Define minimum theta in radians
            amrex::ParticleReal const theta_min = 0.01_prt; //~0.5 degrees. This must be set from the configuration. Also check ElasticBhabhaMollerCollisionFunc.H
            s12 = (16*MathConst::pi*MathConst::pi* PhysConst::r_e* PhysConst::r_e)*(p1sm / m1 / PhysConst::c)*(p1sm / m1 / PhysConst::c)* TreeLevelElasticBhabhaCrossSectionAngularPart(theta_min);

        }

        // Get random numbers
        amrex::ParticleReal r = amrex::Random(engine);

        // Compute scattering angle
        amrex::ParticleReal cosXs;
        amrex::ParticleReal sinXs;


        // Newton-Raphson method between 0 and s12.
        amrex::ParticleReal const sample_point = r * s12;
        // Iteratively find angle which corresponds to the sample point.
        amrex::ParticleReal theta_guess_old = 0.5_prt * MathConst::pi;
        amrex::ParticleReal theta_guess_new = 0.5_prt * MathConst::pi;
        amrex::ParticleReal const tol = 1e-6_prt;
        amrex::ParticleReal rel_error = 1._prt;
        while (rel_error > tol) {
            amrex::ParticleReal const f = TreeLevelElasticBhabhaCrossSectionAngularPart(theta_guess_old) - sample_point;
            //Beware of the sign. TreeLevelElasticBhabhaCrossSectionAngularPart is integrated from 2Pi to theta_min.
            amrex::ParticleReal const df = -TreeLevelElasticBhabhaCrossSectionAngularPartDerivative(theta_guess_old);
            //For theta_min = 0.05, sample_point must range from 0 to 31.5
            theta_guess_new = theta_guess_old - f/df;
            rel_error = std::abs(theta_guess_new-theta_guess_old)/std::abs(theta_guess_old);
            theta_guess_old = theta_guess_new;
        }
        cosXs = std::cos(theta_guess_old);
        sinXs = std::sin(theta_guess_old);

        // Get random azimuthal angle
        amrex::ParticleReal const phis = amrex::Random(engine) * 2.0_prt * MathConst::pi;
        amrex::ParticleReal const cosphis = std::cos(phis);
        amrex::ParticleReal const sinphis = std::sin(phis);

        // Compute post-collision momenta pfs in COM
        amrex::ParticleReal p1fsx;
        amrex::ParticleReal p1fsy;
        amrex::ParticleReal p1fsz;
        // p1sp is the p1s perpendicular
        amrex::ParticleReal p1sp = std::sqrt( p1sx*p1sx + p1sy*p1sy );
        // Make sure p1sp is not almost zero
        if ( p1sp > std::numeric_limits<amrex::ParticleReal>::min() )
        {
            p1fsx = ( p1sx*p1sz/p1sp ) * sinXs*cosphis +
                    ( p1sy*p1sm/p1sp ) * sinXs*sinphis +
                    ( p1sx           ) * cosXs;
            p1fsy = ( p1sy*p1sz/p1sp ) * sinXs*cosphis +
                    (-p1sx*p1sm/p1sp ) * sinXs*sinphis +
                    ( p1sy           ) * cosXs;
            p1fsz = (-p1sp           ) * sinXs*cosphis +
                    ( 0.0_prt        ) * sinXs*sinphis +
                    ( p1sz           ) * cosXs;
        }
        else
        {
            p1sp = std::sqrt( p1sy*p1sy + p1sz*p1sz );
            p1fsy = ( p1sy*p1sx/p1sp ) * sinXs*cosphis +
                    ( p1sz*p1sm/p1sp ) * sinXs*sinphis +
                    ( p1sy           ) * cosXs;
            p1fsz = ( p1sz*p1sx/p1sp ) * sinXs*cosphis +
                    (-p1sy*p1sm/p1sp ) * sinXs*sinphis +
                    ( p1sz           ) * cosXs;
            p1fsx = (-p1sp           ) * sinXs*cosphis +
                    ( 0.0_prt        ) * sinXs*sinphis +
                    ( p1sx           ) * cosXs;
        }

        amrex::ParticleReal const p2fsx = -p1fsx;
        amrex::ParticleReal const p2fsy = -p1fsy;
        amrex::ParticleReal const p2fsz = -p1fsz;

        // Transform from COM to lab frame
        amrex::ParticleReal const vcDp1fs = vcx*p1fsx + vcy*p1fsy + vcz*p1fsz;
        amrex::ParticleReal const vcDp2fs = vcx*p2fsx + vcy*p2fsy + vcz*p2fsz;
        /* factor = (gc-1.0)/vcms; Rewrite to avoid subtraction losing precision when gc is close to 1 */
        amrex::ParticleReal const factor = gc*gc*inv_c2/(gc+amrex::ParticleReal(1.0));
        amrex::ParticleReal const factor1 = factor*vcDp1fs + m1*g1s*gc;
        amrex::ParticleReal const factor2 = factor*vcDp2fs + m2*g2s*gc;
        amrex::ParticleReal const p1fx = p1fsx + vcx * factor1;
        amrex::ParticleReal const p1fy = p1fsy + vcy * factor1;
        amrex::ParticleReal const p1fz = p1fsz + vcz * factor1;
        amrex::ParticleReal const p2fx = p2fsx + vcx * factor2;
        amrex::ParticleReal const p2fy = p2fsy + vcy * factor2;
        amrex::ParticleReal const p2fz = p2fsz + vcz * factor2;

        // Fill momentum of product species (note that we actually
        // create 4 products, 2 at the position of each incident particle)
        // - Scattered electron
        soa1_out.m_rdata[PIdx::ux][idx1_out_start] = p1fx / m1;
        soa1_out.m_rdata[PIdx::uy][idx1_out_start] = p1fy / m1;
        soa1_out.m_rdata[PIdx::uz][idx1_out_start] = p1fz / m1;
        soa1_out.m_rdata[PIdx::ux][idx1_out_start + 1] = p1fx / m1;
        soa1_out.m_rdata[PIdx::uy][idx1_out_start + 1] = p1fy / m1;
        soa1_out.m_rdata[PIdx::uz][idx1_out_start + 1] = p1fz / m1;
        // - Scattered electron
        soa2_out.m_rdata[PIdx::ux][idx2_out_start] = p2fx / m2;
        soa2_out.m_rdata[PIdx::uy][idx2_out_start] = p2fy / m2;
        soa2_out.m_rdata[PIdx::uz][idx2_out_start] = p2fz / m2;
        soa2_out.m_rdata[PIdx::ux][idx2_out_start + 1] = p2fx / m2;
        soa2_out.m_rdata[PIdx::uy][idx2_out_start + 1] = p2fy / m2;
        soa2_out.m_rdata[PIdx::uz][idx2_out_start + 1] = p2fz / m2;
    }
}

#endif // ELASTIC_BHABHA_INITIALIZE_MOMENTUM_H
