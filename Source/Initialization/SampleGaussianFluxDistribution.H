/* Copyright 2024 Remi Lehe, Revathi Jambunathan
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_SAMPLE_GAUSSIAN_FLUX_DISTRIBUTION_H
#define WARPX_SAMPLE_GAUSSIAN_FLUX_DISTRIBUTION_H

#include <AMReX_Random.H>

namespace {
    /** This function returns u sampled according to the probability distribution:
      * p(u) \propto u \exp(-(u-u_m)^2/2u_th^2)
      *
      * @param u_m Central momentum
      * @param u_th Momentum spread
      * @param engine Object used to generate random numbers
      */
    [[nodiscard]]
    AMREX_FORCE_INLINE
    AMREX_GPU_HOST_DEVICE
    amrex::Real
    generateGaussianFluxDist( amrex::Real u_m, amrex::Real u_th, amrex::RandomEngine const& engine ) {

        using namespace amrex::literals;

        // Momentum to be returned at the end of this function
        amrex::Real u = 0._rt;

        const amrex::Real abs_u_m = std::abs(u_m);

        if (u_th == 0._rt) {
            u = u_m; // Trivial case ; avoids division by 0 in the rest of the code below
        } else if (abs_u_m < 0.6*u_th) {
            // Mean velocity magnitude is less than thermal velocity
            // Use the distribution u*exp(-u**2*(1-abs(u_m)/u_th)/(2*u_th**2)) as an approximation
            // and then use the rejection method to correct it
            // ( stop rejecting with probability exp(-abs(u_m)/(2*u_th**3)*(u-sign(u_m)*u_th)**2) )
            // Note that this is the method that is used in the common case u_m=0
            const amrex::Real umsign = std::copysign(1._rt, u_m);
            const amrex::Real approx_u_th = u_th/std::sqrt( 1._rt - abs_u_m/u_th );
            const amrex::Real reject_prefactor = (abs_u_m/u_th)/(2._rt*u_th*u_th); // To save computation
            bool reject = true;
            while (reject) {
                // Generates u according to u*exp(-u**2/(2*approx_u_th**2)),
                // using the method of the inverse cumulative function
                amrex::Real xrand = 1._rt - amrex::Random(engine); // ensures urand > 0
                u = approx_u_th * std::sqrt(2._rt*std::log(1._rt/xrand));
                // Rejection method
                xrand = amrex::Random(engine);
                if (xrand < std::exp(-reject_prefactor*(u - umsign*u_th)*(u - umsign*u_th))) { reject = false; }
            }
        } else {
            // Mean velocity magnitude is greater than thermal velocity
            // Use the distribution exp(-(u-u_m-u_th**2/abs(u_m))**2/(2*u_th**2)) as an approximation
            // and then use the rejection method to correct it
            // ( stop rejecting with probability (u/abs(u_m))*exp(1-(u/abs(u_m))) ; note
            // that this number is always between 0 and 1 )
            // Note that in the common case `u_m = 0`, this rejection method
            // is not used, and the above rejection method is used instead.
            bool reject = true;
            const amrex::Real approx_u_m = u_m + u_th*u_th/abs_u_m;
            const amrex::Real inv_um = 1._rt/abs_u_m; // To save computation
            while (reject) {
                // Approximate distribution: normal distribution, where we only retain positive u
                u = -1._rt;
                while (u < 0) {
                    u = amrex::RandomNormal(approx_u_m, u_th, engine);
                }
                // Rejection method
                const amrex::Real xrand = amrex::Random(engine);
                if (xrand < u*inv_um* std::exp(1._rt - u*inv_um)) { reject = false; }
            }
        }
        return u;
    }
}

#endif //WARPX_SAMPLE_GAUSSIAN_FLUX_DISTRIBUTION_H
