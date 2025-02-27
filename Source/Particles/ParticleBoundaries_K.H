/* Copyright 2021 David Grote, Remi Lehe, Revathi Jambunathan
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLEBOUNDARIES_K_H_
#define WARPX_PARTICLEBOUNDARIES_K_H_

#include "ParticleBoundaries.H"
#include "Initialization/SampleGaussianFluxDistribution.H"

#include <AMReX_AmrCore.H>

namespace ApplyParticleBoundaries {
    using namespace amrex::literals;
    /* \brief Applies the boundary condition on a specific axis
     *        This is called by apply_boundaries.
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void
    apply_boundary (amrex::ParticleReal& x, amrex::Real xmin, amrex::Real xmax,
                    bool& change_sign_ux, bool& rethermalize_x, bool& particle_lost,
                    bool& SEE_particle_added, int& SEE_num_particles_added,
                    ParticleBoundaryType xmin_bc, ParticleBoundaryType xmax_bc,
                    amrex::Real refl_probability_xmin, amrex::Real refl_probability_xmax,
                    amrex::Real SEE_probability_xmin, amrex::Real SEE_probability_xmax,
                    amrex::RandomEngine const& engine )
    {
        if (x < xmin) {
            if (xmin_bc == ParticleBoundaryType::Open) {
                particle_lost = true;
            }
            else if (xmin_bc == ParticleBoundaryType::Absorbing) {
                if (refl_probability_xmin == 0 || amrex::Random(engine) > refl_probability_xmin) {
                    particle_lost = true;
                }
                else
                {
                    x = 2*xmin - x;
                    change_sign_ux = true;
                }
                if (SEE_probability_xmin > 0) {
                    // Configure SEE boundary emission
                    amrex::Real SEE_probability = SEE_probability_xmin;
                    if (SEE_probability_xmin >= 1) {
                        SEE_particle_added = true;
                        SEE_num_particles_added = static_cast<int>(SEE_probability_xmin);

                        SEE_probability = SEE_probability_xmin - static_cast<amrex::Real>(SEE_num_particles_added);
                    }
                    if (amrex::Random(engine) < SEE_probability) {
                        SEE_particle_added = true;
                        SEE_num_particles_added += 1;
                    }
                }
            }
            else if (xmin_bc == ParticleBoundaryType::Reflecting) {
                x = 2*xmin - x;
                change_sign_ux = true;
            }
            else if (xmin_bc == ParticleBoundaryType::Thermal) {
                x = 2*xmin - x;
                rethermalize_x = true;
            }
        }
        else if (x > xmax) {
            if (xmax_bc == ParticleBoundaryType::Open) {
                particle_lost = true;
            }
            else if (xmax_bc == ParticleBoundaryType::Absorbing) {
                if (refl_probability_xmax == 0 || amrex::Random(engine) > refl_probability_xmax) {
                    particle_lost = true;
                }
                else
                {
                    x = 2*xmax - x;
                    change_sign_ux = true;
                }
                if (SEE_probability_xmax > 0) {
                    // Configure SEE boundary emission
                    amrex::Real SEE_probability = SEE_probability_xmax;
                    if (SEE_probability_xmax >= 1) {
                        SEE_particle_added = true;
                        SEE_num_particles_added = static_cast<int>(SEE_probability_xmax);

                        SEE_probability = SEE_probability_xmax - static_cast<amrex::Real>(SEE_num_particles_added);
                    }
                    if (amrex::Random(engine) < SEE_probability) {
                        SEE_particle_added = true;
                        SEE_num_particles_added += 1;
                    }
                }
            }
            else if (xmax_bc == ParticleBoundaryType::Reflecting) {
                x = 2*xmax - x;
                change_sign_ux = true;
            }
            else if (xmax_bc == ParticleBoundaryType::Thermal) {
                x = 2*xmax - x;
                rethermalize_x = true;
            }
        }
    }

    /* \brief Thermalize particles that have been identified to cross the boundary.
     *        The normal component samples from a half-Maxwellian,
     *        and the two tangential components will sample from full Maxwellian disbutions
     *        with thermal velocity uth
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void thermalize_boundary_particle (amrex::ParticleReal& u_norm, amrex::ParticleReal& u_tang1,
                                       amrex::ParticleReal& u_tang2, amrex::Real uth,
                                       amrex::RandomEngine const& engine)
    {
        u_tang1 = (uth > 0._rt) ? PhysConst::c * amrex::RandomNormal(0._rt, uth, engine) : 0._rt;
        u_tang2 = (uth > 0._rt) ? PhysConst::c * amrex::RandomNormal(0._rt, uth, engine) : 0._rt;
        u_norm = (uth > 0._rt) ? std::copysign(1._prt, -u_norm) * PhysConst::c * generateGaussianFluxDist(0._rt, uth, engine) : 0._rt;
    }


    /* \brief Applies absorbing, reflecting or thermal boundary condition to the input particles, along all axis.
     *        For reflecting boundaries, the position of the particle is changed appropriately and
     *        the sign of the velocity is changed (depending on the reflect_all_velocities flag).
     *        For absorbing, a flag is set whether the particle has been lost (it is up to the calling
     *        code to take appropriate action to remove any lost particles). Absorbing boundaries can
     *        be given a reflection coefficient for stochastic reflection of particles, this
     *        coefficient is zero by default.
     *        For thermal boundaries, the particle is first reflected and the position of the particle
     *        is changed appropriately.
     *        For SEE boundaries, the particle is removed and a new SEE particle is added.
     *        Note that periodic boundaries are handled in AMReX code.
     *
     * \param x, xmin, xmax: particle x position, location of x boundary
     * \param y, ymin, ymax: particle y position, location of y boundary (3D only)
     * \param z, zmin, zmax: particle z position, location of z boundary
     * \param ux, uy, uz: particle momenta
     * \param particle_lost: output, flags whether the particle was lost
     * \param SEE_particle_added: output, flags whether a SEE particle was added
     * \param boundaries: object with boundary condition settings
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void
    apply_boundaries ([[maybe_unused]] amrex::ParticleReal& x,
                      [[maybe_unused]] amrex::ParticleReal& y,
                      [[maybe_unused]] amrex::ParticleReal& z,
                      amrex::XDim3 gridmin, amrex::XDim3 gridmax,
                      amrex::ParticleReal& ux, amrex::ParticleReal& uy, amrex::ParticleReal& uz,
                      bool& particle_lost, bool& SEE_particle_added, int& SEE_num_particles_added,
                      ParticleBoundaries::ParticleBoundariesData const& boundaries,
                      amrex::RandomEngine const& engine)
    {
        bool change_sign_ux = false;
        bool change_sign_uy = false;
        bool change_sign_uz = false;

#ifndef WARPX_DIM_1D_Z
        bool rethermalize_x = false; // stores if particle crosses x boundary and needs to be thermalized
        apply_boundary(x, gridmin.x, gridmax.x, change_sign_ux, rethermalize_x, particle_lost,
                       SEE_particle_added, SEE_num_particles_added,
                       boundaries.xmin_bc, boundaries.xmax_bc,
                       boundaries.reflection_model_xlo(-ux), boundaries.reflection_model_xhi(ux),
                       boundaries.SEE_probability_xlo, boundaries.SEE_probability_xhi,
                       engine);
        if (rethermalize_x) {
            thermalize_boundary_particle(ux, uy, uz, boundaries.m_uth, engine);
        }
#endif
#ifdef WARPX_DIM_3D
        bool rethermalize_y = false; // stores if particle crosses y boundary and needs to be thermalized
        apply_boundary(y, gridmin.y, gridmax.y, change_sign_uy, rethermalize_y, particle_lost,
                       SEE_particle_added, SEE_num_particles_added,
                       boundaries.ymin_bc, boundaries.ymax_bc,
                       boundaries.reflection_model_ylo(-uy), boundaries.reflection_model_yhi(uy),
                       boundaries.SEE_probability_ylo, boundaries.SEE_probability_yhi,
                       engine);
        if (rethermalize_y) {
            thermalize_boundary_particle(uy, uz, ux, boundaries.m_uth, engine);
        }
#endif
        bool rethermalize_z = false; // stores if particle crosses z boundary and needs to be thermalized
        apply_boundary(z, gridmin.z, gridmax.z, change_sign_uz, rethermalize_z, particle_lost,
                       SEE_particle_added, SEE_num_particles_added,
                       boundaries.zmin_bc, boundaries.zmax_bc,
                       boundaries.reflection_model_zlo(-uz), boundaries.reflection_model_zhi(uz),
                       boundaries.SEE_probability_zlo, boundaries.SEE_probability_zhi,
                       engine);
        if (rethermalize_z) {
            thermalize_boundary_particle(uz, ux, uy, boundaries.m_uth, engine);
        }

        if (boundaries.reflect_all_velocities && (change_sign_ux | change_sign_uy | change_sign_uz)) {
            change_sign_ux = true;
            change_sign_uy = true;
            change_sign_uz = true;
        }
#ifdef WARPX_DIM_RZ
        // Note that the reflection of the position does "r = 2*rmax - r", but this is only approximate.
        // The exact calculation requires the position at the start of the step.
        if (change_sign_ux && change_sign_uy) {
            ux = -ux;
            uy = -uy;
        } else if (change_sign_ux) {
            // Reflect only ur
            // Note that y is theta
            amrex::Real ur = ux*std::cos(y) + uy*std::sin(y);
            const amrex::Real ut = -ux*std::sin(y) + uy*std::cos(y);
            ur = -ur;
            ux = ur*std::cos(y) - ut*std::sin(y);
            uy = ur*std::sin(y) + ut*std::cos(y);
        }
#else
        if (change_sign_ux) { ux = -ux; }
        if (change_sign_uy) { uy = -uy; }
#endif
        if (change_sign_uz) { uz = -uz; }
    }


    /* \brief Applies absorbing, reflecting or thermal boundary condition to the input particles, along all axis.
     *        For reflecting boundaries, the position of the particle is changed appropriately and
     *        the sign of the velocity is changed (depending on the reflect_all_velocities flag).
     *        For absorbing, a flag is set whether the particle has been lost (it is up to the calling
     *        code to take appropriate action to remove any lost particles). Absorbing boundaries can
     *        be given a reflection coefficient for stochastic reflection of particles, this
     *        coefficient is zero by default.
     *        For thermal boundaries, the particle is first reflected and the position of the particle
     *        is changed appropriately.
     *        For SEE boundaries, the particle is removed and a new SEE particle is added.
     *        Note that periodic boundaries are handled in AMReX code.
     *
     * \param xpos, ypos, zpos: particle x, y, z position before boundary conditions were applied
     * \param gridmin, gridmax: location of boundaries
     * \param max_SEE_probability: maximum probability of adding a SEE particle at any boundary
     * \param SEE_num_particles_added: number of SEE particles added
     * \param engine: random number generator
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void
    do_SEE(amrex::ParticleReal& xpos, amrex::ParticleReal& ypos, amrex::ParticleReal& zpos,
           amrex::XDim3 gridmin, amrex::XDim3 gridmax,
           amrex::Real const& max_SEE_probability, amrex::Real const& v_SEE,
           int& SEE_num_particles_added, amrex::RandomEngine const& engine)
    {
        // Make sure there are not more particles added (in case two boundaries were crossed by one particle)
        if (SEE_num_particles_added > static_cast<int>(max_SEE_probability) + 1) {
            SEE_num_particles_added = 0;
            amrex::Real SEE_probability = max_SEE_probability;

            // Redraw a random number to determine if SEE particles are added
            if (max_SEE_probability >= 1) {
                SEE_num_particles_added = static_cast<int>(max_SEE_probability);

                SEE_probability = max_SEE_probability - static_cast<amrex::Real>(SEE_num_particles_added);
            }
            if (amrex::Random(engine) < SEE_probability) {
                SEE_num_particles_added += 1;
            }
        }

        // Determine normal vector pointing inward from the boundary that was hit
        amrex::ParticleReal nx = 0.0;
        amrex::ParticleReal ny = 0.0;
        amrex::ParticleReal nz = 0.0;

        // Determine which boundaries were crossed
#ifndef WARPX_DIM_1D_Z
        if (xpos < gridmin.x) {
            xpos = gridmin.x;
            nx = 1.0;
        }
        else if (xpos > gridmax.x) {
            xpos = gridmax.x;
            nx = -1.0;
        }
#endif
#ifdef WARPX_DIM_3D
        if (ypos < gridmin.y) {
            ypos = gridmin.y;
            ny = 1.0;
        }
        else if (ypos > gridmax.y) {
            ypos = gridmax.y;
            ny = -1.0;
        }
#endif
        if (zpos < gridmin.z) {
            zpos = gridmin.z;
            nz = 1.0;
        }
        else if (zpos > gridmax.z) {
            zpos = gridmax.z;
            nz = -1.0;
        }

        // Do we need to do anything special to account for RZ simulations?

        // Initialize arrays to hold new particle data
        amrex::Vector<amrex::ParticleReal> xp_new(SEE_num_particles_added, xpos);
        amrex::Vector<amrex::ParticleReal> yp_new(SEE_num_particles_added, ypos);
        amrex::Vector<amrex::ParticleReal> zp_new(SEE_num_particles_added, zpos);
        amrex::Vector<amrex::ParticleReal> uxp_new(SEE_num_particles_added);
        amrex::Vector<amrex::ParticleReal> uyp_new(SEE_num_particles_added);
        amrex::Vector<amrex::ParticleReal> uzp_new(SEE_num_particles_added);
        amrex::Vector<amrex::ParticleReal> wp_new(SEE_num_particles_added, wp[i]); // Use same weight as parent

        // Generate velocities pointing inward from the boundary
        for (int n = 0; n < SEE_num_particles_added; n++) {
            // Generate random angles in the hemisphere pointing inward
            const amrex::ParticleReal cos_theta = amrex::Random(engine); // Range [0,1] for hemisphere
            const amrex::ParticleReal sin_theta = std::sqrt(1.0 - cos_theta*cos_theta);
            const amrex::ParticleReal phi = 2.0 * MathConst::pi * amrex::Random(engine);

            // Calculate velocity components in local hemisphere coordinate system
            const amrex::ParticleReal vr = v_SEE * sin_theta;
            const amrex::ParticleReal vz_local = v_SEE * cos_theta;
            const amrex::ParticleReal vx_local = vr * std::cos(phi);
            const amrex::ParticleReal vy_local = vr * std::sin(phi);

            // TODO: Align z velocity component to boundary normal and then confirm only one boundary was crossed. If
            //      multiple boundaries were crossed, the particle needs to ensure it is directed inwards.
        }

        // TODO: Add the SEE particle, maybe with WarpXParticleContainer::AddNParticles
    }
}
#endif //WARPX_PARTICLEBOUNDARIES_K_H_
