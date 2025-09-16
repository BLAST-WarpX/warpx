/* Copyright 2024 David Grote
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_BREMSSTRAHLUNG_FUNC_H_
#define WARPX_BREMSSTRAHLUNG_FUNC_H_

#include "Particles/Collision/BinaryCollision/BinaryCollisionUtils.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/ParticleUtils.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <AMReX_Algorithm.H>
#include <AMReX_DenseBins.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Random.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

/**
 * \brief This functor does binary Bremsstrahlung reactions in a single cell.
 *  Particles of the two interacting species are paired with each other and for each pair we compute
 *  the energy of the resulting photon if Bremsstrahlung happens.
 *  We fill p_mask with true if it happens so that the new photon corresponding to a given pair can be
 *  effectively created in the particle creation functor.
 */
class BremsstrahlungFunc
{
    // Define shortcuts for frequently-used type names
    using ParticleType = WarpXParticleContainer::ParticleType;
    using ParticleTileType = WarpXParticleContainer::ParticleTileType;
    using ParticleTileDataType = ParticleTileType::ParticleTileDataType;
    using ParticleBins = amrex::DenseBins<ParticleTileDataType>;
    using index_type = ParticleBins::index_type;
    using SoaData_type = WarpXParticleContainer::ParticleTileType::ParticleTileDataType;

public:
    /*
     * \brief Default constructor of the BremsstrahlungFunc class.
     */
    BremsstrahlungFunc () = default;

    /**
     * \brief Constructor of the BremsstrahlungFunc class
     *
     * @param[in] collision_name the name of the collision
     * @param[in] mypc pointer to the MultiParticleContainer
     * @param[in] isSameSpecies whether the two colliding species are the same (should always be false)
     */
    BremsstrahlungFunc (std::string const& collision_name, MultiParticleContainer const * mypc,
                        bool /*isSameSpecies*/);

    struct Executor {
        /**
         * \brief Executor of the BremsstrahlungFunc class. Produces Bremsstrahlung photons at the cell level.
         * Note that this function does not yet create the resulting photons, but
         * instead sets p_mask and calculates the weight and energy
         *
         * @param[in] I1s, I1e is the start and step index for I1 (start inclusive, step exclusive)
         * @param[in] I2s, I2e is the start and step index for I2 (start inclusive, step exclusive)
         * @param[in] I1, I2 index arrays. They determine all elements that will be used
         * @param[in] soa_1, soa_2 contain the struct of array data of the two species
         * @param[in] n1, n2 the species number densities
         * @param[in] m1, m2 the species masses
         * @param[in] dt is the time step length between two collision calls
         * @param[in] coll_idx is the collision index offset
         * @param[in] cell_start_pair is the start index of the pairs in that cell
         * @param[out] p_mask is a mask that will be set to true for all pairs
         * @param[out] p_pair_indices_1, p_pair_indices_2 arrays that store the indices of the
         * particles of a given pair. Only p_pair_indices_1 will be used when creating photons
         * @param[out] p_pair_reaction_weight stores the weight of the resulting photons
         * @param[out] p_product_data stores the energy of the resulting photons
         * @param[in] engine the random engine
         */
        AMREX_GPU_DEVICE AMREX_INLINE
        void operator() (
            index_type const I1s, index_type const I1e,
            index_type const I2s, index_type const I2e,
            index_type const* AMREX_RESTRICT I1,
            index_type const* AMREX_RESTRICT I2,
            SoaData_type const& soa_1, const SoaData_type& soa_2,
            GetParticlePosition<PIdx> /*get_position_1*/, GetParticlePosition<PIdx> /*get_position_2*/,
            amrex::ParticleReal const n1, amrex::ParticleReal const n2,
            amrex::ParticleReal const /*T1*/, amrex::ParticleReal const /*T2*/,
            amrex::Real const /*global_lamdb*/,
            amrex::ParticleReal const  /*q1*/, amrex::ParticleReal const  /*q2*/,
            amrex::ParticleReal const m1, amrex::ParticleReal const m2,
            amrex::Real const dt, amrex::Real const /*dV*/, index_type coll_idx,
            index_type const cell_start_pair, index_type* AMREX_RESTRICT p_mask,
            index_type* AMREX_RESTRICT p_pair_indices_1, index_type* AMREX_RESTRICT p_pair_indices_2,
            amrex::ParticleReal* AMREX_RESTRICT p_pair_reaction_weight,
            amrex::ParticleReal* AMREX_RESTRICT p_product_data,
            amrex::RandomEngine const& engine) const
        {
            amrex::ParticleReal * const AMREX_RESTRICT weight1 = soa_1.m_rdata[PIdx::w];
            amrex::ParticleReal * const AMREX_RESTRICT u1x = soa_1.m_rdata[PIdx::ux];
            amrex::ParticleReal * const AMREX_RESTRICT u1y = soa_1.m_rdata[PIdx::uy];
            amrex::ParticleReal * const AMREX_RESTRICT u1z = soa_1.m_rdata[PIdx::uz];

            amrex::ParticleReal * const AMREX_RESTRICT weight2 = soa_2.m_rdata[PIdx::w];
            amrex::ParticleReal * const AMREX_RESTRICT u2x = soa_2.m_rdata[PIdx::ux];
            amrex::ParticleReal * const AMREX_RESTRICT u2y = soa_2.m_rdata[PIdx::uy];
            amrex::ParticleReal * const AMREX_RESTRICT u2z = soa_2.m_rdata[PIdx::uz];

            // Number of macroparticles of each species
            index_type const NI1 = I1e - I1s;
            index_type const NI2 = I2e - I2s;

            // Only loop over the number of particles in the first species (the electrons)
            index_type const max_N = NI1;
            index_type const min_N = NI2;

            index_type pair_index = cell_start_pair + coll_idx;

#if (defined WARPX_DIM_RZ)
            amrex::ParticleReal * const AMREX_RESTRICT theta1 = soa_1.m_rdata[PIdx::theta];
            amrex::ParticleReal * const AMREX_RESTRICT theta2 = soa_2.m_rdata[PIdx::theta];
#endif

            index_type i1 = I1s + coll_idx;
            index_type const i2 = I2s + coll_idx;
            // we will start from collision number = coll_idx and then add
            // stride (smaller set size) until we do all collisions (larger set size)
            // If there are more atoms than electrons, this loop executes once
            // If there are more electrons than atoms, this loops over the electrons matched to the atom
            for (index_type k = coll_idx; k < max_N; k += min_N)
            {

#if (defined WARPX_DIM_RZ)
                /* In RZ geometry, macroparticles can collide with other macroparticles
                 * in the same *cylindrical* cell. For this reason, collisions between macroparticles
                 * are actually not local in space. In this case, the underlying assumption is that
                 * particles within the same cylindrical cell represent a cylindrically-symmetry
                 * momentum distribution function. Therefore, here, we temporarily rotate the
                 * momentum of one of the macroparticles in agreement with this cylindrical symmetry.
                 * (This is technically only valid if we use only the m=0 azimuthal mode in the simulation;
                 * there is a corresponding assert statement at initialization.) */
                amrex::ParticleReal const theta = theta2[I2[i2]]-theta1[I1[i1]];
                amrex::ParticleReal const u1xbuf = u1x[I1[i1]];
                u1x[I1[i1]] = u1xbuf*std::cos(theta) - u1y[I1[i1]]*std::sin(theta);
                u1y[I1[i1]] = u1xbuf*std::sin(theta) + u1y[I1[i1]]*std::cos(theta);
#endif

                BremsstrahlungEvent(
                    u1x[ I1[i1] ], u1y[ I1[i1] ], u1z[ I1[i1] ],
                    u2x[ I2[i2] ], u2y[ I2[i2] ], u2z[ I2[i2] ],
                    m1, m2, weight1[ I1[i1] ], weight2[ I2[i2] ],
                    n1, n2, dt, static_cast<int>(pair_index), p_mask,
                    p_pair_reaction_weight, p_product_data,
                    engine);

#if (defined WARPX_DIM_RZ)
                amrex::ParticleReal const u1xbuf_new = u1x[I1[i1]];
                u1x[I1[i1]] = u1xbuf_new*std::cos(-theta) - u1y[I1[i1]]*std::sin(-theta);
                u1y[I1[i1]] = u1xbuf_new*std::sin(-theta) + u1y[I1[i1]]*std::cos(-theta);
#endif

                p_pair_indices_1[pair_index] = I1[i1];
                p_pair_indices_2[pair_index] = I2[i2];
                i1 += min_N;
                pair_index += min_N;

            }
        }

        /* \brief Calculate the cross section given the electron energy.
         *        The differential cross section is cut off below the plasma frequency since the
         *        low energy photons would all be immediately absorbed by the surrounding plasma.
         *
         * \param[in] E electron energy in Joules
         * \param[in] n1 the electron number density
         * \param[in] m1 the mass of the electrons
         * \param[out] index the index for the electron energy grid
         * \param[out] i0_cut the index below the cut off for the photon energy grid
         * \param[out] koT1_cut the k of the photon energy cut off
         * \param[out] kdsigdk_cut the ksigdk at the cut off
         * \param[out] w0 the grid weighting at the cut off
         * \result the cross section
         */
        AMREX_GPU_DEVICE AMREX_INLINE
        amrex::ParticleReal
        CalculateCrossSection (amrex::ParticleReal KErel_eV,
                               amrex::ParticleReal wpe,
                               int & index,
                               int & i0_cut,
                               amrex::ParticleReal & koT1_cut, amrex::ParticleReal & kdsigdk_cut,
                               amrex::ParticleReal & w0) const
        {
            using namespace amrex::literals;

            amrex::ParticleReal const KEcut_eV = PhysConst::hbar*wpe/PhysConst::q_e;

            if (KErel_eV < KEcut_eV) {
                // Ignore low energy electrons which would only produce low energy photons
                return 0._prt;
            }

            // find lo-index for koT1 cutoff (will typically be 1)
            koT1_cut = std::max(KEcut_eV/KErel_eV, m_koT1_cut_default);
            i0_cut = 0;
            for (int i=1; i < nkoT1; i++) {
                if (m_koT1_grid[i] > koT1_cut) {
                    break;
                } else {
                    i0_cut = i;
                }
            }
            if (i0_cut == nkoT1 - 1) {
                return 0.0_prt;
            }

            // kdsigdk is linearly weighted in electron energy
            if (KErel_eV >= m_KEgrid_eV[nKE-1]) {
                index = nkoT1 - 2;
                w0 = 0.0_prt;
            } else {
                // find index for electron energy interpolation
                index = amrex::bisect(m_KEgrid_eV, 0, nKE, KErel_eV);

                // compute interpolation weights for k*dsigma/dk
                w0 = (m_KEgrid_eV[index+1] - KErel_eV)/(m_KEgrid_eV[index+1] - m_KEgrid_eV[index]);
            }

            // kdsigdk at the cutoff
            amrex::ParticleReal const w00 = (m_koT1_grid[i0_cut+1] - koT1_cut)/(m_koT1_grid[i0_cut+1] - m_koT1_grid[i0_cut]);
            amrex::ParticleReal const w01 = 1.0_prt - w00;
            kdsigdk_cut = (  w00*(w0*m_kdsigdk[index*nkoT1 + i0_cut  ] + (1.0_prt - w0)*m_kdsigdk[(index+1)*nkoT1 + i0_cut])
                           + w01*(w0*m_kdsigdk[index*nkoT1 + i0_cut+1] + (1.0_prt - w0)*m_kdsigdk[(index+1)*nkoT1 + i0_cut+1]));

            amrex::ParticleReal sigma_total;
            if (KEcut_eV/KErel_eV <= m_koT1_cut_default) {
                // Use precalculated value
                sigma_total = w0*m_sigma_total[index] + (1.0_prt - w0)*m_sigma_total[index+1];
            } else {
                // Integrate the distribution using trapezoidal rule
                amrex::ParticleReal koT1_grid_im1 = koT1_cut;
                amrex::ParticleReal kdsigdk_im1 = kdsigdk_cut;
                amrex::ParticleReal sigma = 0.0_prt;
                for (int i=i0_cut+1; i < nkoT1; i++) {
                    amrex::ParticleReal const kdsigdk_i = w0*m_kdsigdk[index*nkoT1 + i] + (1.0_prt - w0)*m_kdsigdk[(index+1)*nkoT1 + i];
                    amrex::ParticleReal const dk = (m_koT1_grid[i] - koT1_grid_im1);
                    amrex::ParticleReal const k_ave = (m_koT1_grid[i] + koT1_grid_im1)*0.5_prt;
                    amrex::ParticleReal const dsigdk = (kdsigdk_i + kdsigdk_im1)*0.5_prt/k_ave;
                    sigma = sigma + dsigdk*dk;
                    koT1_grid_im1 = m_koT1_grid[i];
                    kdsigdk_im1 = kdsigdk_i;
                }
                sigma_total = sigma;
            }

            return sigma_total;
        }

        /* \brief Generate the photon energy from the differential cross section
         *
         * \param[in] index the index for the electron energy grid
         * \param[in] i0_cut the index below the cut off for the photon energy grid
         * \param[in] koT1_cut the k of the photon energy cut off
         * \param[in] kdsigdk_cut the ksigdk at the cut off
         * \param[in] w0 the grid weighting at the cut off
         * \param[in] sigma_total the total cross section
         * \param[in] random_number the uniformly spaced random number
         * \result the photon energy
         */
        AMREX_GPU_DEVICE AMREX_INLINE
        amrex::ParticleReal
        Photon_energy(int index, int i0_cut, amrex::ParticleReal koT1_cut, amrex::ParticleReal kdsigdk_cut,
                      amrex::ParticleReal w0, amrex::ParticleReal sigma_total, amrex::ParticleReal random_number) const
        {
            using namespace amrex::literals;
            amrex::ParticleReal koT1_grid_im1 = koT1_cut;
            amrex::ParticleReal kdsigdk_im1 = kdsigdk_cut;
            amrex::ParticleReal sigma = 0._prt;
            amrex::ParticleReal kdsigdk_i = kdsigdk_cut;
            amrex::ParticleReal dk = m_koT1_grid[i0_cut+1] - koT1_grid_im1;
            amrex::ParticleReal k_ave = koT1_cut;
            for (int i=i0_cut+1; i < nkoT1; i++) {
                dk = (m_koT1_grid[i] - koT1_grid_im1);
                k_ave = (m_koT1_grid[i] + koT1_grid_im1)*0.5_prt;
                kdsigdk_i = w0*m_kdsigdk[index*nkoT1 + i] + (1.0_prt - w0)*m_kdsigdk[(index+1)*nkoT1 + i];
                amrex::ParticleReal const dsigdk = (kdsigdk_i + kdsigdk_im1)*0.5_prt/k_ave;
                amrex::ParticleReal const next_sigma = sigma + dsigdk*dk;
                if (random_number*sigma_total > next_sigma) {
                    sigma = next_sigma;
                    koT1_grid_im1 = m_koT1_grid[i];
                    kdsigdk_im1 = kdsigdk_i;
                } else {
                    break;
                }
            }

            // k will be between k_im1 and k_i
            amrex::ParticleReal const f_im1 = kdsigdk_im1/k_ave;
            amrex::ParticleReal const f_i = kdsigdk_i/k_ave;
            amrex::ParticleReal const x = (std::sqrt(f_im1*f_im1 + 2._prt*(f_i - f_im1)*(random_number*sigma_total - sigma)/dk)
                                           - f_im1)/(f_i - f_im1);
            amrex::ParticleReal const result = x*dk + koT1_grid_im1;

            return result;
        }

        /* \brief Do the Bremsstrahlung calculation
         *
         * \param[in] u1x, u1y, u1z the gamma*velocity of the first species, the electrons
         * \param[in] u2x, u2y, u2z the gamma*velocity of the second species, the atoms
         * \param[in] m1, m2 the mass of the electrons and ions
         * \param[in] weight1 the particle weight of the electrons
         * \param[in] weight2 the particle weight of the target (unused)
         * \param[in] n1 the number density of the electrons
         * \param[in] n2 the number density of the target
         * \param[in] dt the time step size
         * \param[in] pair_index the index number of the pair colliding
         * \param[out] p_mask flags whether the pair collided
         * \param[out] p_pair_reaction_weight the particle weight of the generated photon
         * \param[out] p_product_data the energy of the generated photon
         * \param[in] engine the random number generating engine
         */
        AMREX_GPU_DEVICE AMREX_INLINE
        void BremsstrahlungEvent (amrex::ParticleReal & u1x, amrex::ParticleReal & u1y,
                                  amrex::ParticleReal & u1z, amrex::ParticleReal & u2x,
                                  amrex::ParticleReal & u2y, amrex::ParticleReal & u2z,
                                  amrex::ParticleReal m1, amrex::ParticleReal m2,
                                  amrex::ParticleReal weight1, amrex::ParticleReal weight2,
                                  amrex::ParticleReal n1, amrex::ParticleReal n2,
                                  amrex::Real dt, int pair_index,
                                  index_type* AMREX_RESTRICT p_mask,
                                  amrex::ParticleReal* AMREX_RESTRICT p_pair_reaction_weight,
                                  amrex::ParticleReal* AMREX_RESTRICT p_product_data,
                                  amrex::RandomEngine const& engine) const
        {
            using namespace amrex::literals;

            constexpr auto c2 = PhysConst::c * PhysConst::c;

            // compute gamma for particles 1 (electron) and 2 (ion)
            amrex::ParticleReal const u1sq = (u1x*u1x + u1y*u1y + u1z*u1z);
            amrex::ParticleReal const u2sq = (u2x*u2x + u2y*u2y + u2z*u2z);
            amrex::ParticleReal const gamma1 = std::sqrt(1.0_prt + u1sq/c2);
            amrex::ParticleReal const gamma2 = std::sqrt(1.0_prt + u2sq/c2);

            // transform electron to frame where ion is at rest
            amrex::ParticleReal u1x_rel = u1x;
            amrex::ParticleReal u1y_rel = u1y;
            amrex::ParticleReal u1z_rel = u1z;
            ParticleUtils::doLorentzTransformWithU(u1x_rel, u1y_rel, u1z_rel, u2x, u2y, u2z);

            // compute electron KE in frame where ion is at rest
            amrex::ParticleReal const u1sq_rel = (u1x_rel*u1x_rel + u1y_rel*u1y_rel + u1z_rel*u1z_rel);
            amrex::ParticleReal const gamma1_rel = std::sqrt(1.0_prt + u1sq_rel/c2);
            amrex::ParticleReal const KE_eV = m1*u1sq_rel/(1.0_prt + gamma1_rel)/PhysConst::q_e;

            amrex::ParticleReal const wpe = PhysConst::q_e*std::sqrt(n1/(m1*PhysConst::ep0));

            int i0_cut, index;
            amrex::ParticleReal koT1_cut, kdsigdk_cut;
            amrex::ParticleReal w0;
            amrex::ParticleReal const sigma_total = CalculateCrossSection(KE_eV, wpe,
                                                                          index, i0_cut, koT1_cut, kdsigdk_cut, w0);

            if (sigma_total == 0._prt) { return; }

            // determine if the pair collide and then do the collision
            amrex::ParticleReal fmulti = m_multiplier; // >= 1.0
            amrex::ParticleReal const gamma_factor = gamma1_rel/(gamma1*gamma2);
            amrex::ParticleReal const v1 = std::sqrt(u1sq_rel)/gamma1_rel; // magnitude electron velocity
            amrex::ParticleReal arg = fmulti*v1*sigma_total*n2*dt*gamma_factor;
            if (arg > 1.0_prt) {
#ifndef AMREX_USE_GPU
                amrex::Print() << "Notice: arg = " << arg << " in interSpeciesBremsstrahlung" << "\n";
                amrex::Print() << "        v1 = " << v1 << "\n";
                amrex::Print() << "        n2 = " << n2 << "\n";
                amrex::Print() << "        sigma_total = " << sigma_total << "\n";
#endif
                arg = 1.0_prt;
                fmulti = fmulti/arg;
                if (fmulti < 1.0_prt) { fmulti = 1.0_prt; }
            }
            //q12 = 1.0 - exp(-arg)
            amrex::ParticleReal const q12 = arg;

            // Get a random number
            amrex::ParticleReal const random_number = amrex::Random(engine);
            if (random_number < q12) {

                // compute weight for photon
                 amrex::ParticleReal const w_photon = weight1/fmulti;

                // get energy of photon (hbar*omega)
                amrex::ParticleReal const random_number2 = amrex::Random(engine);
                amrex::ParticleReal const EphotonoT1 = Photon_energy(index, i0_cut, koT1_cut, kdsigdk_cut,
                                                                     w0, sigma_total, random_number2);
                amrex::ParticleReal Ephoton = EphotonoT1*KE_eV*PhysConst::q_e;

                // limit Ephoton to KE in weighted center-of-momentum frame
                // This is needed for physical solution that conserves both momentum and energy
                amrex::ParticleReal const mime = (weight2*m2)/(weight1*m1);
                amrex::ParticleReal const u1_rel = std::sqrt(u1sq_rel);
                amrex::ParticleReal const gamma_m_1 = (u1sq_rel/c2)/(gamma1_rel + 1._prt);  // gamma - 1
                amrex::ParticleReal const gamma_m_u = 1._prt/(gamma1_rel + u1_rel/PhysConst::c);  // gamma - u
                amrex::ParticleReal const KE_cm = mime*gamma_m_1/(gamma_m_u + mime)*m1*c2;
                Ephoton = std::min(Ephoton, KE_cm);

                // The electron and ion momentum are updated in the same place where the photon is created,
                // in PhotonCreationFunc.

                p_product_data[pair_index] = Ephoton;
                p_pair_reaction_weight[pair_index] = w_photon;
                p_mask[pair_index] = 1;

            }

        }

        bool m_computeSpeciesDensities = true;
        bool m_computeSpeciesTemperatures = false;
        bool m_need_product_data = true;
        amrex::ParticleReal m_multiplier;
        amrex::ParticleReal m_koT1_cut_default;

        int nkoT1;
        int nKE;

        amrex::ParticleReal * m_koT1_grid;
        amrex::ParticleReal * m_KEgrid_eV;
        amrex::ParticleReal * m_kdsigdk;
        amrex::ParticleReal * m_sigma_total;

    };

    [[nodiscard]] Executor const& executor () const { return m_exe; }

    bool use_global_debye_length() { return m_use_global_debye_length; }

private:

    Executor m_exe;

    bool m_use_global_debye_length = false;

    // Cross section data

    void UploadCrossSection (int Z);

    int nkoT1;
    int nKE;

    // relative photon energy grid ( koT1 = Ephoton_eV/KErel_eV )
    amrex::GpuArray<amrex::ParticleReal, 17> m_koT1_grid_h = {0., 0.025, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99, 1.00};

    // energy grid for electrons in eV
    amrex::GpuArray<amrex::ParticleReal, 11> m_KEgrid_eV_h = {1.0e3, 2.0e3, 5.0e3, 1.0e4, 2.0e4, 5.0e4, 1.0e5, 2.0e5, 5.0e5, 1.0e6, 2.0e6};

    // differential total cross section
    amrex::Vector<amrex::ParticleReal> m_kdsigdk_h;
    amrex::Vector<amrex::ParticleReal> m_sigma_total_h;

    amrex::Gpu::DeviceVector<amrex::ParticleReal> m_koT1_grid_d;
    amrex::Gpu::DeviceVector<amrex::ParticleReal> m_KEgrid_eV_d;
    amrex::Gpu::DeviceVector<amrex::ParticleReal> m_kdsigdk_d;
    amrex::Gpu::DeviceVector<amrex::ParticleReal> m_sigma_total_d;

    // scaled energy-weighted differential total cross section for Bremsstrahlung (kdsigdk_n + Z*kdsigdk_e)
    // for e + H from Table I of Seltzer and Berger, ATOMIC DATA AND NUCLEAR DATA TABLES 35,345-418 (1985).
    // The values here are actualy beta1^2/Z^2*k*dsig/dk in units of mBarns = 1e-31 m^2 where Z is the
    // atomic number and beta1 = sqrt(1.0 - 1/gamma1^2) with gamma1 = 1 + T1/(me*c^2). These vectors are
    // converted to k*dsig/dk in units of m^2 on initialization.

    std::map<int, std::vector<std::vector<amrex::ParticleReal>>> m_kdsigdk_map = {

    // atomic number Z = 1
    {1, {
          {7.853, 7.849, 7.833, 7.800, 7.746, 7.446, 7.040, 6.586, 6.124, 5.664, 5.230, 4.841, 4.521, 4.400, 4.372, 4.362, 4.360}, // 1.0 keV
          {8.805, 8.817, 8.801, 8.738, 8.638, 8.059, 7.377, 6.699, 6.052, 5.431, 4.839, 4.263, 3.682, 3.437, 3.374, 3.326, 3.320}, // 2.0 keV
          {10.32, 10.25, 10.15, 9.976, 9.753, 8.703, 7.661, 6.728, 5.899, 5.148, 4.424, 3.697, 2.897, 2.446, 2.286, 2.161, 2.104}, // 5.0 keV
          {11.55, 11.40, 11.19, 10.89, 10.55, 9.087, 7.795, 6.711, 5.776, 4.952, 4.172, 3.387, 2.509, 1.982, 1.762, 1.548, 1.439}, // 10 keV
          {13.07, 12.61, 12.13, 11.62, 11.11, 9.370, 7.915, 6.674, 5.617, 4.734, 3.933, 3.141, 2.239, 1.684, 1.424, 1.129, 0.967}, // 20 keV
          {14.90, 14.17, 13.42, 12.65, 11.92, 9.620, 7.858, 6.426, 5.295, 4.370, 3.575, 2.777, 1.940, 1.401, 1.112, 0.754, 0.551}, // 50 keV
          {16.94, 15.72, 14.55, 13.47, 12.51, 9.686, 7.689, 6.107, 4.937, 3.951, 3.179, 2.438, 1.670, 1.180, 0.903, 0.548, 0.343}, // 100 keV
          {20.31, 18.17, 16.25, 14.71, 13.46, 9.837, 7.509, 5.808, 4.595, 3.549, 2.694, 2.022, 1.362, 0.938, 0.697, 0.380, 0.197}, // 200 keV
          {27.90, 23.39, 19.68, 17.28, 15.68, 10.84, 7.831, 5.945, 4.555, 3.400, 2.386, 1.634, 1.045, 0.706, 0.505, 0.244, 0.094}, // 500 keV
          {35.57, 28.83, 23.45, 20.32, 18.41, 12.21, 8.960, 6.849, 5.241, 3.927, 2.763, 1.732, 1.038, 0.688, 0.483, 0.218, 0.065}, // 1.0 MeV
          {43.42, 34.55, 27.56, 23.66, 21.45, 14.61, 11.03, 8.605, 6.744, 5.195, 3.809, 2.472, 1.306, 0.846, 0.587, 0.250, 0.057}  // 2.0 MeV
    }},

    // atomic number Z = 2
    {2, {
          {7.167, 7.192, 7.206, 7.201, 7.181, 7.001, 6.726, 6.409, 6.077, 5.740, 5.422, 5.150, 4.939, 4.858, 4.837, 4.825, 4.821}, // 1.0 keV
          {8.232, 8.239, 8.229, 8.187, 8.120, 7.713, 7.200, 6.666, 6.147, 5.646, 5.173, 4.729, 4.313, 4.145, 4.099, 4.064, 4.053}, // 2.0 keV
          {9.678, 9.640, 9.570, 9.444, 9.276, 8.439, 7.568, 6.770, 6.055, 5.397, 4.770, 4.135, 3.510, 3.180, 3.070, 2.986, 2.951}, // 5.0 keV
          {10.81, 10.71, 10.56, 10.33, 10.06, 8.834, 7.706, 6.747, 5.914, 5.166, 4.461, 3.762, 3.009, 2.590, 2.430, 2.283, 2.213}, // 10 keV
          {12.18, 11.80, 11.40, 10.98, 10.55, 9.062, 7.784, 6.678, 5.721, 4.897, 4.148, 3.420, 2.605, 2.131, 1.927, 1.711, 1.600}, // 20 keV
          {13.49, 12.92, 12.33, 11.71, 11.11, 9.139, 7.592, 6.336, 5.325, 4.473, 3.707, 2.950, 2.148, 1.657, 1.413, 1.130, 0.974}, // 50 keV
          {14.54, 13.71, 12.88, 12.05, 11.28, 8.928, 7.241, 5.917, 4.892, 4.017, 3.265, 2.544, 1.797, 1.334, 1.093, 0.792, 0.624}, // 100 keV
          {16.12, 14.79, 13.54, 12.44, 11.49, 8.747, 6.868, 5.470, 4.418, 3.513, 2.751, 2.092, 1.433, 1.023, 0.802, 0.525, 0.365}, // 200 keV
          {19.94, 17.44, 16.27, 13.69, 12.50, 9.013, 6.783, 5.288, 4.140, 3.184, 2.345, 1.667, 1.080, 0.745, 0.556, 0.315, 0.178}, // 500 keV
          {24.04, 20.39, 17.38, 15.42, 14.06, 9.865, 7.470, 5.820, 4.535, 3.479, 2.549, 1.714, 1.060, 0.713, 0.517, 0.266, 0.123}, // 1.0 MeV
          {28.11, 23.46, 19.71, 17.43, 15.98, 11.46, 8.864, 7.025, 5.582, 4.375, 3.303, 2.268, 1.320, 0.866, 0.613, 0.292, 0.108}  // 2.0 MeV
    }},

    // atomic number Z = 5
    {5, {
          {5.670, 5.717, 5.760, 5.793, 5.820, 5.871, 5.856, 5.797, 5.713, 5.610, 5.502, 5.411, 5.350, 5.323, 5.311, 5.299, 5.293}, // 1.0 keV
          {6.814, 6.863, 6.903, 6.925, 6.932, 6.863, 6.698, 6.484, 6.250, 6.010, 5.789, 5.596, 5.438, 5.366, 5.340, 5.318, 5.306}, // 2.0 keV
          {8.325, 8.350, 8.360, 8.324, 8.265, 7.870, 7.382, 6.895, 6.441, 6.020, 5.635, 5.285, 4.987, 4.880, 4.847, 4.814, 4.799}, // 5.0 keV
          {9.457, 9.432, 9.377, 9.272, 9.128, 8.384, 7.614, 6.920, 6.310, 5.759, 5.259, 4.795, 4.372, 4.201, 4.150, 4.111, 4.097}, // 10 keV
          {10.56, 10.44, 10.29, 10.07, 9.819, 8.684, 7.651, 6.784, 6.042, 5.381, 4.785, 4.222, 3.676, 3.414, 3.332, 3.275, 3.252}, // 20 keV
          {12.00, 11.56, 11.11, 10.67, 10.23, 8.718, 7.455, 6.396, 5.515, 4.765, 4.097, 3.463, 2.785, 2.431, 2.302, 2.185, 2.136}, // 50 keV
          {12.69, 12.07, 11.45, 10.86, 10.30, 8.452, 7.031, 5.897, 4.996, 4.223, 3.526, 2.867, 2.182, 1.807, 1.650, 1.490, 1.410}, // 100 keV
          {13.49, 12.57, 11.70, 10.91, 10.22, 8.039, 6.503, 5.341, 4.421, 3.625, 2.914, 2.286, 1.651, 1.287, 1.121, 0.940, 0.844}, // 200 keV
          {15.27, 13.74, 12.38, 11.33, 10.49, 7.918, 6.188, 4.945, 3.953, 3.121, 2.392, 1.765, 1.185, 0.865, 0.706, 0.521, 0.419}, // 500 keV
          {17.23, 15.27, 13.58, 12.36, 11.41, 8.400, 6.550, 5.235, 4.166, 3.260, 2.464, 1.749, 1.131, 0.788, 0.617, 0.409, 0.293}, // 1.0 MeV
          {19.34, 16.89, 14.86, 13.50, 12.53, 9.504, 7.545, 6.077, 4.899, 3.908, 3.027, 2.183, 1.368, 0.924, 0.693, 0.414, 0.256}  // 2.0 MeV
    }},

    // atomic number Z = 6
    {6, {
          {5.336, 5.384, 5.427, 5.464, 5.494, 5.570, 5.585, 5.557, 5.501, 5.425, 5.337, 5.259, 5.206, 5.185, 5.175, 5.164, 5.158}, // 1.0 keV
          {6.498, 6.553, 6.600, 6.630, 6.648, 6.628, 6.518, 6.359, 6.174, 5.976, 5.790, 5.619, 5.470, 5.400, 5.375, 5.355, 5.346}, // 2.0 keV
          {8.028, 8.065, 8.084, 8.070, 8.031, 7.721, 7.315, 6.897, 6.502, 6.135, 5.801, 5.503, 5.251, 5.160, 5.129, 5.096, 5.080}, // 5.0 keV
          {9.168, 9.159, 9.123, 9.041, 8.922, 8.281, 7.593, 6.964, 6.411, 5.913, 5.467, 5.064, 4.712, 4.577, 4.536, 4.500, 4.485}, // 10 keV
          {10.27, 10.18, 10.05, 9.865, 9.638, 8.606, 7.644, 6.832, 6.140, 5.522, 4.973, 4.465, 3.994, 3.779, 3.715, 3.671, 3.654}, // 20 keV
          {11.72, 11.31, 10.90, 10.49, 10.08, 8.651, 7.450, 6.436, 5.587, 4.862, 4.221, 3.622, 2.997, 2.684, 2.578, 2.489, 2.455}, // 50 keV
          {12.38, 11.81, 11.24, 10.68, 10.16, 8.400, 7.026, 5.924, 5.045, 4.293, 3.611, 2.971, 2.314, 1.967, 1.830, 1.701, 1.638}, // 100 keV
          {13.10, 12.26, 11.45, 10.72, 10.06, 7.971, 6.481, 5.348, 4.447, 3.668, 2.970, 2.351, 1.726, 1.378, 1.225, 1.068, 0.987}, // 200 keV
          {14.65, 13.27, 12.04, 11.06, 10.26, 7.806, 6.132, 4.921, 3.953, 3.133, 2.417, 1.797, 1.222, 0.904, 0.756, 0.586, 0.494}, // 500 keV
          {16.39, 14.64, 13.12, 11.99, 11.09, 8.243, 6.460, 5.179, 4.135, 3.248, 2.467, 1.769, 1.153, 0.814, 0.650, 0.453, 0.345}, // 1.0 MeV
          {18.14, 16.07, 14.31, 13.09, 12.18, 9.254, 7.367, 5.974, 4.837, 3.869, 3.000, 2.184, 1.385, 0.942, 0.720, 0.453, 0.301}  // 2.0 MeV
    }}

    };

};

#endif // WARPX_BREMSSTRAHLUNG_FUNC_H_
