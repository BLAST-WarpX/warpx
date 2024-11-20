/* Copyright 2021 Revathi Jambunathan
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_BACKTRANSFORMPARTICLEFUNCTOR_H_
#define WARPX_BACKTRANSFORMPARTICLEFUNCTOR_H_

#include "ComputeParticleDiagFunctor.H"
#include "Particles/Pusher/GetAndSetPosition.H"

#include <AMReX.H>
#include <AMReX_AmrParticles.H>




/**
 * \brief Filter to select particles that correspond to a z-slice of the corresponding lab-frame.
 */
struct SelectParticles
{
    using TmpParticles = WarpXParticleContainer::TmpParticles;

    /**
     * \brief Constructor of SelectParticles functor.
     *
     * @param[in] a_pti              WarpX particle iterator
     * @param[in] tmp_particle_data  temporary particle data
     * @param[in] current_z_boost    current z-position of the slice in boosted frame
     * @param[in] old_z_boost        previous z-position of the slice in boosted frame
     * @param[in] a_offset           index offset for particles to be selected
     */
    SelectParticles( const WarpXParIter& a_pti, TmpParticles& tmp_particle_data,
                     amrex::Real current_z_boost, amrex::Real old_z_boost,
                     int a_offset = 0);

    /**
     * \brief Functor call. This method determines if a given particle should be selected
     * for Lorentz transformation in obtaining the lab-frame data. The particles that
     * with positions that correspond to the specific z-slice in boosted frame are selected.
     *
     * @tparam SrcData type of source data
     * @param[in] src SrcData particle tile data
     * @param[in] i       particle index
     * @return 1 if particles is selected for transformation, else 0
     */
    template <typename SrcData>
    AMREX_GPU_HOST_DEVICE
    int operator() (const SrcData& src, int i) const noexcept
    {
        amrex::ignore_unused(src);
        amrex::ParticleReal xp, yp, zp;
        m_get_position(i, xp, yp, zp);
        int Flag = 0;
        if ( ( (zp >= m_current_z_boost) && (zpold[i] <= m_old_z_boost) ) ||
             ( (zp <= m_current_z_boost) && (zpold[i] >= m_old_z_boost) ))
        {    Flag = 1;
        }
        return Flag;
    }

    /** Object to extract the positions of the macroparticles inside a ParallelFor kernel */
    GetParticlePosition<PIdx> m_get_position;
    /** Z coordinate in boosted frame that corresponds to a give snapshot*/
    amrex::Real m_current_z_boost;
    /** Previous Z coordinate in boosted frame that corresponds to a give snapshot*/
    amrex::Real m_old_z_boost;
    /** Particle z coordinate in boosted frame*/
    amrex::ParticleReal* AMREX_RESTRICT zpold = nullptr;
};

/**
 * \brief Transform functor to Lorentz-transform particles and obtain lab-frame data.
 */
struct LorentzTransformParticles
{

    using TmpParticles = WarpXParticleContainer::TmpParticles;

    /**
     * \brief Constructor of the LorentzTransformParticles functor.
     *
     * @param[in] a_pti              WarpX particle iterator
     * @param[in] tmp_particle_data  temporary particle data
     * @param[in] t_boost            time in boosted frame
     * @param[in] dt                 timestep in boosted-frame
     * @param[in] t_lab              time in lab-frame
     * @param[in] a_offset           index offset for particles to be transformed
     */
    LorentzTransformParticles ( const WarpXParIter& a_pti, TmpParticles& tmp_particle_data,
                                amrex::Real t_boost, amrex::Real dt,
                                amrex::Real t_lab, int a_offset = 0);

    /**
     * \brief Functor call. This method computes the Lorentz-transform for particle
     * attributes to obtain the lab-frame snapshot data.
     *
     * @param[out] dst DstData particle tile data that stores the transformed particle data
     * @param[in] src SrcData particle tile data that is selected for transformation
     * @param[in] i_src particle index of the source particles
     * @param[in] i_dst particle index of the target particles (transformed data).
     */
    template <typename DstData, typename SrcData>
    AMREX_GPU_HOST_DEVICE
    void operator () (const DstData& dst, const SrcData& src, int i_src, int i_dst) const noexcept
    {
        amrex::ignore_unused(src);
        using namespace amrex::literals;
        // get current src position
        amrex::ParticleReal xpnew, ypnew, zpnew;
        m_get_position(i_src, xpnew, ypnew, zpnew);
        const amrex::Real gamma_new_p = std::sqrt(1.0_rt + m_inv_c2*
                                        ( m_uxpnew[i_src] * m_uxpnew[i_src]
                                        + m_uypnew[i_src] * m_uypnew[i_src]
                                        + m_uzpnew[i_src] * m_uzpnew[i_src]));
        const amrex::Real gamma_old_p = std::sqrt(1.0_rt + m_inv_c2*
                                        ( m_uxpold[i_src] * m_uxpold[i_src]
                                        + m_uypold[i_src] * m_uypold[i_src]
                                        + m_uzpold[i_src] * m_uzpold[i_src]));
        const amrex::Real t_new_p = m_gammaboost * m_t_boost - m_uzfrm * zpnew * m_inv_c2;
        const amrex::Real z_new_p = m_gammaboost* ( zpnew + m_betaboost * m_Phys_c * m_t_boost);
        const amrex::Real uz_new_p = m_gammaboost * m_uzpnew[i_src] - gamma_new_p * m_uzfrm;
        const amrex::Real t_old_p = m_gammaboost * (m_t_boost - m_dt)
                                    - m_uzfrm * m_zpold[i_src] * m_inv_c2;
        const amrex::Real z_old_p = m_gammaboost * ( m_zpold[i_src] + m_betaboost
                                                     * m_Phys_c * (m_t_boost - m_dt ) );
        const amrex::Real uz_old_p = m_gammaboost * m_uzpold[i_src] - gamma_old_p * m_uzfrm;
        // interpolate in time to t_lab
        const amrex::Real weight_old = (t_new_p - m_t_lab)
                                     / (t_new_p - t_old_p);
        const amrex::Real weight_new = (m_t_lab - t_old_p)
                                     / (t_new_p - t_old_p);
        // weighted sum of old and new values
        const amrex::ParticleReal xp = m_xpold[i_src] * weight_old + xpnew * weight_new;
        const amrex::ParticleReal yp = m_ypold[i_src] * weight_old + ypnew * weight_new;
        const amrex::ParticleReal zp = z_old_p * weight_old + z_new_p * weight_new;
        const amrex::ParticleReal uxp = m_uxpold[i_src] * weight_old
                                      + m_uxpnew[i_src] * weight_new;
        const amrex::ParticleReal uyp = m_uypold[i_src] * weight_old
                                      + m_uypnew[i_src] * weight_new;
        const amrex::ParticleReal uzp = uz_old_p * weight_old
                                      + uz_new_p * weight_new;
#if defined (WARPX_DIM_3D)
        dst.m_rdata[PIdx::x][i_dst] = xp;
        dst.m_rdata[PIdx::y][i_dst] = yp;
        dst.m_rdata[PIdx::z][i_dst] = zp;
#elif defined (WARPX_DIM_RZ)
        dst.m_rdata[PIdx::x][i_dst] = std::sqrt(xp*xp + yp*yp);
        dst.m_rdata[PIdx::z][i_dst] = zp;
        dst.m_rdata[PIdx::theta][i_dst] = std::atan2(yp, xp);
#elif defined (WARPX_DIM_XZ)
        dst.m_rdata[PIdx::x][i_dst] = xp;
        dst.m_rdata[PIdx::z][i_dst] = zp;
        amrex::ignore_unused(yp);
#elif defined (WARPX_DIM_1D_Z)
        dst.m_rdata[PIdx::z][i_dst] = zp;
        amrex::ignore_unused(xp, yp);
#else
        amrex::ignore_unused(xp, yp, zp);
#endif
        dst.m_rdata[PIdx::w][i_dst] = m_wpnew[i_src];
        dst.m_rdata[PIdx::ux][i_dst] = uxp;
        dst.m_rdata[PIdx::uy][i_dst] = uyp;
        dst.m_rdata[PIdx::uz][i_dst] = uzp;
    }

    GetParticlePosition<PIdx> m_get_position;

    amrex::ParticleReal* AMREX_RESTRICT m_xpold = nullptr;
    amrex::ParticleReal* AMREX_RESTRICT m_ypold = nullptr;
    amrex::ParticleReal* AMREX_RESTRICT m_zpold = nullptr;

    amrex::ParticleReal* AMREX_RESTRICT m_uxpold = nullptr;
    amrex::ParticleReal* AMREX_RESTRICT m_uypold = nullptr;
    amrex::ParticleReal* AMREX_RESTRICT m_uzpold = nullptr;

    const amrex::ParticleReal* AMREX_RESTRICT m_uxpnew = nullptr;
    const amrex::ParticleReal* AMREX_RESTRICT m_uypnew = nullptr;
    const amrex::ParticleReal* AMREX_RESTRICT m_uzpnew = nullptr;
    const amrex::ParticleReal* AMREX_RESTRICT m_wpnew = nullptr;

    amrex::Real m_gammaboost;
    amrex::Real m_betaboost;
    amrex::Real m_Phys_c;
    amrex::Real m_uzfrm;
    amrex::Real m_inv_c2;
    amrex::Real m_t_boost;
    amrex::Real m_dt;
    amrex::Real m_t_lab;
};

/**
 * \brief BackTransform functor to select particles and Lorentz Transform them
 *  and store in particle buffers
 */
class BackTransformParticleFunctor final : public ComputeParticleDiagFunctor
{
public:
    BackTransformParticleFunctor(WarpXParticleContainer *pc_src, std::string species_name, int num_buffers);
    /** Computes the Lorentz transform of source particles to obtain lab-frame data in pc_dst*/
    void operator () (PinnedMemoryParticleContainer& pc_dst, int &TotalParticleCounter, int i_buffer) const override;
    void InitData() override;
    /** \brief Prepare data required to back-transform particle attributes for lab-frame snapshot, i_buffer
     *
     * \param[in] i_buffer           index of the snapshot
     * \param[in] z_slice_in_domain  if the z-slice at current_z_boost is within the bounds of
     *            the boosted-frame and lab-frame domain. The particles are transformed
     *            only if this value is true.
     * \param[in] old_z_boost        previous z-position of the slice in boosted frame
     * \param[in] current_z_boost    z co-ordinate of the slice selected in boosted-frame.
     * \param[in] t_lab              current time in lab-frame for snapshot, i_buffer.
     * \param[in] snapshot_full      if the current snapshot, with index, i_buffer, is
     *            full (1) or not (0). If it is full, then Lorentz-transform is not performed
     *            by setting m_perform_backtransform to 0 for the corresponding ith snapshot.
     */
    void PrepareFunctorData ( int i_buffer, bool z_slice_in_domain, amrex::Real old_z_boost,
                              amrex::Real current_z_boost, amrex::Real t_lab,
                              int snapshot_full) override;
private:
    /** Source particle container */
    WarpXParticleContainer* m_pc_src = nullptr;
    /** String containing species name of particles being transformed*/
    std::string m_species_name;
    /** Number of buffers or snapshots*/
    int m_num_buffers;
    /** Vector of current z co-ordinate in the boosted frame for each snapshot*/
    amrex::Vector<amrex::Real> m_current_z_boost;
    /** Vector of previous z co-ordinate in the boosted frame for each snapshot*/
    amrex::Vector<amrex::Real> m_old_z_boost;
    /** Vector of lab-frame time for each snapshot*/
    amrex::Vector<amrex::Real> m_t_lab;
    /** Vector of 0s and 1s stored to check if back-transformation is to be performed for
     *  the ith snapshot. The value is set to 0 (false) or 1 (true) using the
     *  boolean ZSliceInDomain in PrepareFunctorData()
     */
    amrex::Vector<int> m_perform_backtransform;
};




#endif //
