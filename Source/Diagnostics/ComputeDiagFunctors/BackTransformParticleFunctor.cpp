/* Copyright 2021 Revathi Jambunathan
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "BackTransformParticleFunctor.H"

#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_BaseFwd.H>

SelectParticles::SelectParticles (const WarpXParIter& a_pti, TmpParticles& tmp_particle_data,
                                  amrex::Real current_z_boost, amrex::Real old_z_boost,
                                  int a_offset)
    : m_current_z_boost(current_z_boost), m_old_z_boost(old_z_boost)
{
    m_get_position = GetParticlePosition<PIdx>(a_pti, a_offset);

    const auto lev = a_pti.GetLevel();
    const auto index = a_pti.GetPairIndex();

    zpold = tmp_particle_data[lev][index][TmpIdx::zold].dataPtr();
}


LorentzTransformParticles::LorentzTransformParticles ( const WarpXParIter& a_pti,
                                TmpParticles& tmp_particle_data,
                                amrex::Real t_boost, amrex::Real dt,
                                amrex::Real t_lab, int a_offset)
    : m_t_boost(t_boost), m_dt(dt), m_t_lab(t_lab)
{
    using namespace amrex::literals;

    if (tmp_particle_data.empty()) { return; }
    m_get_position = GetParticlePosition<PIdx>(a_pti, a_offset);

    const auto& attribs = a_pti.GetAttribs();
    m_wpnew = attribs[PIdx::w].dataPtr();
    m_uxpnew = attribs[PIdx::ux].dataPtr();
    m_uypnew = attribs[PIdx::uy].dataPtr();
    m_uzpnew = attribs[PIdx::uz].dataPtr();

    const auto lev = a_pti.GetLevel();
    const auto index = a_pti.GetPairIndex();

    m_xpold = tmp_particle_data[lev][index][TmpIdx::xold].dataPtr();
    m_ypold = tmp_particle_data[lev][index][TmpIdx::yold].dataPtr();
    m_zpold = tmp_particle_data[lev][index][TmpIdx::zold].dataPtr();
    m_uxpold = tmp_particle_data[lev][index][TmpIdx::uxold].dataPtr();
    m_uypold = tmp_particle_data[lev][index][TmpIdx::uyold].dataPtr();
    m_uzpold = tmp_particle_data[lev][index][TmpIdx::uzold].dataPtr();

    m_betaboost = WarpX::beta_boost;
    m_gammaboost = WarpX::gamma_boost;
    m_Phys_c = PhysConst::c;
    m_inv_c2 = 1._rt/(m_Phys_c * m_Phys_c);
    m_uzfrm = -m_gammaboost*m_betaboost*m_Phys_c;
}

/**
 * \brief Functor to compute Lorentz Transform and store the selected particles in existing
 * particle buffers
 */
BackTransformParticleFunctor::BackTransformParticleFunctor (
                              WarpXParticleContainer *pc_src,
                              std::string species_name,
                              int num_buffers)
    : m_pc_src{pc_src}, m_species_name{std::move(species_name)}, m_num_buffers{num_buffers}
{
    InitData();
}


void
BackTransformParticleFunctor::operator () (PinnedMemoryParticleContainer& pc_dst, int &totalParticleCounter, int i_buffer) const
{
    if (m_perform_backtransform[i_buffer] == 0) { return; }
    auto &warpx = WarpX::GetInstance();
    // get particle slice
    const int nlevs = std::max(0, m_pc_src->finestLevel()+1);
    auto tmp_particle_data = m_pc_src->getTmpParticleData();
    for (int lev = 0; lev < nlevs; ++lev) {
        const amrex::Real t_boost = warpx.gett_new(0);
        const amrex::Real dt = warpx.getdt(0);

        for (WarpXParIter pti(*m_pc_src, lev); pti.isValid(); ++pti) {
            pc_dst.DefineAndReturnParticleTile(lev, pti.index(), pti.LocalTileIndex() );
        }

        auto& particles = m_pc_src->GetParticles(lev);
#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
        {
            // Temporary arrays to store copy_flag and copy_index for particles
            // that cross the z-slice
            amrex::Gpu::DeviceVector<int> FlagForPartCopy;
            amrex::Gpu::DeviceVector<int> IndexForPartCopy;

            for (WarpXParIter pti(*m_pc_src, lev); pti.isValid(); ++pti) {

                auto index = std::make_pair(pti.index(), pti.LocalTileIndex());

                const auto GetParticleFilter = SelectParticles(pti, tmp_particle_data,
                                               m_current_z_boost[i_buffer],
                                               m_old_z_boost[i_buffer]);
                const auto GetParticleLorentzTransform = LorentzTransformParticles(
                                                         pti, tmp_particle_data,
                                                         t_boost, dt,
                                                         m_t_lab[i_buffer]);

                long const np = pti.numParticles();

                FlagForPartCopy.resize(np);
                IndexForPartCopy.resize(np);

                int* const AMREX_RESTRICT Flag = FlagForPartCopy.dataPtr();
                int* const AMREX_RESTRICT IndexLocation = IndexForPartCopy.dataPtr();

                const auto& ptile_src = particles.at(index);
                auto src_data = ptile_src.getConstParticleTileData();
                // Flag particles that need to be copied if they cross the z-slice
                // setting this to 1 for testing (temporarily)
                amrex::ParallelFor(np,
                [=] AMREX_GPU_DEVICE(int i)
                {
                    Flag[i] = GetParticleFilter(src_data, i);
                });

                const int total_partdiag_size = amrex::Scan::ExclusiveSum(np,Flag,IndexLocation);
                auto& ptile_dst = pc_dst.DefineAndReturnParticleTile(lev, pti.index(), pti.LocalTileIndex() );
                auto old_size = ptile_dst.numParticles();
                ptile_dst.resize(old_size + total_partdiag_size);
                amrex::filterParticles(ptile_dst, ptile_src, GetParticleFilter, 0, old_size, np);
                auto dst_data = ptile_dst.getParticleTileData();
                amrex::ParallelFor(np,
                [=] AMREX_GPU_DEVICE(int i)
                {
                   if (Flag[i] == 1) { GetParticleLorentzTransform(dst_data, src_data, i,
                                                                 old_size + IndexLocation[i]);
                   }
                });
                amrex::Gpu::synchronize();
            }
        }
    }
    totalParticleCounter = static_cast<int>(pc_dst.TotalNumberOfParticles());
}


void
BackTransformParticleFunctor::InitData()
{
    m_current_z_boost.resize(m_num_buffers);
    m_old_z_boost.resize(m_num_buffers);
    m_t_lab.resize(m_num_buffers);
    m_perform_backtransform.resize(m_num_buffers);
}

void
BackTransformParticleFunctor::PrepareFunctorData ( int i_buffer, bool z_slice_in_domain,
                              amrex::Real old_z_boost, amrex::Real current_z_boost,
                              amrex::Real t_lab, int snapshot_full)
{
    m_old_z_boost.at(i_buffer) = old_z_boost;
    m_current_z_boost.at(i_buffer) = current_z_boost;
    m_t_lab.at(i_buffer) = t_lab;
    m_perform_backtransform.at(i_buffer) = 0;
    if (z_slice_in_domain && (snapshot_full == 0)) { m_perform_backtransform.at(i_buffer) = 1; }
}
