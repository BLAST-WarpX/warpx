/* Copyright 2019-2020 Andrew Myers, Axel Huebl,
 * Maxence Thevenet
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_DEFAULTINITIALIZATION_H_
#define WARPX_DEFAULTINITIALIZATION_H_

#include <WarpX.H>
#ifdef WARPX_QED
#   include "Particles/ElementaryProcess/QEDInternals/BreitWheelerEngineWrapper.H"
#   include "Particles/ElementaryProcess/QEDInternals/QuantumSyncEngineWrapper.H"
#endif

#include <AMReX_GpuContainers.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <map>
#include <string>

/**
 * \brief This set of initialization policies describes what happens
 * when we need to create a new particle due to an elementary process.
 * For example, when an ionization event creates an electron, these
 * policies control the initial values of the electron's components.
 * These can always be over-written later.
 *
 * The specific meanings are as follows:
 *     Zero         - set the component to zero
 *     One          - set the component to one
 *     RandomExp    - a special flag for the optical depth component used by
 *                    certain QED processes, which gets a random initial value
 *                    extracted from an exponential distribution
 *
 */
enum struct InitializationPolicy {Zero=0, One, RandomExp};

/**
 * \brief This map sets the initialization policy for each particle component
 * used in WarpX.
 */
static std::map<std::string, InitializationPolicy> initialization_policies = {
    {"w",     InitializationPolicy::Zero },
    {"ux",    InitializationPolicy::Zero },
    {"uy",    InitializationPolicy::Zero },
    {"uz",    InitializationPolicy::Zero },
#ifdef WARPX_DIM_RZ
    {"theta", InitializationPolicy::Zero},
#endif

#ifdef WARPX_QED
    {"opticalDepthBW",   InitializationPolicy::RandomExp},
    {"opticalDepthQSR",   InitializationPolicy::RandomExp}
#endif

};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::ParticleReal initializeRealValue (const InitializationPolicy policy, amrex::RandomEngine const& engine) noexcept
{
    switch (policy) {
        case InitializationPolicy::Zero : return 0.0;
        case InitializationPolicy::One  : return 1.0;
        case InitializationPolicy::RandomExp : {
            return -std::log(amrex::Random(engine));
        }
        default : {
            amrex::Abort("Initialization Policy not recognized");
            return 1.0;
        }
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int initializeIntValue (const InitializationPolicy policy) noexcept
{
    switch (policy) {
        case InitializationPolicy::Zero : return 0;
        case InitializationPolicy::One  : return 1;
        default : {
            amrex::Abort("Initialization Policy not recognized");
            return 1;
        }
    }
}

namespace ParticleCreation {

    /**
     * \brief Default initialize runtime attributes in a tile. This routine does not initialize the
     * first n_external_attr_real real attributes and the first n_external_attr_int integer
     * attributes, which have been in principle externally set elsewhere.
     *
     * @tparam PTile The type of the particle tile to operate on (e.g. could use different allocators)
     * @param[inout] ptile the tile in which attributes are initialized
     * @param[in] n_external_attr_real The number of real attributes that have been externally set.
     * These are NOT initialized by this function.
     * @param[in] n_external_attr_int The number of integer attributes that have been externally set.
     * These are NOT initialized by this function.
     * @param[in] user_real_attribs The names of the real components for this particle tile
     * @param[in] user_int_attribs The names of the int components for this particle tile
     * @param[in] particle_comps particle component names for real comps
     * @param[in] particle_icomps particle component names for int comps
     * @param[in] user_real_attrib_parser the parser functions used to initialize the user real components
     * @param[in] user_int_attrib_parser the parser functions used to initialize the user int components
     * @param[in] do_qed_comps whether to initialize the qed components (these are usually handled by
     * SmartCopy, but NOT when adding particles in AddNParticles)
     * @param[in] p_bw_engine the engine to use for setting the breit-wheeler component for QED
     * @param[in] p_qs_engine the engine to use for setting the quantum synchrotron component for QED
     * @param[in] ionization_initial_level the ionization level particles created should start at
     * @param[in] start the index to start initializing particles
     * @param[in] stop the index to stop initializing particles
     */
template <typename PTile>
void DefaultInitializeRuntimeAttributes (PTile& ptile,
                                         const int n_external_attr_real,
                                         const int n_external_attr_int,
                                         const std::vector<std::string>& user_real_attribs,
                                         const std::vector<std::string>& user_int_attribs,
                                         const std::vector<std::string>& particle_comps,
                                         const std::vector<std::string>& particle_icomps,
                                         const std::vector<amrex::Parser*>& user_real_attrib_parser,
                                         const std::vector<amrex::Parser*>& user_int_attrib_parser,
#ifdef WARPX_QED
                                         const bool do_qed_comps,
                                         BreitWheelerEngine* p_bw_engine,
                                         QuantumSynchrotronEngine* p_qs_engine,
#endif
                                         const int ionization_initial_level,
                                         int start, int stop)
{
        using namespace amrex::literals;

        // Preparing data needed for user defined attributes
        const auto n_user_real_attribs = static_cast<int>(user_real_attribs.size());
        const auto n_user_int_attribs = static_cast<int>(user_int_attribs.size());
        const auto get_position = GetParticlePosition<PIdx>(ptile);
        const auto soa = ptile.getParticleTileData();
        const amrex::ParticleReal* AMREX_RESTRICT ux = soa.m_rdata[PIdx::ux];
        const amrex::ParticleReal* AMREX_RESTRICT uy = soa.m_rdata[PIdx::uy];
        const amrex::ParticleReal* AMREX_RESTRICT uz = soa.m_rdata[PIdx::uz];
        constexpr int lev = 0;
        const amrex::Real t = WarpX::GetInstance().gett_new(lev);

        // Initialize the last NumRuntimeRealComps() - n_external_attr_real runtime real attributes
        for (int j = PIdx::nattribs + n_external_attr_real; j < ptile.NumRealComps() ; ++j)
        {
            auto attr_ptr = ptile.GetStructOfArrays().GetRealData(j).data();
#ifdef WARPX_QED
            // Current runtime comp is quantum synchrotron optical depth
            auto const it_qsr = std::find(particle_comps.begin(), particle_comps.end(), "opticalDepthQSR");
            if (it_qsr != particle_comps.end() &&
                std::distance(particle_comps.begin(), it_qsr) == j)
            {
                if (!do_qed_comps) { continue; }
                const QuantumSynchrotronGetOpticalDepth quantum_sync_get_opt =
                                                p_qs_engine->build_optical_depth_functor();
                // If the particle tile was allocated in a memory pool that can run on GPU, launch GPU kernel
                if constexpr (amrex::RunOnGpu<typename PTile::template AllocatorType<amrex::Real>>::value) {
                        amrex::ParallelForRNG(stop - start,
                                              [=] AMREX_GPU_DEVICE (int i, amrex::RandomEngine const& engine) noexcept {
                                                  const int ip = i + start;
                                                  attr_ptr[ip] = quantum_sync_get_opt(engine);
                                              });
                // Otherwise (e.g. particle tile allocated in pinned memory), run on CPU
                } else {
                    for (int ip = start; ip < stop; ++ip) {
#ifdef AMREX_USE_GPU
                        attr_ptr[ip] = quantum_sync_get_opt(amrex::RandomEngine{nullptr});
#else
                        attr_ptr[ip] = quantum_sync_get_opt(amrex::RandomEngine{});
#endif
                    }
                }
            }

            // Current runtime comp is Breit-Wheeler optical depth
            auto const it_bw = std::find(particle_comps.begin(), particle_comps.end(), "opticalDepthBW");
            if (it_bw != particle_comps.end() &&
                std::distance(particle_comps.begin(), it_bw) == j)
            {
                if (!do_qed_comps) { continue; }
                const BreitWheelerGetOpticalDepth breit_wheeler_get_opt =
                                                p_bw_engine->build_optical_depth_functor();;
                // If the particle tile was allocated in a memory pool that can run on GPU, launch GPU kernel
                if constexpr (amrex::RunOnGpu<typename PTile::template AllocatorType<amrex::Real>>::value) {
                        amrex::ParallelForRNG(stop - start,
                                              [=] AMREX_GPU_DEVICE (int i, amrex::RandomEngine const& engine) noexcept {
                                                  const int ip = i + start;
                                                  attr_ptr[ip] = breit_wheeler_get_opt(engine);
                                              });
                // Otherwise (e.g. particle tile allocated in pinned memory), run on CPU
                } else {
                    for (int ip = start; ip < stop; ++ip) {
#ifdef AMREX_USE_GPU
                        attr_ptr[ip] = breit_wheeler_get_opt(amrex::RandomEngine{nullptr});
#else
                        attr_ptr[ip] = breit_wheeler_get_opt(amrex::RandomEngine{});
#endif
                    }
                }
            }
#endif

            for (int ia = 0; ia < n_user_real_attribs; ++ia)
            {
                // Current runtime comp is ia-th user defined attribute
                auto const it_ura = std::find(particle_comps.begin(), particle_comps.end(), user_real_attribs[ia]);
                if (it_ura != particle_comps.end() &&
                    std::distance(particle_comps.begin(), it_ura) == j)
                {
                    const amrex::ParserExecutor<7> user_real_attrib_parserexec =
                                             user_real_attrib_parser[ia]->compile<7>();
                    // If the particle tile was allocated in a memory pool that can run on GPU, launch GPU kernel
                    if constexpr (amrex::RunOnGpu<typename PTile::template AllocatorType<amrex::Real>>::value) {
                        amrex::ParallelFor(stop - start,
                                           [=] AMREX_GPU_DEVICE (int i) noexcept {
                                               const int ip = i + start;
                                               amrex::ParticleReal xp, yp, zp;
                                               get_position(ip, xp, yp, zp);
                                               attr_ptr[ip] = user_real_attrib_parserexec(xp, yp, zp,
                                                                                          ux[ip], uy[ip], uz[ip], t);
                                           });
                    // Otherwise (e.g. particle tile allocated in pinned memory), run on CPU
                    } else {
                        for (int ip = start; ip < stop; ++ip) {
                            amrex::ParticleReal xp, yp, zp;
                            get_position(ip, xp, yp, zp);
                            attr_ptr[ip] = user_real_attrib_parserexec(xp, yp, zp,
                                                                       ux[ip], uy[ip], uz[ip], t);
                        }
                    }
                }
            }
        }

        // Initialize the last NumRuntimeIntComps() - n_external_attr_int runtime int attributes
        for (int j = n_external_attr_int; j < ptile.NumIntComps() ; ++j)
        {
            auto attr_ptr = ptile.GetStructOfArrays().GetIntData(j).data();

            // Current runtime comp is ionization level
            auto const it_ioniz = std::find(particle_icomps.begin(), particle_icomps.end(), "ionizationLevel");
            if (it_ioniz != particle_icomps.end() &&
                std::distance(particle_icomps.begin(), it_ioniz) == j)
            {
                if constexpr (amrex::RunOnGpu<typename PTile::template AllocatorType<int>>::value) {
                        amrex::ParallelFor(stop - start,
                                           [=] AMREX_GPU_DEVICE (int i) noexcept {
                                               const int ip = i + start;
                                               attr_ptr[ip] = ionization_initial_level;
                                           });
                } else {
                    for (int ip = start; ip < stop; ++ip) {
                        attr_ptr[ip] = ionization_initial_level;
                    }
                }
            }

            for (int ia = 0; ia < n_user_int_attribs; ++ia)
            {
                // Current runtime comp is ia-th user defined attribute
                auto const it_uia = std::find(particle_icomps.begin(), particle_icomps.end(), user_int_attribs[ia]);
                if (it_uia != particle_icomps.end() &&
                    std::distance(particle_icomps.begin(), it_uia) == j)
                {
                     const amrex::ParserExecutor<7> user_int_attrib_parserexec =
                                             user_int_attrib_parser[ia]->compile<7>();
                if constexpr (amrex::RunOnGpu<typename PTile::template AllocatorType<int>>::value) {
                        amrex::ParallelFor(stop - start,
                                           [=] AMREX_GPU_DEVICE (int i) noexcept {
                                               const int ip = i + start;
                                               amrex::ParticleReal xp, yp, zp;
                                               get_position(ip, xp, yp, zp);
                                               attr_ptr[ip] = static_cast<int>(
                                                                               user_int_attrib_parserexec(xp, yp, zp, ux[ip], uy[ip], uz[ip], t));
                                           });
                } else {
                    for (int ip = start; ip < stop; ++ip) {
                        amrex::ParticleReal xp, yp, zp;
                        get_position(ip, xp, yp, zp);
                        attr_ptr[ip] = static_cast<int>(
                                user_int_attrib_parserexec(xp, yp, zp, ux[ip], uy[ip], uz[ip], t));
                    }
                }
                }
            }
        }
}

}

#endif //WARPX_DEFAULTINITIALIZATION_H_
