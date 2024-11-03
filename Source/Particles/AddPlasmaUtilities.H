/* Copyright 2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 * Authors: Andrew Myers
 */
#ifndef WARPX_ADDPLASMAUTILITIES_H_
#define WARPX_ADDPLASMAUTILITIES_H_

#include "Initialization/PlasmaInjector.H"

#ifdef WARPX_QED
#   include "Particles/ElementaryProcess/QEDInternals/BreitWheelerEngineWrapper.H"
#   include "Particles/ElementaryProcess/QEDInternals/QuantumSyncEngineWrapper.H"
#endif

#include <AMReX_Array.H>
#include <AMReX_Box.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_IntVect.H>
#include <AMReX_REAL.H>
#include <AMReX_RealBox.H>

struct PDim3 {
    amrex::ParticleReal x, y, z;

    AMREX_GPU_HOST_DEVICE
    PDim3(const amrex::XDim3& a):
        x{static_cast<amrex::ParticleReal>(a.x)},
        y{static_cast<amrex::ParticleReal>(a.y)},
        z{static_cast<amrex::ParticleReal>(a.z)}
    {}

    AMREX_GPU_HOST_DEVICE
    ~PDim3() = default;

    AMREX_GPU_HOST_DEVICE
    PDim3(PDim3 const &)            = default;
    AMREX_GPU_HOST_DEVICE
    PDim3& operator=(PDim3 const &) = default;
    AMREX_GPU_HOST_DEVICE
    PDim3(PDim3&&)                  = default;
    AMREX_GPU_HOST_DEVICE
    PDim3& operator=(PDim3&&)       = default;
};

/*
  Finds the overlap region between the given tile_realbox and part_realbox, returning true
  if an overlap exists and false if otherwise. This also sets the parameters overlap_realbox,
  overlap_box, and shifted to the appropriate values.
 */
bool find_overlap (const amrex::RealBox& tile_realbox, const amrex::RealBox& part_realbox,
                   const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dx,
                   const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_lo,
                   amrex::RealBox& overlap_realbox, amrex::Box& overlap_box, amrex::IntVect& shifted);

/*
  Finds the overlap region between the given tile_realbox, part_realbox and the surface used for
  flux injection, returning true if an overlap exists and false if otherwise. This also sets the
  parameters overlap_realbox, overlap_box, and shifted to the appropriate values.
 */
bool find_overlap_flux (const amrex::RealBox& tile_realbox, const amrex::RealBox& part_realbox,
                        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dx,
                        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_lo,
                        const PlasmaInjector& plasma_injector,
                        amrex::RealBox& overlap_realbox, amrex::Box& overlap_box, amrex::IntVect& shifted);

/*
  This computes the scale_fac (used for setting the particle weights) on a volumetric basis.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_scale_fac_volume (const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dx,
                                      const amrex::Long pcount) {
    using namespace amrex::literals;
    return (pcount != 0) ? AMREX_D_TERM(dx[0],*dx[1],*dx[2])/pcount : 0.0_rt;
}

/*
  Given a refinement ratio, this computes the total increase in resolution for a plane
  defined by the normal_axis.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int compute_area_weights (const amrex::IntVect& iv, const int normal_axis) {
    int r = AMREX_D_TERM(iv[0],*iv[1],*iv[2]);
#if defined(WARPX_DIM_3D)
    r /= iv[normal_axis];
#elif defined(WARPX_DIM_RZ) || defined(WARPX_DIM_XZ)
    if (normal_axis == 0) { r /= iv[0]; }
    else if (normal_axis == 2) { r /= iv[1]; }
#elif defined(WARPX_DIM_1D_Z)
    if (normal_axis == 2) { r /= iv[0]; }
#endif
    return r;
}


#ifdef AMREX_USE_EB
/*
 * \brief This computes the scale_fac (used for setting the particle weights) on a on area basis
 * (used for flux injection from the embedded boundary).
 *
 * \param[in] dx: cell size in each direction
 * \param[in] num_ppc_real: number of particles per cell
 * \param[in] eb_bnd_normal_arr: array containing the normal to the embedded boundary
 * \param[in] i, j, k: indices of the cell
 *
 * \return scale_fac: the scaling factor to be applied to the weight of the particles
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_scale_fac_area_eb (
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dx,
    const amrex::Real num_ppc_real,
    amrex::Array4<const amrex::Real> const& eb_bnd_normal_arr,
    int i, int j, int k ) {
    using namespace amrex::literals;
    // Scale particle weight by the area of the emitting surface, within one cell
    // By definition, eb_bnd_area_arr is normalized (unitless).
    // Here we undo the normalization (i.e. multiply by the surface used for normalization in amrex:
    // see https://amrex-codes.github.io/amrex/docs_html/EB.html#embedded-boundary-data-structures)
#if defined(WARPX_DIM_3D)
    const amrex::Real nx = eb_bnd_normal_arr(i,j,k,0);
    const amrex::Real ny = eb_bnd_normal_arr(i,j,k,1);
    const amrex::Real nz = eb_bnd_normal_arr(i,j,k,2);
    amrex::Real scale_fac = std::sqrt(amrex::Math::powi<2>(nx*dx[1]*dx[2]) +
                  amrex::Math::powi<2>(ny*dx[0]*dx[2]) +
                  amrex::Math::powi<2>(nz*dx[0]*dx[1]));

#elif defined(WARPX_DIM_RZ) || defined(WARPX_DIM_XZ)
    const amrex::Real nx = eb_bnd_normal_arr(i,j,k,0);
    const amrex::Real nz = eb_bnd_normal_arr(i,j,k,1);
    amrex::Real scale_fac = std::sqrt(amrex::Math::powi<2>(nx*dx[1]) +
                  amrex::Math::powi<2>(nz*dx[0]));
#else
    amrex::ignore_unused(dx, eb_bnd_normal_arr, i, j, k);
    amrex::Real scale_fac = 0.0_rt;
#endif
    // Do not multiply by eb_bnd_area_arr(i,j,k) here because this
    // already taken into account by emitting a number of macroparticles
    // that is proportional to the area of eb_bnd_area_arr(i,j,k).
    scale_fac /= num_ppc_real;
    return scale_fac;
}

/* \brief Rotate the momentum of the particle so that the z direction
 * transforms to the normal of the embedded boundary.
 *
 * More specifically, before calling this function, `pu.z` has the
 * momentum distribution that is meant for the direction normal to the
 * embedded boundary, and `pu.x`/`pu.y` have the momentum distribution that
 * is meant for the tangentional direction. After calling this function,
 * `pu.x`, `pu.y`, `pu.z` will have the correct momentum distribution,
 * consistent with the local normal to the embedded boundary.
 *
 * \param[inout] pu momentum of the particle
 * \param[in] eb_bnd_normal_arr: array containing the normal to the embedded boundary
 * \param[in] i, j, k: indices of the cell
 * */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void rotate_momentum_eb (
    PDim3 & pu,
    amrex::Array4<const amrex::Real> const& eb_bnd_normal_arr,
    int i, int j, int k )
{
    using namespace amrex::literals;

#if defined(WARPX_DIM_3D)
    // The minus sign below takes into account the fact that eb_bnd_normal_arr
    // points towards the covered region, while particles are to be emitted
    // *away* from the covered region.
    amrex::Real const nx = -eb_bnd_normal_arr(i,j,k,0);
    amrex::Real const ny = -eb_bnd_normal_arr(i,j,k,1);
    amrex::Real const nz = -eb_bnd_normal_arr(i,j,k,2);

    // Rotate the momentum in theta and phi
    amrex::Real const cos_theta = nz;
    amrex::Real const sin_theta = std::sqrt(1._rt-nz*nz);
    amrex::Real const nperp = std::sqrt(nx*nx + ny*ny);
    amrex::Real cos_phi = 1;
    amrex::Real sin_phi = 0;
    if ( nperp > 0.0 ) {
        cos_phi = nx/nperp;
        sin_phi = ny/nperp;
    }
    // Apply rotation matrix
    amrex::Real const ux = pu.x*cos_theta*cos_phi - pu.y*sin_phi + pu.z*sin_theta*cos_phi;
    amrex::Real const uy = pu.x*cos_theta*sin_phi + pu.y*cos_phi + pu.z*sin_theta*sin_phi;
    amrex::Real const uz = -pu.x*sin_theta + pu.z*cos_theta;
    pu.x = ux;
    pu.y = uy;
    pu.z = uz;

#elif defined(WARPX_DIM_RZ) || defined(WARPX_DIM_XZ)
    // The minus sign below takes into account the fact that eb_bnd_normal_arr
    // points towards the covered region, while particles are to be emitted
    // *away* from the covered region.
    amrex::Real const sin_theta = -eb_bnd_normal_arr(i,j,k,0);
    amrex::Real const cos_theta = -eb_bnd_normal_arr(i,j,k,1);
    amrex::Real const uz = pu.z*cos_theta - pu.x*sin_theta;
    amrex::Real const ux = pu.x*cos_theta + pu.z*sin_theta;
    pu.x = ux;
    pu.z = uz;
#else
    amrex::ignore_unused(pu, eb_bnd_normal_arr, i, j, k);
#endif
}
#endif //AMREX_USE_EB

/*
  This computes the scale_fac (used for setting the particle weights) on a on area basis
  (used for flux injection from a plane).
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_scale_fac_area_plane (const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dx,
                                    const amrex::Real num_ppc_real, const int flux_normal_axis) {
    using namespace amrex::literals;
    amrex::Real scale_fac = AMREX_D_TERM(dx[0],*dx[1],*dx[2])/num_ppc_real;
    // Scale particle weight by the area of the emitting surface, within one cell
#if defined(WARPX_DIM_3D)
    scale_fac /= dx[flux_normal_axis];
#elif defined(WARPX_DIM_RZ) || defined(WARPX_DIM_XZ)
    // When emission is in the r direction, the emitting surface is a cylinder.
    // The factor 2*pi*r is added later below.
    if (flux_normal_axis == 0) { scale_fac /= dx[0]; }
    // When emission is in the z direction, the emitting surface is an annulus
    // The factor 2*pi*r is added later below.
    if (flux_normal_axis == 2) { scale_fac /= dx[1]; }
    // When emission is in the theta direction (flux_normal_axis == 1),
    // the emitting surface is a rectangle, within the plane of the simulation
#elif defined(WARPX_DIM_1D_Z)
    if (flux_normal_axis == 2) { scale_fac /= dx[0]; }
#endif
    return scale_fac;
}

/*
  These structs encapsulates several data structures needed for using the parser during plasma
  injection.
 */
struct PlasmaParserWrapper
{
    PlasmaParserWrapper (std::size_t a_num_user_int_attribs,
                         std::size_t a_num_user_real_attribs,
                         const amrex::Vector< std::unique_ptr<amrex::Parser> >& a_user_int_attrib_parser,
                         const amrex::Vector< std::unique_ptr<amrex::Parser> >& a_user_real_attrib_parser);

    amrex::Gpu::PinnedVector< amrex::ParserExecutor<7> > m_user_int_attrib_parserexec_pinned;
    amrex::Gpu::PinnedVector< amrex::ParserExecutor<7> > m_user_real_attrib_parserexec_pinned;
};

struct PlasmaParserHelper
{
    template <typename SoAType>
    PlasmaParserHelper (SoAType& a_soa, std::size_t old_size,
                        const std::vector<std::string>& a_user_int_attribs,
                        const std::vector<std::string>& a_user_real_attribs,
                        std::map<std::string,int>& a_particle_icomps,
                        std::map<std::string,int>& a_particle_comps,
                        const PlasmaParserWrapper& wrapper) :
        m_wrapper_ptr(&wrapper) {
        m_pa_user_int_pinned.resize(a_user_int_attribs.size());
        m_pa_user_real_pinned.resize(a_user_real_attribs.size());

#ifdef AMREX_USE_GPU
        m_d_pa_user_int.resize(a_user_int_attribs.size());
        m_d_pa_user_real.resize(a_user_real_attribs.size());
        m_d_user_int_attrib_parserexec.resize(a_user_int_attribs.size());
        m_d_user_real_attrib_parserexec.resize(a_user_real_attribs.size());
#endif

        for (std::size_t ia = 0; ia < a_user_int_attribs.size(); ++ia) {
            m_pa_user_int_pinned[ia] = a_soa.GetIntData(a_particle_icomps[a_user_int_attribs[ia]]).data() + old_size;
        }
        for (std::size_t ia = 0; ia < a_user_real_attribs.size(); ++ia) {
            m_pa_user_real_pinned[ia] = a_soa.GetRealData(a_particle_comps[a_user_real_attribs[ia]]).data() + old_size;
        }

#ifdef AMREX_USE_GPU
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_pa_user_int_pinned.begin(),
                              m_pa_user_int_pinned.end(), m_d_pa_user_int.begin());
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_pa_user_real_pinned.begin(),
                              m_pa_user_real_pinned.end(), m_d_pa_user_real.begin());
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, wrapper.m_user_int_attrib_parserexec_pinned.begin(),
                              wrapper.m_user_int_attrib_parserexec_pinned.end(), m_d_user_int_attrib_parserexec.begin());
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, wrapper.m_user_real_attrib_parserexec_pinned.begin(),
                              wrapper.m_user_real_attrib_parserexec_pinned.end(), m_d_user_real_attrib_parserexec.begin());
#endif
    }

    int** getUserIntDataPtrs ();
    amrex::ParticleReal** getUserRealDataPtrs ();
    [[nodiscard]] amrex::ParserExecutor<7> const* getUserIntParserExecData () const;
    [[nodiscard]] amrex::ParserExecutor<7> const* getUserRealParserExecData () const;

    amrex::Gpu::PinnedVector<int*> m_pa_user_int_pinned;
    amrex::Gpu::PinnedVector<amrex::ParticleReal*> m_pa_user_real_pinned;

#ifdef AMREX_USE_GPU
    // To avoid using managed memory, we first define pinned memory vector, initialize on cpu,
    // and them memcpy to device from host
    amrex::Gpu::DeviceVector<int*> m_d_pa_user_int;
    amrex::Gpu::DeviceVector<amrex::ParticleReal*> m_d_pa_user_real;
    amrex::Gpu::DeviceVector< amrex::ParserExecutor<7> > m_d_user_int_attrib_parserexec;
    amrex::Gpu::DeviceVector< amrex::ParserExecutor<7> > m_d_user_real_attrib_parserexec;
#endif
    const PlasmaParserWrapper* m_wrapper_ptr;
};

#ifdef WARPX_QED
struct QEDHelper
{
    template <typename SoAType>
    QEDHelper (SoAType& a_soa, std::size_t old_size,
               std::map<std::string,int>& a_particle_comps,
               bool a_has_quantum_sync, bool a_has_breit_wheeler,
               const std::shared_ptr<QuantumSynchrotronEngine>& a_shr_p_qs_engine,
               const std::shared_ptr<BreitWheelerEngine>& a_shr_p_bw_engine)
        : has_quantum_sync(a_has_quantum_sync), has_breit_wheeler(a_has_breit_wheeler)
    {
        if(has_quantum_sync){
            quantum_sync_get_opt =
                a_shr_p_qs_engine->build_optical_depth_functor();
            p_optical_depth_QSR = a_soa.GetRealData(
                a_particle_comps["opticalDepthQSR"]).data() + old_size;
        }
        if(has_breit_wheeler){
            breit_wheeler_get_opt =
                a_shr_p_bw_engine->build_optical_depth_functor();
            p_optical_depth_BW = a_soa.GetRealData(
                a_particle_comps["opticalDepthBW"]).data() + old_size;
        }
    }

    amrex::ParticleReal* p_optical_depth_QSR = nullptr;
    amrex::ParticleReal* p_optical_depth_BW  = nullptr;

    const bool has_quantum_sync;
    const bool has_breit_wheeler;

    QuantumSynchrotronGetOpticalDepth quantum_sync_get_opt;
    BreitWheelerGetOpticalDepth breit_wheeler_get_opt;
};
#endif

#endif /*WARPX_ADDPLASMAUTILITIES_H_*/
