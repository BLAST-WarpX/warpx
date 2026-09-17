/* Copyright 2021 Andrew Myers, Eya Dammak
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "WarpX.H"
#include "EmbeddedBoundary/Enabled.H"
#include "EmbeddedBoundary/DistanceToEB.H"
#include "Particles/ParticleBoundaryBuffer.H"
#include "Particles/MultiParticleContainer.H"
#include "Utils/TextMsg.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/Pusher/UpdatePosition.H"

#include <ablastr/particles/NodalFieldGather.H>
#include <ablastr/profiler/ProfilerWrapper.H>

#include <AMReX_Geometry.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Reduce.H>
#include <AMReX_Tuple.H>
#include <AMReX.H>
#include <AMReX_Algorithm.H>
#include <AMReX_RealVect.H>

using namespace amrex::literals;


struct IsOutsideDomainBoundary {
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_plo;
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_phi;
    int m_idim;
    int m_iside;

    template <typename SrcData>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int operator() (const SrcData& src,
                    int ip, const amrex::RandomEngine& /*engine*/) const noexcept
    {
        const auto& p = src.getSuperParticle(ip);
        if (m_iside == 0) {
            if (p.pos(m_idim) < m_plo[m_idim]) { return 1; }
        } else {
            if (p.pos(m_idim) >= m_phi[m_idim]) { return 1; }
        }
        return 0;
    }
};

constexpr std::size_t MAX_FIELDS = 8;
template <std::size_t NFields = MAX_FIELDS>
struct FindBoundaryIntersection {
    int m_step_index;
    int m_delta_index;
    int m_time_index;
    int m_normal_index;
    int m_step;
    amrex::Real m_cur_time;
    amrex::Real m_dt;
    amrex::GpuArray<amrex::Array4<const amrex::Real>, NFields> m_phiarrs{};
    int m_num_fields{1};
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_dxi;
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_plo;
    amrex::ParticleReal m_mass;

    // --- Constructor 1: Multi-field (EB + Sinks) ---
    AMREX_GPU_HOST_DEVICE
    FindBoundaryIntersection (
        int step_index, int delta_index, int time_index, int normal_index,
        int step, amrex::Real cur_time, amrex::Real dt,
        amrex::GpuArray<amrex::Array4<const amrex::Real>, NFields> const& phiarrs,
        int num_fields,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dxi,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& plo,
        amrex::ParticleReal mass)
      : m_step_index(step_index), m_delta_index(delta_index),
        m_time_index(time_index), m_normal_index(normal_index),
        m_step(step), m_cur_time(cur_time), m_dt(dt),
        m_phiarrs(phiarrs), m_num_fields(num_fields),
        m_dxi(dxi), m_plo(plo), m_mass(mass) {}

    // --- Constructor 2: Single-field overload (EB legacy) ---
    AMREX_GPU_HOST_DEVICE
    FindBoundaryIntersection (
        int step_index, int delta_index, int time_index, int normal_index,
        int step, amrex::Real cur_time, amrex::Real dt,
        amrex::Array4<const amrex::Real> const& phiarr,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dxi,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& plo,
        amrex::ParticleReal mass)
      : m_step_index(step_index), m_delta_index(delta_index),
        m_time_index(time_index), m_normal_index(normal_index),
        m_step(step), m_cur_time(cur_time), m_dt(dt),
        m_dxi(dxi), m_plo(plo), m_mass(mass)
    {
        m_phiarrs[0] = phiarr;
    }

    template <typename DstData, typename SrcData>
    AMREX_GPU_HOST_DEVICE
    void operator() (const DstData& dst, const SrcData& src,
                     int src_i, int dst_i) const noexcept
    {
        // Copy all particle attributes from source to destination
        dst.m_idcpu[dst_i] = src.m_idcpu[src_i];
        for (int j = 0; j < SrcData::NAR; ++j) {
            dst.m_rdata[j][dst_i] = src.m_rdata[j][src_i];
        }
        for (int j = 0; j < src.m_num_runtime_real; ++j) {
            dst.m_runtime_rdata[j][dst_i] = src.m_runtime_rdata[j][src_i];
        }
        for (int j = 0; j < src.m_num_runtime_int; ++j) {
            dst.m_runtime_idata[j][dst_i] = src.m_runtime_idata[j][src_i];
        }

        const auto& p = dst.getSuperParticle(dst_i);
        amrex::ParticleReal xp, yp, zp;
        get_particle_position( p, xp, yp, zp );
        amrex::ParticleReal const ux = dst.m_rdata[PIdx::ux][dst_i];
        amrex::ParticleReal const uy = dst.m_rdata[PIdx::uy][dst_i];
        amrex::ParticleReal const uz = dst.m_rdata[PIdx::uz][dst_i];

        // Temporary variables to avoid implicit capture
        amrex::Real const dt = m_dt;
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dxi = m_dxi;
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const plo = m_plo;
        amrex::ParticleReal const mass = m_mass;

        // Determine which distance field (EB or Sink) was entered
        int selected_field = 0;
        amrex::Real min_phi = ablastr::particles::doGatherScalarFieldNodal(
            xp, yp, zp, m_phiarrs[0], dxi, plo);

        for (int f = 1; f < m_num_fields; ++f) {
            amrex::Real const phi_val = ablastr::particles::doGatherScalarFieldNodal(
                xp, yp, zp, m_phiarrs[f], dxi, plo);
            if (phi_val < min_phi) {
                min_phi = phi_val;
                selected_field = f;
            }
        }

        // Use the selected boundary field array for ray-tracing and normal calculations
        amrex::Array4<amrex::Real const> const phiarr = m_phiarrs[selected_field];

        // Bisection algorithm to find point where phi(x,y,z)=0 for the crossed boundary
        amrex::Real const dt_fraction = amrex::bisect( 0.0, 1.0,
            [=] (amrex::Real dt_frac) {
                int i, j, k;
                amrex::Real W[AMREX_SPACEDIM][2];
                amrex::ParticleReal x_temp=xp, y_temp=yp, z_temp=zp;
                UpdatePosition(x_temp, y_temp, z_temp, ux, uy, uz, -dt_frac*dt, mass);
                ablastr::particles::compute_weights<amrex::IndexType::NODE>(
                    x_temp, y_temp, z_temp, plo, dxi, i, j, k, W);
                amrex::Real const phi_value = ablastr::particles::interp_field_nodal(i, j, k, W, phiarr);
                return phi_value;
            } );

        // Record timing data on destination particle
        dst.m_runtime_idata[m_step_index][dst_i] = m_step;
        dst.m_runtime_rdata[m_delta_index][dst_i] = (1._rt - dt_fraction)*m_dt;
        dst.m_runtime_rdata[m_time_index][dst_i] = m_cur_time + (1._rt - dt_fraction)*m_dt;

        // Save position of the particle at the exact surface boundary
        amrex::ParticleReal x_temp=xp, y_temp=yp, z_temp=zp;
        UpdatePosition(x_temp, y_temp, z_temp, ux, uy, uz, -dt_fraction*m_dt, m_mass);

        // Compute surface normal using the selected field
        auto const n3d = DistanceToEB::interp_normal(x_temp, y_temp, z_temp, plo, dxi, phiarr);

        // Record geometry/position on destination particle
#if (defined WARPX_DIM_3D)
        dst.m_rdata[PIdx::x][dst_i] = x_temp;
        dst.m_rdata[PIdx::y][dst_i] = y_temp;
        dst.m_rdata[PIdx::z][dst_i] = z_temp;
#elif (defined WARPX_DIM_XZ)
        dst.m_rdata[PIdx::x][dst_i] = x_temp;
        dst.m_rdata[PIdx::z][dst_i] = z_temp;
#elif (defined WARPX_DIM_RZ)
        dst.m_rdata[PIdx::r][dst_i] = std::sqrt(x_temp*x_temp + y_temp*y_temp);
        dst.m_rdata[PIdx::z][dst_i] = z_temp;
        dst.m_rdata[PIdx::theta][dst_i] = std::atan2(y_temp, x_temp);
#elif (defined WARPX_DIM_1D_Z)
        dst.m_rdata[PIdx::z][dst_i] = z_temp;
#elif (defined WARPX_DIM_RCYLINDER)
        dst.m_rdata[PIdx::r][dst_i] = std::sqrt(x_temp*x_temp + y_temp*y_temp);
#elif (defined WARPX_DIM_RSPHERE)
        dst.m_rdata[PIdx::r][dst_i] = std::sqrt(x_temp*x_temp + y_temp*y_temp + z_temp*z_temp);
#endif

        // Record normal vector components
        dst.m_runtime_rdata[m_normal_index][dst_i]   = n3d[0];
        dst.m_runtime_rdata[m_normal_index+1][dst_i] = n3d[1];
        dst.m_runtime_rdata[m_normal_index+2][dst_i] = n3d[2];

        // Mark particle as valid in destination buffer
        amrex::ParticleIDWrapper{dst.m_idcpu[dst_i]}.make_valid();
    }
};


struct CopyAndTimestamp {
    int m_step_index;
    int m_delta_index;
    int m_time_index;
    int m_normal_index;
    int m_step;
    amrex::Real m_cur_time;
    amrex::Real m_dt;
    int m_idim;
    int m_iside;


    template <typename DstData, typename SrcData>
    AMREX_GPU_HOST_DEVICE
    void operator() (const DstData& dst, const SrcData& src,
                     int src_i, int dst_i) const noexcept
    {
        dst.m_idcpu[dst_i] = src.m_idcpu[src_i];
        for (int j = 0; j < SrcData::NAR; ++j) {
            dst.m_rdata[j][dst_i] = src.m_rdata[j][src_i];
        }
        for (int j = 0; j < src.m_num_runtime_real; ++j) {
            dst.m_runtime_rdata[j][dst_i] = src.m_runtime_rdata[j][src_i];
        }
        for (int j = 0; j < src.m_num_runtime_int; ++j) {
            dst.m_runtime_idata[j][dst_i] = src.m_runtime_idata[j][src_i];
        }

        dst.m_runtime_idata[m_step_index][dst_i] = m_step;
        dst.m_runtime_rdata[m_delta_index][dst_i] = 0._rt; //delta_fraction is initialized to zero
        dst.m_runtime_rdata[m_time_index][dst_i] = m_cur_time;

        //calculation of the normal to the boundary
        std::array<double, 3> n = {0.0, 0.0, 0.0};
        n[m_idim]=1-2*m_iside;
        dst.m_runtime_rdata[m_normal_index][dst_i]= n[0];
        dst.m_runtime_rdata[m_normal_index+1][dst_i]= n[1];
        dst.m_runtime_rdata[m_normal_index+2][dst_i]= n[2];

        // flip id to positive in destination
        amrex::ParticleIDWrapper{dst.m_idcpu[dst_i]}.make_valid();
    }
};


ParticleBoundaryBuffer::ParticleBoundaryBuffer ()
{
    m_particle_containers.resize(numBoundaries());
    m_do_boundary_buffer.resize(numBoundaries());
    m_do_any_boundary.resize(numBoundaries(), 0);
    m_boundary_names.resize(numBoundaries());

    for (int i = 0; i < numBoundaries(); ++i)
    {
        m_particle_containers[i].resize(numSpecies());
        m_do_boundary_buffer[i].resize(numSpecies(), 0);
    }

#if defined(WARPX_DIM_1D_Z)
    constexpr auto idx_zlo = 0;
    constexpr auto idx_zhi = 1;
#elif (defined WARPX_DIM_RCYLINDER) || (defined WARPX_DIM_RSPHERE)
    constexpr auto idx_xlo = 0;
    constexpr auto idx_xhi = 1;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    constexpr auto idx_xlo = 0;
    constexpr auto idx_xhi = 1;
    constexpr auto idx_zlo = 2;
    constexpr auto idx_zhi = 3;
#else
    constexpr auto idx_xlo = 0;
    constexpr auto idx_xhi = 1;
    constexpr auto idx_ylo = 2;
    constexpr auto idx_yhi = 3;
    constexpr auto idx_zlo = 4;
    constexpr auto idx_zhi = 5;
#endif

    bool const eb_enabled = EB::enabled();
    bool const particle_sink_enabled = ParticleSink::enabled();

    for (int ispecies = 0; ispecies < numSpecies(); ++ispecies)
    {
        const amrex::ParmParse pp_species(getSpeciesNames()[ispecies]);
#if defined(WARPX_DIM_1D_Z)
        pp_species.query("save_particles_at_zlo", m_do_boundary_buffer[idx_zlo][ispecies]);
        pp_species.query("save_particles_at_zhi", m_do_boundary_buffer[idx_zhi][ispecies]);
#elif (defined WARPX_DIM_RCYLINDER) || (defined WARPX_DIM_RSPHERE)
        pp_species.query("save_particles_at_xlo", m_do_boundary_buffer[idx_xlo][ispecies]);
        pp_species.query("save_particles_at_xhi", m_do_boundary_buffer[idx_xhi][ispecies]);
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
        pp_species.query("save_particles_at_xlo", m_do_boundary_buffer[idx_xlo][ispecies]);
        pp_species.query("save_particles_at_xhi", m_do_boundary_buffer[idx_xhi][ispecies]);
        pp_species.query("save_particles_at_zlo", m_do_boundary_buffer[idx_zlo][ispecies]);
        pp_species.query("save_particles_at_zhi", m_do_boundary_buffer[idx_zhi][ispecies]);
#else
        pp_species.query("save_particles_at_xlo", m_do_boundary_buffer[idx_xlo][ispecies]);
        pp_species.query("save_particles_at_xhi", m_do_boundary_buffer[idx_xhi][ispecies]);
        pp_species.query("save_particles_at_ylo", m_do_boundary_buffer[idx_ylo][ispecies]);
        pp_species.query("save_particles_at_yhi", m_do_boundary_buffer[idx_yhi][ispecies]);
        pp_species.query("save_particles_at_zlo", m_do_boundary_buffer[idx_zlo][ispecies]);
        pp_species.query("save_particles_at_zhi", m_do_boundary_buffer[idx_zhi][ispecies]);
#endif

        if (eb_enabled || particle_sink_enabled) { pp_species.query("save_particles_at_eb", m_do_boundary_buffer[AMREX_SPACEDIM*2][ispecies]); }

        // Set the flag whether the boundary is active or any species
        for (int i = 0; i < numBoundaries(); ++i) {
            if (m_do_boundary_buffer[i][ispecies]) { m_do_any_boundary[i] = 1; }
        }
    }

#if defined(WARPX_DIM_1D_Z)
    m_boundary_names[idx_zlo] = "zlo";
    m_boundary_names[idx_zhi] = "zhi";
#elif (defined WARPX_DIM_RCYLINDER) || (defined WARPX_DIM_RSPHERE)
    m_boundary_names[idx_xlo] = "xlo";
    m_boundary_names[idx_xhi] = "xhi";
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    m_boundary_names[idx_xlo] = "xlo";
    m_boundary_names[idx_xhi] = "xhi";
    m_boundary_names[idx_zlo] = "zlo";
    m_boundary_names[idx_zhi] = "zhi";
#else
    m_boundary_names[idx_xlo] = "xlo";
    m_boundary_names[idx_xhi] = "xhi";
    m_boundary_names[idx_ylo] = "ylo";
    m_boundary_names[idx_yhi] = "yhi";
    m_boundary_names[idx_zlo] = "zlo";
    m_boundary_names[idx_zhi] = "zhi";
#endif
    if (eb_enabled) { m_boundary_names[AMREX_SPACEDIM*2] = "eb"; }
}

void ParticleBoundaryBuffer::printNumParticles () const {
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        for (int iside = 0; iside < 2; ++iside)
        {
            const auto& buffer = m_particle_containers[2*idim+iside];
            for (int i = 0; i < numSpecies(); ++i)
            {
                const auto np = buffer[i].isDefined() ? buffer[i].TotalNumberOfParticles(false) : 0;
                amrex::Print() << Utils::TextMsg::Info(
                    "Species " + getSpeciesNames()[i] + " has "
                    + std::to_string(np) + " particles in the boundary buffer "
                    + "for side " + std::to_string(iside) + " of dim " + std::to_string(idim)
                );
            }
        }
    }

    if (EB::enabled()) {
        auto const & buffer = m_particle_containers[2 * AMREX_SPACEDIM];
        for (int i = 0; i < numSpecies(); ++i) {
            const auto np = buffer[i].isDefined() ? buffer[i].TotalNumberOfParticles(false) : 0;
            amrex::Print() << Utils::TextMsg::Info(
                    "Species " + getSpeciesNames()[i] + " has "
                    + std::to_string(np) + " particles in the EB boundary buffer"
            );
        }
    }
}

void ParticleBoundaryBuffer::redistribute () {
    for (int i = 0; i < numBoundaries(); ++i)
    {
        auto& buffer = m_particle_containers[i];
        for (int ispecies = 0; ispecies < numSpecies(); ++ispecies)
        {
            auto& species_buffer = buffer[ispecies];
            if (species_buffer.isDefined()) {
                // do not remove particles with negative ids
                species_buffer.Redistribute(0, -1, 0, 0, false);
            }
        }
    }
}

const std::vector<std::string>& ParticleBoundaryBuffer::getSpeciesNames() const
{
    if (!m_species_names_initialized)
    {
        const amrex::ParmParse pp_particles("particles");
        pp_particles.queryarr("species_names", m_species_names);
        m_species_names_initialized = true;
    }
    return m_species_names;
}

void ParticleBoundaryBuffer::clearParticles () {
    for (int i = 0; i < numBoundaries(); ++i)
    {
        clearParticles(i);
    }
}

void ParticleBoundaryBuffer::clearParticles (int const i) {
    auto& buffer = m_particle_containers[i];
    for (int ispecies = 0; ispecies < numSpecies(); ++ispecies)
    {
        auto& species_buffer = buffer[ispecies];
        if (species_buffer.isDefined()) { species_buffer.clearParticles(); }
    }
}

void ParticleBoundaryBuffer::gatherParticlesFromDomainBoundaries (MultiParticleContainer& mypc, amrex::Real cur_time)
{
    ABLASTR_PROFILE("ParticleBoundaryBuffer::gatherParticles");

    using PIter = amrex::ParConstIterSoA<PIdx::nattribs, 0, amrex::PolymorphicArenaAllocator>;
    const auto& warpx_instance = WarpX::GetInstance();
    const amrex::Geometry& geom = warpx_instance.Geom(0);
    auto plo = geom.ProbLoArray();
    auto phi = geom.ProbHiArray();
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        if (geom.isPeriodic(idim)) { continue; }
        for (int iside = 0; iside < 2; ++iside)
        {
            auto& buffer = m_particle_containers[2*idim+iside];
            for (int i = 0; i < numSpecies(); ++i)
            {
                if (!m_do_boundary_buffer[2*idim+iside][i]) { continue; }
                const WarpXParticleContainer& pc = mypc.GetParticleContainer(i);
                if (!buffer[i].isDefined())
                {
                    buffer[i] = pc.make_alike<>();
                    buffer[i].SetArena(amrex::The_Pinned_Arena());
                    buffer[i].AddIntComp("stepScraped", true);
                    buffer[i].AddRealComp("deltaTimeScraped", true);
                    buffer[i].AddRealComp("timeScraped", true);
                    buffer[i].AddRealComp("nx", true);
                    buffer[i].AddRealComp("ny", true);
                    buffer[i].AddRealComp("nz", true);
                }

                auto& species_buffer = buffer[i];
                for (int lev = 0; lev < pc.numLevels(); ++lev){
                    for(PIter pti(pc, lev); pti.isValid(); ++pti){
                        species_buffer.DefineAndReturnParticleTile(
                            lev, pti.index(), pti.LocalTileIndex());
                    }
                }

                for (int lev = 0; lev < pc.numLevels(); ++lev)
                {
                    for (PIter pti(pc, lev); pti.isValid(); ++pti) {
                        species_buffer.DefineAndReturnParticleTile(
                            lev, pti.index(), pti.LocalTileIndex());
                    }

                    const auto& plevel = pc.GetParticles(lev);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
                    for(PIter pti(pc, lev); pti.isValid(); ++pti)
                    {
                        auto index = std::make_pair(pti.index(), pti.LocalTileIndex());

                        auto& ptile_buffer =
                            species_buffer.ParticlesAt(lev, pti.index(), pti.LocalTileIndex());

                        const auto& ptile = plevel.at(index);
                        auto np = ptile.numParticles();
                        if (np == 0) { continue; }

                        auto predicate = IsOutsideDomainBoundary{plo, phi, idim, iside};

                        const auto ptile_data = ptile.getConstParticleTileData();

                        amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
                        amrex::ReduceData<int> reduce_data(reduce_op);
                        {
                          ABLASTR_PROFILE("ParticleBoundaryBuffer::gatherParticles::count_out_of_bounds");
#ifdef AMREX_USE_GPU
                          const amrex::RandomEngine rng{nullptr};
#else
                          const amrex::RandomEngine rng{};
#endif
                          reduce_op.eval(np, reduce_data, [=] AMREX_GPU_HOST_DEVICE (int ip)
                                         { return predicate(ptile_data, ip, rng) ? 1 : 0; });
                        }

                        auto dst_index = ptile_buffer.numParticles();
                        {
                          ABLASTR_PROFILE("ParticleBoundaryBuffer::gatherParticles::resize");
                          auto np_to_add = amrex::get<0>(reduce_data.value());
                          auto new_np = dst_index + np_to_add;
                          const amrex::Long capacity = ptile_buffer.capacity() / species_buffer.superParticleSize();
                          // reserve space to avoid many small resize operations for performance reasons
                          // the resize below will not shrink the capacity
                          if (new_np > capacity) { ptile_buffer.reserve(2*new_np); }
                          ptile_buffer.resize(new_np);
                        }
                        {
                          ABLASTR_PROFILE("ParticleBoundaryBuffer::gatherParticles::filterAndTransform");
                          auto& warpx = WarpX::GetInstance();
                          const auto dt = warpx.getdt(pti.GetLevel());
                          auto & buf = buffer[i];
                          const int step_scraped_index = buf.GetIntCompIndex("stepScraped") - WarpXParticleContainer::NArrayInt;
                          const int delta_index = buf.GetRealCompIndex("deltaTimeScraped") - WarpXParticleContainer::NArrayReal;
                          const int time_scraped_index = buf.GetRealCompIndex("timeScraped") - WarpXParticleContainer::NArrayReal;
                          const int normal_index = buf.GetRealCompIndex("nx") - WarpXParticleContainer::NArrayReal;
                          const int step = warpx_instance.getistep(0);
                          amrex::filterAndTransformParticles(ptile_buffer, ptile,
                                                             predicate,
                                                             CopyAndTimestamp{step_scraped_index, delta_index, time_scraped_index, normal_index,
                                                                              step, cur_time, dt, idim, iside},
                                                             0, dst_index);
                        }
                    }
                }
            }
        }
    }
}


void ParticleBoundaryBuffer::gatherParticlesFromEmbeddedBoundaries (
    MultiParticleContainer& mypc,
    ablastr::fields::MultiLevelScalarField const& distance_to_eb,
    amrex::Real const cur_time)
{
    std::vector<ablastr::fields::MultiLevelScalarField const*> const fields = { &distance_to_eb };
    gatherParticlesFromDistanceFields(mypc, fields, cur_time);
}

void ParticleBoundaryBuffer::gatherParticlesFromDistanceFields (
    MultiParticleContainer& mypc,
    std::vector<ablastr::fields::MultiLevelScalarField const*> const& distance_fields,
    amrex::Real cur_time)
{
    if (distance_fields.empty()) { return; }

    AMREX_ALWAYS_ASSERT(distance_fields.size() <= MAX_FIELDS);
    int const num_fields = static_cast<int>(distance_fields.size());

    using PIter = amrex::ParConstIterSoA<PIdx::nattribs, 0, amrex::PolymorphicArenaAllocator>;
    const auto &warpx_instance = WarpX::GetInstance();
    const amrex::Geometry &geom = warpx_instance.Geom(0);
    auto plo = geom.ProbLoArray();

    auto& buffer = m_particle_containers.back();

    for (int i = 0; i < numSpecies(); ++i) {
        if (!m_do_boundary_buffer[AMREX_SPACEDIM*2][i]) { continue; }
        const auto& pc = mypc.GetParticleContainer(i);

        if (!buffer[i].isDefined()) {
            buffer[i] = pc.make_alike<>();
            buffer[i].SetArena(amrex::The_Pinned_Arena());
            buffer[i].AddIntComp("stepScraped", true);
            buffer[i].AddRealComp("deltaTimeScraped", true);
            buffer[i].AddRealComp("timeScraped", true);
            buffer[i].AddRealComp("nx", true);
            buffer[i].AddRealComp("ny", true);
            buffer[i].AddRealComp("nz", true);
        }

        auto& species_buffer = buffer[i];
        for (int lev = 0; lev < pc.numLevels(); ++lev) {
            for (PIter pti(pc, lev); pti.isValid(); ++pti) {
                species_buffer.DefineAndReturnParticleTile(lev, pti.index(), pti.LocalTileIndex());
            }
        }

        for (int lev = 0; lev < pc.numLevels(); ++lev) {
            const auto& plevel = pc.GetParticles(lev);
            auto dxi = warpx_instance.Geom(lev).InvCellSizeArray();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (PIter pti(pc, lev); pti.isValid(); ++pti) {
                auto index = std::make_pair(pti.index(), pti.LocalTileIndex());
                if (!plevel.contains(index)) { continue; }

                // Pack Array4 tile views for EB and all sinks
                amrex::GpuArray<amrex::Array4<amrex::Real const>, MAX_FIELDS> phi_arrs;
                for (int f = 0; f < num_fields; ++f) {
                    phi_arrs[f] = (*(*distance_fields[f])[lev])[pti].array();
                }

                const auto getPosition = GetParticlePosition<PIdx>(pti);
                auto &ptile_buffer = species_buffer.DefineAndReturnParticleTile(
                    lev, pti.index(), pti.LocalTileIndex());
                const auto &ptile = plevel.at(index);
                auto np = ptile.numParticles();
                if (np == 0) { continue; }

                using SrcData = WarpXParticleContainer::ParticleTileType::ConstParticleTileDataType;

                // Evaluates all fields in a single GPU thread
                auto predicate = [=] AMREX_GPU_HOST_DEVICE(const SrcData &, const int ip) {
                    amrex::ParticleReal xp, yp, zp;
                    getPosition(ip, xp, yp, zp);

                    for (int f = 0; f < num_fields; ++f) {
                        amrex::Real const phi_val = ablastr::particles::doGatherScalarFieldNodal(
                            xp, yp, zp, phi_arrs[f], dxi, plo);
                        if (phi_val < 0.0) { return 1; } // Inside EB or Sink
                    }
                    return 0;
                };

                const auto ptile_data = ptile.getConstParticleTileData();
                amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
                amrex::ReduceData<int> reduce_data(reduce_op);
                reduce_op.eval(np, reduce_data, [=] AMREX_GPU_HOST_DEVICE(int ip) {
                    return predicate(ptile_data, ip) ? 1 : 0;
                });

                auto dst_index = ptile_buffer.numParticles();
                auto np_to_add = amrex::get<0>(reduce_data.value());
                auto new_np = dst_index + np_to_add;
                const amrex::Long capacity = ptile_buffer.capacity() / species_buffer.superParticleSize();
                if (new_np > capacity) { ptile_buffer.reserve(2 * new_np); }
                ptile_buffer.resize(new_np);

                const auto dt = warpx_instance.getdt(pti.GetLevel());
                auto &buf = buffer[i];
                const int step_scraped_index = buf.GetIntCompIndex("stepScraped") - WarpXParticleContainer::NArrayInt;
                const int delta_index = buf.GetRealCompIndex("deltaTimeScraped") - WarpXParticleContainer::NArrayReal;
                const int time_scraped_index = buf.GetRealCompIndex("timeScraped") - WarpXParticleContainer::NArrayReal;
                const int normal_index = buf.GetRealCompIndex("nx") - WarpXParticleContainer::NArrayReal;
                const int step = warpx_instance.getistep(0);

                amrex::filterAndTransformParticles(ptile_buffer, ptile, predicate,
                                                   FindBoundaryIntersection<MAX_FIELDS>{
                                                                                 step_scraped_index, delta_index,
                                                                                 time_scraped_index, normal_index,
                                                                                 step, cur_time, dt,
                                                                                 phi_arrs, num_fields, dxi,
                                                                                 plo, pc.getMass()},
                                                    0, dst_index
                );
            }
        }
    }
}


int ParticleBoundaryBuffer::getNumParticlesInContainer(
        const std::string& species_name, int boundary, bool local) {

    auto& buffer = m_particle_containers[boundary];
    auto index = WarpX::GetInstance().GetPartContainer().getSpeciesID(species_name);

    if (buffer[index].isDefined()){
        return static_cast<int>(buffer[index].TotalNumberOfParticles(false, local));
    }
    else{
        return 0;
    }
}

WarpXParticleContainer::Base &
ParticleBoundaryBuffer::getParticleBuffer(const std::string& species_name, int boundary) {

    auto& buffer = m_particle_containers[boundary];
    auto index = WarpX::GetInstance().GetPartContainer().getSpeciesID(species_name);

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_do_boundary_buffer[boundary][index],
                                     "Attempted to get particle buffer for boundary "
                                     + std::to_string(boundary) + ", which is not used!");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(buffer[index].isDefined(),
                                     "Tried to get a buffer that is not defined!");

    return buffer[index];
}

WarpXParticleContainer::Base *
ParticleBoundaryBuffer::getParticleBufferPointer(const std::string& species_name, int boundary) {

    auto& buffer = m_particle_containers[boundary];
    auto index = WarpX::GetInstance().GetPartContainer().getSpeciesID(species_name);

    return &buffer[index];
}
