/* Copyright 2021 Andrew Myers
 * modified by Remi Lehe, Eya Dammak 2023
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
#include "Utils/WarpXProfilerWrapper.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/Pusher/UpdatePosition.H"

#include <ablastr/particles/NodalFieldGather.H>

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

struct FindEmbeddedBoundaryIntersection {
    const int m_step_index;
    const int m_delta_index;
    const int m_normal_index;
    const int m_step;
    const amrex::Real m_dt;
    amrex::Array4<const amrex::Real> m_phiarr;
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_dxi;
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_plo;

    template <typename DstData, typename SrcData>
    AMREX_GPU_HOST_DEVICE
    void operator() (const DstData& dst, const SrcData& src,
                     int src_i, int dst_i) const noexcept
    {
        // Copy all particle attributes, from the source to the destination
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

        // Modify the position of the destination particle:
        // Move it to the point of intersection with the embedded boundary
        // (which is found by using a bisection algorithm)

        const auto& p = dst.getSuperParticle(dst_i);
        amrex::ParticleReal xp, yp, zp;
        get_particle_position( p, xp, yp, zp );
        amrex::ParticleReal const ux = dst.m_rdata[PIdx::ux][dst_i];
        amrex::ParticleReal const uy = dst.m_rdata[PIdx::uy][dst_i];
        amrex::ParticleReal const uz = dst.m_rdata[PIdx::uz][dst_i];

        // Temporary variables to avoid implicit capture
        amrex::Real const dt = m_dt;
        amrex::Array4<amrex::Real const> const phiarr = m_phiarr;
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dxi = m_dxi;
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const plo = m_plo;

        // Bisection algorithm to find the point where phi(x,y,z)=0 (i.e. on the embedded boundary)
        amrex::Real const dt_fraction = amrex::bisect( 0.0, 1.0,
            [=] (amrex::Real dt_frac) {
                int i, j, k;
                amrex::Real W[AMREX_SPACEDIM][2];
                amrex::ParticleReal x_temp=xp, y_temp=yp, z_temp=zp;
                UpdatePosition(x_temp, y_temp, z_temp, ux, uy, uz, -dt_frac*dt);
                ablastr::particles::compute_weights<amrex::IndexType::NODE>(
                    x_temp, y_temp, z_temp, plo, dxi, i, j, k, W);
                amrex::Real const phi_value = ablastr::particles::interp_field_nodal(i, j, k, W, phiarr);
                return phi_value;
            } );

        // Also record the real time on the destination
        dst.m_runtime_idata[m_step_index][dst_i] = m_step;
        dst.m_runtime_rdata[m_delta_index][dst_i] = (1._rt- dt_fraction)*m_dt;

        // Now that dt_fraction has be obtained (with bisect)
        // Save the corresponding position of the particle at the boundary
        amrex::ParticleReal x_temp=xp, y_temp=yp, z_temp=zp;
        UpdatePosition(x_temp, y_temp, z_temp, ux, uy, uz, -dt_fraction*m_dt);

        // record the components of the normal on the destination
        int i, j, k;
        amrex::Real W[AMREX_SPACEDIM][2];
        ablastr::particles::compute_weights<amrex::IndexType::NODE>(
            x_temp, y_temp, z_temp, plo, dxi, i, j, k, W);
        int ic, jc, kc; // Cell-centered indices
        amrex::Real Wc[AMREX_SPACEDIM][2]; // Cell-centered weight
        ablastr::particles::compute_weights<amrex::IndexType::CELL>(
            x_temp, y_temp, z_temp, plo, dxi, ic, jc, kc, Wc);
        amrex::RealVect normal = DistanceToEB::interp_normal(i, j, k, W, ic, jc, kc, Wc, phiarr, dxi);
        DistanceToEB::normalize(normal);

#if (defined WARPX_DIM_3D)
        dst.m_rdata[PIdx::x][dst_i] = x_temp;
        dst.m_rdata[PIdx::y][dst_i] = y_temp;
        dst.m_rdata[PIdx::z][dst_i] = z_temp;
        //save normal components
        dst.m_runtime_rdata[m_normal_index][dst_i] = normal[0];
        dst.m_runtime_rdata[m_normal_index+1][dst_i] = normal[1];
        dst.m_runtime_rdata[m_normal_index+2][dst_i] = normal[2];
#elif (defined WARPX_DIM_XZ)
        dst.m_rdata[PIdx::x][dst_i] = x_temp;
        dst.m_rdata[PIdx::z][dst_i] = z_temp;
        amrex::ignore_unused(y_temp);
        //save normal components
        dst.m_runtime_rdata[m_normal_index][dst_i] = normal[0];
        dst.m_runtime_rdata[m_normal_index+1][dst_i] = 0.0;
        dst.m_runtime_rdata[m_normal_index+2][dst_i] = normal[1];
#elif (defined WARPX_DIM_RZ)
        dst.m_rdata[PIdx::x][dst_i] = std::sqrt(x_temp*x_temp + y_temp*y_temp);
        dst.m_rdata[PIdx::z][dst_i] = z_temp;
        dst.m_rdata[PIdx::theta][dst_i] = std::atan2(y_temp, x_temp);
        //save normal components
        amrex::Real const theta = std::atan2(y_temp, x_temp);
        dst.m_runtime_rdata[m_normal_index][dst_i] = normal[0]*std::cos(theta);
        dst.m_runtime_rdata[m_normal_index+1][dst_i] = normal[0]*std::sin(theta);
        dst.m_runtime_rdata[m_normal_index+2][dst_i] = normal[1];
#elif (defined WARPX_DIM_1D_Z)
        dst.m_rdata[PIdx::z][dst_i] = z_temp;
        amrex::ignore_unused(x_temp, y_temp);
        //normal not defined
        dst.m_runtime_rdata[m_normal_index][dst_i] = 0.0;
        dst.m_runtime_rdata[m_normal_index+1][dst_i] = 0.0;
        dst.m_runtime_rdata[m_normal_index+2][dst_i] = 0.0;
#else
        amrex::ignore_unused(x_temp, y_temp, z_temp,normal);
#endif

        // flip id to positive in destination
        amrex::ParticleIDWrapper{dst.m_idcpu[dst_i]}.make_valid();
    }
};

struct CopyAndTimestamp {
    int m_step_index;
    int m_delta_index;
    int m_normal_index;
    int m_step;
    const amrex::Real m_dt;
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

    for (int ispecies = 0; ispecies < numSpecies(); ++ispecies)
    {
        const amrex::ParmParse pp_species(getSpeciesNames()[ispecies]);
#if defined(WARPX_DIM_1D_Z)
        pp_species.query("save_particles_at_zlo", m_do_boundary_buffer[idx_zlo][ispecies]);
        pp_species.query("save_particles_at_zhi", m_do_boundary_buffer[idx_zhi][ispecies]);
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

        if (eb_enabled) { pp_species.query("save_particles_at_eb", m_do_boundary_buffer[AMREX_SPACEDIM*2][ispecies]); }

        // Set the flag whether the boundary is active or any species
        for (int i = 0; i < numBoundaries(); ++i) {
            if (m_do_boundary_buffer[i][ispecies]) { m_do_any_boundary[i] = 1; }
        }
    }

#if defined(WARPX_DIM_1D_Z)
    m_boundary_names[idx_zlo] = "zlo";
    m_boundary_names[idx_zhi] = "zhi";
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
    static bool initialized = false;
    if (!initialized)
    {
        const amrex::ParmParse pp_particles("particles");
        pp_particles.queryarr("species_names", m_species_names);
        initialized = true;
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

void ParticleBoundaryBuffer::gatherParticlesFromDomainBoundaries (MultiParticleContainer& mypc)
{
    WARPX_PROFILE("ParticleBoundaryBuffer::gatherParticles");

    using PIter = amrex::ParConstIterSoA<PIdx::nattribs, 0>;
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
                    buffer[i] = pc.make_alike<amrex::PinnedArenaAllocator>();
                    buffer[i].NewIntComp("stepScraped", false);
                    buffer[i].NewRealComp("deltaTimeScraped", false);
                    buffer[i].NewRealComp("nx", false);
                    buffer[i].NewRealComp("ny", false);
                    buffer[i].NewRealComp("nz", false);
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
                          WARPX_PROFILE("ParticleBoundaryBuffer::gatherParticles::count_out_of_bounds");
                          const amrex::RandomEngine rng{};
                          reduce_op.eval(np, reduce_data, [=] AMREX_GPU_HOST_DEVICE (int ip)
                                         { return predicate(ptile_data, ip, rng) ? 1 : 0; });
                        }

                        auto dst_index = ptile_buffer.numParticles();
                        {
                          WARPX_PROFILE("ParticleBoundaryBuffer::gatherParticles::resize");
                          ptile_buffer.resize(dst_index + amrex::get<0>(reduce_data.value()));
                        }
                        {
                          WARPX_PROFILE("ParticleBoundaryBuffer::gatherParticles::filterAndTransform");
                          auto& warpx = WarpX::GetInstance();
                          const auto dt = warpx.getdt(pti.GetLevel());
                          auto string_to_index_intcomp = buffer[i].getParticleRuntimeiComps();
                          const int step_scraped_index = string_to_index_intcomp.at("stepScraped");
                          auto string_to_index_realcomp = buffer[i].getParticleRuntimeComps();
                          const int delta_index = string_to_index_realcomp.at("deltaTimeScraped");
                          const int normal_index = string_to_index_realcomp.at("nx");
                          const int step = warpx_instance.getistep(0);
                          amrex::filterAndTransformParticles(ptile_buffer, ptile,
                                                             predicate,
                                                             CopyAndTimestamp{step_scraped_index, delta_index, normal_index, step, dt, idim, iside},
                                                             0, dst_index);
                        }
                    }
                }
            }
        }
    }
}

void ParticleBoundaryBuffer::gatherParticlesFromEmbeddedBoundaries (
    MultiParticleContainer& mypc, ablastr::fields::MultiLevelScalarField const& distance_to_eb)
{
    if (EB::enabled()) {
        WARPX_PROFILE("ParticleBoundaryBuffer::gatherParticles::EB");


        using PIter = amrex::ParConstIterSoA<PIdx::nattribs, 0>;
        const auto &warpx_instance = WarpX::GetInstance();
        const amrex::Geometry &geom = warpx_instance.Geom(0);
        auto plo = geom.ProbLoArray();

        auto& buffer = m_particle_containers[m_particle_containers.size()-1];
        for (int i = 0; i < numSpecies(); ++i)
        {
            if (!m_do_boundary_buffer[AMREX_SPACEDIM*2][i]) { continue; }
            const auto& pc = mypc.GetParticleContainer(i);
            if (!buffer[i].isDefined())
            {
                buffer[i] = pc.make_alike<amrex::PinnedArenaAllocator>();
                buffer[i].NewIntComp("stepScraped", false);
                buffer[i].NewRealComp("deltaTimeScraped", false);
                buffer[i].NewRealComp("nx", false);
                buffer[i].NewRealComp("ny", false);
                buffer[i].NewRealComp("nz", false);

            }

            auto& species_buffer = buffer[i];
            for (int lev = 0; lev < pc.numLevels(); ++lev) {
                for (PIter pti(pc, lev); pti.isValid(); ++pti) {
                    species_buffer.DefineAndReturnParticleTile(
                        lev, pti.index(), pti.LocalTileIndex());
                }
            }

            for (int lev = 0; lev < pc.numLevels(); ++lev)
            {
                const auto& plevel = pc.GetParticles(lev);
                auto dxi = warpx_instance.Geom(lev).InvCellSizeArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
                for (PIter pti(pc, lev); pti.isValid(); ++pti) {
                    auto phiarr = (*distance_to_eb[lev])[pti].array();  // signed distance function
                    auto index = std::make_pair(pti.index(), pti.LocalTileIndex());
                    if (plevel.find(index) == plevel.end()) { continue; }

                    const auto getPosition = GetParticlePosition<PIdx>(pti);
                    auto &ptile_buffer = species_buffer.DefineAndReturnParticleTile(lev, pti.index(),
                                                                                    pti.LocalTileIndex());
                    const auto &ptile = plevel.at(index);
                    auto np = ptile.numParticles();
                    if (np == 0) { continue; }

                    using SrcData = WarpXParticleContainer::ParticleTileType::ConstParticleTileDataType;
                    auto predicate = [=] AMREX_GPU_HOST_DEVICE(const SrcData & /*src*/, const int ip)
                            /* NVCC 11.3.109 chokes in C++17 on this: noexcept */
                    {
                        amrex::ParticleReal xp, yp, zp;
                        getPosition(ip, xp, yp, zp);

                        amrex::Real const phi_value = ablastr::particles::doGatherScalarFieldNodal(
                            xp, yp, zp, phiarr, dxi, plo
                        );
                        return phi_value < 0.0 ? 1 : 0;
                    };

                    const auto ptile_data = ptile.getConstParticleTileData();

                    amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
                    amrex::ReduceData<int> reduce_data(reduce_op);
                    {
                        WARPX_PROFILE("ParticleBoundaryBuffer::gatherParticles::count_out_of_boundsEB");
                        reduce_op.eval(np, reduce_data,
                                       [=] AMREX_GPU_HOST_DEVICE(int ip) { return predicate(ptile_data, ip) ? 1 : 0; });
                    }

                    auto dst_index = ptile_buffer.numParticles();
                    {
                        WARPX_PROFILE("ParticleBoundaryBuffer::gatherParticles::resize_eb");
                        ptile_buffer.resize(dst_index + amrex::get<0>(reduce_data.value()));
                    }
                    auto &warpx = WarpX::GetInstance();
                    const auto dt = warpx.getdt(pti.GetLevel());
                    auto string_to_index_intcomp = buffer[i].getParticleRuntimeiComps();
                    const int step_scraped_index = string_to_index_intcomp.at("stepScraped");
                    auto string_to_index_realcomp = buffer[i].getParticleRuntimeComps();
                    const int delta_index = string_to_index_realcomp.at("deltaTimeScraped");
                    const int normal_index = string_to_index_realcomp.at("nx");
                    const int step = warpx_instance.getistep(0);

                    {
                        WARPX_PROFILE("ParticleBoundaryBuffer::gatherParticles::filterTransformEB");
                        amrex::filterAndTransformParticles(ptile_buffer, ptile, predicate,
                                                           FindEmbeddedBoundaryIntersection{step_scraped_index,
                                                                                            delta_index, normal_index,
                                                                                            step, dt, phiarr, dxi, plo},
                                                           0, dst_index);

                    }
                }
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

PinnedMemoryParticleContainer &
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

PinnedMemoryParticleContainer *
ParticleBoundaryBuffer::getParticleBufferPointer(const std::string& species_name, int boundary) {

    auto& buffer = m_particle_containers[boundary];
    auto index = WarpX::GetInstance().GetPartContainer().getSpeciesID(species_name);

    return &buffer[index];
}
