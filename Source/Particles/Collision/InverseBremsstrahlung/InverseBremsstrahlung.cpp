/* Copyright 2025 David Grote
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "InverseBremsstrahlung.H"

#include "Particles/ParticleCreation/FilterCopyTransform.H"
#include "Particles/ParticleCreation/SmartCopy.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"
#include "Utils/ParticleUtils.H"
#include "Utils/WarpXProfilerWrapper.H"
#include "WarpX.H"

#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <string>

InverseBremsstrahlung::InverseBremsstrahlung (std::string const& collision_name, MultiParticleContainer const* mypc)
    : CollisionBase(collision_name)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_species_names.size() == 2,
                                     "Inverse Bremsstrahlung must have exactly two species.");

    auto& species_1 = mypc->GetParticleContainerFromName(m_species_names[0]);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(species_1.AmIA<PhysicalSpecies::photon>(),
                                     "InverseBremsstrahlung: The first species must be photons");
}

void
InverseBremsstrahlung::doCollisions (amrex::Real /*cur_time*/, amrex::Real dt, MultiParticleContainer* mypc)
{
    WARPX_PROFILE("InverseBremsstrahlung::doCollisions()");
    using namespace amrex::literals;

    auto& species_1 = mypc->GetParticleContainerFromName(m_species_names[0]);
    auto& species_2 = mypc->GetParticleContainerFromName(m_species_names[1]);

    mypc->CalculateNuei(m_species_names[0]);

    // Enable tiling
    amrex::MFItInfo info;
    if (amrex::Gpu::notInLaunchRegion()) { info.EnableTiling(species_1.tile_size); }

    // Loop over refinement levels
    auto const flvl = species_1.finestLevel();
    for (int lev = 0; lev <= flvl; ++lev) {

        auto *cost = WarpX::getCosts(lev);

        // Loop over all grids/tiles at this level
#ifdef AMREX_USE_OMP
        info.SetDynamic(true);
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi = species_1.MakeMFIter(lev, info); mfi.isValid(); ++mfi){
            if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
            {
                amrex::Gpu::synchronize();
            }
            auto wt = static_cast<amrex::Real>(amrex::second());

            doInverseBremsstrahlungWithinTile(dt, lev, mfi, species_1, species_2);

            if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
            {
                amrex::Gpu::synchronize();
                wt = static_cast<amrex::Real>(amrex::second()) - wt;
                amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
            }
        }
    }
}


void InverseBremsstrahlung::doInverseBremsstrahlungWithinTile(
        amrex::Real dt, int lev, amrex::MFIter const& mfi,
        WarpXParticleContainer& species_1,
        WarpXParticleContainer& species_2)
{
    using namespace amrex::literals;

   ParticleTileType& ptile_1 = species_1.ParticlesAt(lev, mfi);
   ParticleTileType& ptile_2 = species_2.ParticlesAt(lev, mfi);

   // Find the particles that are in each cell of this tile
   ParticleBins bins_1 = ParticleUtils::findParticlesInEachCell(species_1.Geom(lev), mfi, ptile_1);
   ParticleBins bins_2 = ParticleUtils::findParticlesInEachCell(species_1.Geom(lev), mfi, ptile_2);

    // Extract low-level data
    auto const n_cells = static_cast<int>(bins_1.numBins());

    // - Species 1
    const auto soa_1 = ptile_1.getParticleTileData();
    index_type* AMREX_RESTRICT indices_1 = bins_1.permutationPtr();
    index_type const* AMREX_RESTRICT cell_offsets_1 = bins_1.offsetsPtr();

    // - Species 2
    const auto soa_2 = ptile_2.getParticleTileData();
    index_type* AMREX_RESTRICT indices_2 = bins_2.permutationPtr();
    index_type const* AMREX_RESTRICT cell_offsets_2 = bins_2.offsetsPtr();
    /* const amrex::ParticleReal q2 = species_2.getCharge(); */
    /* const amrex::ParticleReal m2 = species_2.getMass(); */

    WarpX & warpx = WarpX::GetInstance();
    amrex::MultiFab const & nuei = *warpx.m_fields.get("nuei_" + species_1.getName(), lev);
    amrex::FArrayBox const & nuei_fab = nuei[mfi];
    amrex::Real const * nuei_data = nuei_fab.dataPtr();

    amrex::Geometry const& geom = warpx.Geom(lev);
    auto const dV = AMREX_D_TERM(geom.CellSize(0), *geom.CellSize(1), *geom.CellSize(2));
#if defined WARPX_DIM_RZ
    amrex::Box const& cbx = mfi.tilebox(amrex::IntVect::TheZeroVector()); //Cell-centered box
    auto const lo = lbound(cbx);
    auto const hi = ubound(cbx);
    int const nz = hi.y - lo.y + 1;
    auto const dr = geom.CellSize(0);
#endif

    auto volume_factor = [=] AMREX_GPU_DEVICE(int i_cell) noexcept {
#if defined WARPX_DIM_RZ
        // Return the radial factor for the volume element, dV
        int const ri = (i_cell - i_cell%nz)/nz;
        return MathConst::pi*(2.0_prt*ri + 1.0_prt)*dr;
#else
        // No factor is needed for Cartesian
        amrex::ignore_unused(i_cell);
        return 1._prt;
#endif
    };

    // Loop over cells
    amrex::ParallelFor(n_cells,
        [=] AMREX_GPU_DEVICE (int i_cell) noexcept
        {
            // The particles from species_1 that are in the cell `i_cell` are
            // given by the `indices_1[cell_start_1:cell_stop_1]`
            index_type const cell_start_1 = cell_offsets_1[i_cell];
            index_type const cell_stop_1 = cell_offsets_1[i_cell+1];
            // Same for species 2
            index_type const cell_start_2 = cell_offsets_2[i_cell];
            index_type const cell_stop_2  = cell_offsets_2[i_cell+1];

            amrex::ParticleReal * const AMREX_RESTRICT w1 = soa_1.m_rdata[PIdx::w];
            amrex::ParticleReal * const AMREX_RESTRICT w2 = soa_2.m_rdata[PIdx::w];

            amrex::ParticleReal * const AMREX_RESTRICT u1x = soa_1.m_rdata[PIdx::ux];
            amrex::ParticleReal * const AMREX_RESTRICT u1y = soa_1.m_rdata[PIdx::uy];
            amrex::ParticleReal * const AMREX_RESTRICT u1z = soa_1.m_rdata[PIdx::uz];

            uint64_t * AMREX_RESTRICT idcpu1 = soa_1.m_idcpu;

            // Calculate electron number density (now only the colliding species
            // but presumably should be all electrons?)
            amrex::ParticleReal cell_ne = 0._prt;
            for (index_type i2 = cell_start_2 ; i2 < cell_stop_2 ; ++i2) {
                cell_ne += w2[indices_2[i2]];
            }
            cell_ne /= dV*volume_factor(i_cell);

            amrex::ParticleReal const cell_wpe = PhysConst::q_e*std::sqrt(cell_ne/PhysConst::m_e/PhysConst::ep0);
            amrex::ParticleReal const cell_Ewpe_eV = PhysConst::hbar*cell_wpe/PhysConst::q_e;

            amrex::ParticleReal const cell_nuei = nuei_data[i_cell];

            // loop over photons, adjust weight based on abosrpotion
            // tabulate total momentum and energy absorbed
            /* amrex::ParticleReal sum_deltaE_eV = 0.0_prt; */
            /* amrex::ParticleReal sum_dPx = 0.0_prt; */
            /* amrex::ParticleReal sum_dPy = 0.0_prt; */
            /* amrex::ParticleReal sum_dPz = 0.0_prt; */

            for (index_type i1=cell_start_1; i1<cell_stop_1; ++i1) {

                amrex::ParticleReal const upx = u1x[indices_1[i1]];
                amrex::ParticleReal const upy = u1y[indices_1[i1]];
                amrex::ParticleReal const upz = u1z[indices_1[i1]];

                // compute photon energy in [eV]
                amrex::ParticleReal const up = std::sqrt(upx*upx + upy*upy + upz*upz);
                amrex::ParticleReal const Ephoton_eV = up*PhysConst::m_e*PhysConst::c/PhysConst::q_e;
                amrex::ParticleReal const nuIB = std::pow(cell_Ewpe_eV/Ephoton_eV, 2)*cell_nuei*0.5_prt;

                // update photon weight based on absorption
                amrex::ParticleReal const wp1 = w1[indices_1[i1]];
                amrex::ParticleReal dw = wp1*std::min(nuIB*dt, 1.0_prt); // weight to be removed from photon

                if (dw < wp1) {
                    // Remove weight from the photon
                    w1[indices_1[i1]] -= dw;
                } else {
                    // set kill tag for photon if weight is too small
                    dw = wp1;
                    w1[indices_1[i1]] = 0.0_prt;
                    idcpu1[indices_1[i1]] = amrex::ParticleIdCpus::Invalid;
                }

                // update total energy lost
                /* sum_deltaE_eV += dw*Ephoton_eV; */

                // update total momentum lost
                /* sum_dPx += dw*upx; */
                /* sum_dPy += dw*upy; */
                /* sum_dPz += dw*upz; */

            }

            // update probe for energy gain via absorption
            /* m_deltaE_IBremsstrahlung += sum_deltaE_eV*PhysConst::q_e; // Joules */

        });

}
