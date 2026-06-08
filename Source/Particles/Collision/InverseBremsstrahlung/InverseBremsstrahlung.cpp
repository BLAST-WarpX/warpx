/* Copyright 2025 David Grote
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "InverseBremsstrahlung.H"

#include "Particles/Collision/BinaryCollision/ShuffleFisherYates.H"
#include "Particles/ParticleCreation/FilterCopyTransform.H"
#include "Particles/ParticleCreation/SmartCopy.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"
#include "Utils/ParticleUtils.H"
#include "WarpX.H"

#include <ablastr/profiler/ProfilerWrapper.H>

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

    const amrex::ParmParse pp_collision_name(collision_name);

    pp_collision_name.query("energy_fraction", m_energy_fraction);

}

void
InverseBremsstrahlung::doCollisions (amrex::Real /*cur_time*/, amrex::Real dt, MultiParticleContainer* mypc)
{
    ABLASTR_PROFILE("InverseBremsstrahlung::doCollisions()");
    using namespace amrex::literals;

    auto& photons = mypc->GetParticleContainerFromName(m_species_names[0]);
    auto& electrons = mypc->GetParticleContainerFromName(m_species_names[1]);

    mypc->CalculateNuei(m_species_names[1]);

    // Enable tiling
    amrex::MFItInfo info;
    if (amrex::Gpu::notInLaunchRegion()) { info.EnableTiling(WarpXParticleContainer::tile_size); }

    // Loop over refinement levels
    auto const flvl = photons.finestLevel();
    for (int lev = 0; lev <= flvl; ++lev) {

        auto *cost = WarpX::getCosts(lev);

        // Loop over all grids/tiles at this level
#ifdef AMREX_USE_OMP
        info.SetDynamic(true);
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi = photons.MakeMFIter(lev, info); mfi.isValid(); ++mfi){
            if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
            {
                amrex::Gpu::synchronize();
            }
            auto wt = static_cast<amrex::Real>(amrex::second());

            doInverseBremsstrahlungWithinTile(dt, lev, mfi, photons, electrons);

            if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
            {
                amrex::Gpu::synchronize();
                wt = static_cast<amrex::Real>(amrex::second()) - wt;
                amrex::HostDevice::Atomic::Add(&(*cost)[mfi.index()], wt);
            }
        }
    }
}


void InverseBremsstrahlung::doInverseBremsstrahlungWithinTile (
        amrex::Real dt, int lev, amrex::MFIter const& mfi,
        WarpXParticleContainer& photons,
        WarpXParticleContainer& electrons) const
{
    using namespace amrex::literals;

    ParticleTileType& ptile_photons = photons.ParticlesAt(lev, mfi);
    ParticleTileType& ptile_electrons = electrons.ParticlesAt(lev, mfi);
    auto np_photons = ptile_photons.numParticles();
    auto np_electrons = ptile_electrons.numParticles();

    // Find the particles that are in each cell of this tile
    ParticleBins bins_photons = ParticleUtils::findParticlesInEachCell(photons.Geom(lev), mfi, ptile_photons);
    ParticleBins bins_electrons = ParticleUtils::findParticlesInEachCell(electrons.Geom(lev), mfi, ptile_electrons);

    auto const n_cells = static_cast<int>(bins_photons.numBins());

    const auto soa_photons = ptile_photons.getParticleTileData();
    const auto soa_electrons = ptile_electrons.getParticleTileData();
    index_type const* AMREX_RESTRICT bins_photons_ptr = bins_photons.binsPtr();
    index_type const* AMREX_RESTRICT bins_electrons_ptr = bins_electrons.binsPtr();
    index_type const* AMREX_RESTRICT cell_offsets_electrons = bins_electrons.offsetsPtr();
    index_type* AMREX_RESTRICT indices_electrons = bins_electrons.permutationPtr();

    const amrex::ParticleReal m_electrons = electrons.getMass();

    WarpX & warpx = WarpX::GetInstance();
    amrex::MultiFab const & nuei = *warpx.m_fields.get("nuei_" + electrons.getName(), lev);
    amrex::FArrayBox const & nuei_fab = nuei[mfi];
    amrex::Real const * nuei_data = nuei_fab.dataPtr();

    amrex::MultiFab const & N_e = *warpx.m_fields.get("N_" + electrons.getName(), lev);
    amrex::FArrayBox const & N_e_fab = N_e[mfi];
    amrex::Real const * N_e_data = N_e_fab.dataPtr();

    amrex::Geometry const& geom = warpx.Geom(lev);
    auto const dV = AMREX_D_TERM(geom.CellSize(0), *geom.CellSize(1), *geom.CellSize(2));
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    amrex::Box const& cbx = mfi.tilebox(amrex::IntVect::TheZeroVector()); //Cell-centered box
#if defined(WARPX_DIM_RZ)
    auto const lo = lbound(cbx);
    auto const hi = ubound(cbx);
    int const nr = hi.x - lo.x + 1;
#endif
    amrex::XDim3 const xyzmin = WarpX::LowerCorner(cbx, lev, 0._rt);
    amrex::Real const rmin = xyzmin.x;
    auto const dr = geom.CellSize(0);
#endif

    auto volume_factor = [=] AMREX_GPU_DEVICE(int i_cell) noexcept {
#if defined(WARPX_DIM_RZ)
        // Return the radial factor for the volume element, dV
        int const ri = i_cell % nr;
        // rr is radius at the cell center
        amrex::ParticleReal const rr = rmin + (ri + 0.5_prt)*dr;
        return 2.0_prt*static_cast<amrex::ParticleReal>(MathConst::pi)*rr;
#elif defined(WARPX_DIM_RCYLINDER)
        int const ri = i_cell;
        // rr is radius at the cell center
        amrex::ParticleReal const rr = rmin + (ri + 0.5_prt)*dr;
        return 2.0_prt*static_cast<amrex::ParticleReal>(MathConst::pi)*rr;
#elif defined(WARPX_DIM_RSPHERE)
        // This needs double checking
        int const ri = i_cell;
        // rr is radius at the cell center
        amrex::ParticleReal const rr = rmin + (ri + 0.5_prt)*dr;
        return 4.0_prt*static_cast<amrex::ParticleReal>(MathConst::pi)*rr*rr;
#else
        // No factor is needed for Cartesian
        amrex::ignore_unused(i_cell);
        return 1.0_prt;
#endif
    };

    amrex::Gpu::DeviceVector<amrex::ParticleReal> KE_vec(n_cells, 0._prt);
    amrex::Gpu::DeviceVector<amrex::ParticleReal> px_vec(n_cells, 0._prt);
    amrex::Gpu::DeviceVector<amrex::ParticleReal> py_vec(n_cells, 0._prt);
    amrex::Gpu::DeviceVector<amrex::ParticleReal> pz_vec(n_cells, 0._prt);
    amrex::Gpu::DeviceVector<amrex::ParticleReal> w_sum_electrons_vec(n_cells, 0._prt);

    amrex::ParticleReal* AMREX_RESTRICT KE_in_each_cell = KE_vec.dataPtr();
    amrex::ParticleReal* AMREX_RESTRICT px_in_each_cell = px_vec.dataPtr();
    amrex::ParticleReal* AMREX_RESTRICT py_in_each_cell = py_vec.dataPtr();
    amrex::ParticleReal* AMREX_RESTRICT pz_in_each_cell = pz_vec.dataPtr();
    amrex::ParticleReal* AMREX_RESTRICT w_sum_electrons_in_each_cell = w_sum_electrons_vec.dataPtr();

    amrex::ParticleReal * const AMREX_RESTRICT w_photons = soa_photons.m_rdata[PIdx::w];
    amrex::ParticleReal * const AMREX_RESTRICT ux_photons = soa_photons.m_rdata[PIdx::ux];
    amrex::ParticleReal * const AMREX_RESTRICT uy_photons = soa_photons.m_rdata[PIdx::uy];
    amrex::ParticleReal * const AMREX_RESTRICT uz_photons = soa_photons.m_rdata[PIdx::uz];
    uint64_t * AMREX_RESTRICT idcpu_photons = soa_photons.m_idcpu;

    amrex::ParticleReal * const AMREX_RESTRICT w_electrons = soa_electrons.m_rdata[PIdx::w];
    amrex::ParticleReal * const AMREX_RESTRICT ux_electrons = soa_electrons.m_rdata[PIdx::ux];
    amrex::ParticleReal * const AMREX_RESTRICT uy_electrons = soa_electrons.m_rdata[PIdx::uy];
    amrex::ParticleReal * const AMREX_RESTRICT uz_electrons = soa_electrons.m_rdata[PIdx::uz];

    // Loop over photons
    amrex::ParallelFor(np_photons,
        [=] AMREX_GPU_DEVICE (int ip) noexcept
        {
            const int i_cell = bins_photons_ptr[ip];

            // Calculate electron number density (now only the colliding species
            // but presumably should be all electrons?)
            amrex::ParticleReal const cell_ne = N_e_data[i_cell]/(dV*volume_factor(i_cell));

            amrex::ParticleReal const cell_wpe = PhysConst::q_e*std::sqrt(cell_ne/PhysConst::m_e/PhysConst::epsilon_0);
            amrex::ParticleReal const cell_Ewpe_J = PhysConst::hbar*cell_wpe;

            amrex::ParticleReal const cell_nuei = nuei_data[i_cell];

            amrex::ParticleReal const wp = w_photons[ip];
            amrex::ParticleReal const upx = ux_photons[ip];
            amrex::ParticleReal const upy = uy_photons[ip];
            amrex::ParticleReal const upz = uz_photons[ip];

            // adjust weight based on abosrpotion
            // tabulate total momentum and energy absorbed

            // compute photon energy
            amrex::ParticleReal const Ephoton_J = Algorithms::KineticEnergyPhotons(upx, upy, upz);
            amrex::ParticleReal const nuIB = std::pow(cell_Ewpe_J/Ephoton_J, 2._prt)*cell_nuei*0.5_prt;

            // update photon weight based on absorption
            auto const dt_prt = static_cast<amrex::ParticleReal>(dt);
            amrex::ParticleReal dw = wp*std::min(nuIB*dt_prt, 1.0_prt); // weight to be removed from photon

            if (dw < wp) {
                // Remove weight from the photon
                w_photons[ip] -= dw;
            } else {
                // set kill tag for photon if weight is too small
                dw = wp;
                w_photons[ip] = 0.0_prt;
                idcpu_photons[ip] = amrex::ParticleIdCpus::Invalid;
            }

            // update total energy and momentum lost
            // note that the photon upx, y, z is scaled by m_e
            auto constexpr m_e_prt = static_cast<amrex::ParticleReal>(PhysConst::m_e);
            amrex::Gpu::Atomic::AddNoRet(&KE_in_each_cell[i_cell], dw*Ephoton_J);
            amrex::Gpu::Atomic::AddNoRet(&px_in_each_cell[i_cell], dw*upx*m_e_prt);
            amrex::Gpu::Atomic::AddNoRet(&py_in_each_cell[i_cell], dw*upy*m_e_prt);
            amrex::Gpu::Atomic::AddNoRet(&pz_in_each_cell[i_cell], dw*upz*m_e_prt);

            // update probe for energy gain via absorption
            /* m_deltaE_IBremsstrahlung += sum_deltaE_J*PhysConst::q_e; // Joules */

        });

    // Need total electron weight to determine how much momentum is given to each electron
    amrex::ParallelFor(np_electrons,
        [=] AMREX_GPU_DEVICE (int ie) noexcept
        {
            const int i_cell = bins_electrons_ptr[ie];
            amrex::Gpu::Atomic::AddNoRet(&w_sum_electrons_in_each_cell[i_cell], w_electrons[ie]);
        });

    // Distribute momentum absorbed from photons to electrons
    amrex::ParallelFor(np_electrons,
        [=] AMREX_GPU_DEVICE (int ie) noexcept
        {
            const int i_cell = bins_electrons_ptr[ie];

            amrex::ParticleReal const KE_before = Algorithms::KineticEnergy(ux_electrons[ie],
                                                                            uy_electrons[ie],
                                                                            uz_electrons[ie],
                                                                            m_electrons);

            ux_electrons[ie] += px_in_each_cell[i_cell]/(w_sum_electrons_in_each_cell[i_cell]*m_electrons);
            uy_electrons[ie] += py_in_each_cell[i_cell]/(w_sum_electrons_in_each_cell[i_cell]*m_electrons);
            uz_electrons[ie] += pz_in_each_cell[i_cell]/(w_sum_electrons_in_each_cell[i_cell]*m_electrons);

            amrex::ParticleReal const KE_after = Algorithms::KineticEnergy(ux_electrons[ie],
                                                                           uy_electrons[ie],
                                                                           uz_electrons[ie],
                                                                           m_electrons);

            // Update the energy in the cell by subracting off the energy added to the electron
            amrex::ParticleReal const KE_change = KE_after - KE_before;
            amrex::Gpu::Atomic::AddNoRet(&KE_in_each_cell[i_cell], -w_electrons[ie]*KE_change);

        });

    amrex::ParallelForRNG(n_cells,
        [=] AMREX_GPU_DEVICE (int i_cell, amrex::RandomEngine const& engine) noexcept
        {
            // The particles from species1 that are in the cell `i_cell` are
            // given by the `indices_electrons[cell_start_electrons:cell_stop_electrons]`
            index_type const cell_start_electrons = cell_offsets_electrons[i_cell];
            index_type const cell_stop_electrons  = cell_offsets_electrons[i_cell+1];

            // Do not shuffle if there is only one particle in the cell
            if (cell_stop_electrons - cell_start_electrons <= 1) { return; }

            ShuffleFisherYates(indices_electrons, cell_start_electrons, cell_stop_electrons, engine);
        }
    );

    amrex::Gpu::Buffer<amrex::Long> failed_corrections({0});
    amrex::Long* failed_corrections_ptr = failed_corrections.data();

    const amrex::ParticleReal energy_fraction = m_energy_fraction;

    // Distribute any remaining energy to the electrons using the pairwise
    // operation (that does not affect the total momentum)
    amrex::ParallelFor(n_cells,
        [=] AMREX_GPU_DEVICE (int i_cell) noexcept
        {

            index_type const cell_start_electrons = cell_offsets_electrons[i_cell];
            index_type const cell_stop_electrons  = cell_offsets_electrons[i_cell + 1];
            amrex::ParticleReal deltaEp_subtract = -KE_in_each_cell[i_cell];

            if (deltaEp_subtract != 0.) {
                // Adjust electrons to absorb deltaEp_subtract.
                const bool correction_failed =
                     ParticleUtils::ModifyEnergyPairwise(ux_electrons, uy_electrons, uz_electrons, w_electrons,
                                                         indices_electrons,
                                                         cell_start_electrons, cell_stop_electrons, m_electrons,
                                                         energy_fraction, deltaEp_subtract);
                KE_in_each_cell[i_cell] = -deltaEp_subtract;
                if (correction_failed) {
                    amrex::Gpu::Atomic::Add(failed_corrections_ptr, amrex::Long(1));
                }
            }

        }
    );

    amrex::Long const num_failed_corrections = *(failed_corrections.copyToHost());

    if (num_failed_corrections > 0) {
        ablastr::warn_manager::WMRecordWarning("InverseBremsstrahlung::doInverseBremsstrahlungWithinTile",
            "Pair-wise energy distribution failed, resorting to method that does not conserve momentum");

        // Distribute the remaining energy among the electrons without conserving momentum
        // Note that this only works if the particle energy is nonzero
        amrex::ParallelFor(np_electrons,
            [=] AMREX_GPU_DEVICE (int ie) noexcept
            {
                const int i_cell = bins_electrons_ptr[ie];
                amrex::ParticleReal const dKE = KE_in_each_cell[i_cell]/w_sum_electrons_in_each_cell[i_cell];
                if (dKE == 0.) { return; }

                amrex::ParticleReal constexpr c2 = PhysConst::c2;

                amrex::ParticleReal const dgamma = dKE/(m_electrons*c2);

                amrex::ParticleReal const u2 = ux_electrons[ie]*ux_electrons[ie]
                                             + uy_electrons[ie]*uy_electrons[ie]
                                             + uz_electrons[ie]*uz_electrons[ie];
                if (u2 == 0.) { return; }

                amrex::ParticleReal const gamma = std::sqrt(1._prt + u2/c2);
                amrex::ParticleReal const gamma_new = gamma + dgamma;
                amrex::ParticleReal const fsq = (gamma_new + 1._prt)*(u2/(1._prt + gamma) + dgamma*c2)/u2;
                amrex::ParticleReal const f = std::sqrt(fsq);

                ux_electrons[ie] *= f;
                uy_electrons[ie] *= f;
                uz_electrons[ie] *= f;

            });

        }

}
