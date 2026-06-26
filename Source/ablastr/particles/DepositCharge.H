/* Copyright 2019-2021 Axel Huebl, Andrew Myers
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef ABLASTR_DEPOSIT_CHARGE_H_
#define ABLASTR_DEPOSIT_CHARGE_H_

#include "ablastr/profiler/ProfilerWrapper.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/Deposition/ChargeDeposition.H"
#include "ablastr/utils/TextMsg.H"

#include <AMReX.H>

#include <optional>


namespace ablastr::particles
{

/** Perform charge deposition for the particles on a tile.
 *
 * \tparam T_PC a type of amrex::ParticleContainer
 *
 * \param pti an amrex::ParIter pointing to the tile to operate on
 * \param wp vector of the particle weights for those particles.
 * \param charge charge of the particle species
 * \param ion_lev pointer to array of particle ionization level. This is
                  required to have the charge of each macroparticle
                  since q is a scalar. For non-ionizable species,
                  ion_lev is a null pointer.
 * \param rho MultiFab of the charge density
 * \param local_rho temporary FArrayBox for deposition with OpenMP
 * \param particle_shape shape factor in each direction
 * \param dinv cell spacing inverses at level lev
 * \param xyzmin lo corner of the current tile in physical coordinates.
 * \param n_rz_azimuthal_modes number of azimuthal modes in use, irrelevant outside RZ geometry (default: 0)
 * \param num_rho_deposition_guards number of ghost cells to use for rho (default: rho.nGrowVect())
 * \param depos_lev the level to deposit the particles to (default: lev)
 * \param rel_ref_ratio mesh refinement ratio between lev and depos_lev (default: 1)
 * \param offset index to start at when looping over particles to deposit (default: 0)
 * \param np_to_deposit number of particles to deposit (default: pti.numParticles())
 * \param icomp component in MultiFab to start depositing to
 * \param nc number of components to deposit
 */
template< typename T_PC >
void
deposit_charge (typename T_PC::ParIterType& pti,
                typename T_PC::RealVector const& wp,
                amrex::Real const charge,
                int const * const ion_lev,
                amrex::MultiFab* rho,
                amrex::FArrayBox& local_rho,
                int const particle_shape,
                const amrex::XDim3 dinv,
                const amrex::XDim3 xyzmin,
                int const n_rz_azimuthal_modes = 0,
                std::optional<amrex::IntVect> num_rho_deposition_guards = std::nullopt,
                std::optional<int> depos_lev = std::nullopt,
                std::optional<amrex::IntVect> rel_ref_ratio = std::nullopt,
                long const offset = 0,
                std::optional<long> np_to_deposit = std::nullopt,
                int const icomp = 0, int const nc = 1)
{
    // deposition guards
    amrex::IntVect ng_rho = rho->nGrowVect();
    if (num_rho_deposition_guards.has_value()) {
        ng_rho = num_rho_deposition_guards.value();
    }
    ABLASTR_ALWAYS_ASSERT_WITH_MESSAGE(ng_rho.allLE(rho->nGrowVect()),
                                       "num_rho_deposition_guards are larger than allocated!");
    // particle shape
    auto const[nox, noy, noz] = std::array<int, 3>{particle_shape, particle_shape, particle_shape};

    // used for MR when we want to deposit for a subset of the particles on the level in the
    // current box; with offset, we start at a later particle index
    if (!np_to_deposit.has_value()) {
        np_to_deposit = pti.numParticles();
    }
    ABLASTR_ALWAYS_ASSERT_WITH_MESSAGE(np_to_deposit.value() + offset <= pti.numParticles(),
                                       "np_to_deposit + offset are out-of-bounds for particle iterator");

    int const lev = pti.GetLevel();
    if (!depos_lev.has_value()) {
        depos_lev = lev;
    }
    ABLASTR_ALWAYS_ASSERT_WITH_MESSAGE((depos_lev.value() == (lev-1)) ||
                                       (depos_lev.value() == (lev  )),
                                       "Deposition buffers only work for lev or lev-1");
    if (!rel_ref_ratio.has_value()) {
        ABLASTR_ALWAYS_ASSERT_WITH_MESSAGE(lev == depos_lev,
                                           "rel_ref_ratio must be set if lev != depos_lev");
        rel_ref_ratio = amrex::IntVect(AMREX_D_DECL(1, 1, 1));
    }

    // If no particles, do not do anything
    if (np_to_deposit == 0) { return; }

    // Extract deposition order and check that particles shape fits within the guard cells.
    // NOTE: In specific situations where the staggering of rho and the charge deposition algorithm
    // are not trivial, this check might be too strict and we might need to relax it, as currently
    // done for the current deposition.

#if   defined(WARPX_DIM_1D_Z)
    amrex::ignore_unused(nox);
    amrex::ignore_unused(noy);
    const amrex::IntVect shape_extent = amrex::IntVect(static_cast<int>(noz/2+1));
#elif defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    const amrex::IntVect shape_extent = amrex::IntVect(static_cast<int>(nox/2+1));
    amrex::ignore_unused(noy);
    amrex::ignore_unused(noz);
#elif   defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    amrex::ignore_unused(noy);
    const amrex::IntVect shape_extent = amrex::IntVect(static_cast<int>(nox/2+1),
                                                       static_cast<int>(noz/2+1));
#elif defined(WARPX_DIM_3D)
    const amrex::IntVect shape_extent = amrex::IntVect(static_cast<int>(nox/2+1),
                                                       static_cast<int>(noy/2+1),
                                                       static_cast<int>(noz/2+1));
#endif

    // On CPU: particles deposit on tile arrays, which have a small number of guard cells ng_rho
    // On GPU: particles deposit directly on the rho array, which usually have a larger number of guard cells
#ifndef AMREX_USE_GPU
    const amrex::IntVect range = ng_rho - shape_extent;
#else
    const amrex::IntVect range = rho->nGrowVect() - shape_extent;
#endif
    amrex::ignore_unused(range); // for release builds
    ABLASTR_ALWAYS_ASSERT_WITH_MESSAGE(
        amrex::numParticlesOutOfRange(pti, range) == 0,
        "Particles shape does not fit within tile (CPU) or guard cells (GPU) used for charge deposition");

    ABLASTR_PROFILE_VAR_NS("ablastr::particles::deposit_charge::ChargeDeposition", blp_ppc_chd);
    ABLASTR_PROFILE_VAR_NS("ablastr::particles::deposit_charge::Accumulate", blp_accumulate);

    // Get tile box where charge is deposited.
    // The tile box is different when depositing in the buffers (depos_lev<lev)
    // or when depositing inside the level (depos_lev=lev)
    amrex::Box tilebox;
    if (lev == depos_lev) {
        tilebox = pti.tilebox();
    } else {
        tilebox = amrex::coarsen(pti.tilebox(), rel_ref_ratio.value());
    }

#ifndef AMREX_USE_GPU
    // Staggered tile box
    amrex::Box tb = amrex::convert( tilebox, rho->ixType().toIntVect() );
#endif

    tilebox.grow(ng_rho);

#ifdef AMREX_USE_GPU
    amrex::ignore_unused(local_rho);
    // GPU, no tiling: rho_fab points to the full rho array
    amrex::MultiFab rhoi(*rho, amrex::make_alias, icomp*nc, nc);
    auto & rho_fab = rhoi.get(pti);
#else
    tb.grow(ng_rho);

    // CPU, tiling: rho_fab points to local_rho
    local_rho.resize(tb, nc);

    // local_rho is set to zero
    local_rho.setVal(0.0);

    auto & rho_fab = local_rho;
#endif

    const auto GetPosition = GetParticlePosition<PIdx>(pti, offset);

    // Indices of the lower bound
    const amrex::Dim3 lo = lbound(tilebox);

    ABLASTR_PROFILE_VAR_START(blp_ppc_chd);

    if        (nox == 1){
        doChargeDepositionShapeN<1>(GetPosition, wp.dataPtr()+offset, ion_lev,
                                    rho_fab, np_to_deposit.value(), dinv, xyzmin, lo, charge,
                                    n_rz_azimuthal_modes);
    } else if (nox == 2){
        doChargeDepositionShapeN<2>(GetPosition, wp.dataPtr()+offset, ion_lev,
                                    rho_fab, np_to_deposit.value(), dinv, xyzmin, lo, charge,
                                    n_rz_azimuthal_modes);
    } else if (nox == 3){
        doChargeDepositionShapeN<3>(GetPosition, wp.dataPtr()+offset, ion_lev,
                                    rho_fab, np_to_deposit.value(), dinv, xyzmin, lo, charge,
                                    n_rz_azimuthal_modes);
    } else if (nox == 4){
        doChargeDepositionShapeN<4>(GetPosition, wp.dataPtr()+offset, ion_lev,
                                    rho_fab, np_to_deposit.value(), dinv, xyzmin, lo, charge,
                                    n_rz_azimuthal_modes);
    }
    ABLASTR_PROFILE_VAR_STOP(blp_ppc_chd);

#ifndef AMREX_USE_GPU
    // CPU, tiling: atomicAdd local_rho into rho
    ABLASTR_PROFILE_VAR_START(blp_accumulate);
    (*rho)[pti].lockAdd(local_rho, tb, tb, 0, icomp*nc, nc);
    ABLASTR_PROFILE_VAR_STOP(blp_accumulate);
#endif
}

} // namespace ablastr::particles

#endif // ABLASTR_DEPOSIT_CHARGE_H_
