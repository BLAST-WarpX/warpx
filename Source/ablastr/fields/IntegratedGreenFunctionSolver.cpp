/* Copyright 2019-2024 Arianna Formenti, Remi Lehe
 *
 * This file is part of ABLASTR.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "IntegratedGreenFunctionSolver.H"

#include <ablastr/constant.H>
#include <ablastr/warn_manager/WarnManager.H>

#include <AMReX_Array4.H>
#include <AMReX_BaseFab.H>
#include <AMReX_BLassert.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Config.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FabArray.H>
#include <AMReX_FFT.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_MLLinOp.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace ablastr::fields {

namespace {

    using OpenBCSolverT = amrex::FFT::OpenBCSolver<amrex::Real>;
    using SpectralMF = OpenBCSolverT::cMF;

    /** @brief Default for how closely two cell-size ratios must agree to be one grid.
     *
     * Not exact equality, because a caller cannot always hand us a bit-identical cell
     * size for what is the same grid. ImpactX, for instance, sizes its mesh so that the
     * longitudinal cell size times the Lorentz factor is a chosen length, but the factor
     * reaches this solver as a velocity and is turned back into a stretch by
     * sqrt(1-beta^2), which costs a few ulp. Demanding bit equality there rejects every
     * reuse. A few dozen epsilon absorbs it while staying far below the accuracy of the
     * Green's function itself, so this default never trades accuracy for reuse.
     *
     * Some callers need more. Reconstructing the stretch from a velocity loses about
     * eps*gamma^2, which is a few ulp at gamma of a thousand but reaches 1e-5 at gamma
     * of a few hundred thousand, and a velocity measured from the particles carries its
     * own sampling noise on top. Those runs can raise ablastr.igf_cache_tolerance to
     * reuse a Green's function across grids that differ by that much. The error this
     * introduces is of the order of the tolerance itself, so raise it deliberately.
     *
     * There are two ways to make a Green's function come back, and they are not
     * interchangeable. Which one applies depends on whether the caller owns the quantity
     * that varies:
     *
     *   - The caller chooses its own grid. Then it can snap the grid to a repeating set,
     *     the Green's function stays exact for the grid it is used on, and reuse is
     *     bit-identical. ImpactX does this with geometry.prob_relative_max, paying in
     *     resolution rather than in accuracy, and needs no tolerance at all.
     *   - The quantity that varies is measured rather than chosen, as the velocity is in
     *     the relativistic electrostatic solver. Nothing can be snapped, and only this
     *     tolerance recovers reuse, by letting a Green's function serve a grid slightly
     *     different from the one it was built for.
     *
     * Prefer the first where it is available: it buys reuse with resolution instead of
     * with a small wrongness.
     */
    constexpr amrex::Real
    default_grid_match_tolerance ()
    {
        return amrex::Real(64.) * std::numeric_limits<amrex::Real>::epsilon();
    }

    /** @brief Do @p a and @p b agree to within @p tol, relatively? */
    bool
    same_grid_ratio (amrex::Real a, amrex::Real b, amrex::Real tol)
    {
        amrex::Real const scale = std::max(std::abs(a), std::abs(b));
        return std::abs(a - b) <= tol * scale;
    }

    /** @brief Is @p r a power of two to within @p tol?
     *
     * On success @p snapped returns the exact power of two, so that the rescaling of phi
     * stays exact even though the ratio that got us here was only approximate.
     */
    bool
    is_power_of_two_approx (amrex::Real r, amrex::Real tol, amrex::Real & snapped)
    {
        if (!(r > amrex::Real(0.)) || !std::isfinite(r)) { return false; }
        amrex::Real const exponent = std::log2(r);
        amrex::Real const nearest = std::round(exponent);
        if (std::abs(exponent - nearest) > tol) { return false; }
        snapped = std::exp2(nearest);
        return true;
    }

    /** @brief One cached spectral Green's function, keyed on the shape of the grid.
     *
     * The Green's function depends on the grid only through the cell size, and it is
     * homogeneous of degree two, so an entry is identified by the two cell-size ratios
     * and can serve any overall scale reachable by an exact power of two.
     */
    struct IGFCacheEntry
    {
        amrex::Real ry = 0.;      //!< dy/dx of the grid this was built for
        amrex::Real rz = 0.;      //!< dz/dx; always 1 in 2D, where G does not depend on dz
        amrex::Real dx = 0.;      //!< cell size this was built at, for the scale factor
        SpectralMF g;             //!< spectral G; empty in 3D while loaded in the solver
        bool built = false;       //!< false until it has been filled once
        std::uint64_t used = 0;   //!< LRU stamp, bumped whenever the entry is used
    };

    /** @brief Solver plus its cache of spectral Green's functions. */
    struct IGFCache
    {
        std::unique_ptr<OpenBCSolverT> solver;
        // identity of the solver: a change in any of these invalidates every entry
        amrex::Box domain;
        bool twod = false;
        int nprocs = 0;

        std::vector<IGFCacheEntry> entries;
        int loaded = -1;              //!< entry backing the solver's G, -1 if none does
        // Shape and scale of the Green's function the solver currently holds. This is
        // tracked independently of the store, so the rebuild can be skipped even when
        // the store is disabled.
        bool loaded_valid = false;
        amrex::Real loaded_ry = 0.;
        amrex::Real loaded_rz = 0.;
        amrex::Real loaded_dx = 0.;
        std::uint64_t clock = 0;
        std::uint64_t hits = 0;
        std::uint64_t misses = 0;
        std::uint64_t evictions = 0;

        int max_entries = 8;
        std::size_t max_bytes = 0;    //!< 0: no byte limit
        amrex::Real tolerance = default_grid_match_tolerance();  //!< grid match
        bool rebuild_always = false;  //!< reference behavior: rebuild on every call
        int verbose = 0;
        bool finalize_registered = false;

        void clearEntries () { entries.clear(); loaded = -1; loaded_valid = false; }
        void reset () { clearEntries(); solver.reset(); }
    };

    IGFCache&
    getIGFCache ()
    {
        static IGFCache cache;
        return cache;
    }

    /** @brief Re-read the cache controls.
     *
     * Read on every call rather than once, so that a Python or otherwise interactive
     * caller can change them between tracking runs, as it can for the other parameters
     * this solver honors.
     */
    void
    readIGFCacheParams (IGFCache & cache)
    {
        amrex::ParmParse pp("ablastr");
        pp.queryAdd("igf_cache_max_entries", cache.max_entries);
        pp.queryAdd("igf_cache_verbose", cache.verbose);
        pp.queryAdd("igf_cache_tolerance", cache.tolerance);
        if (cache.tolerance < amrex::Real(0.)) {
            cache.tolerance = default_grid_match_tolerance();
        }

        int rebuild_always = 0;
        pp.queryAdd("igf_rebuild_always", rebuild_always);
        cache.rebuild_always = (rebuild_always != 0);

        // Default byte budget: a quarter of the device memory that was free at startup,
        // since a single 3D entry can be a sizable fraction of a GPU. Host memory is left
        // unbounded and governed by the entry count instead. Sampled once, because the
        // free memory shrinks as the run allocates and a moving budget would make the
        // number of resident entries depend on when they happened to be created.
        static amrex::Long const default_max_bytes = [] () {
#ifdef AMREX_USE_GPU
            return static_cast<amrex::Long>(amrex::Gpu::Device::freeMemAvailable() / 4);
#else
            return amrex::Long(0);
#endif
        }();

        amrex::Long max_bytes = default_max_bytes;
        pp.queryAdd("igf_cache_max_bytes", max_bytes);
        cache.max_bytes = (max_bytes > 0) ? static_cast<std::size_t>(max_bytes) : std::size_t(0);
    }

    /** @brief Local size in bytes of one spectral Green's function. */
    std::size_t
    entryBytes (SpectralMF const & proto)
    {
        std::size_t bytes = 0;
        for (amrex::MFIter mfi(proto); mfi.isValid(); ++mfi) {
            bytes += proto[mfi].nBytes();
        }
        return bytes;
    }

    /** @brief Give the solver the Green's function of entry \p idx.
     *
     * In 3D this is a swap, so no data is copied: the solver takes the entry's buffer
     * and the previously loaded entry takes its own data back. The slot of the loaded
     * entry is therefore always empty.
     *
     * In 2D the solver's spectral Green's function is an alias of the Green's function
     * R2C's spectral data, which a later setGreensFunction() writes into, so it must be
     * copied through rather than swapped away. 2D entries keep their data at all times.
     */
    void
    loadEntry (IGFCache & cache, int idx)
    {
        if (cache.loaded == idx) { return; }

        auto & g_solver = cache.solver->greensFunctionFFT();
        auto & g_entry = cache.entries[idx].g;

        if (cache.twod) {
            if (cache.entries[idx].built) {
                g_solver.LocalCopy(g_entry, 0, 0, g_solver.nComp(), g_solver.nGrowVect());
            }
        } else {
            std::swap(g_solver, g_entry);
            if (cache.loaded >= 0) {
                std::swap(cache.entries[cache.loaded].g, g_entry);
            } else {
                // Discard the buffer the solver allocated for itself; from here on the
                // solver always holds a buffer owned by the cache.
                g_entry.clear();
            }
        }

        cache.loaded = idx;
    }

    /** @brief Drop least-recently-used entries until one more fits, then add it. */
    int
    makeEntry (IGFCache & cache, amrex::Real ry, amrex::Real rz, amrex::Real dx)
    {
        auto const& proto = cache.solver->greensFunctionFFT();
        std::size_t const bytes = entryBytes(proto);

        auto const room = [&] () {
            if (static_cast<int>(cache.entries.size()) + 1 > cache.max_entries) { return false; }
            if (cache.max_bytes > 0) {
                std::size_t const used = (cache.entries.size() + 1) * bytes;
                if (used > cache.max_bytes) { return false; }
            }
            return true;
        };

        while (!room()) {
            // Evict the least recently used entry other than the one in the solver.
            int victim = -1;
            for (int i = 0; i < static_cast<int>(cache.entries.size()); ++i) {
                if (i == cache.loaded) { continue; }
                if (victim < 0 || cache.entries[i].used < cache.entries[victim].used) { victim = i; }
            }
            if (victim < 0) { break; }  // only the loaded entry is left; keep it
            cache.entries.erase(cache.entries.begin() + victim);
            if (cache.loaded > victim) { --cache.loaded; }
            ++cache.evictions;
        }

        IGFCacheEntry e;
        e.ry = ry;
        e.rz = rz;
        e.dx = dx;
        e.g.define(proto.boxArray(), proto.DistributionMap(), proto.nComp(), proto.nGrowVect());
        cache.entries.push_back(std::move(e));
        return static_cast<int>(cache.entries.size()) - 1;
    }

} // namespace

void
computePhiIGF ( amrex::MultiFab const & rho,
                amrex::MultiFab & phi,
                std::array<amrex::Real, 3> const & cell_size,
                bool const is_igf_2d_slices)
{
    using namespace amrex::literals;

    BL_PROFILE("ablastr::fields::computePhiIGF");

    // Define box that encompasses the valid (nodal) domain of the source.
    //   Note: we intentionally do NOT grow by phi.nGrowVect() here.
    //   Guard/ghost values of phi are filled by the caller later on,
    //   e.g., in FillBoundary.
    amrex::Box const domain = rho.boxArray().minimalBox();

    int nprocs = amrex::ParallelDescriptor::NProcs();
    {
        amrex::ParmParse pp("ablastr");
        pp.queryAdd("nprocs_igf_fft", nprocs);
        nprocs = std::max(1,std::min(nprocs, amrex::ParallelDescriptor::NProcs()));
    }

    IGFCache & cache = getIGFCache();
    readIGFCacheParams(cache);
    if (!cache.finalize_registered) {
        cache.finalize_registered = true;
        amrex::ExecOnFinalize([] () {
            IGFCache & c = getIGFCache();
            if (c.verbose > 0 && (c.hits + c.misses) > 0) {
                amrex::Print() << "IGF Green's function cache: " << c.hits << " reused, "
                               << c.misses << " built, " << c.evictions << " evicted, "
                               << c.entries.size() << " resident\n";
            }
            c.reset();
        });
    }

    // A change of any of these invalidates the solver and every cached Green's function.
    if (!cache.solver || cache.domain != domain || cache.twod != is_igf_2d_slices ||
        cache.nprocs != nprocs)
    {
        cache.clearEntries();
        amrex::FFT::Info info{};
        if (is_igf_2d_slices) { info.setTwoDMode(true); } // do 2D FFTs
        info.setNumProcs(nprocs);
        cache.solver = std::make_unique<OpenBCSolverT>(domain, info);
        cache.domain = domain;
        cache.twod = is_igf_2d_slices;
        cache.nprocs = nprocs;
    }
    auto & obc_solver = cache.solver;

    auto const& lo = domain.smallEnd();
    amrex::Real const dx = cell_size[0];
    amrex::Real const dy = cell_size[1];
    amrex::Real const dz = cell_size[2];

    auto const build_greens_function = [&] ()
    {
        if (!is_igf_2d_slices){
            // fully 3D solver
            obc_solver->setGreensFunction(
            [=] AMREX_GPU_DEVICE (int i, int j, int k) -> amrex::Real
            {
                int const i0 = i - lo[0];
                int const j0 = j - lo[1];
                int const k0 = k - lo[2];
                amrex::Real const x = i0*dx;
                amrex::Real const y = j0*dy;
                amrex::Real const z = k0*dz;

                return SumOfIntegratedPotential3D(x, y, z, dx, dy, dz);
            });
        }else{
            // 2D sliced solver
            obc_solver->setGreensFunction(
            [=] AMREX_GPU_DEVICE (int i, int j, int k) -> amrex::Real
            {
                int const i0 = i - lo[0];
                int const j0 = j - lo[1];
                amrex::Real const x = i0*dx;
                amrex::Real const y = j0*dy;
                amrex::ignore_unused(k);

                return SumOfIntegratedPotential2D(x, y, dx, dy);
            });

        }
    };

    // Scale factor to apply to phi when a cached Green's function built at a different
    // cell size is reused. The integrated Green's function is homogeneous of degree two.
    amrex::Real scale = amrex::Real(1.);

    if (cache.rebuild_always) {
        // Reference behavior: rebuild the Green's function on every call. The solver may
        // be holding a buffer that belongs to a stored Green's function, which this
        // rebuild overwrites, so drop the store rather than leave it holding data that
        // no longer matches the key it is filed under.
        cache.clearEntries();
        build_greens_function();
    } else {
        // Shape of the grid, compared to within grid_match_tolerance() rather than
        // exactly, so that a caller returning to the same grid is recognized even when
        // the cell size it hands us differs in the last bits.
        amrex::Real const ry = dy / dx;
        amrex::Real const rz = is_igf_2d_slices ? amrex::Real(1.) : dz / dx;

        // Can this Green's function serve a grid of cell size `cell_dx`, and if so at
        // what scale? Reusing across scales rescales phi, which is exact only for powers
        // of two. The 2D Green's function is homogeneous only up to an additive constant,
        // and that constant is observable wherever the potential itself is gathered, so
        // 2D reuses a Green's function at its own scale only.
        auto const usable = [&] (amrex::Real g_ry, amrex::Real g_rz, amrex::Real g_dx,
                                 amrex::Real & scale_out)
        {
            if (!same_grid_ratio(g_ry, ry, cache.tolerance) ||
                !same_grid_ratio(g_rz, rz, cache.tolerance)) { return false; }
            amrex::Real const ratio = dx / g_dx;
            if (same_grid_ratio(ratio, amrex::Real(1.), cache.tolerance)) {
                scale_out = amrex::Real(1.);
                return true;
            }
            amrex::Real snapped = amrex::Real(1.);
            if (!is_igf_2d_slices && is_power_of_two_approx(ratio, cache.tolerance, snapped)) {
                scale_out = snapped;
                return true;
            }
            return false;
        };

        // The guard: the solver may already hold a Green's function we can use, in which
        // case there is nothing to rebuild, restore or even look up. This is by far the
        // most common case and it works whether or not the store is enabled.
        bool const already_loaded = cache.loaded_valid &&
            usable(cache.loaded_ry, cache.loaded_rz, cache.loaded_dx, scale);

        if (already_loaded) {
            ++cache.hits;
            if (cache.loaded >= 0) { cache.entries[cache.loaded].used = ++cache.clock; }
        } else {
            int idx = -1;
            for (int i = 0; i < static_cast<int>(cache.entries.size()); ++i) {
                auto const& e = cache.entries[i];
                if (e.built && usable(e.ry, e.rz, e.dx, scale)) { idx = i; break; }
            }

            if (idx >= 0) {
                ++cache.hits;
                loadEntry(cache, idx);
            } else {
                ++cache.misses;
                scale = amrex::Real(1.);
                if (cache.max_entries > 0) {
                    idx = makeEntry(cache, ry, rz, dx);
                    loadEntry(cache, idx);
                    build_greens_function();    // fills the buffer the entry owns
                    cache.entries[idx].built = true;
                    if (is_igf_2d_slices) {
                        // 2D cannot build through the alias, so take a copy of the result.
                        auto const& g_solver = obc_solver->greensFunctionFFT();
                        cache.entries[idx].g.LocalCopy(g_solver, 0, 0, g_solver.nComp(),
                                                       g_solver.nGrowVect());
                    }
                } else {
                    // Store disabled: the solver keeps the only Green's function, which
                    // the guard above still lets us reuse for as long as it fits.
                    build_greens_function();
                    cache.loaded = -1;
                }
            }

            if (idx >= 0) { cache.entries[idx].used = ++cache.clock; }

            // Remember what the solver now holds, and at which scale it was built.
            cache.loaded_valid = true;
            cache.loaded_ry = ry;
            cache.loaded_rz = rz;
            cache.loaded_dx = (idx >= 0) ? cache.entries[idx].dx : dx;
        }
    }

    obc_solver->solve(phi, rho);

    if (scale != amrex::Real(1.)) {
        // Rescale over the same region solve() wrote, valid cells plus all ghosts.
        amrex::Real const factor = scale * scale;
        amrex::IntVect const ng = phi.nGrowVect();
        int const ncomp = phi.nComp();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(phi, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            amrex::Box const& bx = mfi.growntilebox(ng);
            auto const& phi_arr = phi.array(mfi);
            amrex::ParallelFor(bx, ncomp,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
            {
                phi_arr(i,j,k,n) *= factor;
            });
        }
    }
} // computePhiIGF

} // namespace ablastr::fields
