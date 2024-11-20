#ifndef WARPX_RHOFUNCTOR_H_
#define WARPX_RHOFUNCTOR_H_

#include "ComputeDiagFunctor.H"

#include <AMReX_BaseFwd.H>

/**
 * \brief Functor to compute charge density rho into mf_out
 */
class RhoFunctor final : public ComputeDiagFunctor
{

public:

    /**
     * \brief Constructor
     *
     * \param[in] lev level of MultiFab
     * \param[in] crse_ratio coarsening ratio for interpolation of field values
     *                       from simulation MultiFabs to the output MultiFab mf_dst
     * \param[in] species_index Index of species to dump rho per species. This argument
     *                          is optional and defaults to -1 (dump total rho)
     * \param[in] convertRZmodes2cartesian if true, all RZ modes are averaged into one component
     * \param[in] ncomp optional number of component of source MultiFab mf_src
     *                  to be cell-centered in output MultiFab mf_dst
     */
    RhoFunctor (int lev,
                amrex::IntVect crse_ratio,
                bool apply_rz_psatd_filter = false,
                int species_index = -1,
                bool convertRZmodes2cartesian = true,
                int ncomp = 1);

    /**
     * \brief Compute rho directly into mf_dst
     *
     * \param[out] mf_dst output MultiFab where the result is written
     * \param[in]  dcomp  first component of mf_dst in which cell-centered data are stored
     */
    void operator() ( amrex::MultiFab& mf_dst, int dcomp, int /*i_buffer=0*/ ) const override;

private:

    // Level on which source MultiFab mf_src is defined in RZ geometry
    int const m_lev;

    // Whether to apply k-space filtering of charge density in the diagnostics output in RZ PSATD
    bool m_apply_rz_psatd_filter;

    // Species index to dump rho per species
    const int m_species_index;

    // Whether to average all modes into one component in RZ geometry
    bool m_convertRZmodes2cartesian;
};

#endif // WARPX_RHOFUNCTOR_H_
