#ifndef WARPX_PARTICLEREDUCTIONFUNCTOR_H_
#define WARPX_PARTICLEREDUCTIONFUNCTOR_H_

#include "ComputeDiagFunctor.H"

#include <AMReX_Parser.H>
#include <AMReX_BaseFwd.H>

#include <memory>
#include <string>

/**
 * \brief Functor to calculate per-cell averages of particle properties.
 */
class ParticleReductionFunctor final : public ComputeDiagFunctor
{
public:
    /** Constructor.
     * \param[in] mf_src source multifab. Must be nullptr as no source MF is needed
     *            to compute the number of particles per cell, banane.
     * \param[in] lev level of multifab.
     * \param[in] crse_ratio for interpolating field values from simulation MultiFabs
                  to the output diagnostic MultiFab mf_dst.
     * \param[in] fn_str parser string that describes the function to apply to particles
     *            and then average over each cell
     * \param[in] ispec index of the species over which to calculate the average
     * \param[in] do_average Whether to do an average or a sum of the function
     * \param[in] do_filter Whether to apply a filter function to particles before averaging
     * \param[in] filter_str Parser string for filter function to apply before averaging
     * \param[in] ncomp Number of component of mf_src to cell-center in dst multifab.
     */
    ParticleReductionFunctor(const amrex::MultiFab * mf_src, int lev,
                       amrex::IntVect crse_ratio, const std::string& fn_str,
                       int ispec, bool do_average,
                       bool do_filter, const std::string& filter_str,
                       int ncomp=1);

    /** \brief Compute the average of the function m_map_fn over each grid cell.
     *
     * \param[out] mf_dst output MultiFab where the result is written
     * \param[in] dcomp first component of mf_dst in which cell-centered
     *            data is stored
     */
    void operator()(amrex::MultiFab& mf_dst, int dcomp, int /*i_buffer=0*/) const override;
private:
    int const m_lev; /**< level on which mf_src is defined */
    int const m_ispec; /**< index of species to average over */
    bool const m_do_average; /**< Whether to calculate the average of the data */
    bool const m_do_filter; /**< whether to apply #m_filter_fn */
    /** Parser function to be averaged by the functor. Arguments: x, y, z, ux, uy, uz */
    std::unique_ptr<amrex::Parser> m_map_fn_parser;
    /** Parser function to filter particles before pass to map. Arguments: x, y, z, ux, uy, uz */
    std::unique_ptr<amrex::Parser> m_filter_fn_parser;
    /** Compiled #m_map_fn_parser */
    amrex::ParserExecutor<6> m_map_fn;
    /** Compiled #m_filter_fn_parser */
    amrex::ParserExecutor<6> m_filter_fn;
};

#endif // WARPX_PARTICLEREDUCTIONFUNCTOR_H_
