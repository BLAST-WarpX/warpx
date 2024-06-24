#ifndef WARPX_TEMPERATUREFUNCTOR_H_
#define WARPX_TEMPERATUREFUNCTOR_H_

#include "ComputeDiagFunctor.H"

#include <AMReX_BaseFwd.H>

/**
 * \brief Functor to calculate per-cell averages of particle temperature
 */
class TemperatureFunctor final : public ComputeDiagFunctor
{
public:
    /** \brief Constructor
     * \param[in] lev level of multifab
     * \param[in] crse_ratio for interpolating field values from simulation MultiFabs
                  to the output diagnostic MultiFab mf_dst
     * \param[in] ispec index of the species over which to calculate the temperature
     * \param[in] ncomp Number of component of mf_src to cell-center in dst multifab
     */
    TemperatureFunctor(int lev, amrex::IntVect crse_ratio, int ispec, int ncomp=1);

    /** \brief Compute the temperature in each grid cell.
     *
     * \param[out] mf_dst output MultiFab where the result is written
     * \param[in] dcomp first component of mf_dst in which cell-centered
     *            data is stored
     */
    void operator()(amrex::MultiFab& mf_dst, int dcomp, int /*i_buffer=0*/) const override;
private:
    int const m_lev; /**< level on which mf_src is defined */
    int const m_ispec; /**< index of species to average over */
};

#endif // WARPX_TEMPERATUREFUNCTOR_H_
