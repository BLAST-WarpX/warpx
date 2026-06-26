#ifndef WARPX_PHIFUNCTOR_H_
#define WARPX_PHIFUNCTOR_H_

#include "ComputeDiagFunctor.H"

#include <AMReX_BaseFwd.H>

/**
 * \brief Functor to compute electrostatic potential into mf_out
 */
class PhiFunctor final : public ComputeDiagFunctor
{

public:

    /**
     * \brief Constructor
     *
     * \param[in] lev level of MultiFab
     * \param[in] crse_ratio coarsening ratio for interpolation of field values
     *                       from simulation MultiFabs to the output MultiFab mf_dst
     * \param[in] convertRZmodes2cartesian if true, all RZ modes are averaged into one component
     * \param[in] ncomp optional number of component of source MultiFab mf_src
     *                  to be cell-centered in output MultiFab mf_dst
     */
    PhiFunctor (int lev,
                amrex::IntVect crse_ratio,
                bool convertRZmodes2cartesian = true,
                int ncomp = 1);

    /**
     * \brief Compute phi directly into mf_dst
     *
     * \param[out] mf_dst output MultiFab where the result is written
     * \param[in]  dcomp  first component of mf_dst in which cell-centered data are stored
     */
    void operator() ( amrex::MultiFab& mf_dst, int dcomp, int /*i_buffer=0*/ ) const override;

private:

    // Level on which source MultiFab mf_src is defined
    int m_lev;

    // Whether to average all modes into one component in RZ geometry
    bool m_convertRZmodes2cartesian;
};

#endif // WARPX_PHIFUNCTOR_H_
