#ifndef WARPX_PRESSURETENSORFUNCTOR_H_
#define WARPX_PRESSURETENSORFUNCTOR_H_

#include "ComputeDiagFunctor.H"

#include <AMReX_BaseFwd.H>

#include <string>
#include <vector>

/**
 * \brief Functor to calculate per-cell pressure tensor from particle data.
 *
 * Computes the 6 independent components of the symmetric pressure tensor
 * `P_ij = n * mass * <(u_i - <u_i>)(u_j - <u_j>)>` [Pa]
 * where u = gamma*v, n is the number density, and the average is weighted
 * over the macroparticles in each cell (nearest-grid-point assignment).
 * In RZ, RCYLINDER and RSPHERE, u is expressed in the local curvilinear basis.
 *
 * Output components (in order): xx, xy, xz, yy, yz, zz, with the direction
 * labels of the geometry (see ComponentNames()).
 */
class PressureTensorFunctor final : public ComputeDiagFunctor
{
public:
    /** \brief Constructor
     * \param[in] lev level of multifab
     * \param[in] crse_ratio for interpolating field values from simulation MultiFabs
                  to the output diagnostic MultiFab mf_dst
     * \param[in] ispec index of the species over which to calculate the pressure tensor
     * \param[in] ncomp Number of components (must be 6 for the symmetric tensor)
     */
    PressureTensorFunctor(int lev, amrex::IntVect crse_ratio, int ispec, int ncomp=6);

    /** \brief Compute the pressure tensor in each grid cell.
     *
     * \param[out] mf_dst output MultiFab where the result is written
     * \param[in] dcomp first component of mf_dst in which cell-centered
     *            data is stored
     */
    void operator()(amrex::MultiFab& mf_dst, int dcomp, int /*i_buffer=0*/) const override;

    /** \brief Names of the 6 output components, in output order, without the species suffix
     *
     * Pxx, Pxy, Pxz, Pyy, Pyz, Pzz in Cartesian geometries,
     * Prr, Prt, Prz, Ptt, Ptz, Pzz in RZ and RCYLINDER, and
     * Prr, Prt, Prp, Ptt, Ptp, Ppp in RSPHERE.
     */
    static std::vector<std::string> ComponentNames ();
private:
    int m_lev; /**< level on which mf_src is defined */
    int m_ispec; /**< index of species to average over */
};

#endif // WARPX_PRESSURETENSORFUNCTOR_H_
