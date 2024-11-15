/* Copyright 2022 Remi Lehe
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_BOUNDARYSCRAPINGDIAGNOSTICS_H_
#define WARPX_BOUNDARYSCRAPINGDIAGNOSTICS_H_

#include "Diagnostics.H"
#include "Utils/Parser/IntervalsParser.H"

#include <string>

/** collect the particles that are absorbed at the embedded boundary, throughout the simulation
 */
class BoundaryScrapingDiagnostics final : public Diagnostics
{
public:

    /** Constructor
     *
     * @param i index of diagnostics in MultiDiagnostics::alldiags
     * @param name diagnostics name in the inputs file
     */
    BoundaryScrapingDiagnostics (int i, const std::string& name, DiagTypes diag_type);

private:
    /** Read relevant parameters for BoundaryScraping */
    void ReadParameters ();

    /** Determines timesteps at which the particles are written out */
    utils::parser::IntervalsParser m_intervals;

    /** \brief Flush data to file. */
    void Flush (int i_buffer, bool /* force_flush */) override;
    /** \brief Return whether to dump data to file at this time step.
     * (i.e. whether to call Flush)
     *
     * \param[in] step current time step
     * \param[in] i_buffer index of the boundary buffer used (e.g. upper/lower boundary, EB boundary)
     * \param[in] force_flush if true, return true for any step
     */
    bool DoDump (int step, int i_buffer, bool force_flush=false) override;
    /** Return whether to pack field data in output buffers at this time step
     *
     * This is not used for BoundaryScrapingDiagnostics: no field to output
     */
    bool DoComputeAndPack (int step, bool force_flush=false) override;
    /** Initialize buffers that contain field data
     *
     * This is not used for BoundaryScrapingDiagnostics: no field to output
     */
    void InitializeBufferData (int i_buffer, int lev, bool restart=false) override;
    /** Initialize functors that point to the fields requested by the user.
     *
     * This is not used for BoundaryScrapingDiagnostics: no field to output
     */
    void InitializeFieldFunctors (int lev) override;
    /** Initialize the diagnostics by pointing to the `ParticleBoundaryBuffer`
     * that correspond to the species requested by the user.
     */
    void InitializeParticleBuffer () override;

};
#endif // WARPX_BOUNDARYSCRAPINGDIAGNOSTICS_H_
