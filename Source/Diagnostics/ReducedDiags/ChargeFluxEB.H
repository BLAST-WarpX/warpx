#ifndef WARPX_DIAGNOSTICS_REDUCEDDIAGS_CHARGEFLUXEB_H_
#define WARPX_DIAGNOSTICS_REDUCEDDIAGS_CHARGEFLUXEB_H_

#include "ReducedDiags.H"

#include <AMReX_Parser.H>
#include <AMReX_REAL.H>

#include <memory>
#include <string>
#include <vector>

class ChargeFluxEB : public ReducedDiags
{
public:

    explicit ChargeFluxEB (const std::string& rd_name);

    void ComputeDiags (const int step) final;

private:

    // all simulation species, cached at construction; index 0 of m_data
    // is the summed total, indices [1, N] map 1:1 to m_species_names
    std::vector<std::string> m_species_names;

    bool m_do_parser_weighting = false;
    std::unique_ptr<amrex::Parser> m_parser_weighting;

    int m_eb_boundary_index = -1;
};

#endif