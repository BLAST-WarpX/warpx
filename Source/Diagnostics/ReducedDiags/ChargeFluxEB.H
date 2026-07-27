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

    // matches species specific diags
    // int m_selected_species_id = -1;
    std::vector<std::string> m_selected_species_names;

    // matches ChargeOnEB naming
    bool m_do_parser_weighting = false;
    std::unique_ptr<amrex::Parser> m_parser_weighting;

    int m_eb_boundary_index = -1;
};

#endif