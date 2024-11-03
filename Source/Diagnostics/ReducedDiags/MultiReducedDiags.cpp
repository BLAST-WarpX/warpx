/* Copyright 2019-2020 Maxence Thevenet, Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "MultiReducedDiags.H"

#include "BeamRelevant.H"
#include "ChargeOnEB.H"
#include "ColliderRelevant.H"
#include "DifferentialLuminosity.H"
#include "FieldEnergy.H"
#include "FieldMaximum.H"
#include "FieldMomentum.H"
#include "FieldProbe.H"
#include "FieldReduction.H"
#include "LoadBalanceCosts.H"
#include "LoadBalanceEfficiency.H"
#include "ParticleEnergy.H"
#include "ParticleExtrema.H"
#include "ParticleHistogram.H"
#include "ParticleHistogram2D.H"
#include "ParticleMomentum.H"
#include "ParticleNumber.H"
#include "RhoMaximum.H"
#include "Timestep.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXProfilerWrapper.H"

#include <AMReX.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <functional>
#include <iterator>
#include <map>

using namespace amrex;

// constructor
MultiReducedDiags::MultiReducedDiags ()
{
    // read reduced diags names
    const ParmParse pp_warpx("warpx");
    m_plot_rd = pp_warpx.queryarr("reduced_diags_names", m_rd_names);

    // if names are not given, reduced diags will not be done
    if ( m_plot_rd == 0 ) { return; }

    using CS = const std::string& ;
    const auto reduced_diags_dictionary =
        std::map<std::string, std::function<std::unique_ptr<ReducedDiags>(CS)>>{
            {"BeamRelevant",          [](CS s){return std::make_unique<BeamRelevant>(s);}},
            {"ChargeOnEB",            [](CS s){return std::make_unique<ChargeOnEB>(s);}},
            {"ColliderRelevant",      [](CS s){return std::make_unique<ColliderRelevant>(s);}},
            {"DifferentialLuminosity",[](CS s){return std::make_unique<DifferentialLuminosity>(s);}},
            {"ParticleEnergy",        [](CS s){return std::make_unique<ParticleEnergy>(s);}},
            {"ParticleExtrema",       [](CS s){return std::make_unique<ParticleExtrema>(s);}},
            {"ParticleHistogram",     [](CS s){return std::make_unique<ParticleHistogram>(s);}},
            {"ParticleHistogram2D",   [](CS s){return std::make_unique<ParticleHistogram2D>(s);}},
            {"ParticleMomentum",      [](CS s){return std::make_unique<ParticleMomentum>(s);}},
            {"ParticleNumber",        [](CS s){return std::make_unique<ParticleNumber>(s);}},
            {"FieldEnergy",           [](CS s){return std::make_unique<FieldEnergy>(s);}},
            {"FieldMaximum",          [](CS s){return std::make_unique<FieldMaximum>(s);}},
            {"FieldMomentum",         [](CS s){return std::make_unique<FieldMomentum>(s);}},
            {"FieldProbe",            [](CS s){return std::make_unique<FieldProbe>(s);}},
            {"FieldReduction",        [](CS s){return std::make_unique<FieldReduction>(s);}},
            {"LoadBalanceCosts",      [](CS s){return std::make_unique<LoadBalanceCosts>(s);}},
            {"LoadBalanceEfficiency", [](CS s){return std::make_unique<LoadBalanceEfficiency>(s);}},
            {"RhoMaximum",            [](CS s){return std::make_unique<RhoMaximum>(s);}},
            {"Timestep",              [](CS s){return std::make_unique<Timestep>(s);}}
    };
    // loop over all reduced diags and fill m_multi_rd with requested reduced diags
    std::transform(m_rd_names.begin(), m_rd_names.end(), std::back_inserter(m_multi_rd),
        [&](const auto& rd_name){
            const ParmParse pp_rd_name(rd_name);

            // read reduced diags type
            std::string rd_type;
            pp_rd_name.get("type", rd_type);

            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                reduced_diags_dictionary.count(rd_type) != 0,
                rd_type + " is not a valid type for reduced diagnostic " + rd_name
            );

            return reduced_diags_dictionary.at(rd_type)(rd_name);
        });
    // end loop over all reduced diags
}
// end constructor

void MultiReducedDiags::InitData ()
{
    // loop over all reduced diags
    for (int i_rd = 0; i_rd < static_cast<int>(m_rd_names.size()); ++i_rd)
    {
        m_multi_rd[i_rd] -> InitData();
    }
}

void MultiReducedDiags::LoadBalance () {
    // loop over all reduced diags
    for (int i_rd = 0; i_rd < static_cast<int>(m_rd_names.size()); ++i_rd)
    {
        m_multi_rd[i_rd] -> LoadBalance();
    }
}

// call functions to compute diags
void MultiReducedDiags::ComputeDiags (int step)
{
    WARPX_PROFILE("MultiReducedDiags::ComputeDiags()");

    // loop over all reduced diags
    for (int i_rd = 0; i_rd < static_cast<int>(m_rd_names.size()); ++i_rd)
    {
        m_multi_rd[i_rd] -> ComputeDiags(step);
    }
    // end loop over all reduced diags
}
// end void MultiReducedDiags::ComputeDiags

// function to write data
void MultiReducedDiags::WriteToFile (int step)
{
    // Only the I/O rank does
    if ( !ParallelDescriptor::IOProcessor() ) { return; }

    // loop over all reduced diags
    for (int i_rd = 0; i_rd < static_cast<int>(m_rd_names.size()); ++i_rd)
    {
        // Judge if the diags should be done
        if (!m_multi_rd[i_rd]->m_intervals.contains(step+1)) { continue; }

        // call the write to file function
        m_multi_rd[i_rd]->WriteToFile(step);
    }
    // end loop over all reduced diags
}
// end void MultiReducedDiags::WriteToFile
