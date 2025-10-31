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
#include "DifferentialLuminosity2D.H"
#include "FieldEnergy.H"
#include "FieldMaximum.H"
#include "FieldMomentum.H"
#include "FieldPoyntingFlux.H"
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
MultiReducedDiags::MultiReducedDiags (WarpX* warpx)
{
    // read reduced diags names
    const ParmParse pp_warpx("warpx");
    m_plot_rd = pp_warpx.queryarr("reduced_diags_names", m_rd_names);

    // if names are not given, reduced diags will not be done
    if ( m_plot_rd == 0 ) { return; }

    using CS = const std::string& ;
    const auto reduced_diags_dictionary =
        std::map<std::string, std::function<std::unique_ptr<ReducedDiags>(WarpX*,CS)>>{
            {"BeamRelevant",          [](WarpX* w, CS s){return std::make_unique<BeamRelevant>(w, s);}},
            {"ChargeOnEB",            [](WarpX* w, CS s){return std::make_unique<ChargeOnEB>(w, s);}},
            {"ColliderRelevant",      [](WarpX* w, CS s){return std::make_unique<ColliderRelevant>(w, s);}},
            {"DifferentialLuminosity",[](WarpX* w, CS s){return std::make_unique<DifferentialLuminosity>(w, s);}},
            {"DifferentialLuminosity2D",[](WarpX* w, CS s){return std::make_unique<DifferentialLuminosity2D>(w, s);}},
            {"ParticleEnergy",        [](WarpX* w, CS s){return std::make_unique<ParticleEnergy>(w, s);}},
            {"ParticleExtrema",       [](WarpX* w, CS s){return std::make_unique<ParticleExtrema>(w, s);}},
            {"ParticleHistogram",     [](WarpX* w, CS s){return std::make_unique<ParticleHistogram>(w, s);}},
            {"ParticleHistogram2D",   [](WarpX* w, CS s){return std::make_unique<ParticleHistogram2D>(w, s);}},
            {"ParticleMomentum",      [](WarpX* w, CS s){return std::make_unique<ParticleMomentum>(w, s);}},
            {"ParticleNumber",        [](WarpX* w, CS s){return std::make_unique<ParticleNumber>(w, s);}},
            {"FieldEnergy",           [](WarpX* w, CS s){return std::make_unique<FieldEnergy>(w, s);}},
            {"FieldMaximum",          [](WarpX* w, CS s){return std::make_unique<FieldMaximum>(w, s);}},
            {"FieldMomentum",         [](WarpX* w, CS s){return std::make_unique<FieldMomentum>(w, s);}},
            {"FieldPoyntingFlux",     [](WarpX* w, CS s){return std::make_unique<FieldPoyntingFlux>(w, s);}},
            {"FieldProbe",            [](WarpX* w, CS s){return std::make_unique<FieldProbe>(w, s);}},
            {"FieldReduction",        [](WarpX* w, CS s){return std::make_unique<FieldReduction>(w, s);}},
            {"LoadBalanceCosts",      [](WarpX* w, CS s){return std::make_unique<LoadBalanceCosts>(w, s);}},
            {"LoadBalanceEfficiency", [](WarpX* w, CS s){return std::make_unique<LoadBalanceEfficiency>(w, s);}},
            {"RhoMaximum",            [](WarpX* w, CS s){return std::make_unique<RhoMaximum>(w, s);}},
            {"Timestep",              [](WarpX* w, CS s){return std::make_unique<Timestep>(w, s);}}
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

            return reduced_diags_dictionary.at(rd_type)(warpx,rd_name);
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

// call functions to compute diags at the mid step time level
void MultiReducedDiags::ComputeDiagsMidStep (int step)
{
    WARPX_PROFILE("MultiReducedDiags::ComputeDiagsMidStep()");

    // loop over all reduced diags
    for (int i_rd = 0; i_rd < static_cast<int>(m_rd_names.size()); ++i_rd)
    {
        m_multi_rd[i_rd] -> ComputeDiagsMidStep(step);
    }
    // end loop over all reduced diags
}
// end void MultiReducedDiags::ComputeDiagsMidStep

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

// Check if any diagnostics will be done
bool MultiReducedDiags::DoDiags(int step)
{
    bool result = false;
    for (int i_rd = 0; i_rd < static_cast<int>(m_rd_names.size()); ++i_rd)
    {
        result = result || m_multi_rd[i_rd] -> DoDiags(step);
    }
    return result;
}
// end bool MultiReducedDiags::DoDiags

void MultiReducedDiags::WriteCheckpointData (std::string const & dir)
{
    // Only the I/O rank does
    if ( !ParallelDescriptor::IOProcessor() ) { return; }

    // loop over all reduced diags
    for (int i_rd = 0; i_rd < static_cast<int>(m_rd_names.size()); ++i_rd)
    {
        m_multi_rd[i_rd]->WriteCheckpointData(dir);
    }
    // end loop over all reduced diags
}

void MultiReducedDiags::ReadCheckpointData (std::string const & dir)
{
    // loop over all reduced diags
    for (int i_rd = 0; i_rd < static_cast<int>(m_rd_names.size()); ++i_rd)
    {
        m_multi_rd[i_rd]->ReadCheckpointData(dir);
    }
    // end loop over all reduced diags
}
