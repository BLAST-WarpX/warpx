/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: AlexGlock
 * License: BSD-3-Clause-LBNL
 */
#include "ChargeFluxEB.H"

#include "Diagnostics/ReducedDiags/ReducedDiags.H"
#include "EmbeddedBoundary/Enabled.H"
#include "Particles/ParticleBoundaryBuffer.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Utils/TextMsg.H"
#include "Utils/Parser/ParserUtils.H"
#include "WarpX.H"

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Reduce.H>

#include <fstream>
#include <stdexcept>

using namespace amrex;

// constructor
ChargeFluxEB::ChargeFluxEB (const std::string& rd_name)
    : ReducedDiags{rd_name}
{
#if !(defined AMREX_USE_EB)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(false,
        "ChargeFluxEB reduced diagnostics only works when compiling with EB support");
#endif

    if (!EB::enabled()) {
        throw std::runtime_error(
            "ChargeFluxEB reduced diagnostics only works when EBs are enabled at runtime");
    }

    const amrex::ParmParse pp_rd_name(rd_name);

    // optional weighting, same idiom as ChargeOnEB
    std::string buf;
    m_do_parser_weighting = pp_rd_name.query("weighting_function(x,y,z)", buf);
    if (m_do_parser_weighting) {
        std::string weighting_string;
        utils::parser::Store_parserString(
            pp_rd_name, "weighting_function(x,y,z)", weighting_string);
        m_parser_weighting = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(weighting_string, {"x","y","z"}));
    }

    m_eb_boundary_index = AMREX_SPACEDIM*2;

    auto& mypc = WarpX::GetInstance().GetPartContainer();
    m_species_names = mypc.GetSpeciesNames();

    // index 0 = total, then one column per species
    m_data.resize(1 + m_species_names.size(), 0.0_rt);

    if (ParallelDescriptor::IOProcessor())
    {
        if (m_write_header)
        {
            std::ofstream ofs{m_path + m_rd_name + "." + m_extension, std::ofstream::out};
            int c = 0;
            ofs << "#";
            ofs << "[" << c++ << "]step()";
            ofs << m_sep;
            ofs << "[" << c++ << "]time(s)";
            ofs << m_sep;
            ofs << "[" << c++ << "]total_charge_flux(C/s)";
            for (auto const& name : m_species_names) {
                ofs << m_sep;
                ofs << "[" << c++ << "]" << name << "_charge_flux(C/s)";
            }
            ofs << "\n";
            ofs.close();
        }
    }
}
// end constructor

// function that computes the charge flux of particles scraped onto the EB
void ChargeFluxEB::ComputeDiags (const int step)
{
    if (!m_intervals.contains(step+1)) { return; }

    if (!EB::enabled()) {
        throw std::runtime_error("ChargeFluxEB::ComputeDiags only works when EBs are enabled at runtime");
    }

#if defined(AMREX_USE_EB)
    auto& warpx = WarpX::GetInstance();
    ParticleBoundaryBuffer& boundary_buffer = warpx.GetParticleBoundaryBuffer();
    auto& mypc = warpx.GetPartContainer();

    const amrex::Real dt = warpx.getdt(0);
    const int current_step = warpx.getistep(0);

    const bool do_parser_weighting = m_do_parser_weighting;
    auto fun_weightingparser = utils::parser::compileParser<3>(m_parser_weighting.get());

    amrex::Real total_charge_flux = 0.0_rt;

    for (int isp = 0; isp < static_cast<int>(m_species_names.size()); ++isp)
    {
        const std::string& species_name = m_species_names[isp];

        // m_data[0] is reserved for the total; species columns start at 1
        amrex::Real& species_slot = m_data[isp + 1];
        species_slot = 0.0_rt;

        const int n_scraped = boundary_buffer.getNumParticlesInContainer(
            species_name, m_eb_boundary_index, /*local=*/false);
        if (n_scraped == 0) { continue; }

        const WarpXParticleContainer& pc = mypc.GetParticleContainerFromName(species_name);
        const amrex::Real charge = pc.getCharge();

        WarpXParticleContainer::Base& scraped_pc =
            boundary_buffer.getParticleBuffer(species_name, m_eb_boundary_index);

        const int step_scraped_comp = scraped_pc.GetIntCompIndex("stepScraped");

        amrex::Real charge_this_species = 0.0_rt;

        using PIter = amrex::ParConstIterSoA<PIdx::nattribs, 0, amrex::PolymorphicArenaAllocator>;

        for (int lev = 0; lev < scraped_pc.numLevels(); ++lev)
        {
            for (PIter pti(scraped_pc, lev); pti.isValid(); ++pti)
            {
                const auto& soa = pti.GetStructOfArrays();
                const auto* p_w    = soa.GetRealData(PIdx::w).data();
                const auto* p_step = soa.GetIntData(step_scraped_comp).data();

                const auto getPosition = GetParticlePosition<PIdx>(pti);
                const auto np = pti.numParticles();

                amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
                amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
                using ReduceTuple = typename decltype(reduce_data)::Type;

                reduce_op.eval(np, reduce_data,
                    [=] AMREX_GPU_DEVICE (int i) -> ReduceTuple
                    {
                        if (p_step[i] != current_step) { return {0.0_rt}; }

                        amrex::Real w = p_w[i];
                        if (do_parser_weighting) {
                            amrex::ParticleReal xp, yp, zp;
                            getPosition(i, xp, yp, zp);
                            w *= fun_weightingparser(xp, yp, zp);
                        }
                        return { w };
                    });

                charge_this_species += amrex::get<0>(reduce_data.value(reduce_op));
            }
        }
        amrex::ParallelDescriptor::ReduceRealSum(charge_this_species);

        species_slot = charge_this_species * charge / dt;
        total_charge_flux += species_slot;
    }

    m_data[0] = total_charge_flux;
#endif
}
// end void ChargeFluxEB::ComputeDiags
