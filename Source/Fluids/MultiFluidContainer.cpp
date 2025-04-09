/* Copyright 2023 Grant Johnson, Remi Lehe
 *
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "MultiFluidContainer.H"
#include "Fluids/WarpXFluidContainer.H"
#include "Utils/Parser/ParserUtils.H"

#include <string>

using namespace amrex;

MultiFluidContainer::MultiFluidContainer ()
{
    const ParmParse pp_fluids("fluids");
    pp_fluids.queryarr("species_names", species_names);

    const int nspecies = static_cast<int>(species_names.size());

    allcontainers.resize(nspecies);
    for (int i = 0; i < nspecies; ++i) {
        allcontainers[i] = std::make_unique<WarpXFluidContainer>(i, species_names[i]);
    }
}

void
MultiFluidContainer::AllocateLevelMFs (ablastr::fields::MultiFabRegister& m_fields, const BoxArray& ba, const DistributionMapping& dm, int lev)
{
    for (auto& fl : allcontainers) {
        fl->AllocateLevelMFs(m_fields, ba, dm, lev);
    }
}

void
MultiFluidContainer::InitData (ablastr::fields::MultiFabRegister& m_fields, amrex::Box init_box, amrex::Real cur_time, int lev)
{
    for (auto& fl : allcontainers) {
        fl->InitData(m_fields, init_box, cur_time, lev);
    }
}


void
MultiFluidContainer::DepositCharge (ablastr::fields::MultiFabRegister& m_fields, amrex::MultiFab &rho, int lev)
{
    for (auto& fl : allcontainers) {
        fl->DepositCharge(m_fields,rho,lev);
    }
}

void
MultiFluidContainer::DepositCurrent (ablastr::fields::MultiFabRegister& m_fields,
    amrex::MultiFab& jx, amrex::MultiFab& jy, amrex::MultiFab& jz, int lev)
{
    for (auto& fl : allcontainers) {
        fl->DepositCurrent(m_fields,jx,jy,jz,lev);
    }
}

void
MultiFluidContainer::Evolve (ablastr::fields::MultiFabRegister& fields,
                            int lev,
                            std::string const& current_fp_string,
                            amrex::Real cur_time,
                            bool skip_deposition)
{
    for (auto& fl : allcontainers) {
        fl->Evolve(fields, lev, current_fp_string, cur_time, skip_deposition);
    }
}
