/* Copyright 2022 David Grote
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "HardEdgedPlasmaLens.H"
#include "Utils/Parser/ParserUtils.H"

#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>

#include <string>

HardEdgedPlasmaLens::HardEdgedPlasmaLens ()
    : LatticeElementBase("plasmalens")
{
}

void
HardEdgedPlasmaLens::AddElement (amrex::ParmParse & pp_element, amrex::ParticleReal & z_location)
{
    using namespace amrex::literals;

    AddElementBase(pp_element, z_location);

    amrex::ParticleReal dEdx = 0._prt;
    amrex::ParticleReal dBdx = 0._prt;
    utils::parser::queryWithParser(pp_element, "dEdx", dEdx);
    utils::parser::queryWithParser(pp_element, "dBdx", dBdx);

    h_dEdx.push_back(dEdx);
    h_dBdx.push_back(dBdx);
}

void
HardEdgedPlasmaLens::WriteToDevice ()
{
    WriteToDeviceBase();

    d_dEdx.resize(h_dEdx.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_dEdx.begin(), h_dEdx.end(), d_dEdx.begin());
    d_dBdx.resize(h_dBdx.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_dBdx.begin(), h_dBdx.end(), d_dBdx.begin());
}

HardEdgedPlasmaLensDevice
HardEdgedPlasmaLens::GetDeviceInstance () const
{
    HardEdgedPlasmaLensDevice result;
    result.InitHardEdgedPlasmaLensDevice(*this);
    return result;
}

void
HardEdgedPlasmaLensDevice::InitHardEdgedPlasmaLensDevice (HardEdgedPlasmaLens const& h_plasmalens)
{

    nelements = h_plasmalens.nelements;

    if (nelements == 0) { return; }

    d_zs_arr = h_plasmalens.d_zs.data();
    d_ze_arr = h_plasmalens.d_ze.data();

    d_dEdx_arr = h_plasmalens.d_dEdx.data();
    d_dBdx_arr = h_plasmalens.d_dBdx.data();

}
