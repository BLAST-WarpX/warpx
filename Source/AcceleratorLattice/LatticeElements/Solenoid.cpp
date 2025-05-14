/* Copyright 2023 David Grote and David Bizzozero
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "Solenoid.H"
#include "Utils/Parser/ParserUtils.H"

#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>

#include <string>

Solenoid::Solenoid ()
    : LatticeElementBase("solenoid")
{
}

void
Solenoid::AddElement (amrex::ParmParse & pp_element, amrex::ParticleReal & z_location)
{
    using namespace amrex::literals;

    AddElementBase(pp_element, z_location);

    amrex::ParticleReal scale = 0._prt;
    amrex::Vector<amrex::ParticleReal> b_coef = {};

    utils::parser::queryWithParser(pp_element, "scale", scale);
    utils::parser::queryArrWithParser(pp_element, "b_coef", b_coef);

    h_scale.push_back(scale);
    h_b_coef.push_back(b_coef);

}

void
Solenoid::WriteToDevice ()
{
    WriteToDeviceBase();

    d_scale.resize(h_scale.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_scale.begin(), h_scale.end(), d_scale.begin());

    d_b_coef.resize(h_b_coef.size());
    for (auto i = 0lu; i < h_b_coef.size(); ++i) {
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_b_coef[i].begin(), h_b_coef[i].end(), d_b_coef[i].begin());
    }

}

SolenoidDevice
Solenoid::GetDeviceInstance () const
{
    SolenoidDevice result;
    result.InitSolenoidDevice(*this);
    return result;
}

void
SolenoidDevice::InitSolenoidDevice (Solenoid const& h_solenoid)
{

    nelements = h_solenoid.nelements;

    if (nelements == 0) return;

    // remember pointers to device data in simple data structures
    d_zs_arr = h_solenoid.d_zs.data();
    d_ze_arr = h_solenoid.d_ze.data();
    d_scale_arr = h_solenoid.d_scale.data();
    for (auto i = 0lu; i < nelements; ++i) {
        d_num_b_coef[i] = h_solenoid.d_b_coef[i].size();
        d_b_coef_arr[i] = h_solenoid.d_b_coef[i].data();
    }

}
