/* Copyright 2023 David Grote and David Bizzozero
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "SolenoidRF.H"
#include "Utils/Parser/ParserUtils.H"

#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>

#include <string>

SolenoidRF::SolenoidRF ()
    : LatticeElementBase("solenoidrf")
{
}

void
SolenoidRF::AddElement (amrex::ParmParse & pp_element, amrex::ParticleReal & z_location)
{
    using namespace amrex::literals;

    AddElementBase(pp_element, z_location);

    amrex::ParticleReal scale = 0._prt;
    amrex::ParticleReal freq = 0._prt;
    amrex::ParticleReal theta = 0._prt;
    amrex::Vector<amrex::ParticleReal> e_coef = {};
    amrex::Vector<amrex::ParticleReal> b_coef = {};

    utils::parser::queryWithParser(pp_element, "scale", scale);
    utils::parser::queryWithParser(pp_element, "freq", freq);
    utils::parser::queryWithParser(pp_element, "theta", theta);
    utils::parser::queryArrWithParser(pp_element, "e_coef", e_coef);

    h_scale.push_back(scale);
    h_freq.push_back(freq);
    h_theta.push_back(theta);
    h_e_coef.push_back(e_coef);

}

void
SolenoidRF::WriteToDevice ()
{
    WriteToDeviceBase();

    d_scale.resize(h_scale.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_scale.begin(), h_scale.end(), d_scale.begin());
    d_freq.resize(h_freq.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_freq.begin(), h_freq.end(), d_freq.begin());
    d_theta.resize(h_theta.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_theta.begin(), h_theta.end(), d_theta.begin());

    d_e_coef.resize(h_e_coef.size());
    for (auto i = 0lu; i < h_e_coef.size(); ++i) {
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_e_coef[i].begin(), h_e_coef[i].end(), d_e_coef[i].begin());
    }

}

SolenoidRFDevice
SolenoidRF::GetDeviceInstance () const
{
    SolenoidRFDevice result;
    result.InitSolenoidRFDevice(*this);
    return result;
}

void
SolenoidRFDevice::InitSolenoidRFDevice (SolenoidRF const& h_solenoidrf)
{

    nelements = h_solenoidrf.nelements;

    if (nelements == 0) return;

    // remember pointers to device data in simple data structures
    d_zs_arr = h_solenoidrf.d_zs.data();
    d_ze_arr = h_solenoidrf.d_ze.data();
    d_scale_arr = h_solenoidrf.d_scale.data();
    d_freq_arr = h_solenoidrf.d_freq.data();
    d_theta_arr = h_solenoidrf.d_theta.data();
    for (auto i = 0lu; i < nelements; ++i) {
        d_num_e_coef[i] = h_solenoidrf.d_e_coef[i].size();
        d_e_coef_arr[i] = h_solenoidrf.d_e_coef[i].data();
    }

}
