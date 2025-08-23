/* Copyright 2024 David Grote
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "BremsstrahlungFunc.H"

#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

using namespace amrex::literals;

BremsstrahlungFunc::BremsstrahlungFunc (std::string const& collision_name, MultiParticleContainer const * const mypc,
                                        bool isSameSpecies)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!isSameSpecies,
                                     "BremsstrahlungFunc: The two colliding species must be different");

    const amrex::ParmParse pp_collision_name(collision_name);

    // Read in the number of electrons on the target
    int Z;
    pp_collision_name.get("Z", Z);

    std::string product_species_name;
    pp_collision_name.get("product_species", product_species_name);
    auto& product_species = mypc->GetParticleContainerFromName(product_species_name);

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(product_species.AmIA<PhysicalSpecies::photon>(),
                                     "BremsstrahlungFunc: The product species must be photons");

    amrex::ParticleReal multiplier = 1._prt;
    pp_collision_name.query("multiplier", multiplier);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(multiplier >= 1.,
                                     "BremsstrahlungFunc: The multiplier must be greater than or equal to one");
    m_exe.m_multiplier = multiplier;

    amrex::ParticleReal koT1_cut_default = 1.e-4;
    pp_collision_name.query("koT1_cut", koT1_cut_default);
    m_exe.m_koT1_cut_default = koT1_cut_default;

    // Fill in the m_kdsigdk array
    UploadCrossSection(Z);

}

void
BremsstrahlungFunc::UploadCrossSection (int Z)
{

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_kdsigdk_map.count(Z) == 1,
        "Bremsstrahlung cross section not available for Z = " + std::to_string(Z) + "!");

    constexpr auto m_e_eV = PhysConst::m_e*PhysConst::c*PhysConst::c/PhysConst::q_e; // 0.511e6

    nkoT1 = static_cast<int>(m_koT1_grid_h.size());
    nKE = static_cast<int>(m_KEgrid_eV_h.size());

    m_exe.nkoT1 = nkoT1;
    m_exe.nKE = nKE;

    m_sigma_total_h.resize(m_KEgrid_eV_h.size());
    m_kdsigdk_h.resize(m_KEgrid_eV_h.size()*m_koT1_grid_h.size());

    std::vector<std::vector<amrex::ParticleReal>> & kdsigdk = m_kdsigdk_map.at(Z);

    // kdsigdk at the default cutoff
    int const i0_cut = 0;
    amrex::ParticleReal const koT1_cut = m_exe.m_koT1_cut_default;
    amrex::ParticleReal const w00 = (m_koT1_grid_h[i0_cut+1] - koT1_cut)/(m_koT1_grid_h[i0_cut+1] - m_koT1_grid_h[i0_cut]);

    // Convert Seltzer and Berger energy-weighted differential cross section to units of [m^2]
    for (int iee=0; iee < nKE; iee++) {
        amrex::ParticleReal const E = m_KEgrid_eV_h[iee]/m_e_eV;
        amrex::ParticleReal const gamma = 1.0_prt + E;
        /* betaSq = 1.0 - 1.0/gamma/gamma */
        amrex::ParticleReal const betaSq = (E*E + 2._prt*E)/gamma/gamma;
        // 1.0e-31 converts mBarn to m**2
        amrex::ParticleReal const scale_factor = 1.0e-31_prt*Z*Z/betaSq;
        for (int i=0; i < nkoT1; i++) {
            m_kdsigdk_h[iee*nkoT1 + i] = kdsigdk[iee][i]*scale_factor;
        }

        // Calculate the total cross section using the default k-cutoff
        amrex::ParticleReal const kdsigdk_cut = w00*m_kdsigdk_h[iee*nkoT1 + i0_cut] + (1.0_prt - w00)*m_kdsigdk_h[iee*nkoT1 + i0_cut+1];
        amrex::ParticleReal kdsigdk_im1 = kdsigdk_cut;
        amrex::ParticleReal koT1_im1 = koT1_cut;
        amrex::ParticleReal sigma = 0.0_prt;
        for (int i=i0_cut+1; i < nkoT1; i++) {
            amrex::ParticleReal const koT1_i = m_koT1_grid_h[i];
            amrex::ParticleReal const kdsigdk_i = m_kdsigdk_h[iee*nkoT1 + i];
            amrex::ParticleReal const dk = (koT1_i - koT1_im1);
            amrex::ParticleReal const k_ave = (koT1_i + koT1_im1)*0.5_prt;
            amrex::ParticleReal const dsigdk = (kdsigdk_i + kdsigdk_im1)*0.5_prt/k_ave;
            sigma = sigma + dsigdk*dk;
            koT1_im1 = koT1_i;
            kdsigdk_im1 = kdsigdk_i;
        }
        m_sigma_total_h[iee] = sigma;
    }

    // Setup and transfer the data to the GPU device
    m_koT1_grid_d.resize(m_koT1_grid_h.size());
    m_KEgrid_eV_d.resize(m_KEgrid_eV_h.size());
    m_kdsigdk_d.resize(m_kdsigdk_h.size());
    m_sigma_total_d.resize(m_sigma_total_h.size());

    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_koT1_grid_h.begin(), m_koT1_grid_h.end(), m_koT1_grid_d.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_KEgrid_eV_h.begin(), m_KEgrid_eV_h.end(), m_KEgrid_eV_d.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_kdsigdk_h.begin(), m_kdsigdk_h.end(), m_kdsigdk_d.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_sigma_total_h.begin(), m_sigma_total_h.end(), m_sigma_total_d.begin());

    // The kernels access the data via these pointers
    m_exe.m_koT1_grid = m_koT1_grid_d.dataPtr();
    m_exe.m_KEgrid_eV = m_KEgrid_eV_d.dataPtr();
    m_exe.m_kdsigdk = m_kdsigdk_d.dataPtr();
    m_exe.m_sigma_total = m_sigma_total_d.dataPtr();

}
