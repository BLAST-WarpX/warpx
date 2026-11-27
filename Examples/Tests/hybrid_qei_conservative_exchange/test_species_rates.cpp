/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/QeiThermalSupport.H"
#include "Initialization/WarpXInit.H"
#include "Particles/MultiParticleContainer.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Reduce.H>
#include <AMReX_Utility.H>

#include <array>
#include <cmath>
#include <fstream>
#include <map>
#include <memory>
#include <vector>

using namespace amrex::literals;

namespace
{
amrex::Real
kinetic (WarpXParticleContainer& species)
{
    amrex::ReduceOps<amrex::ReduceOpSum> ops;
    amrex::ReduceData<amrex::Real> data(ops);
    using Tuple = typename decltype(data)::Type;
    auto const mass = static_cast<amrex::Real>(species.getMass());
    for (WarpXParIter pti(species, 0); pti.isValid(); ++pti)
    {
        auto const* w = pti.GetAttribs(PIdx::w).dataPtr();
        auto const* ux = pti.GetAttribs(PIdx::ux).dataPtr();
        auto const* uy = pti.GetAttribs(PIdx::uy).dataPtr();
        auto const* uz = pti.GetAttribs(PIdx::uz).dataPtr();
        ops.eval(
            pti.numParticles(), data, [=] AMREX_GPU_DEVICE(int p) -> Tuple
            { return {0.5_rt * mass * w[p] * (ux[p] * ux[p] + uy[p] * uy[p] + uz[p] * uz[p])}; });
    }
    auto result = amrex::get<0>(data.value());
    amrex::ParallelDescriptor::ReduceRealSum(result);
    return result;
}

amrex::Real
electrons (WarpX& simulation)
{
    using warpx::fields::FieldType;
    auto const& te = *simulation.m_fields.get(FieldType::hybrid_electron_temperature_fp, 0);
    auto const& rho = *simulation.m_fields.get(FieldType::rho_fp, 0);
    auto const& geometry = simulation.Geom(0);
    auto const dx = geometry.CellSizeArray();
    auto const hi = amrex::ubound(amrex::surroundingNodes(geometry.Domain()));
    auto const owner = te.OwnerMask(geometry.periodicity());
    amrex::MultiFab energy(te.boxArray(), te.DistributionMap(), 1, 0);
    for (amrex::MFIter mfi(energy); mfi.isValid(); ++mfi)
    {
        auto const out = energy.array(mfi);
        auto const t = te.const_array(mfi), charge = rho.const_array(mfi);
        auto const owned = owner->const_array(mfi);
        amrex::ParallelFor(mfi.validbox(),
                           [=] AMREX_GPU_DEVICE(int i, int j, int k)
                           {
                               amrex::Real volume = MathConst::pi * dx[0] * dx[0] * dx[1] *
                                                    (i == 0 ? 1.0_rt / 3.0_rt : 2.0_rt * i);
                               if (i == hi.x)
                               {
                                   volume *= 0.5_rt;
                               }
                               if (j == 0 || j == hi.y)
                               {
                                   volume *= 0.5_rt;
                               }
                               out(i, j, k) = owned(i, j, k)
                                                  ? 1.5_rt * charge(i, j, k) / PhysConst::q_e *
                                                        PhysConst::kb * t(i, j, k) * volume
                                                  : 0.0_rt;
                           });
    }
    return energy.sum(0);
}
} // namespace

int
main (int argc, char* argv[])
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        bool fallback = false, legacy = false, invalid_rate = false;
        bool composition_conduction = false, ambiguous_conductivity = false;
        bool changed_conduction_signature = false;
        amrex::ParmParse("test").query("fallback", fallback);
        amrex::ParmParse("test").query("legacy", legacy);
        amrex::ParmParse("test").query("invalid_rate", invalid_rate);
        amrex::ParmParse("test").query("composition_conduction", composition_conduction);
        amrex::ParmParse("test").query("ambiguous_conductivity", ambiguous_conductivity);
        amrex::ParmParse("test").query("changed_conduction_signature",
                                       changed_conduction_signature);
        amrex::ParmParse hybrid("hybrid_pic_model");
        if (composition_conduction)
        {
            hybrid.add("electron_heat_conduction", true);
            hybrid.add("electron_conduction_flux_limiter", 0.0_rt);
            hybrid.add("electron_thermal_conductivity(rho,Te,Zbar,Zeff)",
                       std::string("if(Zbar>1.333333333,if(Zbar<1.333333334,"
                                   "if(Zeff>1.499999999,if(Zeff<1.500000001,100,-1),-1),-1),-1)"));
            if (ambiguous_conductivity)
            {
                hybrid.add("electron_thermal_conductivity(rho,Te)", std::string("100"));
            }
        }
        hybrid.add("resolved_qei_support", !legacy);
        hybrid.add("electron_ion_relaxation_rate(rho,Te,Ti,t)",
                   std::string(fallback ? "1.e10" : "0"));
        hybrid.addarr("electron_ion_relaxation_species",
                      fallback ? std::vector<std::string>{"heavy"}
                               : std::vector<std::string>{"ions", "heavy"});
        hybrid.add("electron_ion_relaxation_rate_heavy(rho_s,rho,Te,Ti,t)", std::string("0"));
        if (!fallback)
        {
            // All five arguments must have their declared units/roles. A
            // missing or misordered density/temperature/time argument makes
            // this invalid rather than silently using a constant substitute.
            hybrid.add("electron_ion_relaxation_rate_ions(rho_s,rho,Te,Ti,t)",
                       std::string("if(rho_s/rho>0.49,if(rho_s/rho<0.51,"
                                   "if(Te>70,if(Te<150,if(Ti>=0,if(Ti<80,if(t<1.e-20,1.e10,-1),-1),"
                                   "-1),-1),-1),-1),-1)"));
        }
        if (invalid_rate)
        {
            hybrid.add("electron_ion_relaxation_rate_ions(rho_s,rho,Te,Ti,t)", std::string("-1"));
        }
        auto& simulation = WarpX::GetInstance();
        simulation.InitData();
        simulation.HybridPICPrepareElectronStateForDiagnostics();
        auto& model = *simulation.get_pointer_HybridPICModel();
        auto& particles = simulation.GetPartContainer();
        auto& light = particles.GetParticleContainerFromName("ions");
        auto& heavy = particles.GetParticleContainerFromName("heavy");
        using Velocities = std::array<amrex::Gpu::HostVector<amrex::ParticleReal>, 3>;
        std::map<std::pair<int, int>, Velocities> original;
        for (WarpXParIter pti(heavy, 0); pti.isValid(); ++pti)
        {
            auto& saved = original[{pti.index(), pti.LocalTileIndex()}];
            for (int d = 0; d < 3; ++d)
            {
                auto const& v = pti.GetAttribs(PIdx::ux + d);
                saved[d].resize(v.size());
                amrex::Gpu::copy(amrex::Gpu::deviceToHost, v.begin(), v.end(), saved[d].begin());
            }
        }
        auto const initial_light = kinetic(light);
        auto const initial = initial_light + kinetic(heavy) + electrons(simulation);
        AMREX_ALWAYS_ASSERT(std::isfinite(initial) && initial > 0.0_rt);
        amrex::Print() << "Qei test time=" << simulation.gett_new(0) << '\n';
        {
            auto const& rho = *simulation.m_fields.get(warpx::fields::FieldType::rho_fp, 0);
            auto const& rhos = *simulation.m_fields.get("rho_fp_ions", 0);
            auto const& te = *simulation.m_fields.get(
                warpx::fields::FieldType::hybrid_electron_temperature_fp, 0);
            amrex::MultiFab ratios(rho.boxArray(), rho.DistributionMap(), 1, 0);
            for (amrex::MFIter mfi(ratios); mfi.isValid(); ++mfi)
            {
                auto const out = ratios.array(mfi);
                auto const total = rho.const_array(mfi), species = rhos.const_array(mfi);
                amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k)
                                   { out(i, j, k) = species(i, j, k) / total(i, j, k); });
            }
            amrex::Print() << "Rate inputs: rho_s/rho=" << ratios.min(0) << ".." << ratios.max(0)
                           << " Te_eV=" << te.min(0) * PhysConst::kb / PhysConst::q_e << ".."
                           << te.max(0) * PhysConst::kb / PhysConst::q_e << '\n';
            AMREX_ALWAYS_ASSERT(std::abs(ratios.min(0) - 0.5_rt) < 1.e-13_rt &&
                                std::abs(ratios.max(0) - 0.5_rt) < 1.e-13_rt);
        }
        amrex::Real largest_residual = 0.0_rt;
        for (int step = 0; step < 50; ++step)
        {
            std::map<std::string, std::unique_ptr<amrex::MultiFab>> storage;
            std::map<std::string, amrex::MultiFab*> temperatures;
            for (auto const& name : {std::string("ions"), std::string("heavy")})
            {
                auto& species = particles.GetParticleContainerFromName(name);
                auto moments =
                    warpx::hybrid::depositResolvedQeiMoments(species, 0, simulation.Geom(0));
                auto temperature = std::make_unique<amrex::MultiFab>(
                    moments->boxArray(), moments->DistributionMap(), 1, 0);
                auto const mass = static_cast<amrex::Real>(species.getMass());
                for (amrex::MFIter mfi(*temperature); mfi.isValid(); ++mfi)
                {
                    auto const out = temperature->array(mfi);
                    auto const m = moments->const_array(mfi);
                    amrex::ParallelFor(mfi.validbox(),
                                       [=] AMREX_GPU_DEVICE(int i, int j, int k)
                                       {
                                           out(i, j, k) =
                                               m(i, j, k, 0) > 0
                                                   ? mass * m(i, j, k, 4) /
                                                         (3.0_rt * m(i, j, k, 0) * PhysConst::q_e)
                                                   : 0.0_rt;
                                       });
                }
                temperatures.emplace(name, temperature.get());
                storage.emplace(name, std::move(temperature));
            }
            model.QDSMCAddTemperatureRelaxation(0, 2.e-13_rt, temperatures);
            model.QDSMCApplyIonHeating(0, 2.e-13_rt, nullptr, &temperatures);
            auto const total = electrons(simulation) + kinetic(light) + kinetic(heavy);
            AMREX_ALWAYS_ASSERT(std::isfinite(total));
            largest_residual = std::max(largest_residual, std::abs(total - initial) / initial);
        }
        if (composition_conduction)
        {
            auto const& te = *simulation.m_fields.get(
                warpx::fields::FieldType::hybrid_electron_temperature_fp, 0);
            auto const before_contrast = te.max(0) - te.min(0);
            model.AdvanceElectronHeatConduction(0, 2.e-10_rt);
            auto const& zbar = *simulation.m_fields.get("hybrid_conduction_mean_charge_fp", 0);
            auto const& zeff = *simulation.m_fields.get("hybrid_conduction_effective_charge_fp", 0);
            AMREX_ALWAYS_ASSERT(std::abs(zbar.min(0) - 4.0_rt / 3.0_rt) < 1.e-13_rt &&
                                std::abs(zbar.max(0) - 4.0_rt / 3.0_rt) < 1.e-13_rt &&
                                std::abs(zeff.min(0) - 1.5_rt) < 1.e-13_rt &&
                                std::abs(zeff.max(0) - 1.5_rt) < 1.e-13_rt);
            auto const after_contrast = te.max(0) - te.min(0);
            AMREX_ALWAYS_ASSERT(before_contrast > 0.0_rt &&
                                after_contrast < 0.99_rt * before_contrast);
            auto const total = electrons(simulation) + kinetic(light) + kinetic(heavy);
            AMREX_ALWAYS_ASSERT(std::isfinite(total));
            largest_residual = std::max(largest_residual, std::abs(total - initial) / initial);
            amrex::Print() << "Native composition conduction: contrast ratio="
                           << after_contrast / before_contrast
                           << " energy_residual=" << largest_residual << '\n';
        }
        for (WarpXParIter pti(heavy, 0); pti.isValid(); ++pti)
        {
            auto const& saved = original.at({pti.index(), pti.LocalTileIndex()});
            for (int d = 0; d < 3; ++d)
            {
                auto const& v = pti.GetAttribs(PIdx::ux + d);
                amrex::Gpu::HostVector<amrex::ParticleReal> current(v.size());
                amrex::Gpu::copy(amrex::Gpu::deviceToHost, v.begin(), v.end(), current.begin());
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    current == saved[d],
                    "A species-specific zero Qei rate changed a disabled material's velocities.");
            }
        }
        auto const light_gain = kinetic(light) - initial_light;
        amrex::Print() << "Species Qei rates: fallback=" << fallback << " legacy=" << legacy
                       << " energy_residual=" << largest_residual
                       << " light_gain_fraction=" << light_gain / initial
                       << " heavy_exact_noop=true\n";
        AMREX_ALWAYS_ASSERT(light_gain > 0.001_rt * initial && largest_residual < 1.e-10_rt);
        // Exercise the actual checkpoint contract, including both the global
        // fallback and sorted per-material expressions, without a PIC push.
        if (amrex::ParallelDescriptor::IOProcessor())
        {
            amrex::UtilCreateDirectory("rate_checkpoint", 0755);
        }
        amrex::ParallelDescriptor::Barrier();
        model.WriteMomentHistory("rate_checkpoint");
        if (changed_conduction_signature)
        {
            if (amrex::ParallelDescriptor::IOProcessor())
            {
                std::string record;
                {
                    std::ifstream input("rate_checkpoint/HybridElectronConduction.txt");
                    std::getline(input, record);
                }
                auto const separator = record.find(' ');
                AMREX_ALWAYS_ASSERT(separator != std::string::npos);
                record.replace(0, separator, "ideal_isotropic_lagged_harmonic_v1");
                std::ofstream output("rate_checkpoint/HybridElectronConduction.txt");
                output << record << '\n';
            }
            amrex::ParallelDescriptor::Barrier();
        }
        model.ReadMomentHistory("rate_checkpoint");
    }
    WarpX::ResetInstance();
    warpx::initialization::finalize_external_libraries();
}
