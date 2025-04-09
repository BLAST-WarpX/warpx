/* Copyright 2019-2020 Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "ParticleExtrema.H"

#include "Diagnostics/ReducedDiags/ReducedDiags.H"
#if (defined WARPX_QED)
#   include "Particles/ElementaryProcess/QEDInternals/QedChiFunctions.H"
#endif
#include "Fields.H"
#include "Particles/Gather/FieldGather.H"
#include "Particles/Gather/GetExternalFields.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/SpeciesPhysicalProperties.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_Algorithm.H>
#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_Dim3.H>
#include <AMReX_Extension.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_FabArray.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IndexType.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>
#include <AMReX_PODVector.H>
#include <AMReX_ParIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParticleReduce.H>
#include <AMReX_Particles.H>
#include <AMReX_REAL.H>
#include <AMReX_Reduce.H>
#include <AMReX_Tuple.H>
#include <AMReX_Vector.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <map>
#include <vector>

using namespace amrex::literals;
using warpx::fields::FieldType;

// constructor
ParticleExtrema::ParticleExtrema (const std::string& rd_name)
: ReducedDiags{rd_name}
{
    // read species name
    const amrex::ParmParse pp_rd_name(rd_name);
    pp_rd_name.get("species",m_species_name);

    // get WarpX class object
    auto & warpx = WarpX::GetInstance();

    // get MultiParticleContainer class object
    auto & mypc = warpx.GetPartContainer();

    // get number of species (int)
    const auto nSpecies = mypc.nSpecies();

    // get species names (std::vector<std::string>)
    const auto species_names = mypc.GetSpeciesNames();

    // loop over species
    for (int i_s = 0; i_s < nSpecies; ++i_s)
    {
        // only chosen species does
        if (species_names[i_s] != m_species_name) { continue; }

        // get WarpXParticleContainer class object
        auto & myspc = mypc.GetParticleContainer(i_s);

        auto all_diag_names = std::vector<std::string> {};
        auto add_diag = [&,c=0] (
            const std::string& name, const std::string& header) mutable {
            m_headers_indices[name] = aux_header_index{header, c++};
            all_diag_names.push_back(name);
        };

        add_diag("xmin", "xmin(m)");
        add_diag("xmax", "xmax(m)");
        add_diag("ymin", "ymin(m)");
        add_diag("ymax", "ymax(m)");
        add_diag("zmin", "zmin(m)");
        add_diag("zmax", "zmax(m)");
        add_diag("pxmin", "pxmin(kg*m/s)");
        add_diag("pxmax", "pxmax(kg*m/s)");
        add_diag("pymin", "pymin(kg*m/s)");
        add_diag("pymax", "pymax(kg*m/s)");
        add_diag("pzmin", "pzmin(kg*m/s)");
        add_diag("pzmax", "pzmax(kg*m/s)");
        add_diag("gmin", "gmin()");
        add_diag("gmax", "gmax()");

#if (defined WARPX_DIM_3D)
        add_diag("wmin", "wmin()");
        add_diag("wmax", "wmax()");
#elif (defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ))
        add_diag("wmin", "wmin(1/m)");
        add_diag("wmax", "wmax(1/m)");
#else
        add_diag("wmin", "wmin(1/m^2)");
        add_diag("wmax", "wmax(1/m^2)");
#endif
        if (myspc.DoQED()){
            add_diag("chimin", "chimin()");
            add_diag("chimax", "chimax()");
        }

        m_data.resize(all_diag_names.size());

        if (amrex::ParallelDescriptor::IOProcessor())
        {
            if ( m_write_header )
            {
                // open file
                std::ofstream ofs;
                ofs.open(m_path + m_rd_name + "." + m_extension,
                    std::ofstream::out | std::ofstream::app);
                // write header row
                int off = 0;
                ofs << "#";
                ofs << "[" << off++ << "]step()";
                ofs << m_sep;
                ofs << "[" << off++ << "]time(s)";
                for (const auto& name : all_diag_names){
                    const auto& el = m_headers_indices[name];
                    ofs << m_sep << "[" << el.idx + off << "]" << el.header;
                }
                ofs << "\n";
                // close file
                ofs.close();
            }
        }
    }
}
// end constructor

// function that computes extrema
void ParticleExtrema::ComputeDiags (int step)
{
    // Judge if the diags should be done
    if (!m_intervals.contains(step+1)) { return; }

    // get MultiParticleContainer class object
    auto & mypc = WarpX::GetInstance().GetPartContainer();

    // get number of species (int)
    const auto nSpecies = mypc.nSpecies();

    // get species names (std::vector<std::string>)
    const auto species_names = mypc.GetSpeciesNames();

    // inverse of speed of light squared
    amrex::Real constexpr inv_c2 = 1.0_rt / (PhysConst::c * PhysConst::c);

    // loop over species
    for (int i_s = 0; i_s < nSpecies; ++i_s)
    {
        // only chosen species does
        if (species_names[i_s] != m_species_name) { continue; }

        // get WarpXParticleContainer class object
        auto & myspc = mypc.GetParticleContainer(i_s);

        // get mass (Real)
        auto m = myspc.getMass();
        auto is_photon = myspc.AmIA<PhysicalSpecies::photon>();
        if ( is_photon ) {
            m = PhysConst::m_e;
        }

        using PType = typename WarpXParticleContainer::SuperParticleType;
        using OpMin = amrex::ReduceOpMin;
        using OpMax = amrex::ReduceOpMax;

        amrex::ReduceOps<OpMin, OpMin, OpMin, OpMin,
                         OpMax, OpMax, OpMax, OpMax> reduce_ops;
        auto posminmax = amrex::ParticleReduce<amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real,
                                                                 amrex::Real, amrex::Real, amrex::Real, amrex::Real>>(
            myspc,
            [=] AMREX_GPU_DEVICE(const PType& p) noexcept -> amrex::GpuTuple<amrex::Real, amrex::Real, amrex::Real, amrex::Real,
                                                                             amrex::Real, amrex::Real, amrex::Real, amrex::Real>
            {
                amrex::ParticleReal x, y, z;
                get_particle_position(p, x, y, z);
                amrex::Real const w = p.rdata(PIdx::w);
                return {w, x, y, z, w, x, y, z};
            },
            reduce_ops);

        amrex::Real wmin = amrex::get<0>(posminmax);
        amrex::Real xmin = amrex::get<1>(posminmax);
        amrex::Real ymin = amrex::get<2>(posminmax);
        amrex::Real zmin = amrex::get<3>(posminmax);
        amrex::Real wmax = amrex::get<4>(posminmax);
        amrex::Real xmax = amrex::get<5>(posminmax);
        amrex::Real ymax = amrex::get<6>(posminmax);
        amrex::Real zmax = amrex::get<7>(posminmax);

        amrex::Real const gfactor = (is_photon ? 0._rt : 1._rt);
        auto uminmax = amrex::ParticleReduce<amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real,
                                                               amrex::Real, amrex::Real, amrex::Real, amrex::Real>>(
            myspc,
            [=] AMREX_GPU_DEVICE(const PType& p) noexcept -> amrex::GpuTuple<amrex::Real, amrex::Real, amrex::Real, amrex::Real,
                                                                             amrex::Real, amrex::Real, amrex::Real, amrex::Real>
            {
                amrex::Real const ux = p.rdata(PIdx::ux);
                amrex::Real const uy = p.rdata(PIdx::uy);
                amrex::Real const uz = p.rdata(PIdx::uz);
                amrex::Real const g = std::sqrt(gfactor + (ux*ux + uy*uy + uz*uz)*inv_c2);
                return {g, ux, uy, uz, g, ux, uy, uz};
            },
            reduce_ops);

        amrex::Real gmin = amrex::get<0>(uminmax);
        amrex::Real uxmin = amrex::get<1>(uminmax);
        amrex::Real uymin = amrex::get<2>(uminmax);
        amrex::Real uzmin = amrex::get<3>(uminmax);
        amrex::Real gmax = amrex::get<4>(uminmax);
        amrex::Real uxmax = amrex::get<5>(uminmax);
        amrex::Real uymax = amrex::get<6>(uminmax);
        amrex::Real uzmax = amrex::get<7>(uminmax);

        amrex::ParallelDescriptor::ReduceRealMin({xmin,ymin,zmin,uxmin,uymin,uzmin,gmin,wmin});
        amrex::ParallelDescriptor::ReduceRealMax({xmax,ymax,zmax,uxmax,uymax,uzmax,gmax,wmax});

#if (defined WARPX_QED)
        // get number of level (int)
        const auto level_number = WarpX::GetInstance().finestLevel();

        // compute chimin and chimax
        amrex::Real chimin_f = 0.0_rt;
        amrex::Real chimax_f = 0.0_rt;

        if (myspc.DoQED())
        {
            // declare chi arrays
            std::vector<amrex::Real> chimin, chimax;
            chimin.resize(level_number+1,0.0_rt);
            chimax.resize(level_number+1,0.0_rt);

            // define variables in preparation for field gathering
            auto & warpx = WarpX::GetInstance();
            const int n_rz_azimuthal_modes = WarpX::n_rz_azimuthal_modes;
            const int nox = WarpX::nox;
            const bool galerkin_interpolation = WarpX::galerkin_interpolation;
            const amrex::IntVect ngEB = warpx.getngEB();

            using ablastr::fields::Direction;

            // loop over refinement levels
            for (int lev = 0; lev <= level_number; ++lev)
            {
                // define variables in preparation for field gathering
                const amrex::XDim3 dinv = WarpX::InvCellSize(std::max(lev, 0));

                const amrex::MultiFab & Ex = *warpx.m_fields.get(FieldType::Efield_aux, Direction{0}, lev);
                const amrex::MultiFab & Ey = *warpx.m_fields.get(FieldType::Efield_aux, Direction{1}, lev);
                const amrex::MultiFab & Ez = *warpx.m_fields.get(FieldType::Efield_aux, Direction{2}, lev);
                const amrex::MultiFab & Bx = *warpx.m_fields.get(FieldType::Bfield_aux, Direction{0}, lev);
                const amrex::MultiFab & By = *warpx.m_fields.get(FieldType::Bfield_aux, Direction{1}, lev);
                const amrex::MultiFab & Bz = *warpx.m_fields.get(FieldType::Bfield_aux, Direction{2}, lev);

                // declare reduce_op
                amrex::ReduceOps<amrex::ReduceOpMin, amrex::ReduceOpMax> reduce_op;
                amrex::ReduceData<amrex::Real, amrex::Real> reduce_data(reduce_op);
                using ReduceTuple = typename decltype(reduce_data)::Type;

                // Loop over boxes
                for (WarpXParIter pti(myspc, lev); pti.isValid(); ++pti)
                {
                    const auto GetPosition = GetParticlePosition<PIdx>(pti);
                    // get particle arrays
                    amrex::ParticleReal* const AMREX_RESTRICT ux = pti.GetAttribs()[PIdx::ux].dataPtr();
                    amrex::ParticleReal* const AMREX_RESTRICT uy = pti.GetAttribs()[PIdx::uy].dataPtr();
                    amrex::ParticleReal* const AMREX_RESTRICT uz = pti.GetAttribs()[PIdx::uz].dataPtr();
                    // declare external fields
                    const int offset = 0;
                    const auto getExternalEB = GetExternalEBField(pti, offset);
                    const amrex::ParticleReal Ex_external_particle = myspc.m_E_external_particle[0];
                    const amrex::ParticleReal Ey_external_particle = myspc.m_E_external_particle[1];
                    const amrex::ParticleReal Ez_external_particle = myspc.m_E_external_particle[2];
                    const amrex::ParticleReal Bx_external_particle = myspc.m_B_external_particle[0];
                    const amrex::ParticleReal By_external_particle = myspc.m_B_external_particle[1];
                    const amrex::ParticleReal Bz_external_particle = myspc.m_B_external_particle[2];

                    // define variables in preparation for field gathering
                    amrex::Box box = pti.tilebox();
                    box.grow(ngEB);
                    const amrex::Dim3 lo = amrex::lbound(box);
                    const amrex::XDim3 xyzmin = WarpX::LowerCorner(box, lev, 0._rt);
                    const auto& ex_arr = Ex[pti].array();
                    const auto& ey_arr = Ey[pti].array();
                    const auto& ez_arr = Ez[pti].array();
                    const auto& bx_arr = Bx[pti].array();
                    const auto& by_arr = By[pti].array();
                    const auto& bz_arr = Bz[pti].array();
                    const amrex::IndexType ex_type = Ex[pti].box().ixType();
                    const amrex::IndexType ey_type = Ey[pti].box().ixType();
                    const amrex::IndexType ez_type = Ez[pti].box().ixType();
                    const amrex::IndexType bx_type = Bx[pti].box().ixType();
                    const amrex::IndexType by_type = By[pti].box().ixType();
                    const amrex::IndexType bz_type = Bz[pti].box().ixType();

                    // evaluate reduce_op
                    reduce_op.eval(pti.numParticles(), reduce_data,
                    [=] AMREX_GPU_DEVICE (int i) -> ReduceTuple
                    {
                        // get external fields
                        amrex::ParticleReal xp, yp, zp;
                        GetPosition(i, xp, yp, zp);
                        amrex::ParticleReal ex = Ex_external_particle;
                        amrex::ParticleReal ey = Ey_external_particle;
                        amrex::ParticleReal ez = Ez_external_particle;
                        amrex::ParticleReal bx = Bx_external_particle;
                        amrex::ParticleReal by = By_external_particle;
                        amrex::ParticleReal bz = Bz_external_particle;

                        getExternalEB(i, ex, ey, ez, bx, by, bz);

                        // gather E and B
                        doGatherShapeN(xp, yp, zp,
                            ex, ey, ez, bx, by, bz,
                            ex_arr, ey_arr, ez_arr, bx_arr, by_arr, bz_arr,
                            ex_type, ey_type, ez_type,
                            bx_type, by_type, bz_type,
                            dinv, xyzmin, lo,
                            n_rz_azimuthal_modes, nox, galerkin_interpolation);
                        // compute chi
                        amrex::Real chi = 0.0_rt;
                        if ( is_photon ) {
                            chi = QedUtils::chi_photon(ux[i]*m, uy[i]*m, uz[i]*m,
                                             ex, ey, ez, bx, by, bz);
                        } else {
                            chi = QedUtils::chi_ele_pos(ux[i]*m, uy[i]*m, uz[i]*m,
                                             ex, ey, ez, bx, by, bz);
                        }
                        return {chi,chi};
                    });
                }
                auto val = reduce_data.value();
                chimin[lev] = amrex::get<0>(val);
                chimax[lev] = amrex::get<1>(val);
            }
            chimin_f = *std::min_element(chimin.begin(), chimin.end());
            chimax_f = *std::max_element(chimax.begin(), chimax.end());
            amrex::ParallelDescriptor::ReduceRealMin(chimin_f, amrex::ParallelDescriptor::IOProcessorNumber());
            amrex::ParallelDescriptor::ReduceRealMax(chimax_f, amrex::ParallelDescriptor::IOProcessorNumber());
        }
#endif

        const auto get_idx = [&](const std::string& name){
            return m_headers_indices.at(name).idx;
        };

        m_data[get_idx("xmin")]  = xmin;
        m_data[get_idx("xmax")]  = xmax;
        m_data[get_idx("ymin")]  = ymin;
        m_data[get_idx("ymax")]  = ymax;
        m_data[get_idx("zmin")]  = zmin;
        m_data[get_idx("zmax")]  = zmax;
        m_data[get_idx("pxmin")]  = uxmin*m;
        m_data[get_idx("pxmax")]  = uxmax*m;
        m_data[get_idx("pymin")]  = uymin*m;
        m_data[get_idx("pymax")]  = uymax*m;
        m_data[get_idx("pzmin")] = uzmin*m;
        m_data[get_idx("pzmax")] = uzmax*m;
        m_data[get_idx("gmin")] = gmin;
        m_data[get_idx("gmax")] = gmax;
        m_data[get_idx("wmin")] = wmin;
        m_data[get_idx("wmax")] = wmax;
#if (defined WARPX_QED)
        if (myspc.DoQED())
        {
            m_data[get_idx("chimin")] = chimin_f;
            m_data[get_idx("chimax")] = chimax_f;
        }
#endif
    }
    // end loop over species
}
// end void ParticleEnergy::ComputeDiags
