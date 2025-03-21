/* Copyright 2023 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Arianna Formenti, Yinjian Zhao, Remi Lehe
 * License: BSD-3-Clause-LBNL
 */
#include "DifferentialLuminosity2D.H"

#include "Diagnostics/ReducedDiags/ReducedDiags.H"
#include "Diagnostics/OpenPMDHelpFunction.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/SpeciesPhysicalProperties.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/ParticleUtils.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/WarpXConst.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXProfilerWrapper.H"
#include "WarpX.H"

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
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_Reduce.H>
#include <AMReX_TableData.H>
#include <AMReX_Tuple.H>
#include <AMReX_Vector.H>

#ifdef WARPX_USE_OPENPMD
#   include <openPMD/openPMD.hpp>
#endif

#include <ablastr/warn_manager/WarnManager.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <vector>

using ParticleType = WarpXParticleContainer::ParticleType;
using ParticleTileType = WarpXParticleContainer::ParticleTileType;
using ParticleTileDataType = ParticleTileType::ParticleTileDataType;
using ParticleBins = amrex::DenseBins<ParticleTileDataType>;
using index_type = ParticleBins::index_type;

#ifdef WARPX_USE_OPENPMD
namespace io = openPMD;
#endif

using namespace amrex;

DifferentialLuminosity2D::DifferentialLuminosity2D (const std::string& rd_name)
: ReducedDiags{rd_name}
{
    // RZ coordinate is not supported
#if (defined WARPX_DIM_RZ)
    WARPX_ABORT_WITH_MESSAGE(
        "DifferentialLuminosity2D diagnostics does not work in RZ geometry.");
#endif

    // read colliding species names - must be 2
    amrex::ParmParse pp_rd_name(m_rd_name);
    pp_rd_name.getarr("species", m_beam_name);

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_beam_name.size() == 2u,
        "DifferentialLuminosity2D diagnostics must involve exactly two species");

    pp_rd_name.query("openpmd_backend", m_openpmd_backend);
    pp_rd_name.query("file_min_digits", m_file_min_digits);
    // pick first available backend if default is chosen
    if( m_openpmd_backend == "default" ) {
        m_openpmd_backend = WarpXOpenPMDFileType();
    }
    pp_rd_name.add("openpmd_backend", m_openpmd_backend);

    // read bin parameters for species 1
    int bin_num_1 = 0;
    amrex::Real bin_max_1 = 0.0_rt, bin_min_1 = 0.0_rt;
    utils::parser::getWithParser(pp_rd_name, "bin_number_1", bin_num_1);
    utils::parser::getWithParser(pp_rd_name, "bin_max_1",    bin_max_1);
    utils::parser::getWithParser(pp_rd_name, "bin_min_1",    bin_min_1);
    m_bin_num_1 = bin_num_1;
    m_bin_max_1 = bin_max_1;
    m_bin_min_1 = bin_min_1;
    m_bin_size_1 = (bin_max_1 - bin_min_1) / bin_num_1;

    // read bin parameters for species 2
    int bin_num_2 = 0;
    amrex::Real bin_max_2 = 0.0_rt, bin_min_2 = 0.0_rt;
    utils::parser::getWithParser(pp_rd_name, "bin_number_2", bin_num_2);
    utils::parser::getWithParser(pp_rd_name, "bin_max_2",    bin_max_2);
    utils::parser::getWithParser(pp_rd_name, "bin_min_2",    bin_min_2);
    m_bin_num_2 = bin_num_2;
    m_bin_max_2 = bin_max_2;
    m_bin_min_2 = bin_min_2;
    m_bin_size_2 = (bin_max_2 - bin_min_2) / bin_num_2;

    // resize data array on the host
    Array<int,2> tlo{0,0}; // lower bounds
    Array<int,2> thi{m_bin_num_1-1, m_bin_num_2-1}; // inclusive upper bounds
    m_h_data_2D.resize(tlo, thi, The_Pinned_Arena());

    auto const& h_table_data = m_h_data_2D.table();
    // initialize data on the host
    for (int i = tlo[0]; i <= thi[0]; ++i) {
        for (int j = tlo[1]; j <= thi[1]; ++j) {
            h_table_data(i,j) = 0.0_rt;
        }
    }

    // resize data on the host
    m_d_data_2D.resize(tlo, thi);
    // copy data from host to device
    m_d_data_2D.copy(m_h_data_2D);
    Gpu::streamSynchronize();
} // end constructor

void DifferentialLuminosity2D::ComputeDiags (int step)
{
#if defined(WARPX_DIM_RZ)
    amrex::ignore_unused(step);
#else

    WARPX_PROFILE("DifferentialLuminosity2D::ComputeDiags");

    // Since this diagnostic *accumulates* the luminosity in the
    // table m_d_data_2D, we add contributions at *each timestep*, but
    // we only write the data to file at intervals specified by the user.
    const Real c_sq = PhysConst::c*PhysConst::c;
    const Real c_over_qe = PhysConst::c/PhysConst::q_e;

    // output table data
    auto d_table = m_d_data_2D.table();

    // get a reference to WarpX instance
    auto& warpx = WarpX::GetInstance();
    const Real dt = warpx.getdt(0);
    // get cell volume
    Geometry const & geom = warpx.Geom(0);
    const Real dV = AMREX_D_TERM(geom.CellSize(0), *geom.CellSize(1), *geom.CellSize(2));

    // declare local variables
    auto const num_bins_1 = m_bin_num_1;
    Real const bin_min_1  = m_bin_min_1;
    Real const bin_size_1 = m_bin_size_1;
    auto const num_bins_2 = m_bin_num_2;
    Real const bin_min_2  = m_bin_min_2;
    Real const bin_size_2 = m_bin_size_2;

    // get MultiParticleContainer class object
    const MultiParticleContainer& mypc = warpx.GetPartContainer();

    auto& species_1 = mypc.GetParticleContainerFromName(m_beam_name[0]);
    auto& species_2 = mypc.GetParticleContainerFromName(m_beam_name[1]);

    const ParticleReal m1 = species_1.getMass();
    const ParticleReal m2 = species_2.getMass();

    // Enable tiling
    amrex::MFItInfo info;
    if (amrex::Gpu::notInLaunchRegion()) { info.EnableTiling(WarpXParticleContainer::tile_size); }

    int const nlevs = std::max(0, species_1.finestLevel()+1); // species_1 ?
    for (int lev = 0; lev < nlevs; ++lev) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif

        for (amrex::MFIter mfi = species_1.MakeMFIter(lev, info); mfi.isValid(); ++mfi){

            ParticleTileType& ptile_1 = species_1.ParticlesAt(lev, mfi);
            ParticleTileType& ptile_2 = species_2.ParticlesAt(lev, mfi);

            ParticleBins bins_1 = ParticleUtils::findParticlesInEachCell( lev, mfi, ptile_1 );
            ParticleBins bins_2 = ParticleUtils::findParticlesInEachCell( lev, mfi, ptile_2 );

            // species 1
            const auto soa_1 = ptile_1.getParticleTileData();
            index_type* AMREX_RESTRICT indices_1 = bins_1.permutationPtr();
            index_type const* AMREX_RESTRICT cell_offsets_1 = bins_1.offsetsPtr();

            // extract particle data of species 1 in the current tile/box
            amrex::ParticleReal * const AMREX_RESTRICT w1  = soa_1.m_rdata[PIdx::w];
            amrex::ParticleReal * const AMREX_RESTRICT u1x = soa_1.m_rdata[PIdx::ux]; // u=v*gamma=p/m
            amrex::ParticleReal * const AMREX_RESTRICT u1y = soa_1.m_rdata[PIdx::uy];
            amrex::ParticleReal * const AMREX_RESTRICT u1z = soa_1.m_rdata[PIdx::uz];
            bool const species1_is_photon = species_1.AmIA<PhysicalSpecies::photon>();

            // same for species 2
            const auto soa_2 = ptile_2.getParticleTileData();
            index_type* AMREX_RESTRICT indices_2 = bins_2.permutationPtr();
            index_type const* AMREX_RESTRICT cell_offsets_2 = bins_2.offsetsPtr();

            amrex::ParticleReal * const AMREX_RESTRICT w2  = soa_2.m_rdata[PIdx::w];
            amrex::ParticleReal * const AMREX_RESTRICT u2x = soa_2.m_rdata[PIdx::ux];
            amrex::ParticleReal * const AMREX_RESTRICT u2y = soa_2.m_rdata[PIdx::uy];
            amrex::ParticleReal * const AMREX_RESTRICT u2z = soa_2.m_rdata[PIdx::uz];
            bool const species2_is_photon = species_2.AmIA<PhysicalSpecies::photon>();

            // Extract low-level (cell-level) data
            auto const n_cells = static_cast<int>(bins_1.numBins());

            // Loop over cells
            amrex::ParallelFor( n_cells,
                [=] AMREX_GPU_DEVICE (int i_cell) noexcept
            {

                // The particles from species1 that are in the cell `i_cell` are
                // given by the `indices_1[cell_start_1:cell_stop_1]`
                index_type const cell_start_1 = cell_offsets_1[i_cell];
                index_type const cell_stop_1  = cell_offsets_1[i_cell+1];
                // Same for species 2
                index_type const cell_start_2 = cell_offsets_2[i_cell];
                index_type const cell_stop_2  = cell_offsets_2[i_cell+1];

                for(index_type i_1=cell_start_1; i_1<cell_stop_1; ++i_1){
                    for(index_type i_2=cell_start_2; i_2<cell_stop_2; ++i_2){

                        index_type const j_1 = indices_1[i_1];
                        index_type const j_2 = indices_2[i_2];

                        Real p1t=0, p1x=0, p1y=0, p1z=0; // components of 4-momentum of particle 1
                        Real const u1_sq =  u1x[j_1]*u1x[j_1] + u1y[j_1]*u1y[j_1] + u1z[j_1]*u1z[j_1];
                        if (species1_is_photon) {
                            // photon case (momentum is normalized by m_e in WarpX)
                            p1t = PhysConst::m_e*std::sqrt( u1_sq );
                            p1x = PhysConst::m_e*u1x[j_1];
                            p1y = PhysConst::m_e*u1y[j_1];
                            p1z = PhysConst::m_e*u1z[j_1];
                        } else {
                            p1t = m1*std::sqrt( c_sq + u1_sq );
                            p1x = m1*u1x[j_1];
                            p1y = m1*u1y[j_1];
                            p1z = m1*u1z[j_1];
                        }

                        Real p2t=0, p2x=0, p2y=0, p2z=0; // components of 4-momentum of particle 2
                        Real const u2_sq =  u2x[j_2]*u2x[j_2] + u2y[j_2]*u2y[j_2] + u2z[j_2]*u2z[j_2];
                        if (species2_is_photon) {
                            // photon case (momentum is normalized by m_e in WarpX)
                            p2t = PhysConst::m_e*std::sqrt( u2_sq );
                            p2x = PhysConst::m_e*u2x[j_2];
                            p2y = PhysConst::m_e*u2y[j_2];
                            p2z = PhysConst::m_e*u2z[j_2];
                        } else {
                            p2t = m2*std::sqrt( c_sq + u2_sq );
                            p2x = m2*u2x[j_2];
                            p2y = m2*u2y[j_2];
                            p2z = m2*u2z[j_2];
                        }

                        Real const E_1 = p1t * c_over_qe; // eV
                        Real const E_2 = p2t * c_over_qe; // eV

                        // determine energy bin of particle 1
                        int const bin_1 = int(Math::floor((E_1-bin_min_1)/bin_size_1));
                        if ( bin_1<0 || bin_1>=num_bins_1 ) { continue; } // discard if out-of-range

                        // determine energy bin of particle 2
                        int const bin_2 = int(Math::floor((E_2-bin_min_2)/bin_size_2));
                        if ( bin_2<0 || bin_2>=num_bins_2 ) { continue; } // discard if out-of-range

                        Real const inv_p1t = 1.0_rt/p1t;
                        Real const inv_p2t = 1.0_rt/p2t;

                        Real const beta1_sq = (p1x*p1x + p1y*p1y + p1z*p1z) * inv_p1t*inv_p1t;
                        Real const beta2_sq = (p2x*p2x + p2y*p2y + p2z*p2z) * inv_p2t*inv_p2t;
                        Real const beta1_dot_beta2 = (p1x*p2x + p1y*p2y + p1z*p2z) * inv_p1t*inv_p2t;

                        // Here we use the fact that:
                        // (v1 - v2)^2 = v1^2 + v2^2 - 2 v1.v2
                        // and (v1 x v2)^2 = v1^2 v2^2 - (v1.v2)^2
                        // we also use beta=v/c instead of v
                        Real const radicand = beta1_sq + beta2_sq - 2*beta1_dot_beta2 - beta1_sq*beta2_sq + beta1_dot_beta2*beta1_dot_beta2;

                        Real const d2L_dE1_dE2 = PhysConst::c * std::sqrt( radicand ) * w1[j_1] * w2[j_2] / (dV * bin_size_1 * bin_size_2) * dt; // m^-2 eV^-2

                        amrex::Real &data = d_table(bin_1, bin_2);
                        amrex::HostDevice::Atomic::Add(&data, d2L_dE1_dE2);

                    } // particles species 2
                } // particles species 1
            }); // cells
        } // boxes
    } // levels

    // Only write to file at intervals specified by the user.
    // At these intervals, the data needs to ready on the CPU,
    // so we copy it from the GPU to the CPU and reduce across MPI ranks.
    if (m_intervals.contains(step+1)) {

        // Copy data from GPU memory
        m_h_data_2D.copy(m_d_data_2D);
        Gpu::streamSynchronize();

        // reduced sum over mpi ranks
        const int size = static_cast<int> (m_d_data_2D.size());
        ParallelDescriptor::ReduceRealSum
                (m_h_data_2D.table().p, size, ParallelDescriptor::IOProcessorNumber());
    }

    // Return for all that are not IO processor
    if ( !ParallelDescriptor::IOProcessor() ) { return; }

#endif // not RZ
} // end void DifferentialLuminosity2D::ComputeDiags

void DifferentialLuminosity2D::WriteToFile (int step) const
{
     // Judge if the diags should be done at this step
    if (!m_intervals.contains(step+1)) { return; }

#ifdef WARPX_USE_OPENPMD
    // only IO processor writes
    if ( !ParallelDescriptor::IOProcessor() ) { return; }

    // TODO: support different filename templates
    std::string filename = "openpmd";
    // TODO: support also group-based encoding
    const std::string fileSuffix = std::string("_%0") + std::to_string(m_file_min_digits) + std::string("T");
    filename = filename.append(fileSuffix).append(".").append(m_openpmd_backend);

    // transform paths for Windows
    #ifdef _WIN32
        const std::string filepath = openPMD::auxiliary::replace_all(
            m_path + m_rd_name + "/" + filename, "/", "\\");
    #else
        const std::string filepath = m_path + m_rd_name + "/" + filename;
    #endif

    // Create the OpenPMD series
    auto series = io::Series(
            filepath,
            io::Access::CREATE);
    auto i = series.iterations[step + 1];
    // record
    auto f_mesh = i.meshes["d2L_dE1_dE2"]; // m^-2 eV^-2
    f_mesh.setUnitDimension({
                            {io::UnitDimension::L, -6},
                            {io::UnitDimension::M, -2},
                            {io::UnitDimension::T,  4}
                            });

    // record components
    auto data = f_mesh[io::RecordComponent::SCALAR];

    // meta data
    f_mesh.setAxisLabels({"E2", "E1"}); // eV, eV
    std::vector< double > const& gridGlobalOffset = {m_bin_min_2, m_bin_min_1};
    f_mesh.setGridGlobalOffset(gridGlobalOffset);
    f_mesh.setGridSpacing<amrex::Real>({m_bin_size_2, m_bin_size_1});

    data.setPosition<amrex::Real>({0.5, 0.5});

    auto dataset = io::Dataset(
            io::determineDatatype<double>(),
            {static_cast<unsigned long>(m_bin_num_2), static_cast<unsigned long>(m_bin_num_1)});
    data.resetDataset(dataset);

    // Get time at level 0
    auto & warpx = WarpX::GetInstance();
    auto const time = warpx.gett_new(0);
    i.setTime(time);

    auto const& h_table_data = m_h_data_2D.table();
    data.storeChunkRaw(
            h_table_data.p,
            {0, 0},
            {static_cast<unsigned long>(m_bin_num_2), static_cast<unsigned long>(m_bin_num_1)});

    series.flush();
    i.close();
    series.close();
#else
    amrex::ignore_unused(step);
    WARPX_ABORT_WITH_MESSAGE("DifferentialLuminosity2D: Needs openPMD-api compiled into WarpX, but was not found!");
#endif
}
