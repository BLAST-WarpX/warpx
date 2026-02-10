/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Author: Justin Angus (LLNL)
 *
 * License: BSD-3-Clause-LBNL
 */
#include "PulsedIonization.H"

#include "Particles/Collision/BinaryCollision/BinaryCollisionUtils.H"
#include "Particles/ParticleCreation/SmartCopy.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"
#include "Utils/ParticleUtils.H"
#include "Utils/WarpXProfilerWrapper.H"
#include "WarpX.H"

#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include <AMReX_ParticleTile.H>

#include <string>

PulsedIonization::PulsedIonization (std::string const& collision_name, MultiParticleContainer const * mypc)
    : CollisionBase(collision_name)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_species_names.size() == 1,
                                     "Pulsed Ionization must have exactly one species.");

    const amrex::ParmParse pp_collision_name(collision_name);

    // Get the product electron and ion species
    pp_collision_name.queryarr("product_species", m_product_species);

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE( m_product_species.size() == 2,
        "PulsedIonization: product_species size must be equal to two");

    auto& product_species_0 = mypc->GetParticleContainerFromName(m_product_species[0]);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(product_species_0.AmIA<PhysicalSpecies::electron>(),
        "PulsedIonization: The first product species must be an electron");

    // Get the product electron and ion particle weight
    pp_collision_name.get("fixed_product_weight", m_fixed_product_weight);

    // Parse the direction-dependent electron temperature
    amrex::Vector<amrex::ParticleReal> Te_tmp;
    pp_collision_name.getarr("electron_temperature_eV", Te_tmp);

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE( Te_tmp.size() == 3,
        "PulsedIonizationFunc: electron_temperature_eV must have exactly 3 values");

    for (int i = 0; i < 3; ++i) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE( Te_tmp[i] >= 0.0,
            "PulsedIonizationFunc: electron_temperature_eV must be greater than or equal to zero");
    }

    // Set the direction-dependent electron thermal speed
    const amrex::ParticleReal Vtex = std::sqrt(PhysConst::q_e*Te_tmp[0]/product_species_0.getMass());
    const amrex::ParticleReal Vtey = std::sqrt(PhysConst::q_e*Te_tmp[1]/product_species_0.getMass());
    const amrex::ParticleReal Vtez = std::sqrt(PhysConst::q_e*Te_tmp[2]/product_species_0.getMass());
    m_electron_thermal_speed = {Vtex, Vtey, Vtez};

    // Parse the direction-dependent electron drift velocity [m/s]
    amrex::Vector<amrex::ParticleReal> Vd_tmp;
    pp_collision_name.getarr("electron_drift_velocity", Vd_tmp);

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE( Vd_tmp.size() == 3,
        "PulsedIonizationFunc: electron_drift_velocity must have exactly 3 values");

    m_electron_drift_velocity = {Vd_tmp[0], Vd_tmp[1], Vd_tmp[2]};

    // Parse the ionization rate
    std::string ionization_rate_str;
    utils::parser::Store_parserString(pp_collision_name, "ionization_rate(x,y,z,t)", ionization_rate_str);
    m_ionization_rate_parser = utils::parser::makeParser(ionization_rate_str, {"x", "y", "z", "t"});

    // Compile the ionization rate parser
    m_ionization_rate_func = m_ionization_rate_parser.compile<4>();

}

void
PulsedIonization::doCollisions (amrex::Real cur_time, amrex::Real dt, MultiParticleContainer* mypc)
{
    WARPX_PROFILE("PulsedIonization::doCollisions()");

    using namespace amrex::literals;

    using ParticleTileType = WarpXParticleContainer::ParticleTileType;
    using ParticleTileDataType = ParticleTileType::ParticleTileDataType;
    using ParticleBins = amrex::DenseBins<ParticleTileDataType>;
    using index_type = ParticleBins::index_type;
    using SoaData_type = typename WarpXParticleContainer::ParticleTileType::ParticleTileDataType;

    // Get handles to species particle containters
    auto& species1 = mypc->GetParticleContainerFromName(m_species_names[0]);
    auto& product_ele = mypc->GetParticleContainerFromName(m_product_species[0]);
    auto& product_ion = mypc->GetParticleContainerFromName(m_product_species[1]);

    const SmartCopyFactory copy_factory_ele(species1, product_ele);
    const SmartCopyFactory copy_factory_ion(species1, product_ion);
    const SmartCopy CopyEle = copy_factory_ele.getSmartCopy();
    const SmartCopy CopyIon = copy_factory_ion.getSmartCopy();

#ifdef AMREX_USE_GPU
    amrex::Gpu::DeviceScalar<SmartCopy> d_CopyEle(CopyEle);
    amrex::Gpu::DeviceScalar<SmartCopy> d_CopyIon(CopyIon);

    SmartCopy const* AMREX_RESTRICT CopyElePtr = d_CopyEle.dataPtr();
    SmartCopy const* AMREX_RESTRICT CopyIonPtr = d_CopyIon.dataPtr();
#else
    SmartCopy const* AMREX_RESTRICT CopyElePtr = &CopyEle;
    SmartCopy const* AMREX_RESTRICT CopyIonPtr = &CopyIon;
#endif

    // get parsers for the ionization rate
    auto nu_func = m_ionization_rate_func;

    const amrex::ParticleReal fixed_product_weight = m_fixed_product_weight;
    const amrex::ParticleReal electron_Vdrift_x = m_electron_drift_velocity[0];
    const amrex::ParticleReal electron_Vdrift_y = m_electron_drift_velocity[1];
    const amrex::ParticleReal electron_Vdrift_z = m_electron_drift_velocity[2];
    const amrex::ParticleReal electron_Vtherm_x = m_electron_thermal_speed[0];
    const amrex::ParticleReal electron_Vtherm_y = m_electron_thermal_speed[1];
    const amrex::ParticleReal electron_Vtherm_z = m_electron_thermal_speed[2];

    // Loop over refinement levels
    const int flvl = species1.finestLevel();
    for (int lev = 0; lev <= flvl; ++lev) {

        // Enable tiling
        amrex::MFItInfo info;
        if (amrex::Gpu::notInLaunchRegion()) { info.EnableTiling(species1.tile_size); }

#ifdef AMREX_USE_OMP
        info.SetDynamic(true);
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi = species1.MakeMFIter(lev, info); mfi.isValid(); ++mfi){

            // Extract species 1 particles in the grid tile that `mfi` points to
            ParticleTileType& ptile_1 = species1.ParticlesAt(lev, mfi);

            // Find the particles that are in each cell of this tile
            amrex::Geometry const& geom_lev = WarpX::GetInstance().Geom(lev);
            ParticleBins bins_1 = ParticleUtils::findParticlesInEachCell( geom_lev, mfi, ptile_1 );

            // Compute/store total weight of species 1 in each cell
            auto soa_1 = ptile_1.getParticleTileData();
            const int n_cells = static_cast<int>(bins_1.numBins());
            const auto np1 = ptile_1.numParticles();
            amrex::Gpu::DeviceVector<amrex::ParticleReal> wtot1_vec(n_cells, 0.0_prt);
            index_type const* AMREX_RESTRICT bins_1_ptr = bins_1.binsPtr();
            amrex::ParticleReal* AMREX_RESTRICT wtot1_in_each_cell = wtot1_vec.dataPtr();
            amrex::ParticleReal* AMREX_RESTRICT w1 = soa_1.m_rdata[PIdx::w];

            amrex::ParallelFor( np1,
                [=] AMREX_GPU_DEVICE (int ip) noexcept
                {
                    amrex::Gpu::Atomic::AddNoRet(&wtot1_in_each_cell[bins_1_ptr[ip]],
                                                 w1[ip]);
                }
            );

            // Create DeviceVector to store the number of products to create per cell
            amrex::Gpu::DeviceVector<index_type> num_products_vec(n_cells, 0);
            index_type* AMREX_RESTRICT p_counts = num_products_vec.dataPtr();

            amrex::ParallelForRNG( n_cells,
                [=] AMREX_GPU_DEVICE (int i_cell, amrex::RandomEngine const& engine) noexcept
                {
                    const amrex::ParticleReal wtot1 = wtot1_in_each_cell[i_cell];
                    if (wtot1 == 0.0_prt) { return; }

                    // Compute total weight of products to create in this cell
                    const amrex::ParticleReal nu_izn = nu_func(0.,0.,0.,cur_time);
                    const amrex::ParticleReal total_product = wtot1*(1.0 - std::exp(-nu_izn*dt));

                    // Compute number of products macro particles to create in this cell
                    const amrex::ParticleReal num_expected = total_product/fixed_product_weight;
                    int num_macro_particles = static_cast<int>(std::floor(num_expected));
                    const amrex::Real rand = amrex::Random(engine);
                    if (rand < (num_expected - num_macro_particles)) {
                        num_macro_particles++;
                    }

                    // Do not permit total product weight to exceed wtot1
                    if (num_macro_particles*fixed_product_weight > wtot1) {
                        num_macro_particles--;
                    }

                    p_counts[i_cell] += num_macro_particles;
                }
            );

            // Compute offset array and allocate memory for the produced species
            amrex::Gpu::DeviceVector<index_type> offsets(n_cells);
            const index_type total_new = amrex::Scan::ExclusiveSum(n_cells, num_products_vec.dataPtr(), offsets.dataPtr());

            // Resize product species arrays
            ParticleTileType& ptile_ele = product_ele.ParticlesAt(lev, mfi);
            ParticleTileType& ptile_ion = product_ion.ParticlesAt(lev, mfi);
            const index_type old_np_ele = ptile_ele.numParticles();
            const index_type old_np_ion = ptile_ion.numParticles();
            ptile_ele.resize(old_np_ele + total_new);
            ptile_ion.resize(old_np_ion + total_new);

            // Host-side SoA handles for the product species
            SoaData_type soa_product_ele = ptile_ele.getParticleTileData();
            SoaData_type soa_product_ion = ptile_ion.getParticleTileData();

#ifdef AMREX_USE_GPU
            // Make device copies so the kernel sees device-resident handles
            amrex::Gpu::DeviceScalar<SoaData_type> d_soa_ele(soa_product_ele);
            amrex::Gpu::DeviceScalar<SoaData_type> d_soa_ion(soa_product_ion);

            SoaData_type const* AMREX_RESTRICT soa_ele_ptr = d_soa_ele.dataPtr();
            SoaData_type const* AMREX_RESTRICT soa_ion_ptr = d_soa_ion.dataPtr();
#else
            SoaData_type const* AMREX_RESTRICT soa_ele_ptr = &soa_product_ele;
            SoaData_type const* AMREX_RESTRICT soa_ion_ptr = &soa_product_ion;
#endif

            const index_type* AMREX_RESTRICT offsets_ptr = offsets.dataPtr();
            const index_type* AMREX_RESTRICT nprod_cell_ptr = num_products_vec.dataPtr();

            index_type const* AMREX_RESTRICT cell_offsets_1 = bins_1.offsetsPtr();
            index_type const* AMREX_RESTRICT indices_1 = bins_1.permutationPtr();

            uint64_t* AMREX_RESTRICT idcpu1 = soa_1.m_idcpu;

            amrex::ParticleReal* AMREX_RESTRICT uxe  = soa_product_ele.m_rdata[PIdx::ux];
            amrex::ParticleReal* AMREX_RESTRICT uye  = soa_product_ele.m_rdata[PIdx::uy];
            amrex::ParticleReal* AMREX_RESTRICT uze  = soa_product_ele.m_rdata[PIdx::uz];

            amrex::ParallelForRNG( n_cells,
                [=] AMREX_GPU_DEVICE (int i_cell, amrex::RandomEngine const& engine) noexcept
                {
                    const index_type num_products_in_cell = nprod_cell_ptr[i_cell];
                    if (num_products_in_cell == 0) { return; }

                    const index_type start = offsets_ptr[i_cell];

                    auto& ele = *soa_ele_ptr;
                    auto& ion = *soa_ion_ptr;

                    auto const& CopyIonF = *CopyIonPtr;
                    auto const& CopyEleF = *CopyElePtr;

                    // The particles from species1 that are in the cell `i_cell` are
                    // given by the `indices_1[cell_start_1:cell_stop_1]`
                    const index_type cell_start_1 = cell_offsets_1[i_cell];
                    const index_type cell_stop_1  = cell_offsets_1[i_cell+1];
                    const index_type num_in_cell = cell_stop_1 - cell_start_1;

                    const amrex::ParticleReal wpEI = fixed_product_weight;

                    for (index_type j = 0; j < num_products_in_cell; ++j) {

                        const index_type new_idx = start + j;
                        const index_type i_ele = old_np_ele + new_idx;
                        const index_type i_ion = old_np_ion + new_idx;

                        // Get a random particle from species 1 in this cell
                        const index_type k = static_cast<index_type>(amrex::Random(engine) * amrex::Real(num_in_cell));
                        index_type idx = cell_start_1 + k;

                        // Probe until a valid particle is found (should always be at least one)
                        index_type ip = -1;
                        for (index_type t = 0; t < num_in_cell; ++t) {
                            const index_type cand = indices_1[idx];
                            if (idcpu1[cand] != amrex::ParticleIdCpus::Invalid) {
                                ip = cand;
                                break;
                            }
                            ++idx;
                            if (idx == cell_stop_1) { idx = cell_start_1; }
                        }
                        AMREX_IF_ON_DEVICE((
                            AMREX_DEVICE_ASSERT(ip >= 0);
                        ))
                        AMREX_IF_ON_HOST((
                            if (ip < 0) {
                                amrex::Abort("Error in PulsedIonization: valid species 1 particle not found!");
                            }
                        ))

                        // Create product particles at position of particle from species 1
                        CopyIonF(ion, soa_1, ip, static_cast<int>(i_ion), engine);
                        CopyEleF(ele, soa_1, ip, static_cast<int>(i_ele), engine);

                        // Set electron velocity from specified normal distribution
                        uxe[i_ele] = electron_Vdrift_x;
                        uye[i_ele] = electron_Vdrift_y;
                        uze[i_ele] = electron_Vdrift_z;
                        uxe[i_ele] += electron_Vtherm_x*RandomNormal(0_prt, 1.0_prt, engine);
                        uye[i_ele] += electron_Vtherm_y*RandomNormal(0_prt, 1.0_prt, engine);
                        uze[i_ele] += electron_Vtherm_z*RandomNormal(0_prt, 1.0_prt, engine);

                        // Set the weight of the product particles
                        ion.m_rdata[PIdx::w][i_ion] = wpEI;
                        ele.m_rdata[PIdx::w][i_ele] = wpEI;

                        // Remove product weight from species 1
                        amrex::ParticleReal wp_remove = amrex::min(wpEI, w1[ip]);
                        BinaryCollisionUtils::remove_weight_from_colliding_particle(
                            w1[ip], idcpu1[ip], wp_remove);
                        amrex::ParticleReal wp_remaining = wpEI - wp_remove;
                        if (wp_remaining > 0.0_prt) {
                            for (index_type t = 0; t < num_in_cell; ++t) {
                                ++idx;
                                if (idx == cell_stop_1) { idx = cell_start_1; }
                                const index_type ip2 = indices_1[idx];
                                if (idcpu1[ip2] != amrex::ParticleIdCpus::Invalid) {
                                    wp_remove = amrex::min(wp_remaining, w1[ip2]);
                                    BinaryCollisionUtils::remove_weight_from_colliding_particle(
                                        w1[ip2], idcpu1[ip2], wp_remove);
                                    wp_remaining -= wp_remove;
                                }
                                if (wp_remaining <= 0.0_prt) { break; }
                            }
                        }

                    }

                }
            );

            // Initialize the user runtime components
            if (total_new > 0) {
                amrex::GpuArray<index_type, 2> products_np{old_np_ele, old_np_ion};
                amrex::GpuArray<ParticleTileType*, 2> tile_products{&ptile_ele, &ptile_ion};
                amrex::GpuArray<WarpXParticleContainer*, 2> pc_products{&product_ele, &product_ion};
                for (int i = 0; i < 2; i++) {
                    const auto start_index = int(products_np[i]);
                    const auto stop_index  = int(products_np[i] + total_new);
                    ParticleCreation::DefaultInitializeRuntimeAttributes(*tile_products[i],
                        0, 0,
                        pc_products[i]->getUserRealAttribs(), pc_products[i]->getUserIntAttribs(),
                        pc_products[i]->GetRealSoANames(), pc_products[i]->GetIntSoANames(),
                        pc_products[i]->getUserRealAttribParser(),
                        pc_products[i]->getUserIntAttribParser(),
#ifdef WARPX_QED
                        false, // do not initialize QED quantities, since they were initialized
                               // when calling the SmartCopy functors
                        pc_products[i]->get_breit_wheeler_engine_ptr(),
                        pc_products[i]->get_quantum_sync_engine_ptr(),
#endif
                        pc_products[i]->getIonizationInitialLevel(),
                        start_index, stop_index);
                }
            }
            amrex::Gpu::synchronize();

            // Set new particle IDs
            setNewParticleIDs(ptile_ele, old_np_ele, total_new);
            setNewParticleIDs(ptile_ion, old_np_ion, total_new);

        }

    }

}
