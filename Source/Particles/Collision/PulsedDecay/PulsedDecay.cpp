/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Author: Justin Angus (LLNL)
 *
 * License: BSD-3-Clause-LBNL
 */
#include "PulsedDecay.H"

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

PulsedDecay::PulsedDecay (std::string const& collision_name, MultiParticleContainer const * mypc)
    : CollisionBase(collision_name)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_species_names.size() == 1,
                                     "pulsed_decay  must have exactly one species.");

    const amrex::ParmParse pp_collision_name(collision_name);

    // Get the product species
    pp_collision_name.queryarr("product_species", m_product_species);

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE( m_product_species.size() == 2,
        "PulsedDecay: product_species size must be equal to two");

    auto& parent_species = mypc->GetParticleContainerFromName(m_species_names[0]);
    auto& productA = mypc->GetParticleContainerFromName(m_product_species[0]);
    auto& productB = mypc->GetParticleContainerFromName(m_product_species[1]);

    // Verify that the total charge of the product species matches the charge of the parent species
    const int Z_P = static_cast<int>(amrex::Math::round(parent_species.getCharge() / PhysConst::q_e));
    const int Z_A = static_cast<int>(amrex::Math::round(productA.getCharge() / PhysConst::q_e));
    const int Z_B = static_cast<int>(amrex::Math::round(productB.getCharge() / PhysConst::q_e));

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE( Z_P == Z_A + Z_B,
        "PulsedDecayFunc: total charge of product species must match the parent species charge");

    // Verify that the total mass of the product species matches the mass of the parent species
    const amrex::ParticleReal Mass_P = parent_species.getMass();
    const amrex::ParticleReal Mass_A = productA.getMass();
    const amrex::ParticleReal Mass_B = productB.getMass();

    const amrex::ParticleReal mass_ratio = (Mass_A + Mass_B) / Mass_P;
    const amrex::ParticleReal mass_error = amrex::Math::abs(mass_ratio - 1.0_prt);

    const amrex::ParticleReal eps = std::numeric_limits<amrex::ParticleReal>::epsilon();
    const amrex::ParticleReal rtol = 100.0_prt * eps;

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE( mass_error <= rtol,
        "PulsedDecayFunc: total mass of product species must match the parent species mass");

    // Get the fixed product particle weight
    pp_collision_name.get("fixed_product_weight", m_fixed_product_weight);

    // Parse the direction-dependent temperature for product species A
    amrex::Vector<amrex::ParticleReal> TA_tmp;
    pp_collision_name.getarr("productA_temperature_eV", TA_tmp);

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE( TA_tmp.size() == 3,
        "PulsedDecayFunc: productA_temperature_eV must have exactly 3 values");

    for (int i = 0; i < 3; ++i) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE( TA_tmp[i] >= 0.0,
            "PulsedDecayFunc: productA_temperature_eV must be greater than or equal to zero");
    }

    // Set the direction-dependent thermal speed for product species A
    const amrex::ParticleReal VtAx = std::sqrt(PhysConst::q_e*TA_tmp[0] / productA.getMass());
    const amrex::ParticleReal VtAy = std::sqrt(PhysConst::q_e*TA_tmp[1] / productA.getMass());
    const amrex::ParticleReal VtAz = std::sqrt(PhysConst::q_e*TA_tmp[2] / productA.getMass());
    m_productA_thermal_speed = {VtAx, VtAy, VtAz};

    // Parse the direction-dependent temperature for product species B
    amrex::Vector<amrex::ParticleReal> TB_tmp;
    pp_collision_name.getarr("productB_temperature_eV", TB_tmp);

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE( TB_tmp.size() == 3,
        "PulsedDecayFunc: productB_temperature_eV must have exactly 3 values");

    for (int i = 0; i < 3; ++i) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE( TB_tmp[i] >= 0.0,
            "PulsedDecayFunc: productB_temperature_eV must be greater than or equal to zero");
    }

    // Set the direction-dependent thermal speed for product species B
    const amrex::ParticleReal VtBx = std::sqrt(PhysConst::q_e*TB_tmp[0] / productB.getMass());
    const amrex::ParticleReal VtBy = std::sqrt(PhysConst::q_e*TB_tmp[1] / productB.getMass());
    const amrex::ParticleReal VtBz = std::sqrt(PhysConst::q_e*TB_tmp[2] / productB.getMass());
    m_productB_thermal_speed = {VtBx, VtBy, VtBz};

    // Parse the decay rate
    std::string decay_rate_str;
    utils::parser::Store_parserString(pp_collision_name, "decay_rate(x,y,z,t)", decay_rate_str);
    m_decay_rate_parser = utils::parser::makeParser(decay_rate_str, {"x", "y", "z", "t"});

    // Compile the decay rate parser
    m_decay_rate_func = m_decay_rate_parser.compile<4>();

}

void
PulsedDecay::doCollisions (amrex::Real cur_time, amrex::Real dt, MultiParticleContainer* mypc)
{
    WARPX_PROFILE("PulsedDecay::doCollisions()");

    using namespace amrex::literals;

    using ParticleTileType = WarpXParticleContainer::ParticleTileType;
    using ParticleTileDataType = ParticleTileType::ParticleTileDataType;
    using ParticleBins = amrex::DenseBins<ParticleTileDataType>;
    using index_type = ParticleBins::index_type;
    using SoaData_type = typename WarpXParticleContainer::ParticleTileType::ParticleTileDataType;

    // Get handles to species particle containters
    auto& species1 = mypc->GetParticleContainerFromName(m_species_names[0]);
    auto& productA = mypc->GetParticleContainerFromName(m_product_species[0]);
    auto& productB = mypc->GetParticleContainerFromName(m_product_species[1]);

    const SmartCopyFactory copy_factory_A(species1, productA);
    const SmartCopyFactory copy_factory_B(species1, productB);
    const SmartCopy CopyA = copy_factory_A.getSmartCopy();
    const SmartCopy CopyB = copy_factory_B.getSmartCopy();

#ifdef AMREX_USE_GPU
    amrex::Gpu::DeviceScalar<SmartCopy> d_CopyA(CopyA);
    amrex::Gpu::DeviceScalar<SmartCopy> d_CopyB(CopyB);

    SmartCopy const* AMREX_RESTRICT CopyAPtr = d_CopyA.dataPtr();
    SmartCopy const* AMREX_RESTRICT CopyBPtr = d_CopyB.dataPtr();
#else
    SmartCopy const* AMREX_RESTRICT CopyAPtr = &CopyA;
    SmartCopy const* AMREX_RESTRICT CopyBPtr = &CopyB;
#endif

    // get parsers for the decay rate
    auto nu_func = m_decay_rate_func;

    const amrex::ParticleReal fixed_product_weight = m_fixed_product_weight;
    const amrex::ParticleReal VtA_x = m_productA_thermal_speed[0];
    const amrex::ParticleReal VtA_y = m_productA_thermal_speed[1];
    const amrex::ParticleReal VtA_z = m_productA_thermal_speed[2];
    const amrex::ParticleReal VtB_x = m_productB_thermal_speed[0];
    const amrex::ParticleReal VtB_y = m_productB_thermal_speed[1];
    const amrex::ParticleReal VtB_z = m_productB_thermal_speed[2];

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
                    int num_macro_particles = static_cast<int>(std::floor(num_expected + amrex::Random(engine)));

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
            ParticleTileType& ptileA = productA.ParticlesAt(lev, mfi);
            ParticleTileType& ptileB = productB.ParticlesAt(lev, mfi);
            const index_type old_npA = ptileA.numParticles();
            const index_type old_npB = ptileB.numParticles();
            ptileA.resize(old_npA + total_new);
            ptileB.resize(old_npB + total_new);

            // Host-side SoA handles for the product species
            SoaData_type soa_productA = ptileA.getParticleTileData();
            SoaData_type soa_productB = ptileB.getParticleTileData();

#ifdef AMREX_USE_GPU
            // Make device copies so the kernel sees device-resident handles
            amrex::Gpu::DeviceScalar<SoaData_type> d_soaA(soa_productA);
            amrex::Gpu::DeviceScalar<SoaData_type> d_soaB(soa_productB);

            SoaData_type const* AMREX_RESTRICT soaA_ptr = d_soaA.dataPtr();
            SoaData_type const* AMREX_RESTRICT soaB_ptr = d_soaB.dataPtr();
#else
            SoaData_type const* AMREX_RESTRICT soaA_ptr = &soa_productA;
            SoaData_type const* AMREX_RESTRICT soaB_ptr = &soa_productB;
#endif

            const index_type* AMREX_RESTRICT offsets_ptr = offsets.dataPtr();
            const index_type* AMREX_RESTRICT nprod_cell_ptr = num_products_vec.dataPtr();

            index_type const* AMREX_RESTRICT cell_offsets_1 = bins_1.offsetsPtr();
            index_type const* AMREX_RESTRICT indices_1 = bins_1.permutationPtr();

            uint64_t* AMREX_RESTRICT idcpu1 = soa_1.m_idcpu;

            amrex::ParticleReal const* AMREX_RESTRICT ux1  = soa_1.m_rdata[PIdx::ux];
            amrex::ParticleReal const* AMREX_RESTRICT uy1  = soa_1.m_rdata[PIdx::uy];
            amrex::ParticleReal const* AMREX_RESTRICT uz1  = soa_1.m_rdata[PIdx::uz];

            amrex::ParticleReal* AMREX_RESTRICT wA  = soa_productA.m_rdata[PIdx::w];
            amrex::ParticleReal* AMREX_RESTRICT uAx = soa_productA.m_rdata[PIdx::ux];
            amrex::ParticleReal* AMREX_RESTRICT uAy = soa_productA.m_rdata[PIdx::uy];
            amrex::ParticleReal* AMREX_RESTRICT uAz = soa_productA.m_rdata[PIdx::uz];

            amrex::ParticleReal* AMREX_RESTRICT wB  = soa_productB.m_rdata[PIdx::w];
            amrex::ParticleReal* AMREX_RESTRICT uBx = soa_productB.m_rdata[PIdx::ux];
            amrex::ParticleReal* AMREX_RESTRICT uBy = soa_productB.m_rdata[PIdx::uy];
            amrex::ParticleReal* AMREX_RESTRICT uBz = soa_productB.m_rdata[PIdx::uz];

            amrex::ParallelForRNG( n_cells,
                [=] AMREX_GPU_DEVICE (int i_cell, amrex::RandomEngine const& engine) noexcept
                {
                    const index_type num_products_in_cell = nprod_cell_ptr[i_cell];
                    if (num_products_in_cell == 0) { return; }

                    const index_type start = offsets_ptr[i_cell];

                    auto const& CopyAF = *CopyAPtr;
                    auto const& CopyBF = *CopyBPtr;

                    // The particles from species1 that are in the cell `i_cell` are
                    // given by the `indices_1[cell_start_1:cell_stop_1]`
                    const index_type cell_start_1 = cell_offsets_1[i_cell];
                    const index_type cell_stop_1  = cell_offsets_1[i_cell+1];
                    const index_type num_in_cell = cell_stop_1 - cell_start_1;

                    const amrex::ParticleReal wpAB = fixed_product_weight;

                    for (index_type j = 0; j < num_products_in_cell; ++j) {

                        const index_type new_idx = start + j;
                        const index_type ip_A = old_npA + new_idx;
                        const index_type ip_B = old_npB + new_idx;

                        // Get a random particle index from species 1 in this cell
                        index_type k = static_cast<index_type>(amrex::Random(engine) * amrex::Real(num_in_cell));

                        // Probe until a valid particle is found (should always be at least one)
                        index_type ip  = -1;
                        for (index_type t = 0; t < num_in_cell; ++t) {
                            const index_type cand_k = (k + t) % num_in_cell;
                            const index_type idx = cell_start_1 + cand_k;
                            const index_type cand_ip = indices_1[idx];
                            if (idcpu1[cand_ip] != amrex::ParticleIdCpus::Invalid) {
                                k  = cand_k;
                                ip = cand_ip;
                                break;
                            }
                        }
                        AMREX_IF_ON_DEVICE((
                            AMREX_DEVICE_ASSERT(ip >= 0);
                        ))
                        AMREX_IF_ON_HOST((
                            if (ip < 0) {
                                amrex::Abort("Error in PulsedDecay: valid species 1 particle not found!");
                            }
                        ))

                        // Create product particles at position of particle from species 1
                        CopyAF(*soaA_ptr, soa_1, ip, static_cast<int>(ip_A), engine);
                        CopyBF(*soaB_ptr, soa_1, ip, static_cast<int>(ip_B), engine);

                        // Set product species A particle velocity to parent particle velocity plus thermal
                        uAx[ip_A] = ux1[ip];
                        uAy[ip_A] = uy1[ip];
                        uAz[ip_A] = uz1[ip];
                        uAx[ip_A] += VtA_x*RandomNormal(0_prt, 1.0_prt, engine);
                        uAy[ip_A] += VtA_y*RandomNormal(0_prt, 1.0_prt, engine);
                        uAz[ip_A] += VtA_z*RandomNormal(0_prt, 1.0_prt, engine);

                        // Set product species B velocity to parent particle velocity plus thermal
                        uBx[ip_B] = ux1[ip];
                        uBy[ip_B] = uy1[ip];
                        uBz[ip_B] = uz1[ip];
                        uBx[ip_B] += VtB_x*RandomNormal(0_prt, 1.0_prt, engine);
                        uBy[ip_B] += VtB_y*RandomNormal(0_prt, 1.0_prt, engine);
                        uBz[ip_B] += VtB_z*RandomNormal(0_prt, 1.0_prt, engine);

                        // Set the weight of the product particles
                        wA[ip_A] = wpAB;
                        wB[ip_B] = wpAB;

                        // Remove product weight from species 1
                        amrex::ParticleReal wp_remaining = wpAB;
                        for (index_type t = 0; t < num_in_cell; ++t) {
                            const index_type k2 = (k + t) % num_in_cell;
                            const index_type idx2 = cell_start_1 + k2;
                            const index_type ip2 = indices_1[idx2];

                            if (idcpu1[ip2] == amrex::ParticleIdCpus::Invalid) { continue; }

                            const amrex::ParticleReal wp_remove = amrex::min(wp_remaining, w1[ip2]);
                            BinaryCollisionUtils::remove_weight_from_colliding_particle(
                                w1[ip2], idcpu1[ip2], wp_remove);

                            wp_remaining -= wp_remove;
                            if (wp_remaining <= 0.0_prt) { break; }
                        }

                    }

                }
            );

            // Initialize the user runtime components
            if (total_new > 0) {
                amrex::GpuArray<index_type, 2> products_np{old_npA, old_npB};
                amrex::GpuArray<ParticleTileType*, 2> tile_products{&ptileA, &ptileB};
                amrex::GpuArray<WarpXParticleContainer*, 2> pc_products{&productA, &productB};
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
            setNewParticleIDs(ptileA, old_npA, total_new);
            setNewParticleIDs(ptileB, old_npB, total_new);

        }

    }

}
