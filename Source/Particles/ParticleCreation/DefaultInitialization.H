/* Copyright 2019-2020 Andrew Myers, Axel Huebl,
 * Maxence Thevenet
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

// 条件检查：如果宏未定义
#ifndef WARPX_DEFAULTINITIALIZATION_H_
// 定义宏
#define WARPX_DEFAULTINITIALIZATION_H_

// 在C和C++编程中，#include是一个预处理指令，用于在编译之前将指定文件的内容包含到当前文件中。
// #include <filename>通常用于包含标准库头文件或系统头文件。
#include <WarpX.H>

// #include "filename"通常用于包含用户自定义的头文件，这些头文件通常位于当前项目的目录中。
// 这是一个典型的条件编译（Conditional Compilation）的例子，
// 使用了预处理指令 #ifdef 和 #endif。代码中包含了两个 #include 指令，它们都使用了双引号（""），
//这意味着它们包含的是用户自定义的头文件，而不是标准库头文件。
// #ifdef WARPX_QED：这是一个条件编译指令，检查是否定义了宏 WARPX_QED。
// 如果该宏被定义（通常通过编译器选项如 -D WARPX_QED 或在代码中定义），
// 则编译器会包含下面的代码直到 #endif。否则，这段代码会被忽略。
#ifdef WARPX_QED
#   include "Particles/ElementaryProcess/QEDInternals/BreitWheelerEngineWrapper.H"
#   include "Particles/ElementaryProcess/QEDInternals/QuantumSyncEngineWrapper.H"
#endif


#include <AMReX_GpuContainers.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <map>
#include <string>

/**
 * 这组初始化策略描述了当我们需要通过基本过程创建一个新粒子时会发生的情况。
 * 例如，当电离事件产生一个电子时，这些策略会控制电子成分的初始值。这些内容可以在以后随时改写。
 * \brief This set of initialization policies describes what happens
 * when we need to create a new particle due to an elementary process.
 * For example, when an ionization event creates an electron, these
 * policies control the initial values of the electron's components.
 * These can always be over-written later.
 *
 * 具体含义如下：
 * 0 - 将元件设置为零
 * 1 - 将分量设置为 1
 * RandomExp - 某些 QED 过程使用的光学深度分量的特殊标志，它从指数分布中提取随机初始值
 * The specific meanings are as follows:
 *     Zero         - set the component to zero
 *     One          - set the component to one
 *     RandomExp    - a special flag for the optical depth component used by
 *                    certain QED processes, which gets a random initial value
 *                    extracted from an exponential distribution
 *
 */
enum struct InitializationPolicy {Zero=0, One, RandomExp};

/**
 * \brief This map sets the initialization policy for each particle component
 * used in WarpX.
 * 该映射为 WarpX 中使用的每个粒子组件设置初始化策略。
 */
static std::map<std::string, InitializationPolicy> initialization_policies = {
    {"w",     InitializationPolicy::Zero },
    {"ux",    InitializationPolicy::Zero },
    {"uy",    InitializationPolicy::Zero },
    {"uz",    InitializationPolicy::Zero },
#ifdef WARPX_DIM_RZ
    {"theta", InitializationPolicy::Zero},
#endif

#ifdef WARPX_QED
    {"opticalDepthBW",   InitializationPolicy::RandomExp},
    {"opticalDepthQSR",   InitializationPolicy::RandomExp}
#endif

};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::ParticleReal initializeRealValue (const InitializationPolicy policy, amrex::RandomEngine const& engine) noexcept
{
    switch (policy) {
        case InitializationPolicy::Zero : return 0.0;
        case InitializationPolicy::One  : return 1.0;
        case InitializationPolicy::RandomExp : {
            return -std::log(amrex::Random(engine));
        }
        default : {
            amrex::Abort("Initialization Policy not recognized");
            return 1.0;
        }
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int initializeIntValue (const InitializationPolicy policy) noexcept
{
    switch (policy) {
        case InitializationPolicy::Zero : return 0;
        case InitializationPolicy::One  : return 1;
        default : {
            amrex::Abort("Initialization Policy not recognized");
            return 1;
        }
    }
}

namespace ParticleCreation {

    /**
     * \brief Default initialize runtime attributes in a tile. This routine does not initialize the
     * first n_external_attr_real real attributes and the first n_external_attr_int integer
     * attributes, which have been in principle externally set elsewhere.
     * 默认初始化tile中的运行时runtime属性。
     * 此例程不会初始化第一个 n_external_attr_real 实数属性和第一个 n_external_attr_int 整数属性，
     * 因为这些属性原则上已在其他地方进行了外部设置。
     * @tparam[in] The type of the particle tile to operate on (e.g. could use different allocators)
     * 要操作的粒子tile的类型（例如，可以使用不同的分配器）
     * @param[inout] ptile the tile in which attributes are initialized
     * 初始化属性的tile
     * @param[in] n_external_attr_real The number of real attributes that have been externally set.
     * These are NOT initialized by this function.
     * 已从外部设置的实数属性的数量。本函数不会初始化这些属性。
     * @param[in] n_external_attr_int The number of integer attributes that have been externally set.
     * These are NOT initialized by this function.
     * 已从外部设置的整数属性的数量。本函数不会初始化这些属性。
     * @param[in] user_real_attribs The names of the real components for this particle tile
     * 该粒子tile的real实数成分名称
     * @param[in] user_int_attribs The names of the int components for this particle tile
     * 该粒子瓦的int整数组件名称
     * @param[in] particle_comps particle component names for real comps
     * 用于实数real分量的粒子组件名称
     * @param[in] particle_icomps particle component names for int comps
     * 用于整数int分量的粒子组件名称
     * @param[in] user_real_attrib_parser the parser functions used to initialize the user real components
     * 用于初始化用户实数real分量的解析器函数
     * @param[in] user_int_attrib_parser the parser functions used to initialize the user int components
     * 用于初始化用户整数int分量的解析器函数
     * @param[in] do_qed_comps whether to initialize the qed components (these are usually handled by
     * SmartCopy, but NOT when adding particles in AddNParticles)
     * 是否初始化 qed 组件（这些组件通常由 SmartCopy 处理，但在 AddNParticles 中添加粒子时不需要）。
     * @param[in] p_bw_engine the engine to use for setting the breit-wheeler component for QED
     * 用于设置 QED 的breit-wheeler分量的引擎
     * @param[in] p_qs_engine the engine to use for setting the quantum synchrotron component for QED
     * 用于设置 QED 的量子同步加速器组件的引擎
     * @param[in] ionization_initial_level the ionization level particles created should start at
     * 所产生的电离水平粒子应从
     * @param[in] start the index to start initializing particles
     * 开始初始化粒子的索引
     * @param[in] stop the index to stop initializing particles
     * 开始初始化粒子的索引
     */
template <typename PTile>
void DefaultInitializeRuntimeAttributes (PTile& ptile,
                                         const int n_external_attr_real,
                                         const int n_external_attr_int,
                                         const std::vector<std::string>& user_real_attribs,
                                         const std::vector<std::string>& user_int_attribs,
                                         const std::vector<std::string>& particle_comps,
                                         const std::vector<std::string>& particle_icomps,
                                         const std::vector<amrex::Parser*>& user_real_attrib_parser,
                                         const std::vector<amrex::Parser*>& user_int_attrib_parser,
#ifdef WARPX_QED
                                         const bool do_qed_comps,
                                         BreitWheelerEngine* p_bw_engine,
                                         QuantumSynchrotronEngine* p_qs_engine,
#endif
                                         const int ionization_initial_level,
                                         int start, int stop)
{
        using namespace amrex::literals;

        // Preparing data needed for user defined attributes
        // 准备用户定义属性所需的数据
        const auto n_user_real_attribs = static_cast<int>(user_real_attribs.size());
        const auto n_user_int_attribs = static_cast<int>(user_int_attribs.size());
        const auto get_position = GetParticlePosition<PIdx>(ptile);
        const auto soa = ptile.getParticleTileData();
        const amrex::ParticleReal* AMREX_RESTRICT ux = soa.m_rdata[PIdx::ux];
        const amrex::ParticleReal* AMREX_RESTRICT uy = soa.m_rdata[PIdx::uy];
        const amrex::ParticleReal* AMREX_RESTRICT uz = soa.m_rdata[PIdx::uz];
        constexpr int lev = 0;
        const amrex::Real t = WarpX::GetInstance().gett_new(lev);

        // Initialize the last NumRuntimeRealComps() - n_external_attr_real runtime real attributes
        // 初始化最后一个 NumRuntimeRealComps() - n_external_attr_real 运行时实数属性
        for (int j = PIdx::nattribs + n_external_attr_real; j < ptile.NumRealComps() ; ++j)
        {
            auto attr_ptr = ptile.GetStructOfArrays().GetRealData(j).data();
#ifdef WARPX_QED
            // Current runtime comp is quantum synchrotron optical depth
            // 目前的运行时间是量子同步加速器光学深度
            auto const it_qsr = std::find(particle_comps.begin(), particle_comps.end(), "opticalDepthQSR");
            if (it_qsr != particle_comps.end() &&
                std::distance(particle_comps.begin(), it_qsr) == j)
            {
                if (!do_qed_comps) { continue; }
                const QuantumSynchrotronGetOpticalDepth quantum_sync_get_opt =
                                                p_qs_engine->build_optical_depth_functor();
                // If the particle tile was allocated in a memory pool that can run on GPU, launch GPU kernel
                // 如果粒子tile分配在可在 GPU 上运行的内存池中，则启动 GPU 内核
                if constexpr (amrex::RunOnGpu<typename PTile::template AllocatorType<amrex::Real>>::value) {
                        amrex::ParallelForRNG(stop - start,
                                              [=] AMREX_GPU_DEVICE (int i, amrex::RandomEngine const& engine) noexcept {
                                                  const int ip = i + start;
                                                  attr_ptr[ip] = quantum_sync_get_opt(engine);
                                              });
                // Otherwise (e.g. particle tile allocated in pinned memory), run on CPU
                // 否则（例如，在固定内存中分配的粒子tile），在 CPU 上运行
                } else {
                    for (int ip = start; ip < stop; ++ip) {
#ifdef AMREX_USE_GPU
                        attr_ptr[ip] = quantum_sync_get_opt(amrex::RandomEngine{nullptr});
#else
                        attr_ptr[ip] = quantum_sync_get_opt(amrex::RandomEngine{});
#endif
                    }
                }
            }

            // Current runtime comp is Breit-Wheeler optical depth
            // 目前的runtime分量是 Breit-Wheeler 光学深度
            auto const it_bw = std::find(particle_comps.begin(), particle_comps.end(), "opticalDepthBW");
            if (it_bw != particle_comps.end() &&
                std::distance(particle_comps.begin(), it_bw) == j)
            {
                if (!do_qed_comps) { continue; }
                const BreitWheelerGetOpticalDepth breit_wheeler_get_opt =
                                                p_bw_engine->build_optical_depth_functor();;
                // If the particle tile was allocated in a memory pool that can run on GPU, launch GPU kernel
                // 如果粒子tile分配在可在 GPU 上运行的内存池中，则启动 GPU 内核
                if constexpr (amrex::RunOnGpu<typename PTile::template AllocatorType<amrex::Real>>::value) {
                        amrex::ParallelForRNG(stop - start,
                                              [=] AMREX_GPU_DEVICE (int i, amrex::RandomEngine const& engine) noexcept {
                                                  const int ip = i + start;
                                                  attr_ptr[ip] = breit_wheeler_get_opt(engine);
                                              });
                // Otherwise (e.g. particle tile allocated in pinned memory), run on CPU
                // 否则（例如，在固定内存中分配的粒子tile），在 CPU 上运行
                } else {
                    for (int ip = start; ip < stop; ++ip) {
#ifdef AMREX_USE_GPU
                        attr_ptr[ip] = breit_wheeler_get_opt(amrex::RandomEngine{nullptr});
#else
                        attr_ptr[ip] = breit_wheeler_get_opt(amrex::RandomEngine{});
#endif
                    }
                }
            }
#endif

            for (int ia = 0; ia < n_user_real_attribs; ++ia)
            {
                // Current runtime comp is ia-th user defined attribute
                // 当前runtime的分量是第 ia 个用户定义的属性
                auto const it_ura = std::find(particle_comps.begin(), particle_comps.end(), user_real_attribs[ia]);
                if (it_ura != particle_comps.end() &&
                    std::distance(particle_comps.begin(), it_ura) == j)
                {
                    const amrex::ParserExecutor<7> user_real_attrib_parserexec =
                                             user_real_attrib_parser[ia]->compile<7>();
                    // If the particle tile was allocated in a memory pool that can run on GPU, launch GPU kernel
                    // 如果粒子tile分配在可在 GPU 上运行的内存池中，则启动 GPU 内核
                    if constexpr (amrex::RunOnGpu<typename PTile::template AllocatorType<amrex::Real>>::value) {
                        amrex::ParallelFor(stop - start,
                                           [=] AMREX_GPU_DEVICE (int i) noexcept {
                                               const int ip = i + start;
                                               amrex::ParticleReal xp, yp, zp;
                                               get_position(ip, xp, yp, zp);
                                               attr_ptr[ip] = user_real_attrib_parserexec(xp, yp, zp,
                                                                                          ux[ip], uy[ip], uz[ip], t);
                                           });
                    // Otherwise (e.g. particle tile allocated in pinned memory), run on CPU
                    // 否则（例如，在固定内存中分配的粒子tile），在 CPU 上运行
                    } else {
                        for (int ip = start; ip < stop; ++ip) {
                            amrex::ParticleReal xp, yp, zp;
                            get_position(ip, xp, yp, zp);
                            attr_ptr[ip] = user_real_attrib_parserexec(xp, yp, zp,
                                                                       ux[ip], uy[ip], uz[ip], t);
                        }
                    }
                }
            }
        }

        // Initialize the last NumRuntimeIntComps() - n_external_attr_int runtime int attributes
        // 初始化最后一个 NumRuntimeIntComps() - n_external_attr_int runtime整数属性
        for (int j = n_external_attr_int; j < ptile.NumIntComps() ; ++j)
        {
            auto attr_ptr = ptile.GetStructOfArrays().GetIntData(j).data();

            // Current runtime comp is ionization level
            // 当前的runtime的comp是电离水平
            auto const it_ioniz = std::find(particle_icomps.begin(), particle_icomps.end(), "ionizationLevel");
            if (it_ioniz != particle_icomps.end() &&
                std::distance(particle_icomps.begin(), it_ioniz) == j)
            {
                if constexpr (amrex::RunOnGpu<typename PTile::template AllocatorType<int>>::value) {
                        amrex::ParallelFor(stop - start,
                                           [=] AMREX_GPU_DEVICE (int i) noexcept {
                                               const int ip = i + start;
                                               attr_ptr[ip] = ionization_initial_level;
                                           });
                } else {
                    for (int ip = start; ip < stop; ++ip) {
                        attr_ptr[ip] = ionization_initial_level;
                    }
                }
            }

            for (int ia = 0; ia < n_user_int_attribs; ++ia)
            {
                // Current runtime comp is ia-th user defined attribute
                // 当前runtime的comp是第ia个用户定义的属性
                auto const it_uia = std::find(particle_icomps.begin(), particle_icomps.end(), user_int_attribs[ia]);
                if (it_uia != particle_icomps.end() &&
                    std::distance(particle_icomps.begin(), it_uia) == j)
                {
                     const amrex::ParserExecutor<7> user_int_attrib_parserexec =
                                             user_int_attrib_parser[ia]->compile<7>();
                if constexpr (amrex::RunOnGpu<typename PTile::template AllocatorType<int>>::value) {
                        amrex::ParallelFor(stop - start,
                                           [=] AMREX_GPU_DEVICE (int i) noexcept {
                                               const int ip = i + start;
                                               amrex::ParticleReal xp, yp, zp;
                                               get_position(ip, xp, yp, zp);
                                               attr_ptr[ip] = static_cast<int>(
                                                                               user_int_attrib_parserexec(xp, yp, zp, ux[ip], uy[ip], uz[ip], t));
                                           });
                } else {
                    for (int ip = start; ip < stop; ++ip) {
                        amrex::ParticleReal xp, yp, zp;
                        get_position(ip, xp, yp, zp);
                        attr_ptr[ip] = static_cast<int>(
                                user_int_attrib_parserexec(xp, yp, zp, ux[ip], uy[ip], uz[ip], t));
                    }
                }
                }
            }
        }
}

}

#endif //WARPX_DEFAULTINITIALIZATION_H_  结束条件
