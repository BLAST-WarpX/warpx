/* Copyright 2021 Luca Fedeli
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
// Make.package是构建系统（如 Make 或 CMake）中的关键配置指令
 // VPATH_LOCATIONS += $(WARPX_HOME)/Source/Particles/Algorithms 是构建系统（如 Make 或 CMake）中的关键配置指令，用于管理大型科学计算项目的源代码组织。
 // 条件检查：如果宏未定义
#ifndef WARPX_PARTICLES_KINETIC_ENERGY_H_

// 定义宏
#define WARPX_PARTICLES_KINETIC_ENERGY_H_

// 编写一个头文件，用于计算粒子的动能。这个头文件依赖于一些其他头文件。这些 #include 指令构成了粒子动能计算的基础依赖。
// 这是WarpX软件中定义物理常量的头文件，例如光速 PhysConst::c。
#include "Utils/WarpXConst.H"
// 这个头文件提供AMReX库的一些扩展宏，例如 AMREX_RESTRICT 等。
#include "AMReX_Extension.H"
// 这个头文件定义了用于GPU编程的修饰符宏，例如 AMREX_GPU_HOST_DEVICE 和 AMREX_GPU_DEVICE，这些宏确保函数可以在主机（CPU）和设备（GPU）上运行。
#include "AMReX_GpuQualifiers.H"
// 提供数学函数的GPU安全版本，例如 amrex::Math::sqrt，以及一些数学常量和函数。
#include "AMReX_Math.H"
// 定义浮点类型，例如 amrex::Real，它根据编译选项可能是单精度（float）或双精度（double）。同时，该头文件还包含一些浮点数的操作和常量。
#include "AMReX_REAL.H"
// 在 C++ 中，#include <cmath> 是包含标准数学库的关键指令，它在科学计算中扮演着至关重要的角色。
#include <cmath>

namespace Algorithms{

    /**
     * \brief Computes the kinetic energy of a particle.简介计算粒子的动能。此方法不适用于光子。
     * This method should not be used with photons.
     *
     * @tparam T floating-point type to be used in internal calculations (by default equal to amrex::ParticleReal)T 内部计算使用的浮点类型（默认为 amrex::ParticleReal）
     * @param[in] ux x component of the particle momentum (code units)ux 粒子动量的x分量（代码单位）
     * @param[in] uy y component of the particle momentum (code units)uy 粒子动量的y分量（代码单位）
     * @param[in] uz z component of the particle momentum (code units)uz 粒子动量的z分量（代码单位）
     * @param[in] mass mass of the particle (in S.I. units)mass 粒子质量（SI单位）
     *
     * @return the kinetic energy of the particle (in S.I. units)粒子的动能（SI单位）
     */

     /*在C++中，template <typename T = amrex::ParticleReal> 是一个模板声明，
     它定义了一个模板函数（或类），其中有一个模板参数T，并且为T指定了一个默认类型amrex::ParticleReal。
    */
     template <typename T = amrex::ParticleReal>
    /*AMREX_GPU_HOST_DEVICE AMREX_INLINE 是 AMReX 库中用于支持 GPU 加速计算的宏定义组合,在GPU内核中使用时，确保函数被内联以避免额外调用开销
    AMREX_GPU_HOST_DEVICE​表示该函数可在 ​主机(CPU)​​ 和 ​设备(GPU)​​ 上执行
    AMREX_INLINE强制内联函数，同时根据编译器特性选择最佳的内联修饰符
    这个宏组合是 AMReX 支持异构计算的核心技术，使得粒子动力学计算可以在超算、GPU 集群等异构平台上获得最佳性能，同时保持代码的简洁性和可维护性。
    */
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    /*这个函数声明定义了一个模板函数KineticEnergy，用于计算粒子的动能。
    */

   /*这表示该函数是一个模板函数，带有一个类型参数T。T的默认类型是amrex::ParticleReal（通常是float或double）
   函数返回值的类型是模板参数T，即使用默认类型时为amrex::ParticleReal。
   函数的名称是KineticEnergy
    const amrex::ParticleReal ux: x方向动量分量（常量，不可修改）
    const amrex::ParticleReal uy: y方向动量分量
    const amrex::ParticleReal uz: z方向动量分量
    const amrex::ParticleReal mass: 粒子质量
   */
    T KineticEnergy(
        const amrex::ParticleReal ux, const amrex::ParticleReal uy, const amrex::ParticleReal uz, const amrex::ParticleReal mass)
    /*
    下面这段代码实现了相对论粒子的动能计算。
    引入amrex命名空间，简化后续代码
    // 1. 定义基本常数
    constexpr 确保光速倒数在编译时计算
    定义常量：one：将1.0转换为模板类型T的常量
    Math::powi<2> 使用模板元编程进行整数幂优化
    定义常量：inv_c2：计算光速平方的inverse number播放倒数（1/c²），使用Math::powi<2>进行编译时整数幂优化
    使用的表达式是通过还原表达式 (gamma - 1)*(gamma + 1)/(gamma + 1) 得出的
    // 2. 计算动量平方
    计算动量平方：将三个方向动量分量平方和转换为类型T
    // 3. 计算洛伦兹因子γ
    PhysConst::c 使用WarpX物理常数库中的光速值
    static_cast<T> 确保中间计算使用指定精度
    避免隐式类型转换导致的精度损失
    使用相对论公式：γ = √(1 + (p²)/(m²c²))，但注意这里输入的是u=γv（即动量除以质量），所以实际是γ = √(1 + u²/c²)
    // 4. 使用优化公式计算动能
    使用优化公式：KE = [1/(1+γ)] * m * u²
    这是从原始动能公式推导的数值稳定形式：
        KE = (γ - 1)mc²
        = (γ - 1)(γ + 1)mc² / (γ + 1)
        = (γ² - 1)mc² / (γ + 1)
        = (p²/c² * c²) / (γ + 1)   [因为 γ² = 1 + p²/(m²c²) => p² = m²c²(γ²-1)]
        = (p²) / [m(γ + 1)]        [注意：这里的u=p/m，所以u²=p²/m²]
        = m * u² / (γ + 1)
    */
    {
        using namespace amrex;

        constexpr auto one = static_cast<T>(1.0);
        constexpr auto inv_c2 = one/Math::powi<2>(static_cast<T>(PhysConst::c));

        // The expression used is derived by reducing the expression
        // (gamma - 1)*(gamma + 1)/(gamma + 1)

        const auto u2 = static_cast<T>(ux*ux + uy*uy + uz*uz);
        const auto gamma = std::sqrt(one + u2*inv_c2);
        return one/(one + gamma)*static_cast<T>(mass)*u2;
    }

    /**
     * \brief Computes the kinetic energy of a photon.简介计算光子的动能。
     *
     * @param[in] ux x component of the particle momentum (code units)粒子动量的x分量（代码单位）
     * @param[in] uy y component of the particle momentum (code units)粒子动量的y分量（代码单位）
     * @param[in] uz z component of the particle momentum (code units)粒子动量的z分量（代码单位）
     *
     * @return the kinetic energy of the photon (in S.I. units)光子的动能（SI单位）
     */
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    amrex::ParticleReal KineticEnergyPhotons(
        const amrex::ParticleReal ux, const amrex::ParticleReal uy, const amrex::ParticleReal uz)
    {
        // Photons have zero mass, but ux, uy and uz are calculated assuming a mass equal to the
        // electron mass. Hence, photons need a special treatment to calculate the total energy.
        // 光子的质量为零，但在计算 ux、uy 和 uz 时假定其质量等于电子质量。因此，光子需要特殊处理才能计算出总能量。
        constexpr auto me_c = PhysConst::m_e * PhysConst::c;

        return me_c * std::sqrt(ux*ux + uy*uy + uz*uz);
    }

}

#endif // WARPX_PARTICLES_KINETIC_ENERGY_H_  结束条件
