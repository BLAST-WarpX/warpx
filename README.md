# Lanzi-WarpX-Etching-Module

[![WarpX](https://img.shields.io/badge/based%20on-WarpX-blue)](https://warpx.readthedocs.io)
[![License](https://img.shields.io/badge/license-BSD--3--Clause--LBNL-blue.svg)](LICENSE.txt)
[![Language: C++17](https://img.shields.io/badge/language-C%2B%2B17-orange.svg)](https://isocpp.org/)
[![Platform](https://img.shields.io/badge/platform-Linux%20|%20macOS%20|%20Windows-blue)](https://warpx.readthedocs.io)

## 概述

Lanzi-WarpX-Etching-Module 是基于 [WarpX](https://warpx.readthedocs.io) 的等离子体刻蚀模拟扩展模块，专门用于电容耦合等离子体(CCP)设备的数值模拟。本项目在WarpX的基础上添加了石英边界条件模块，能够准确模拟等离子体刻蚀过程中的绝缘材料效应。

### 主要特性

- 🔬 **石英边界条件**: 专门为CCP设备设计的石英环边界条件
- ⚡ **高性能计算**: 基于WarpX的高性能PIC代码，支持GPU加速
- 🎯 **精确物理**: 包含介电材料效应、边界条件连续性等物理细节
- 📊 **完整分析**: 提供完整的后处理和分析工具
- 🔧 **易于扩展**: 模块化设计，便于添加新的边界条件

### 物理应用

- 等离子体刻蚀设备模拟
- 电容耦合等离子体(CCP)研究
- 半导体制造工艺优化
- 等离子体-材料相互作用研究

## 快速开始

### 环境要求

-Homebrew 安装的依赖库
-系统编译器 (Apple Clang 17.0.0)
-系统MPI (OpenMPI via Homebrew)
- **C++17编译器**: GCC 9.1+, Clang 7+, 或 MSVC 19.15+
- **CMake**: 3.24.0+
- **MPI**: 3.0+ (OpenMPI 或 MPICH)
- **Git**: 2.18+

### 安装依赖

#### 方式1: Conda (推荐)

```bash
# 创建conda环境
conda create -n warpx-etching -c conda-forge \
  cmake compilers git openmpi fftw boost hdf5 \
  numpy matplotlib pandas scipy

# 激活环境
conda activate warpx-etching
```

#### 方式2: 系统包管理器

**Ubuntu/Debian:**
```bash
sudo apt-get install cmake g++ libopenmpi-dev libfftw3-dev \
  libboost-dev libhdf5-dev python3-pip
```

**macOS:**
```bash
brew install cmake open-mpi fftw boost hdf5
```

### 编译安装

```bash
# 克隆项目
git clone <您的gitee仓库地址>
cd Lanzi-WarpX-Etching-Module

# 配置和编译
cmake -S . -B build \
  -DWarpX_DIMS="3" \
  -DWarpX_MPI=ON \
  -DWarpX_COMPUTE=OMP \
  -DWarpX_QED=ON

# 编译 (使用4个线程)
cmake --build build -j 4
```

### 运行示例

```bash
# 运行石英环CCP示例
mpirun -np 4 ./build/bin/warpx.3d \
  Examples/Tests/quartz_ring_ccp/inputs_quartz_ring_ccp

# 分析结果
python Examples/Tests/quartz_ring_ccp/analysis_quartz_ring_ccp.py plt00050
```

## 石英边界条件模块

### 物理背景

石英边界条件模块专门为CCP设备设计，模拟石英环作为绝缘屏障的物理效应：

- **石英材料特性**: εr ≈ 3.8, μr ≈ 1.0, σ ≈ 10⁻¹² S/m
- **边界条件**: E_tangential连续，D_normal连续
- **几何形状**: 环形结构，适用于CCP设备

### 使用方法

在输入文件中配置石英边界条件：

```bash
# 石英环几何参数
quartz.ring_inner_radius = 0.05    # 内半径 (m)
quartz.ring_outer_radius = 0.08    # 外半径 (m)
quartz.ring_height = 0.02          # 高度 (m)
quartz.ring_center_x = 0.0         # 中心x坐标 (m)
quartz.ring_center_y = 0.0         # 中心y坐标 (m)
quartz.ring_bottom_z = -0.01       # 底部z坐标 (m)

# 石英材料参数
quartz.epsilon_r = 3.8             # 相对介电常数
quartz.mu_r = 1.0                  # 相对磁导率
quartz.sigma = 1e-12               # 电导率 (S/m)

# 边界条件设置
boundary.field_lo = quartz quartz quartz
boundary.field_hi = quartz quartz quartz
```

### 示例文件

- `Examples/Tests/quartz_ring_ccp/`: 完整的CCP示例
- `Source/BoundaryConditions/Quartz.H`: 石英边界条件头文件
- `Source/BoundaryConditions/Quartz.cpp`: 石英边界条件实现

## 项目结构

```
Lanzi-WarpX-Etching-Module/
├── Source/
│   ├── BoundaryConditions/
│   │   ├── Quartz.H          # 石英边界条件头文件
│   │   └── Quartz.cpp        # 石英边界条件实现
│   └── ...                   # 其他WarpX源文件
├── Examples/
│   └── Tests/
│       └── quartz_ring_ccp/  # CCP示例
├── Docs/                     # 文档
├── Tools/                    # 工具脚本
└── README.md                 # 本文件
```

## 开发指南

### 添加新的边界条件

1. 在 `Source/BoundaryConditions/` 中创建新的边界条件类
2. 继承自WarpX的边界条件基类
3. 实现必要的接口方法
4. 在 `Source/Make.package` 中添加新文件
5. 更新文档和示例

### 代码规范

- 遵循WarpX的代码风格
- 使用C++17标准
- 添加适当的注释和文档
- 编写单元测试

## 常见问题

### 编译问题

**Q: 编译时找不到依赖库？**
A: 确保所有依赖都已正确安装，可以使用conda环境确保一致性。

**Q: MPI相关错误？**
A: 检查MPI是否正确安装，确保使用mpicxx编译器。

### 运行问题

**Q: 石英边界条件不生效？**
A: 检查输入文件中的边界条件设置是否正确。

**Q: 数值不稳定？**
A: 尝试减小时间步长或增加网格分辨率。

## 贡献指南

我们欢迎各种形式的贡献：

1. **报告问题**: 在Gitee上提交Issue
2. **功能请求**: 提出新功能建议
3. **代码贡献**: 提交Pull Request
4. **文档改进**: 完善文档和示例
5. **测试**: 帮助测试和验证

### 开发环境设置

```bash
# 克隆开发分支
git clone -b development <仓库地址>
cd Lanzi-WarpX-Etching-Module

# 设置开发环境
conda create -n warpx-dev -c conda-forge \
  cmake compilers git openmpi fftw boost hdf5 \
  clang-tidy ninja

# 编译开发版本
cmake -S . -B build -GNinja \
  -DWarpX_DIMS="1;2;3;RZ" \
  -DWarpX_MPI=ON \
  -DWarpX_COMPUTE=OMP \
  -DWarpX_QED=ON

cmake --build build -j 4
```

## 许可证

本项目基于 [BSD-3-Clause-LBNL](LICENSE.txt) 许可证开源。

## 致谢

- 基于 [WarpX](https://warpx.readthedocs.io) 项目开发
- 感谢WarpX开发团队提供的优秀基础框架
- 感谢所有贡献者的支持

## 联系方式

- **项目主页**: [Gitee仓库地址]
- **问题反馈**: 在Gitee上提交Issue
- **讨论交流**: 使用Gitee的讨论功能

---

**注意**: 本项目是WarpX的扩展模块，使用前请确保了解WarpX的基本使用方法。更多详细信息请参考 [WarpX官方文档](https://warpx.readthedocs.io)。
