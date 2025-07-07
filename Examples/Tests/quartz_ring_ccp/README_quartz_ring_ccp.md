# 石英环CCP (Capacitively Coupled Plasma) 边界条件示例

## 概述

这个示例演示了如何在WarpX中使用石英环边界条件来模拟电容耦合等离子体(CCP)设备。石英环在等离子体刻蚀和沉积过程中作为绝缘屏障，将电极与等离子体分离。

## 物理背景

### CCP设备结构
- **石英环**: 位于上下电极之间，作为绝缘屏障
- **电极**: 上下两个金属电极，施加射频电压
- **等离子体**: 在电极间形成的电离气体

### 石英材料特性
- **相对介电常数**: εr ≈ 3.8
- **相对磁导率**: μr ≈ 1.0
- **电导率**: σ ≈ 10⁻¹² S/m (极低)

### 边界条件
- **电场**: E_tangential连续，D_normal连续
- **磁场**: B_normal连续，H_tangential连续

## 文件说明

### 输入文件
- `inputs_quartz_ring_ccp`: 主输入文件，包含所有模拟参数

### 分析脚本
- `analysis_quartz_ring_ccp.py`: 分析模拟结果，生成可视化图表

## 使用方法

### 1. 运行模拟
```bash
# 编译WarpX (如果还没有编译)
make -j4

# 运行模拟
mpirun -np 4 ./main3d.llvm.TPROF.MTMPI.OMP.QED.ex inputs_quartz_ring_ccp
```

### 2. 分析结果
```bash
# 分析特定时间步的结果
python analysis_quartz_ring_ccp.py plt00050
```

## 参数配置

### 石英环几何参数
```bash
quartz.ring_inner_radius = 0.05    # 内半径 (m)
quartz.ring_outer_radius = 0.08    # 外半径 (m)
quartz.ring_height = 0.02          # 高度 (m)
quartz.ring_center_x = 0.0         # 中心x坐标 (m)
quartz.ring_center_y = 0.0         # 中心y坐标 (m)
quartz.ring_bottom_z = -0.01       # 底部z坐标 (m)
```

### 石英材料参数
```bash
quartz.epsilon_r = 3.8             # 相对介电常数
quartz.mu_r = 1.0                  # 相对磁导率
quartz.sigma = 1e-12               # 电导率 (S/m)
```

### 边界条件设置
```bash
boundary.field_lo = quartz quartz quartz
boundary.field_hi = quartz quartz quartz
```

## 预期结果

### 电场分布
- 石英环内外的电场强度应该有明显差异
- 石英环内的电场强度约为外部的3.8倍 (εr)
- 电场线在石英环边界处会发生折射

### 磁场分布
- 由于石英是非磁性材料，磁场分布相对均匀
- 石英环对磁场的影响较小

### 等离子体行为
- 等离子体密度在石英环附近会有变化
- 电荷分离效应在石英环边界处明显

## 验证方法

### 1. 介电常数验证
分析脚本会自动计算石英环内外的电场比值，并与理论值(εr=3.8)比较。

### 2. 边界条件验证
检查电场和磁场的连续性条件是否满足。

### 3. 物理合理性
- 电场强度是否在合理范围内
- 等离子体密度分布是否合理
- 能量守恒是否满足

## 扩展应用

### 1. 不同几何形状
可以修改石英环的几何参数来模拟不同的CCP设备配置。

### 2. 材料参数变化
可以调整石英的介电常数、磁导率等参数来模拟不同的绝缘材料。

### 3. 多物理场耦合
可以结合其他物理过程，如化学反应、热传导等。

## 故障排除

### 常见问题
1. **编译错误**: 确保Quartz.cpp已添加到Make.package中
2. **边界条件错误**: 检查边界条件设置是否正确
3. **数值不稳定**: 调整时间步长或网格分辨率

### 调试技巧
1. 使用较小的网格进行测试
2. 检查输出文件中的场分布
3. 验证物理参数是否合理

## 参考文献

1. Lieberman, M. A., & Lichtenberg, A. J. (2005). Principles of plasma discharges and materials processing. John Wiley & Sons.
2. Chen, F. F. (2016). Introduction to plasma physics and controlled fusion. Springer.
3. Chabert, P., & Braithwaite, N. (2011). Physics of radio-frequency plasmas. Cambridge University Press. 