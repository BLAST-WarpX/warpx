# CCP石英环边界条件测试

这个测试演示了如何在WarpX中使用新添加的石英边界条件，专门针对CCP（电容耦合等离子体）设备中的石英环应用。

## CCP石英环边界条件特性

石英环边界条件专门为CCP等离子体设备设计，模拟石英环的电磁特性：

### 石英环几何结构
- **内径**: 可配置的内半径（默认0.05m）
- **外径**: 可配置的外半径（默认0.08m）
- **高度**: 可配置的环高度（默认0.02m）
- **位置**: 可配置的中心位置和底部位置

### 材料特性

- **相对介电常数 (εr)**: 默认值为3.8（石英的典型值）
- **相对磁导率 (μr)**: 默认值为1.0（石英是非磁性材料）
- **电导率 (σ)**: 默认值为0.0（石英是绝缘体）

## 边界条件实现

### 电场边界条件
- **切向分量**: 连续（E_tangential_outside = E_tangential_inside）
- **法向分量**: 不连续，满足 D_normal 连续性
  - E_normal_outside = εr × E_normal_inside

### 磁场边界条件
- **所有分量**: 连续（石英是非磁性材料，μr ≈ 1）
- B_outside = B_inside

## 使用方法

### 1. 设置边界条件类型
在输入文件中设置边界条件为 `quartz`（石英环）：

```bash
boundary.field_lo = quartz quartz quartz
boundary.field_hi = quartz quartz quartz
```

### 2. 配置石英环参数
可以自定义石英环的材料属性和几何参数：

```bash
# 基本材料参数
quartz.epsilon_r = 3.8    # 相对介电常数
quartz.mu_r = 1.0         # 相对磁导率
quartz.sigma = 0.0        # 电导率

# 石英环几何参数
quartz.ring_inner_radius = 0.05    # 内半径 (m)
quartz.ring_outer_radius = 0.08    # 外半径 (m)
quartz.ring_height = 0.02          # 高度 (m)
quartz.ring_center_x = 0.0         # 中心x坐标 (m)
quartz.ring_center_y = 0.0         # 中心y坐标 (m)
quartz.ring_bottom_z = 0.0         # 底部z坐标 (m)

# 空间相关参数（可选）
quartz.epsilon_r_function(x,y,z) = "3.8 + 0.1*sin(2*pi*x)"
quartz.mu_r_function(x,y,z) = "1.0"
quartz.sigma_function(x,y,z) = "0.0"

# 自定义几何函数（可选）
quartz.ring_geometry_function(x,y,z) = "sqrt((x-0.05)^2 + y^2) - 0.03"
```

### 3. 运行测试
```bash
# 编译WarpX（如果还没有编译）
cd /path/to/warpx
make -j4

# 运行测试
cd Examples/Tests/quartz_boundary_test
../../../build/bin/warpx.3d inputs_quartz_test
```

## 物理原理

石英边界条件基于麦克斯韦方程组的边界条件：

1. **电场边界条件**:
   - 电位移矢量的法向分量连续：D_normal_outside = D_normal_inside
   - 电场的切向分量连续：E_tangential_outside = E_tangential_inside

2. **磁场边界条件**:
   - 磁感应强度的法向分量连续：B_normal_outside = B_normal_inside
   - 磁场强度的切向分量连续：H_tangential_outside = H_tangential_inside

## CCP石英环应用场景

石英环边界条件专门适用于CCP等离子体设备：

1. **等离子体刻蚀**: 石英环作为电极间的绝缘隔离
2. **等离子体沉积**: 控制等离子体区域和气体流动
3. **等离子体诊断**: 石英环作为诊断窗口和探针支撑
4. **等离子体约束**: 限制等离子体在特定区域内
5. **温度控制**: 作为热隔离层，控制电极温度

1. **光学器件模拟**: 石英透镜、棱镜等光学元件
2. **等离子体诊断**: 石英窗口、探针等诊断设备
3. **激光系统**: 石英激光器、光学谐振腔
4. **材料研究**: 石英材料的电磁特性研究

## 注意事项

1. 石英边界条件适用于低频到光学频率范围
2. 对于高频应用，可能需要考虑色散效应
3. 空间相关的材料属性需要谨慎使用，确保物理合理性
4. 与其他边界条件（如PML）的组合使用需要测试验证

## 验证方法

可以通过以下方式验证石英边界条件的正确性：

1. **解析解对比**: 与已知解析解进行对比
2. **能量守恒**: 检查电磁场能量是否守恒
3. **边界条件检查**: 验证边界处的场连续性
4. **收敛性测试**: 检查网格细化时的收敛性 