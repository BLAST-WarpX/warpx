#!/usr/bin/env python3
"""
石英环CCP (Capacitively Coupled Plasma) 模拟分析脚本
分析等离子体在石英环边界条件下的行为
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import yt
import sys
import os

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

def analyze_quartz_ring_simulation(plotfile_path):
    """
    分析石英环CCP模拟结果
    
    Parameters:
    -----------
    plotfile_path : str
        WarpX输出文件的路径
    """
    
    print(f"正在分析文件: {plotfile_path}")
    
    # 加载数据
    ds = yt.load(plotfile_path)
    
    # 获取网格信息
    grid = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, 
                           dims=ds.domain_dimensions)
    
    # 提取场数据
    ex = grid['Ex'].to_ndarray()
    ey = grid['Ey'].to_ndarray()
    ez = grid['Ez'].to_ndarray()
    bx = grid['Bx'].to_ndarray()
    by = grid['By'].to_ndarray()
    bz = grid['Bz'].to_ndarray()
    rho = grid['rho'].to_ndarray()
    
    # 计算电场强度
    e_magnitude = np.sqrt(ex**2 + ey**2 + ez**2)
    b_magnitude = np.sqrt(bx**2 + by**2 + bz**2)
    
    # 创建分析图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('石英环CCP等离子体模拟分析', fontsize=16)
    
    # 1. 电场强度分布 (xy平面，z=0)
    im1 = axes[0,0].imshow(e_magnitude[:,:,ez.shape[2]//2], 
                          extent=[ds.domain_left_edge[0], ds.domain_right_edge[0],
                                 ds.domain_left_edge[1], ds.domain_right_edge[1]],
                          cmap='plasma', aspect='equal')
    axes[0,0].set_title('电场强度分布 (xy平面)')
    axes[0,0].set_xlabel('x (m)')
    axes[0,0].set_ylabel('y (m)')
    plt.colorbar(im1, ax=axes[0,0], label='|E| (V/m)')
    
    # 2. 磁场强度分布 (xy平面，z=0)
    im2 = axes[0,1].imshow(b_magnitude[:,:,bz.shape[2]//2],
                          extent=[ds.domain_left_edge[0], ds.domain_right_edge[0],
                                 ds.domain_left_edge[1], ds.domain_right_edge[1]],
                          cmap='viridis', aspect='equal')
    axes[0,1].set_title('磁场强度分布 (xy平面)')
    axes[0,1].set_xlabel('x (m)')
    axes[0,1].set_ylabel('y (m)')
    plt.colorbar(im2, ax=axes[0,1], label='|B| (T)')
    
    # 3. 电荷密度分布 (xy平面，z=0)
    im3 = axes[0,2].imshow(rho[:,:,rho.shape[2]//2],
                          extent=[ds.domain_left_edge[0], ds.domain_right_edge[0],
                                 ds.domain_left_edge[1], ds.domain_right_edge[1]],
                          cmap='RdBu_r', aspect='equal')
    axes[0,2].set_title('电荷密度分布 (xy平面)')
    axes[0,2].set_xlabel('x (m)')
    axes[0,2].set_ylabel('y (m)')
    plt.colorbar(im3, ax=axes[0,2], label='ρ (C/m³)')
    
    # 4. 电场强度分布 (xz平面，y=0)
    im4 = axes[1,0].imshow(e_magnitude[:,ey.shape[1]//2,:],
                          extent=[ds.domain_left_edge[0], ds.domain_right_edge[0],
                                 ds.domain_left_edge[2], ds.domain_right_edge[2]],
                          cmap='plasma', aspect='equal')
    axes[1,0].set_title('电场强度分布 (xz平面)')
    axes[1,0].set_xlabel('x (m)')
    axes[1,0].set_ylabel('z (m)')
    plt.colorbar(im4, ax=axes[1,0], label='|E| (V/m)')
    
    # 5. 磁场强度分布 (xz平面，y=0)
    im5 = axes[1,1].imshow(b_magnitude[:,by.shape[1]//2,:],
                          extent=[ds.domain_left_edge[0], ds.domain_right_edge[0],
                                 ds.domain_left_edge[2], ds.domain_right_edge[2]],
                          cmap='viridis', aspect='equal')
    axes[1,1].set_title('磁场强度分布 (xz平面)')
    axes[1,1].set_xlabel('x (m)')
    axes[1,1].set_ylabel('z (m)')
    plt.colorbar(im5, ax=axes[1,1], label='|B| (T)')
    
    # 6. 电荷密度分布 (xz平面，y=0)
    im6 = axes[1,2].imshow(rho[:,rho.shape[1]//2,:],
                          extent=[ds.domain_left_edge[0], ds.domain_right_edge[0],
                                 ds.domain_left_edge[2], ds.domain_right_edge[2]],
                          cmap='RdBu_r', aspect='equal')
    axes[1,2].set_title('电荷密度分布 (xz平面)')
    axes[1,2].set_xlabel('x (m)')
    axes[1,2].set_ylabel('z (m)')
    plt.colorbar(im6, ax=axes[1,2], label='ρ (C/m³)')
    
    plt.tight_layout()
    plt.savefig('quartz_ring_ccp_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印统计信息
    print("\n=== 石英环CCP模拟统计信息 ===")
    print(f"电场强度范围: {e_magnitude.min():.2e} - {e_magnitude.max():.2e} V/m")
    print(f"磁场强度范围: {b_magnitude.min():.2e} - {b_magnitude.max():.2e} T")
    print(f"电荷密度范围: {rho.min():.2e} - {rho.max():.2e} C/m³")
    print(f"平均电场强度: {e_magnitude.mean():.2e} V/m")
    print(f"平均磁场强度: {b_magnitude.mean():.2e} T")
    print(f"平均电荷密度: {rho.mean():.2e} C/m³")
    
    # 分析石英环区域
    analyze_quartz_ring_region(ds, e_magnitude, b_magnitude, rho)

def analyze_quartz_ring_region(ds, e_magnitude, b_magnitude, rho):
    """
    分析石英环区域的特殊性质
    """
    print("\n=== 石英环区域分析 ===")
    
    # 石英环参数 (从输入文件读取)
    ring_inner_radius = 0.05
    ring_outer_radius = 0.08
    ring_center_x = 0.0
    ring_center_y = 0.0
    ring_bottom_z = -0.01
    ring_height = 0.02
    
    # 创建石英环掩码
    x_coords = np.linspace(ds.domain_left_edge[0], ds.domain_right_edge[0], e_magnitude.shape[0])
    y_coords = np.linspace(ds.domain_left_edge[1], ds.domain_right_edge[1], e_magnitude.shape[1])
    z_coords = np.linspace(ds.domain_left_edge[2], ds.domain_right_edge[2], e_magnitude.shape[2])
    
    X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    
    # 计算到中心的距离
    R = np.sqrt((X - ring_center_x)**2 + (Y - ring_center_y)**2)
    
    # 石英环掩码
    in_radius = (R >= ring_inner_radius) & (R <= ring_outer_radius)
    in_height = (Z >= ring_bottom_z) & (Z <= ring_bottom_z + ring_height)
    quartz_mask = in_radius & in_height
    
    # 分析石英环内外的差异
    if np.any(quartz_mask):
        e_quartz = e_magnitude[quartz_mask]
        b_quartz = b_magnitude[quartz_mask]
        rho_quartz = rho[quartz_mask]
        
        e_outside = e_magnitude[~quartz_mask]
        b_outside = b_magnitude[~quartz_mask]
        rho_outside = rho[~quartz_mask]
        
        print(f"石英环内平均电场强度: {e_quartz.mean():.2e} V/m")
        print(f"石英环外平均电场强度: {e_outside.mean():.2e} V/m")
        print(f"电场强度比值 (内/外): {e_quartz.mean()/e_outside.mean():.3f}")
        
        print(f"石英环内平均磁场强度: {b_quartz.mean():.2e} T")
        print(f"石英环外平均磁场强度: {b_outside.mean():.2e} T")
        print(f"磁场强度比值 (内/外): {b_quartz.mean()/b_outside.mean():.3f}")
        
        print(f"石英环内平均电荷密度: {rho_quartz.mean():.2e} C/m³")
        print(f"石英环外平均电荷密度: {rho_outside.mean():.2e} C/m³")
        
        # 验证介电边界条件
        expected_e_ratio = 3.8  # 石英的相对介电常数
        actual_e_ratio = e_quartz.mean() / e_outside.mean()
        print(f"理论电场比值 (ε_r): {expected_e_ratio}")
        print(f"实际电场比值: {actual_e_ratio:.3f}")
        print(f"相对误差: {abs(actual_e_ratio - expected_e_ratio)/expected_e_ratio*100:.2f}%")
    else:
        print("警告: 未检测到石英环区域")

def main():
    """主函数"""
    if len(sys.argv) != 2:
        print("用法: python analysis_quartz_ring_ccp.py <plotfile>")
        print("示例: python analysis_quartz_ring_ccp.py plt00050")
        sys.exit(1)
    
    plotfile = sys.argv[1]
    
    if not os.path.exists(plotfile):
        print(f"错误: 文件 {plotfile} 不存在")
        sys.exit(1)
    
    try:
        analyze_quartz_ring_simulation(plotfile)
        print("\n分析完成！结果已保存为 quartz_ring_ccp_analysis.png")
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 