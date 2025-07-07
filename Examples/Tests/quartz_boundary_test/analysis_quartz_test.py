#!/usr/bin/env python3
"""
Analysis script for quartz boundary condition test
This script analyzes the results of the quartz boundary condition test
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from openpmd_viewer import OpenPMDTimeSeries

def analyze_quartz_boundary_test(diag_path):
    """
    Analyze the results of the quartz boundary condition test
    
    Parameters:
    -----------
    diag_path : str
        Path to the diagnostic files
    """
    
    print("分析石英边界条件测试结果...")
    
    # Load the time series
    try:
        ts = OpenPMDTimeSeries(diag_path)
        print(f"找到 {len(ts.iterations)} 个时间步")
    except Exception as e:
        print(f"无法加载诊断文件: {e}")
        return
    
    # Get the last iteration
    iteration = ts.iterations[-1]
    time = ts.t[iteration]
    print(f"分析时间步 {iteration}, 时间 = {time:.6f}")
    
    # Get field data
    fields = ['E', 'B']
    field_components = ['x', 'y', 'z']
    
    # Analyze field values at boundaries
    print("\n边界场值分析:")
    
    for field in fields:
        print(f"\n{field} 场:")
        for comp in field_components:
            field_name = f"{field}{comp}"
            try:
                # Get field data
                F, info = ts.get_field(field=field_name, iteration=iteration)
                
                if F is not None:
                    # Check boundary values
                    nx, ny, nz = F.shape
                    
                    # Lower boundary (x=0)
                    lower_boundary = F[0, :, :]
                    # Upper boundary (x=nx-1)
                    upper_boundary = F[-1, :, :]
                    
                    print(f"  {field_name}:")
                    print(f"    下边界平均值: {np.mean(lower_boundary):.6e}")
                    print(f"    上边界平均值: {np.mean(upper_boundary):.6e}")
                    print(f"    下边界标准差: {np.std(lower_boundary):.6e}")
                    print(f"    上边界标准差: {np.std(upper_boundary):.6e}")
                    
            except Exception as e:
                print(f"    无法获取 {field_name}: {e}")
    
    # Check for quartz boundary effects
    print("\n石英边界条件验证:")
    
    # For quartz boundary, we expect:
    # 1. E_normal is discontinuous: E_outside = epsilon_r * E_inside
    # 2. E_tangential is continuous
    # 3. B field is continuous (since mu_r = 1)
    
    try:
        # Check E field normal component (Ex)
        Ex, info = ts.get_field(field='Ex', iteration=iteration)
        if Ex is not None:
            nx, ny, nz = Ex.shape
            
            # Get values just inside and outside the boundary
            E_inside = Ex[1, :, :]  # One cell inside
            E_outside = Ex[0, :, :]  # At boundary
            
            # Calculate the ratio
            ratio = np.mean(E_outside) / np.mean(E_inside)
            expected_ratio = 3.8  # epsilon_r for quartz
            
            print(f"  Ex 边界比值 (outside/inside): {ratio:.3f}")
            print(f"  期望比值 (epsilon_r): {expected_ratio:.3f}")
            print(f"  相对误差: {abs(ratio - expected_ratio) / expected_ratio * 100:.2f}%")
        
        # Check E field tangential component (Ey)
        Ey, info = ts.get_field(field='Ey', iteration=iteration)
        if Ey is not None:
            nx, ny, nz = Ey.shape
            
            E_inside = Ey[1, :, :]
            E_outside = Ey[0, :, :]
            
            ratio = np.mean(E_outside) / np.mean(E_inside)
            expected_ratio = 1.0  # Should be continuous
            
            print(f"  Ey 边界比值 (outside/inside): {ratio:.3f}")
            print(f"  期望比值 (continuous): {expected_ratio:.3f}")
            print(f"  相对误差: {abs(ratio - expected_ratio) / expected_ratio * 100:.2f}%")
            
    except Exception as e:
        print(f"  边界条件验证失败: {e}")
    
    # Create a simple plot
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot Ex field
        Ex, info = ts.get_field(field='Ex', iteration=iteration)
        if Ex is not None:
            im1 = axes[0,0].imshow(Ex[:, :, nz//2], cmap='RdBu_r')
            axes[0,0].set_title('Ex Field (y-z plane)')
            axes[0,0].set_xlabel('y')
            axes[0,0].set_ylabel('x')
            plt.colorbar(im1, ax=axes[0,0])
        
        # Plot Ey field
        Ey, info = ts.get_field(field='Ey', iteration=iteration)
        if Ey is not None:
            im2 = axes[0,1].imshow(Ey[:, :, nz//2], cmap='RdBu_r')
            axes[0,1].set_title('Ey Field (y-z plane)')
            axes[0,1].set_xlabel('y')
            axes[0,1].set_ylabel('x')
            plt.colorbar(im2, ax=axes[0,1])
        
        # Plot Bx field
        Bx, info = ts.get_field(field='Bx', iteration=iteration)
        if Bx is not None:
            im3 = axes[1,0].imshow(Bx[:, :, nz//2], cmap='RdBu_r')
            axes[1,0].set_title('Bx Field (y-z plane)')
            axes[1,0].set_xlabel('y')
            axes[1,0].set_ylabel('x')
            plt.colorbar(im3, ax=axes[1,0])
        
        # Plot By field
        By, info = ts.get_field(field='By', iteration=iteration)
        if By is not None:
            im4 = axes[1,1].imshow(By[:, :, nz//2], cmap='RdBu_r')
            axes[1,1].set_title('By Field (y-z plane)')
            axes[1,1].set_xlabel('y')
            axes[1,1].set_ylabel('x')
            plt.colorbar(im4, ax=axes[1,1])
        
        plt.tight_layout()
        plt.savefig('quartz_boundary_analysis.png', dpi=150, bbox_inches='tight')
        print("\n分析图已保存为 'quartz_boundary_analysis.png'")
        
    except Exception as e:
        print(f"绘图失败: {e}")
    
    print("\n分析完成!")

def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("用法: python analysis_quartz_test.py <diagnostic_path>")
        print("例如: python analysis_quartz_test.py diags/diag1000100")
        sys.exit(1)
    
    diag_path = sys.argv[1]
    analyze_quartz_boundary_test(diag_path)

if __name__ == "__main__":
    main() 