#!/usr/bin/env python3
"""
石英边界条件 vs PEC边界条件物理效果对比分析
验证石英边界条件的物理正确性
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import yt


def load_warpx_data(plotfile_path):
    """加载WarpX数据"""
    try:
        ds = yt.load(plotfile_path)
        return ds
    except Exception as e:
        print(f"无法加载数据文件 {plotfile_path}: {e}")
        return None


def analyze_boundary_effects(ds, boundary_type):
    """分析边界条件对电场的影响"""
    if ds is None:
        return None

    # 获取电场数据
    ad = ds.all_data()

    # 提取边界附近的电场数据
    # 选择靠近边界的区域进行分析
    boundary_region = ad.cut_region(
        "obj['index', 'x'] < -0.08 or obj['index', 'x'] > 0.08 or "
        "obj['index', 'y'] < -0.08 or obj['index', 'y'] > 0.08 or "
        "obj['index', 'z'] < -0.08 or obj['index', 'z'] > 0.08"
    )

    # 计算边界电场的统计信息
    E_fields = [("boxlib", "Ex"), ("boxlib", "Ey"), ("boxlib", "Ez")]
    stats = {}

    for field in E_fields:
        try:
            field_data = boundary_region[field].in_units("V/m")
            stats[field[1]] = {
                "mean": float(field_data.mean()),
                "std": float(field_data.std()),
                "max": float(field_data.max()),
                "min": float(field_data.min()),
                "rms": float(np.sqrt((field_data**2).mean())),
            }
        except Exception as e:
            print(f"字段 {field} 读取失败: {e}")

    return stats


def plot_field_comparison(quartz_ds, pec_ds):
    """生成电场对比图"""
    if quartz_ds is None or pec_ds is None:
        print("数据加载失败，无法生成对比图")
        return

    # 创建输出目录
    output_dir = "comparison_plots"
    os.makedirs(output_dir, exist_ok=True)

    # 设置中文字体
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 创建子图
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("石英边界条件 vs PEC边界条件物理效果对比", fontsize=16)

    # 电场对比
    E_fields = [("boxlib", "Ex"), ("boxlib", "Ey"), ("boxlib", "Ez")]

    for i, field in enumerate(E_fields):
        # 石英边界条件
        ax = axes[0, i]
        plot = yt.SlicePlot(quartz_ds, "z", field, center=[0, 0, 0], width=[0.2, 0.2])
        plot.set_cmap(field, "RdBu_r")
        plot_frb = plot.frb[field]
        im = ax.imshow(plot_frb, origin="lower")
        ax.set_title(f"石英边界 - {field[1]}")

        # PEC边界条件
        ax = axes[1, i]
        plot = yt.SlicePlot(pec_ds, "z", field, center=[0, 0, 0], width=[0.2, 0.2])
        plot.set_cmap(field, "RdBu_r")
        plot_frb = plot.frb[field]
        im = ax.imshow(plot_frb, origin="lower")
        ax.set_title(f"PEC边界 - {field[1]}")

    # 电荷密度对比
    # 石英边界条件
    ax = axes[0, 3]
    plot = yt.SlicePlot(
        quartz_ds, "z", ("boxlib", "rho"), center=[0, 0, 0], width=[0.2, 0.2]
    )
    plot.set_cmap(("boxlib", "rho"), "viridis")
    plot_frb = plot.frb[("boxlib", "rho")]
    im = ax.imshow(plot_frb, origin="lower")
    ax.set_title("石英边界 - 电荷密度")

    # PEC边界条件
    ax = axes[1, 3]
    plot = yt.SlicePlot(
        pec_ds, "z", ("boxlib", "rho"), center=[0, 0, 0], width=[0.2, 0.2]
    )
    plot.set_cmap(("boxlib", "rho"), "viridis")
    plot_frb = plot.frb[("boxlib", "rho")]
    im = ax.imshow(plot_frb, origin="lower")
    ax.set_title("PEC边界 - 电荷密度")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/field_comparison.png", dpi=300, bbox_inches="tight")
    print(f"对比图已保存到: {output_dir}/field_comparison.png")

    # 生成数值对比表
    generate_comparison_table(quartz_ds, pec_ds, output_dir)


def generate_comparison_table(quartz_ds, pec_ds, output_dir):
    """生成数值对比表"""
    if quartz_ds is None or pec_ds is None:
        return

    # 获取全局数据
    quartz_ad = quartz_ds.all_data()
    pec_ad = pec_ds.all_data()

    # 计算统计信息
    fields = [("boxlib", "Ex"), ("boxlib", "Ey"), ("boxlib", "Ez"), ("boxlib", "rho")]

    print("\n" + "=" * 60)
    print("石英边界条件 vs PEC边界条件数值对比")
    print("=" * 60)

    for field in fields:
        field_name = field[1]
        print(f"\n{field_name} 字段对比:")
        print("-" * 40)

        try:
            quartz_data = quartz_ad[field]
            pec_data = pec_ad[field]

            print(
                f"石英边界 - 范围: {float(quartz_data.min()):.2e} 到 {float(quartz_data.max()):.2e}"
            )
            print(
                f"PEC边界  - 范围: {float(pec_data.min()):.2e} 到 {float(pec_data.max()):.2e}"
            )
            print(
                f"差异     - 最大值差异: {float(pec_data.max() - quartz_data.max()):.2e}"
            )

        except Exception as e:
            print(f"字段 {field_name} 分析失败: {e}")


def main():
    """主函数"""
    if len(sys.argv) != 3:
        print("用法: python analysis_comparison.py <quartz_plotfile> <pec_plotfile>")
        sys.exit(1)

    quartz_plotfile = sys.argv[1]
    pec_plotfile = sys.argv[2]

    print("石英边界条件物理效果验证")
    print("=" * 40)

    # 加载数据
    print("加载石英边界条件数据...")
    quartz_ds = load_warpx_data(quartz_plotfile)

    print("加载PEC边界条件数据...")
    pec_ds = load_warpx_data(pec_plotfile)

    # 分析边界效应
    print("分析石英边界条件效应...")
    quartz_stats = analyze_boundary_effects(quartz_ds, "quartz")

    print("分析PEC边界条件效应...")
    pec_stats = analyze_boundary_effects(pec_ds, "pec")

    print("=" * 60)
    print("石英边界条件 vs PEC边界条件物理效果对比")
    print("=" * 60)

    # 生成对比图
    print("\n生成对比图...")
    plot_field_comparison(quartz_ds, pec_ds)

    print("\n物理验证总结:")
    print("1. 石英边界条件应该表现出介电材料的特性")
    print("2. 相比PEC边界，石英边界的电场应该更连续")
    print("3. 石英的介电常数(εr=3.8)应该影响边界处的电场分布")
    print("4. 磁场在石英边界处应该保持连续(μr≈1)")


if __name__ == "__main__":
    main()
