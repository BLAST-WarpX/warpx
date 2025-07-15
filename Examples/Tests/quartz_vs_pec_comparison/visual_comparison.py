#!/usr/bin/env python3
"""
Quartz vs PEC Boundary Visual Comparison Script
Generate intuitive comparison plots to highlight differences
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import yt


def load_data(quartz_file, pec_file):
    """Load quartz and PEC boundary data"""
    print(f"Loading quartz boundary data: {quartz_file}")
    quartz_ds = yt.load(quartz_file)

    print(f"Loading PEC boundary data: {pec_file}")
    pec_ds = yt.load(pec_file)

    return quartz_ds, pec_ds


def extract_field_data(ds, field_name):
    """Extract specified field data"""
    ad = ds.all_data()
    field_data = ad[("boxlib", field_name)]
    return field_data


def create_difference_histograms(quartz_ds, pec_ds, output_dir="comparison_plots"):
    """Create difference histograms"""
    os.makedirs(output_dir, exist_ok=True)

    fields = ["Ex", "Ey", "Ez", "Bx", "By", "Bz"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        "Quartz vs PEC Boundary - Field Difference Histograms",
        fontsize=16,
        fontweight="bold",
    )

    for i, field in enumerate(fields):
        row = i // 3
        col = i % 3

        quartz_field = extract_field_data(quartz_ds, field)
        pec_field = extract_field_data(pec_ds, field)

        diff = quartz_field - pec_field
        relative_diff = np.where(pec_field != 0, diff / np.abs(pec_field), 0)

        max_diff = np.max(np.abs(diff))
        mean_diff = np.mean(np.abs(diff))
        max_rel_diff = np.max(np.abs(relative_diff)) * 100

        axes[row, col].hist(diff, bins=50, alpha=0.7, color="red", edgecolor="black")
        axes[row, col].axvline(
            x=0, color="blue", linestyle="--", linewidth=2, label="Zero"
        )
        axes[row, col].set_xlabel(f"{field} Difference")
        axes[row, col].set_ylabel("Frequency")
        axes[row, col].set_title(
            f"{field}\nMax: {max_diff:.2e}\nMean: {mean_diff:.2e}\nRel: {max_rel_diff:.2f}%"
        )
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/field_differences_histogram.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(f"Difference histograms saved: {output_dir}/field_differences_histogram.png")


def create_profile_comparison(quartz_ds, pec_ds, output_dir="comparison_plots"):
    """Create profile comparison plots"""
    os.makedirs(output_dir, exist_ok=True)

    fields = ["Ex", "Ey", "Ez"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Quartz vs PEC Boundary - Center Line Profiles", fontsize=16, fontweight="bold"
    )

    for i, field in enumerate(fields):
        quartz_line = quartz_ds.ortho_ray(axis="x", coords=(0.0, 0.0))
        pec_line = pec_ds.ortho_ray(axis="x", coords=(0.0, 0.0))

        quartz_field = quartz_line[("boxlib", field)]
        pec_field = pec_line[("boxlib", field)]
        x_coords = quartz_line["x"]

        axes[i].plot(
            x_coords, quartz_field, "r-", linewidth=2, label="Quartz", alpha=0.8
        )
        axes[i].plot(x_coords, pec_field, "b--", linewidth=2, label="PEC", alpha=0.8)
        axes[i].plot(
            x_coords,
            quartz_field - pec_field,
            "g-",
            linewidth=1,
            label="Diff",
            alpha=0.6,
        )

        axes[i].set_xlabel("X Coordinate")
        axes[i].set_ylabel(f"{field} Field")
        axes[i].set_title(f"{field} Profile")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

        correlation = np.corrcoef(quartz_field, pec_field)[0, 1]
        axes[i].text(
            0.05,
            0.95,
            f"Corr: {correlation:.4f}",
            transform=axes[i].transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    plt.savefig(f"{output_dir}/profile_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Profile comparison saved: {output_dir}/profile_comparison.png")


def create_statistics_summary(quartz_ds, pec_ds, output_dir="comparison_plots"):
    """Create statistics summary"""
    os.makedirs(output_dir, exist_ok=True)

    fields = ["Ex", "Ey", "Ez", "Bx", "By", "Bz", "rho"]

    stats_data = []
    for field in fields:
        quartz_field = extract_field_data(quartz_ds, field)
        pec_field = extract_field_data(pec_ds, field)

        diff = quartz_field - pec_field
        relative_diff = np.where(np.abs(pec_field) > 1e-12, diff / np.abs(pec_field), 0)

        stats = {
            "field": field,
            "max_diff": np.max(np.abs(diff)),
            "mean_diff": np.mean(np.abs(diff)),
            "max_rel_diff": np.max(np.abs(relative_diff)) * 100,
            "correlation": np.corrcoef(quartz_field, pec_field)[0, 1],
        }
        stats_data.append(stats)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Quartz vs PEC Boundary - Statistics Summary", fontsize=16, fontweight="bold"
    )

    fields_names = [s["field"] for s in stats_data]

    # Max absolute difference
    max_diffs = [s["max_diff"] for s in stats_data]
    ax1.bar(fields_names, max_diffs, color="red", alpha=0.7)
    ax1.set_title("Max Absolute Difference")
    ax1.set_ylabel("Max Difference")
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(True, alpha=0.3)

    # Max relative difference
    max_rel_diffs = [s["max_rel_diff"] for s in stats_data]
    ax2.bar(fields_names, max_rel_diffs, color="blue", alpha=0.7)
    ax2.set_title("Max Relative Difference (%)")
    ax2.set_ylabel("Max Rel Diff (%)")
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(True, alpha=0.3)

    # Correlation
    correlations = [s["correlation"] for s in stats_data]
    ax3.bar(fields_names, correlations, color="green", alpha=0.7)
    ax3.set_title("Correlation Coefficient")
    ax3.set_ylabel("Correlation")
    ax3.tick_params(axis="x", rotation=45)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=1.0, color="black", linestyle="--", alpha=0.5)

    # Mean difference
    mean_diffs = [s["mean_diff"] for s in stats_data]
    ax4.bar(fields_names, mean_diffs, color="orange", alpha=0.7)
    ax4.set_title("Mean Absolute Difference")
    ax4.set_ylabel("Mean Difference")
    ax4.tick_params(axis="x", rotation=45)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/statistics_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Save table
    with open(f"{output_dir}/statistics_table.txt", "w") as f:
        f.write("Quartz vs PEC Boundary Statistics\n")
        f.write("=" * 40 + "\n\n")
        f.write(
            f"{'Field':<8} {'Max Diff':<12} {'Mean Diff':<12} {'Max Rel(%)':<12} {'Corr':<8}\n"
        )
        f.write("-" * 60 + "\n")
        for stats in stats_data:
            f.write(
                f"{stats['field']:<8} {stats['max_diff']:<12.2e} {stats['mean_diff']:<12.2e} "
                f"{stats['max_rel_diff']:<12.2f} {stats['correlation']:<8.4f}\n"
            )

    print(f"Statistics summary saved: {output_dir}/statistics_summary.png")
    print(f"Statistics table saved: {output_dir}/statistics_table.txt")


def verify_control_variables():
    """Verify only boundary conditions are different"""
    print("\n=== Control Variables Check ===")

    with open("inputs_quartz_laser", "r") as f:
        quartz_input = f.read()

    with open("inputs_pec_laser", "r") as f:
        pec_input = f.read()

    quartz_lines = quartz_input.split("\n")
    pec_lines = pec_input.split("\n")

    quartz_boundary = [line for line in quartz_lines if "boundary.field" in line]
    pec_boundary = [line for line in pec_lines if "boundary.field" in line]

    print("Boundary settings:")
    print("Quartz:", quartz_boundary)
    print("PEC:", pec_boundary)

    quartz_params = {}
    pec_params = {}

    for line in quartz_lines:
        if "=" in line and not line.strip().startswith("#"):
            key, value = line.split("=", 1)
            quartz_params[key.strip()] = value.strip()

    for line in pec_lines:
        if "=" in line and not line.strip().startswith("#"):
            key, value = line.split("=", 1)
            pec_params[key.strip()] = value.strip()

    differences = []
    for key in set(quartz_params.keys()) | set(pec_params.keys()):
        if key not in quartz_params or key not in pec_params:
            differences.append(f"Parameter {key} only in one file")
        elif quartz_params[key] != pec_params[key]:
            differences.append(f"Parameter {key} differs")

    print(f"\nParameter differences: {len(differences)}")
    if differences:
        print("Found differences:")
        for diff in differences[:5]:  # Show first 5
            print(f"  - {diff}")
        if len(differences) > 5:
            print(f"  ... and {len(differences) - 5} more")
    else:
        print("✅ All parameters identical except boundary conditions!")

    return len(differences) == 0


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 visual_comparison.py <quartz_diag_file> <pec_diag_file>")
        sys.exit(1)

    quartz_file = sys.argv[1]
    pec_file = sys.argv[2]

    print("=== Quartz vs PEC Boundary Visual Comparison ===")

    control_verified = verify_control_variables()

    quartz_ds, pec_ds = load_data(quartz_file, pec_file)

    print("\n=== Generating Plots ===")
    create_difference_histograms(quartz_ds, pec_ds)
    create_profile_comparison(quartz_ds, pec_ds)
    create_statistics_summary(quartz_ds, pec_ds)

    print("\n=== Analysis Complete ===")
    print("All plots saved to comparison_plots/ directory")
    if control_verified:
        print("✅ Control variables verified: Only boundary conditions differ")
    else:
        print("⚠️  Other parameter differences exist")


if __name__ == "__main__":
    main()
