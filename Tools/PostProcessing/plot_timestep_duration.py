#!/usr/bin/env python3

import argparse
import re

import matplotlib.pyplot as plt
import numpy as np


def extract_data(filename):
    regex_step = re.compile(
        r"STEP [0-9]* ends.*\n.* Avg\. per step = ([0-9]*[.])?[0-9]+ s", re.MULTILINE
    )

    string_data = []

    print("Processing " + filename + " ...", end="")
    with open(filename) as f:
        text = f.read()

        string_data = [s.group(0) for s in regex_step.finditer(text)]

    regex_real = re.compile(r" -?[\d.]+(?:e-?\d+)?", re.MULTILINE)

    time_data = np.zeros([len(string_data), 6])
    for i, ss in enumerate(string_data):
        numbers = regex_real.findall(ss)
        time_data[i, :] = np.array(numbers)

    print("...done!")
    return time_data


def plot_timestep_duration(time_data, name):
    fig_name = name + "_ts_duration.png"
    print("Generating " + fig_name + "...", end="")

    plt.rcParams.update({"font.size": 20})
    plt.rcParams["axes.linewidth"] = 3

    f, ax = plt.subplots(figsize=(12, 6))

    ax.set_ylabel("timestep duration [s]")
    ax.set_xlabel("step [#]")

    ax.semilogy(time_data[:, 0], time_data[:, 4])

    ax.spines["bottom"].set_color("gray")
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_color("gray")
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    plt.savefig(fig_name, transparent=False, dpi=300)
    print("...done!")


def plot_cumulative_duration(time_data, name):
    fig_name = name + "_cumulative_duration.png"
    print("Generating " + fig_name + "...", end="")

    plt.rcParams.update({"font.size": 20})
    plt.rcParams["axes.linewidth"] = 3

    f, ax = plt.subplots(figsize=(12, 6))

    ax.set_ylabel("cumulative duration [s]")
    ax.set_xlabel("step [#]")

    ax.plot(time_data[:, 0], np.cumsum(time_data[:, 4]))

    ax.spines["bottom"].set_color("gray")
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_color("gray")
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    plt.savefig(name + "_cumulative_duration.png", transparent=False, dpi=300)
    print("...done!")


def do_plot_timestep_duration():
    parser = argparse.ArgumentParser(
        description="Generates plots of timestep duration from WarpX standard output logs"
    )
    parser.add_argument(
        "file_name",
        metavar="file_name",
        type=str,
        nargs=1,
        help="the name of the WarpX output log to process",
    )

    args = parser.parse_args()
    log_file_name = args.file_name[0]

    time_data = extract_data(log_file_name)

    plot_timestep_duration(time_data, log_file_name)
    plot_cumulative_duration(time_data, log_file_name)


if __name__ == "__main__":
    do_plot_timestep_duration()
