#!/usr/bin/env python3
#
# Copyright 2023 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: Ryan Sandberg
# License: BSD-3-Clause-LBNL
#

import os
import zipfile
from urllib import request

import numpy as np
import torch
from openpmd_viewer import OpenPMDTimeSeries, ParticleTracker

c = 2.998e8
###############


def sanitize_dir_strings(*dir_strings):
    """append '/' to a string for concatenation in building up file tree descriptions"""
    dir_strings = list(dir_strings)
    for ii, dir_string in enumerate(dir_strings):
        if dir_string[-1] != "/":
            dir_strings[ii] = dir_string + "/"

    return dir_strings


def download_and_unzip(url, data_dir):
    request.urlretrieve(url, data_dir)
    with zipfile.ZipFile(data_dir, "r") as zip_dataset:
        zip_dataset.extractall()


def create_source_target_data(
    data_dir,
    species,
    source_index=0,
    target_index=-1,
    survivor_select_index=-1,
    particle_selection=None,
):
    """Create dataset from openPMD files

    Parameters
    ---
    data_dir : string, location of diagnostic data
    source_index : int, which index to take source data from
    target_index : int, which index to take target data from
    particle_selection: dictionary, optional, selection criterion for dataset

    Returns
    ---
    source_data:  Nx6 array of source particle data
    source_means: 6 element array of source particle coordinate means
    source_stds:  6 element array of source particle coordinate standard deviations
    target_data:  Nx6 array of target particle data
    target_means: 6 element array of target particle coordinate means
    target_stds:  6 element array of source particle coordinate standard deviations
    relevant times: 2 element array of source and target times
    """
    (data_dir,) = sanitize_dir_strings(data_dir)
    data_path = data_dir
    print("loading openPMD data from", data_path)
    ts = OpenPMDTimeSeries(data_path)
    relevant_times = [ts.t[source_index], ts.t[target_index]]

    # Manual: Particle tracking START
    iteration = ts.iterations[survivor_select_index]
    pt = ParticleTracker(
        ts, species=species, iteration=iteration, select=particle_selection
    )
    # Manual: Particle tracking END

    #### create normalized source, target data sets ####
    print("creating data sets")

    # Manual: Load openPMD START
    iteration = ts.iterations[source_index]
    source_data = ts.get_particle(
        species=species,
        iteration=iteration,
        var_list=["x", "y", "z", "ux", "uy", "uz"],
        select=pt,
    )

    iteration = ts.iterations[target_index]
    target_data = ts.get_particle(
        species=species,
        iteration=iteration,
        var_list=["x", "y", "z", "ux", "uy", "uz"],
        select=pt,
    )
    # Manual: Load openPMD END

    # Manual: Normalization START
    target_means = np.zeros(6)
    target_stds = np.zeros(6)
    source_means = np.zeros(6)
    source_stds = np.zeros(6)
    for jj in range(6):
        source_means[jj] = source_data[jj].mean()
        source_stds[jj] = source_data[jj].std()
        source_data[jj] -= source_means[jj]
        source_data[jj] /= source_stds[jj]

    for jj in range(6):
        target_means[jj] = target_data[jj].mean()
        target_stds[jj] = target_data[jj].std()
        target_data[jj] -= target_means[jj]
        target_data[jj] /= target_stds[jj]
    # Manual: Normalization END

    # Manual: Format data START
    source_data = torch.tensor(np.column_stack(source_data))
    target_data = torch.tensor(np.column_stack(target_data))
    # Manual: Format data END

    return (
        source_data,
        source_means,
        source_stds,
        target_data,
        target_means,
        target_stds,
        relevant_times,
    )


def save_warpx_surrogate_data(
    dataset_fullpath_filename,
    diag_dir,
    species,
    training_frac,
    batch_size,
    source_index,
    target_index,
    survivor_select_index,
    particle_selection=None,
):
    source_target_data = create_source_target_data(
        data_dir=diag_dir,
        species=species,
        source_index=source_index,
        target_index=target_index,
        survivor_select_index=survivor_select_index,
        particle_selection=particle_selection,
    )
    (
        source_data,
        source_means,
        source_stds,
        target_data,
        target_means,
        target_stds,
        times,
    ) = source_target_data

    # Manual: Save dataset START
    full_dataset = torch.utils.data.TensorDataset(
        source_data.float(), target_data.float()
    )

    n_samples = full_dataset.tensors[0].size(0)
    n_train = int(training_frac * n_samples)
    n_test = n_samples - n_train

    train_data, test_data = torch.utils.data.random_split(
        full_dataset, [n_train, n_test]
    )

    torch.save(
        {
            "dataset": full_dataset,
            "train_indices": train_data.indices,
            "test_indices": test_data.indices,
            "source_means": source_means,
            "source_stds": source_stds,
            "target_means": target_means,
            "target_stds": target_stds,
            "times": times,
        },
        dataset_fullpath_filename,
    )
    # Manual: Save dataset END


######## end utility functions #############
######## start dataset creation ############

data_url = "https://zenodo.org/records/10810754/files/lab_particle_diags.zip?download=1"
download_and_unzip(data_url, "lab_particle_diags.zip")
data_dir = "lab_particle_diags/lab_particle_diags/"

# create data set

source_index = 0
target_index = 1
survivor_select_index = 1
batch_size = 1200
training_frac = 0.7

os.makedirs("datasets", exist_ok=True)

# improve stage 0 dataset
stage_i = 0
select = {"z": [0.280025, None]}
species = f"beam_stage_{stage_i}"
dataset_filename = f"dataset_{species}.pt"
dataset_file = "datasets/" + dataset_filename
save_warpx_surrogate_data(
    dataset_fullpath_filename=dataset_file,
    diag_dir=data_dir,
    species=species,
    training_frac=training_frac,
    batch_size=batch_size,
    source_index=source_index,
    target_index=target_index,
    survivor_select_index=survivor_select_index,
    particle_selection=select,
)

for stage_i in range(1, 15):
    species = f"beam_stage_{stage_i}"
    dataset_filename = f"dataset_{species}.pt"
    dataset_file = "datasets/" + dataset_filename
    save_warpx_surrogate_data(
        dataset_fullpath_filename=dataset_file,
        diag_dir=data_dir,
        species=species,
        training_frac=training_frac,
        batch_size=batch_size,
        source_index=source_index,
        target_index=target_index,
        survivor_select_index=survivor_select_index,
    )
