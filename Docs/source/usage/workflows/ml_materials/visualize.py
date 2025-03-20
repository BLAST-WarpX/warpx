#!/usr/bin/env python3
#
# Copyright 2023 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: Ryan Sandberg
# License: BSD-3-Clause-LBNL
#
import neural_network_classes as mynn
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

c = 2.998e8


# open model file
stage_i = 0
species = f"beam_stage_{stage_i}"
model_data = torch.load(f"models/{species}_model.pt", map_location=torch.device("cpu"))
data_dim = 6
n_in = data_dim
n_out = data_dim
n_hidden_layers = model_data["n_hidden_layers"]
n_hidden_nodes = model_data["n_hidden_nodes"]
activation_type = model_data["activation"]
train_loss_list = model_data["train_loss_list"]
test_loss_list = model_data["test_loss_list"]
training_time = model_data["training_time"]
loss_fun = F.mse_loss


n_epochs = len(train_loss_list)
train_counter = np.arange(n_epochs) + 1
test_counter = train_counter

do_log_plot = False
fig, ax = plt.subplots()
if do_log_plot:
    ax.semilogy(
        train_counter, train_loss_list, ".-", color="blue", label="training loss"
    )
    ax.semilogy(test_counter, test_loss_list, color="green", label="testing loss")
else:
    ax.plot(train_counter, train_loss_list, ".-", color="blue", label="training loss")
    ax.plot(test_counter, test_loss_list, color="green", label="testing loss")
ax.set_xlabel("number of epochs seen")
ax.set_ylabel(" loss")
ax.legend()
fig_dir = "figures/"
ax.set_title(f"final test error = {test_loss_list[-1]:.3e} ")
ax.grid()
plt.tight_layout()
plt.savefig(f"{species}_training_testing_error.png")


######### plot phase space comparison #######
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}")

model = mynn.OneActNN(
    n_in=n_in,
    n_out=n_out,
    n_hidden_nodes=n_hidden_nodes,
    n_hidden_layers=n_hidden_layers,
    act=activation_type,
)
model.load_state_dict(model_data["model_state_dict"])
model.to(device=device)

###### load model data ###############
dataset_filename = f"dataset_{species}.pt"
dataset_dir = "datasets/"
model_input_data = torch.load(dataset_dir + dataset_filename)
dataset = model_input_data["dataset"]
train_indices = model_input_data["train_indices"]
test_indices = model_input_data["test_indices"]
source_means = model_input_data["source_means"]
source_stds = model_input_data["source_stds"]
target_means = model_input_data["target_means"]
target_stds = model_input_data["target_stds"]
source_time, target_time = model_input_data["times"]


source = dataset.tensors[0]
test_source = source[test_indices]
test_source_device = test_source.to(device)
with torch.no_grad():
    evaluation_device = model(test_source_device.float())
eval_cpu = evaluation_device.to("cpu")

target = dataset.tensors[1]
test_target = target[test_indices]

target_si = test_target * target_stds + target_means
eval_cpu_si = eval_cpu * target_stds + target_means
target_mu = np.copy(target_si)
eval_cpu_mu = np.copy(eval_cpu_si)
target_mu[:, 2] -= c * target_time
eval_cpu_mu[:, 2] -= c * target_time
target_mu[:, :3] *= 1e6
eval_cpu_mu[:, :3] *= 1e6


loss_tensor = torch.sum(loss_fun(eval_cpu, test_target, reduction="none"), axis=1) / 6
loss_array = loss_tensor.detach().numpy()

tinds = np.nonzero(loss_array > 0.0)[0]
skip = 10

plt.figure()
fig, axT = plt.subplots(3, 3)
axes_label = {
    0: r"x [$\mu$m]",
    1: r"y [$\mu$m]",
    2: r"z - %.2f cm [$\mu$m]" % (c * target_time),
    3: r"$p_x$",
    4: r"$p_y$",
    5: r"$p_z$",
}
xy_inds = [(0, 1), (2, 0), (2, 1)]


def set_axes(ax, indx, indy):
    ax.scatter(
        target_mu[::skip, indx], target_mu[::skip, indy], s=8, c="k", label="simulation"
    )
    ax.scatter(
        eval_cpu_mu[::skip, indx],
        eval_cpu_mu[::skip, indy],
        marker="*",
        c=loss_array[::skip],
        s=0.02,
        label="surrogate",
        cmap="YlOrRd",
    )
    ax.set_xlabel(axes_label[indx])
    ax.set_ylabel(axes_label[indy])
    # return


for ii in range(3):
    ax = axT[0, ii]
    indx, indy = xy_inds[ii]
    set_axes(ax, indx, indy)

for ii in range(2):
    indx, indy = xy_inds[ii]
    ax = axT[1, ii]
    set_axes(ax, indx + 3, indy + 3)

for ii in range(3):
    ax = axT[2, ii]
    indx = ii
    indy = ii + 3
    set_axes(ax, indx, indy)


ax = axT[1, 2]
indx = 5
indy = 4
ax.scatter(
    target_mu[::skip, indx], target_mu[::skip, indy], s=8, c="k", label="simulation"
)
evalplt = ax.scatter(
    eval_cpu_mu[::skip, indx],
    eval_cpu_mu[::skip, indy],
    marker="*",
    c=loss_array[::skip],
    s=2,
    label="surrogate",
    cmap="YlOrRd",
)
ax.set_xlabel(axes_label[indx])
ax.set_ylabel(axes_label[indy])

cb = plt.colorbar(evalplt, ax=ax)
cb.set_label("MSE loss")

fig.suptitle(f"stage {stage_i} prediction")

plt.tight_layout()

plt.savefig(f"{species}_model_evaluation.png")
