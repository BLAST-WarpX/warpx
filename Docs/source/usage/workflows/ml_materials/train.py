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
import time

import neural_network_classes as mynn
import torch
import torch.nn.functional as F
import torch.optim as optim

############# set model parameters #################

stage_i = 0
species = f"beam_stage_{stage_i}"
source_index = 0
target_index = 1
survivor_select_index = 1

data_dim = 6
n_in = data_dim
n_out = data_dim

learning_rate = 0.0001
n_epochs = 10
batch_size = 1200

loss_fun = F.mse_loss

n_hidden_nodes = 20
n_hidden_layers = 3
activation_type = "ReLU"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}")
#################### load dataset ################
dataset_filename = f"dataset_{species}.pt"
dataset_file = "datasets/" + dataset_filename

print(f"trying to load dataset+test-train split in {dataset_file}")

dataset_with_indices = torch.load(dataset_file)
train_data = torch.utils.data.dataset.Subset(
    dataset_with_indices["dataset"], dataset_with_indices["train_indices"]
)
test_data = torch.utils.data.dataset.Subset(
    dataset_with_indices["dataset"], dataset_with_indices["test_indices"]
)
source_data = dataset_with_indices["dataset"]
source_means = dataset_with_indices["source_means"]
source_stds = dataset_with_indices["source_stds"]
target_means = dataset_with_indices["target_means"]
target_stds = dataset_with_indices["target_stds"]
print("able to load data and test/train split")

###### move data to device (GPU) if available ########
source_device = train_data.dataset.tensors[0].to(
    device
)  # equivalently, test_data.tensors[0].to(device)
target_device = train_data.dataset.tensors[1].to(device)
full_dataset_device = torch.utils.data.TensorDataset(
    source_device.float(), target_device.float()
)

train_data_device = torch.utils.data.dataset.Subset(
    full_dataset_device, train_data.indices
)
test_data_device = torch.utils.data.dataset.Subset(
    full_dataset_device, test_data.indices
)

train_loader_device = torch.utils.data.DataLoader(
    train_data_device, batch_size=batch_size, shuffle=True
)
test_loader_device = torch.utils.data.DataLoader(
    test_data_device, batch_size=batch_size, shuffle=True
)

test_source_device = test_data_device.dataset.tensors[0]
test_target_device = test_data_device.dataset.tensors[1]

training_set_size = len(train_data_device.indices)
testing_set_size = len(test_data_device.indices)

###### create model ###########

model = mynn.OneActNN(
    n_in=n_in,
    n_out=n_out,
    n_hidden_nodes=n_hidden_nodes,
    n_hidden_layers=n_hidden_layers,
    act=activation_type,
)

training_time = 0
train_loss_list = []
test_loss_list = []

model.to(device=device)


########## train and test functions ####
# Manual: Train function START
def train(model, optimizer, train_loader, loss_fun):
    model.train()
    total_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        # evaluate network with data
        output = model(data)
        # compute loss
        # sum the differences squared, take mean afterward
        loss = loss_fun(output, target, reduction="sum")
        # backpropagation: step optimizer and reset gradients
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss


# Manual: Train function END


def test(model, test_loader, loss_fun):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            output = model(data)
            total_loss += loss_fun(output, target, reduction="sum").item()
    return total_loss


# Manual: Test function START
def test_dataset(model, test_source, test_target, loss_fun):
    model.eval()
    with torch.no_grad():
        output = model(test_source)
        return loss_fun(output, test_target, reduction="sum").item()


# Manual: Test function END

######## training loop ########

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

do_print = True

t3 = time.time()
# Manual: Training loop START
for epoch in range(n_epochs):
    if do_print:
        t1 = time.time()
    ave_train_loss = (
        train(model, optimizer, train_loader_device, loss_fun)
        / data_dim
        / training_set_size
    )
    ave_test_loss = (
        test_dataset(model, test_source_device, test_target_device, loss_fun)
        / data_dim
        / training_set_size
    )
    train_loss_list.append(ave_train_loss)
    test_loss_list.append(ave_test_loss)

    if do_print:
        t2 = time.time()
        print(
            "Train Epoch: {:04d} \tTrain Loss: {:.6f} \tTest Loss: {:.6f}, this epoch: {:.3f} s".format(
                epoch + 1, ave_train_loss, ave_test_loss, t2 - t1
            )
        )
# Manual: Training loop END
t4 = time.time()
print(f"total training time: {t4 - t3:.3f}s")

######### save model #########

os.makedirs("models", exist_ok=True)

# Manual: Save model START
model.to(device="cpu")
torch.save(
    {
        "n_hidden_layers": n_hidden_layers,
        "n_hidden_nodes": n_hidden_nodes,
        "activation": activation_type,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss_list": train_loss_list,
        "test_loss_list": test_loss_list,
        "training_time": training_time,
    },
    f"models/{species}_model.pt",
)
# Manual: Save model END
