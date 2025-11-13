#!/usr/bin/env python3
#
# Copyright 2023 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: Ryan Sandberg
# License: BSD-3-Clause-LBNL
#
from enum import Enum

from torch import nn


class ActivationType(Enum):
    """
    Activation class provides an enumeration type for the supported activation layers
    """

    ReLU = 1
    Tanh = 2
    PReLU = 3
    Sigmoid = 4


def get_enum_type(type_to_test, EnumClass):
    """
    Returns the enumeration type associated to type_to_test in EnumClass

    Parameters
    ----------
    type_to_test: EnumClass, int or str
        object whose Enum class is to be obtained
    EnumClass: Enum class
        Enum class to test
    """
    if type(type_to_test) is EnumClass:
        return type_to_test
    if type(type_to_test) is int:
        return EnumClass(type_to_test)
    if type(type_to_test) is str:
        return getattr(EnumClass, type_to_test)
    else:
        raise Exception("unsupported type entered")


class ConnectedNN(nn.Module):
    """
    ConnectedNN is a class of fully connected neural networks
    """

    def __init__(self, layers):
        super().__init__()
        self.stack = nn.Sequential(*layers)

    def forward(self, x):
        return self.stack(x)


class OneActNN(ConnectedNN):
    """
    OneActNN is class of fully connected neural networks admitting only one activation function
    """

    def __init__(self, n_in, n_out, n_hidden_nodes, n_hidden_layers, act):
        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_nodes = n_hidden_nodes

        self.act = get_enum_type(act, ActivationType)

        layers = [nn.Linear(self.n_in, self.n_hidden_nodes)]

        for ii in range(self.n_hidden_layers):
            if self.act is ActivationType.ReLU:
                layers += [nn.ReLU()]
            if self.act is ActivationType.Tanh:
                layers += [nn.Tanh()]
            if self.act is ActivationType.PReLU:
                layers += [nn.PReLU()]
            if self.act is ActivationType.Sigmoid:
                layers += [nn.Sigmoid()]

            if ii < self.n_hidden_layers - 1:
                layers += [nn.Linear(self.n_hidden_nodes, self.n_hidden_nodes)]

        layers += [nn.Linear(self.n_hidden_nodes, self.n_out)]

        super().__init__(layers)
