#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__author__ = "Guillermo Torres, Debora Gil and Pau Cano"
__license__ = "GPLv3"
__email__ = "gtorres,debora,pau@cvc.uab.cat"
__year__ = "2023"
"""

import torch
import torch.nn as nn

# Define a layer with a convolution 2D with 3 channels as input, 16 channels of output, kernel_size=3 and padding=1
conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)

# Define a ReLU layer
relu_layer = nn.ReLU()

# Define MaxPool2d layer with a kernel_size=2 and stride=1
maxpool_layer = nn.MaxPool2d(kernel_size=2, stride=2)

# Create an input tensor randomly with size of (1, 1, 6, 6)
x = torch.randn(1, 1, 6, 6)
print(x.shape)
print(x)

# Pass the input through the convolution 2D, ReLU and MaxPool2d layers
out = conv_layer(x)
print(out)

out = relu_layer(out)
print(out)

out = maxpool_layer(out)
print(out)

# Show the shape of the output tensor.
print(out.shape)

# Print the wights of the convolution layer.
print(conv_layer.weight)