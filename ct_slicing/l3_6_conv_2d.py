__author__ = "Debora Gil, Guillermo Torres, Carles Sanchez, Pau Cano"
__license__ = "GPL"
__email__ = "debora,gtorres,csanchez,pcano@cvc.uab.es"
__year__ = "2023"
__doc__ = """Source code for volume visualization

Computer Vision Center
Universitat Autonoma de Barcelona
"""

if __name__ != "__main__":
    raise ImportError(f"Script {__file__} should not be imported as a module")

import torch
import torch.nn as nn

# Define a layer with a convolution 2D with 3 channels as input, 16 channels of output, kernel_size=3 and padding=1
conv_layer = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)

# Define a ReLU layer
relu_layer = nn.ReLU()

# Define MaxPool2d layer with a kernel_size=2 and stride=1
maxpool_layer = nn.MaxPool2d(kernel_size=2, stride=2)

# Create an input tensor randomly with size of (16, 3, 6, 6)
x = torch.randn(16, 3, 6, 6)
print(x.shape)
print(x)

# Exercise 2.a - Add a fully connected layer and show its weights.
# Define a fully connected layer with input features 16*3*6*6
# (output of previous layers) and output features 64
fc_layer = nn.Linear(3, 3)

# Pass the input through the convolution 2D, ReLU and MaxPool2d layers
out = conv_layer(x)
out = relu_layer(out)
out = maxpool_layer(out)

# Show the shape of the output tensor.
print(out)
print(out.shape)

# Print the wights of the convolution layer.
print(conv_layer.weight)

# Exercise 2.a - Add a fully connected layer and show its weights.
out = fc_layer(out)
print("Weights of the fully connected layer:")
print(fc_layer.weight)

"""
# Exercise

## Exercise 2. Understanding a pre-trained network.

a) To understand how a random input tensor goes through a convolutional layer, activation
function, and max pooling layer examine the Conv2D.py code. In the provided code, add
a fully connected layer and show its weights.
    The input tensor is a 16x3x6x6 tensor, where 16 is the batch size, 3 is the
    number of channels. So we define a layer with 3 input channels and 3 output
    channels. The kernel size is 3x3. The weights of the new channel changes
    randomly after each execution.
See for the next part, Exercise 2.b in `l3_7_vgg_architecture.py`.
"""
