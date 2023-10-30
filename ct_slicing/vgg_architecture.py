#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__author__ = "Guillermo Torres, Debora Gil and Pau Cano"
__license__ = "GPLv3"
__email__ = "gtorres,debora,pau@cvc.uab.cat"
__year__ = "2023"
"""

import torch
import torchvision.models as models

# Load the pre-trained model VGG16
model = models.vgg16(pretrained=True)

# Show the architecture of the VGG16 model
print(model)

# Show the layer's weights of the VGG16 model
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)