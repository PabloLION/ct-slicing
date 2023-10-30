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
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
# from NiftyIO import readNifty


# Create a transformation for processing the image
transform = transforms.Compose([
    transforms.ToPILImage(),  # Convertir el tensor a una imagen PIL
    transforms.Resize((224, 224)),  # Redimensionar la imagen a 224x224 píxeles
    transforms.ToTensor(),  # Convertir la imagen a un tensor de PyTorch
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizar el tensor
])

# Load a pre-trained VGG16 or VGG19 model
model = models.vgg16(pretrained=True)  
#model = models.vgg19(pretrained=True)

##############################################


#### Parts of the VGG model ####
vgg_features = model.features
vgg_avgpool = model.avgpool
# ":-2" includes the classifier layers of the VGG up to the penultimate layer
vgg_classifier = nn.Sequential(*list(model.classifier.children())[:-2])
################################

########################
## Insert here your code ... 
## "ONE SLICE" and "ANOTHER SLICE" must be replaced by your code.
## Add a loop to read the nodules and pass a slice at a time through the VGG network 
## to make the features extraction from the first ReLU of the classifier sequence.
##
## print(model)
## ...
## (classifier): Sequential(
##  (0): Linear(in_features=25088, out_features=4096, bias=True)
##  (1): ReLU(inplace=True)  <------- extract features from this layer
##

# image, meta1 = readNifty(imageName, CoordinateOrder='xyz')

## Apply the same preprocesing used in featuresExtraction.py
### PREPROCESSING
# image = ShiftValues(image, value=1024)
# image = SetRange(image, in_min=0, in_max=4000)
# image = SetGrayLevel(image, levels=24)

########################


####### ONE SLICE ########
# Create a random numpy array
X = np.random.rand(224, 224)  # IT SHOULD BE REPLACED BY THE SLICE OF THE NODULE

# Replicate the arry in three channels
X = np.stack([X] * 3, axis=2)   # (224, 224, 3)

# Transpose the axis: (3, 224, 224)
X = X.transpose((2, 0, 1))
# print(X.shape)

# Convert from numpy array to a tensor of PyTorch
tensor = torch.from_numpy(X)
# print(type(tensor))

# Apply the transform to the tensor
tensor = transform(tensor)
# print(tensor.shape)  # torch.Size([3, 224, 224])

# Expand one dimension
tensor = tensor.unsqueeze(0)
# print(tensor.shape)   # torch.Size([1, 3, 224, 224])

# List with the ground truth of each slice
y = []

# Extract features using the VGG model
with torch.no_grad():
    out = model.features(tensor)
    out = model.avgpool(out)
    out = out.view(1, -1)
    out = vgg_classifier(out)

# Convert the tensor to numpy array
array1 = out.numpy()
y.append(0)

#####################################

####### ANOTHER SLICE ########
X = np.random.rand(224, 224) # IT SHOULD BE REPLACED BY THE SLICE OF THE NODULE
X = np.stack([X] * 3, axis=2)   # (224, 224, 3)
X = X.transpose((2, 0, 1))
tensor = torch.from_numpy(X)
tensor = transform(tensor)
tensor = tensor.unsqueeze(0)
print(tensor.shape) 

# Extract features using the VGG model
with torch.no_grad():
    out = model.features(tensor)
    out = model.avgpool(out)
    out = out.view(1, -1)
    out = vgg_classifier(out)

# Convert the tensor to numpy array
array2 = out.numpy()
y.append(1)

#####################################

# Stack the extracted featues
all_features = np.vstack((array1, array2))  # extracted features
y = np.array(y)  # convert the ground truth list to a numpy array.

#####################################

# # DATA SPLITTING
# X_train, X_test, y_train, y_test = train_test_split(all_features, y, test_size=0.3, random_state=42)

# #####################################

# # Create and training a SVM classifier
# clf2 = svm.SVC(probability=True, class_weight='balanced')
# clf2.fit(X_train, y_train)

# y_pred_uncalib = clf2.predict(X_train)

# train_report_dict = classification_report(
# 	y_train,
# 	y_pred_uncalib,
# 	labels = [0, 1],
# 	target_names=['benign', 'malign'],
# 	sample_weight=None,
# 	digits=3,
# 	output_dict=False,
# 	zero_division=0
# 	)

# print(train_report_dict)


# # Show the probabilities of the prediction
# print(clf2.predict_proba(X_train))


# # Use the probabilities to calibrate a new model
# calibrated_classifier = CalibratedClassifierCV(clf2, n_jobs=-1)
# calibrated_classifier.fit(X_train, y_train)


# y_pred_calib = calibrated_classifier.predict(X_train)

# train_report_dict = classification_report(
# 	y_train,
# 	y_pred_calib,
# 	labels = [0, 1],
# 	target_names=['benign', 'malign'],
# 	sample_weight=None,
# 	digits=3,
# 	output_dict=False,
# 	zero_division=0
# 	)

# print(train_report_dict)