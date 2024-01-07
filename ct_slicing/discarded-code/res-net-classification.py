if __name__ != "__main__":
    raise ImportError(f"Script {__file__} should not be imported as a module")


from enum import Enum
import logging
from math import prod
from typing import Iterable, Iterator
from joblib import dump, load
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models, utils

import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
from ct_slicing.config.data_path import (
    CALIBRATED_CLASSIFIER_JOBLIB,
    UNCALIBRATED_CLASSIFIER_JOBLIB,
    extracted_features_npy_path_with_threshold,
)

from ct_slicing.ct_logger import logger
from ct_slicing.data_util.metadata_access import load_all_metadata
from ct_slicing.data_util.nii_file_access import load_nodule_id_pickle, nii_file
from ct_slicing.image_process import process_image
from ct_slicing.vis_lib.nifty_io import CoordinateOrder, read_nifty
from ct_slicing.data_util.slice_diagnosis_pair import load_voi_slice_truth_pairs

logger.setLevel(logging.INFO)

model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# try also:
# criterion = nn.NLLLoss()
# criterion = nn.BCELoss()
# criterion = nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


## prepare the data
from torch.utils.data import IterableDataset, DataLoader


class TensorTransformingIterableDataset(IterableDataset):
    """
    Create an iterable dataset that applies a transformation to the data.
    This transform should accept an arbitrary size of 2d grayscale image (like
    numpy 2d array) and return a 1x3x224x224 tensor (like the input of vgg16, but
    for resnet).

    The self.transform was equivalent to "tensor_from_2d_array" in
    "l3_8_vgg_features_extraction.py", but now it suits better for grayscale
    images.
    """

    def __init__(self, data_iter):
        self.data_iter = data_iter
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                # transforms.Resize(224),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Lambda(
                    lambda x: x.repeat(3, 1, 1)
                ),  # Repeat the single channel to create a 3-channel image
                transforms.Normalize(mean=[0.5], std=[0.5]),  # Adjust for grayscale
            ]
        )

    def __iter__(self):
        for data, label in self.data_iter:
            yield self.transform(data), label


data_iter = load_voi_slice_truth_pairs()
iterable_dataset = TensorTransformingIterableDataset(data_iter)
data_loader = DataLoader(iterable_dataset, batch_size=32)

# setup for training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_epochs = 10  # number of training epochs
# set the output layer to 2 classes: benign and malign, keeping in_features same
model.fc = nn.Linear(model.fc.in_features, 2)
model.to(device)

logger.info("Start training")
for epoch in range(n_epochs):
    for batch, (data, label) in enumerate(data_loader):
        ...
        logger.info(f"Epoch: {epoch}, Batch: {batch:03}, Loss: ")
    logger.info(f"Epoch: {epoch} finished")

torch.save(model.state_dict(), "output/trained-model/resnet152_model.pth")


# X_train, X_test, y_train, y_test = split_data_from_features_and_diagnoses()
# classification_report(
#     y_train,
#     y_pred,
#     labels=[0, 1],
#     target_names=["benign", "malign"],
#     digits=3,
# )
