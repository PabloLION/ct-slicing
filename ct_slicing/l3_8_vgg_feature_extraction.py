__author__ = "Debora Gil, Guillermo Torres, Carles Sanchez, Pau Cano"
__license__ = "GPL"
__email__ = "debora,gtorres,csanchez,pcano@cvc.uab.es"
__year__ = "2023"
__doc__ = """Source code for volume visualization

Computer Vision Center
Universitat Autonoma de Barcelona
"""

# Unit: Visual Feature extraction exploration and selection / PyRadiomics

"""
Renaming:
| Old name | New name |
|----------|----------|
| X | one_slice |
| tensor | tensor_one_slice |
|array1 | feature_one_slice |
| X | another_slice |
| tensor | tensor_another_slice |
|array2 | feature_another_slice |

Improvements:
- Copying an 224x224 image three times along the new first axis:
    - Old version:
        X = np.stack([X] * 3, axis=2)  # (224, 224, 3)
        X = X.transpose((2, 0, 1))
    - New version:
        X = np.tile(X, (3, 1, 1))  # (3, 224, 224)
- Removing the tensor to PIL image conversion:
    # ...
    X = X.transpose((2, 0, 1)) # now X is a (3, 224, 224) np.ndarray
    tensor = torch.from_numpy(X) # this is not needed anymore
    tensor = transform(tensor) # because here we are converting to PIL.Image
                            #  from either a np.ndarray or tensor
"""

if __name__ != "__main__":
    raise ImportError(
        "This file is not meant to be imported. "
        "Please run it directly with python3 -m ct_slicing.l3_8_vgg_feature_extraction"
    )


from enum import Enum
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report

from ct_slicing.ct_logger import logger


class ModelName(Enum):
    VGG16 = "vgg16"
    VGG19 = "vgg19"


## Choose parameters for this script
MODEL_NAME: ModelName = ModelName.VGG16  # property to be configured

# Load a pre-trained VGG16 or VGG19 model. NOT using dict for performance
if MODEL_NAME == ModelName.VGG16:
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
elif MODEL_NAME == ModelName.VGG19:
    model = models.vgg19(pretrained=True)
else:
    raise ValueError(f"Unknown model name: {MODEL_NAME}")


def tensor_from_2d_array(array_2d: np.ndarray) -> torch.Tensor:
    """Create a 1x3x224x224 tensor for vgg16 from an 2d intensity array
    of any size.

    Args:
        image (np.ndarray): Image to be converted.

    Returns:
        torch.Tensor: Tensor created from the image.

    Refactored from:
        It was constructed with ``transform = transforms.Compose()`` but
        here it is done in a typed function.
        This change is tested with ``assert torch.all(torch.eq(old, new))``
    """
    # Convert the image to a PIL image

    array_2d_uint8 = (array_2d * 255).astype(np.uint8)
    array_3d_uint8 = np.stack([array_2d_uint8] * 3, axis=2)  # shape: (H,W,3)
    pil_img = transforms.ToPILImage()(array_3d_uint8)

    ## old version of these 3 lines:
    # array_3d = np.tile(array_2d, (3, 1, 1))
    # array_3d = torch.from_numpy(array_3d)
    # pil_img = transforms.ToPILImage()(array_3d)

    ## reason for the change:
    # transforms.ToPILImage parses float to int8 by `pic = pic.mul(255).byte()`
    # then transposes the array of (3,H,W) to (H,W,3) with dtype = int8.
    # this is a lot of overhead for a simple parse.

    # Resize the image to 224x224 pixels
    pil_img_224x224 = transforms.Resize((224, 224))(pil_img)
    # Convert the image to a PyTorch tensor then normalize it

    tensor: torch.Tensor = transforms.ToTensor()(pil_img_224x224)
    tensor = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )(tensor)
    # Expand one dimension from 3x224x224 to 1x3x224x224
    tensor = tensor.unsqueeze(0)
    return tensor


def vgg_extract_features(slice_tensor: torch.Tensor) -> np.ndarray:
    """
    Extract features as a numpy array from a single slice tensor using the
    VGG model.
    """
    with torch.no_grad():
        features_tensor: torch.Tensor = model.features(slice_tensor)
        features_tensor = model.avgpool(features_tensor)
        features_tensor = features_tensor.view(1, -1)
        features_tensor = vgg_classifier(features_tensor)
        features = features_tensor.numpy()  # Convert tensor to numpy array
    return features


#### Parts of the VGG model ####
# print(c for c in model.classifier.children())
vgg_classifier = nn.Sequential(*list(model.classifier.children())[:-2])
# ":-2" removes the last 2 layers of the classifier
# #TODO: why should we use "-2"?
################################

########################
## Insert your code here...
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

## Apply the same preprocessing used in featuresExtraction.py
### PREPROCESSING
# image = ShiftValues(image, value=1024)
# image = SetRange(image, in_min=0, in_max=4000)
# image = SetGrayLevel(image, levels=24)

########################
rng = np.random.Generator(np.random.PCG64(123))  # better practice of random.seed

y = []  # List with the ground truth of each slice

####### ONE SLICE ########
one_slice = rng.uniform(size=(224, 224))
# #TODO: replace with a real slice (any size)
tensor_one_slice = tensor_from_2d_array(one_slice)
feature_one_slice = vgg_extract_features(tensor_one_slice)
y.append(0)

####### ANOTHER SLICE ########
another_slice = rng.uniform(size=(224, 224))
# #TODO: replace with a real slice (any size)
tensor_another_slice = tensor_from_2d_array(another_slice)
feature_another_slice = vgg_extract_features(tensor_another_slice)
y.append(1)

#####################################

# Stack the extracted features
extracted_features = np.vstack((feature_one_slice, feature_another_slice))
y = np.array(y)  # convert the ground truth list to a numpy array.

#####################################

# DATA SPLITTING
# X_train, X_test, y_train, y_test = train_test_split(
#     all_features, y, test_size=0.3, random_state=42
# )
X_train, X_test, y_train, y_test = extracted_features, extracted_features, y, y

#####################################

# Create and training a SVM classifier
clf2 = svm.SVC(probability=True, class_weight="balanced")
clf2.fit(X_train, y_train)

y_pred_uncalib = clf2.predict(X_train)

train_report_dict = classification_report(
    y_train,
    y_pred_uncalib,
    labels=[0, 1],
    target_names=["benign", "malign"],
    sample_weight=None,
    digits=3,
    output_dict=False,
    zero_division=0,
)

logger.info(train_report_dict)


# Show the probabilities of the prediction
logger.info(f"Probabilities of the prediction:\n{clf2.predict_proba(X_train)}")

# Use the probabilities to calibrate a new model
calibrated_classifier = CalibratedClassifierCV(clf2, n_jobs=-1, cv=2)
# calibrated_classifier.fit(X_train, y_train) # not fixable with 2 samples only


# y_pred_calib = calibrated_classifier.predict(X_train)

# train_report_dict = classification_report(
#     y_train,
#     y_pred_calib,
#     labels=[0, 1],
#     target_names=["benign", "malign"],
#     sample_weight=None,
#     digits=3,
#     output_dict=False,
#     zero_division=0,
# )

# print(train_report_dict)

## Unused code
vgg_features = model.features
vgg_avgpool = model.avgpool

"""
# Exercise

## Exercise 3. Extract features with a pre-trained network and make the
classification of the slices.
a) The code in vgg_features_extraction1.py simulates the extraction of features for two
slices. Modify the vgg_features_extraction1.py to read each (intensity) nodule and
passing each of its slices through the VGG model. Extract the features from the first
ReLU layer of the classification sequence.
    This is not done in the code.

b) After extracting the features, split the data into a train set (70%) and a test set (30%) set
and train a SVM classifier (using the train set). Later calibrate the classifier using the
output probabilities.
    This is not done in the code.
"""
