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
## Renaming

| Old name     | New name              | comment                                                        |
| ------------ | --------------------- | -------------------------------------------------------------- |
| X            | one_slice             | replaced with real data returned by load_voi_slice_truth_pairs |
| tensor       | tensor_one_slice      | replaced with real data returned by load_voi_slice_truth_pairs |
| array1       | feature_one_slice     | moved to get_extracted_features_and_diagnosis_value            |
| X            | another_slice         | replaced with real data returned by load_voi_slice_truth_pairs |
| tensor       | tensor_another_slice  | replaced with real data returned by load_voi_slice_truth_pairs |
| array2       | feature_another_slice | moved to get_extracted_features_and_diagnosis_value            |
| y            | diagnosis_value       | moved to get_extracted_features_and_diagnosis_value            |
| all_features | extracted_features    | moved to get_extracted_features_and_diagnosis_value            |


## Improvements

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
- Correct VGG classifier:
    - old: vgg_classifier = nn.Sequential(*list(model.classifier.children())[:-2])
    - new: vgg_classifier = nn.Sequential(*list(model.classifier.children())[:2])
    - reason: 
        - it is said in the comment "Add a loop to read the nodules and pass
            a slice at a time through the VGG network to make the feature 
            extraction from the first ReLU of the classifier sequence."
        - The layers are:
            (0): Linear(in_features=25088, out_features=4096, bias=True)
            (1): ReLU(inplace=True)
            (2): Dropout(p=0.5, inplace=False)
            (3): Linear(in_features=4096, out_features=4096, bias=True)
            (4): ReLU(inplace=True)
            (5): Dropout(p=0.5, inplace=False)
            (6): Linear(in_features=4096, out_features=1000, bias=True)
        - The old version removes the last 2 layers, but we want to the result
            of the first ReLU layer.
"""

if __name__ != "__main__":
    raise ImportError(f"Script {__file__} should not be imported as a module")


from enum import Enum
import logging
from typing import Iterable
from joblib import dump, load
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
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
from ct_slicing.data_util.slice_view import load_voi_slice_truth_pairs

logger.setLevel(logging.INFO)


class ModelName(Enum):
    VGG16 = "vgg16"
    VGG19 = "vgg19"


## Choose parameters for this script
MODEL_NAME: ModelName = ModelName.VGG16  # property to be configured
empty_slice_threshold = 0.1  # Using 0.1 and 0.0 gives the same result

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

# Old: vgg_classifier = nn.Sequential(*list(model.classifier.children())[:-2])
# ":-2" removes the last 2 layers of the classifier, they are:
# (5): Dropout(p=0.5, inplace=False)
# (6): Linear(in_features=4096, out_features=1000, bias=True)
# Dropout randomly drops some neurons to prevent overfitting.
# Linear is the last layer of the classifier, it maps the 4096 features to
# 1000 classes. Here we want to extract the 4096 features without classifying.

# Add a loop to read the nodules and pass a slice at a time through the VGG network
# to make the features extraction from the first ReLU of the classifier sequence.
## print(model)
## ...
## (classifier): Sequential(
##  (0): Linear(in_features=25088, out_features=4096, bias=True)
##  (1): ReLU(inplace=True)  <------- extract features from this layer
# according to the comment above, extract features from the first ReLU
vgg_classifier = nn.Sequential(*list(model.classifier.children())[:2])


def extract_feature_from_slice_diagnosis_pairs(
    slice_diagnosis_pairs: Iterable[tuple[np.ndarray, int]]
) -> tuple[np.ndarray, np.ndarray]:
    """Extract features from multiple slices.

    Args:
        slices_diagnosis_pairs (Iterable[tuple[np.ndarray, int]]): An iterable

    Returns:
        tuple[np.ndarray, np.ndarray]: The extracted features and the corresponding diagnoses.
    """
    extracted_features = []  # List with the extracted features of each slice
    diagnosis_value = []  # List with the ground truth of each slice

    for process_idx, (img_slice, truth) in enumerate(slice_diagnosis_pairs):
        extracted_features.append(vgg_extract_features(tensor_from_2d_array(img_slice)))
        diagnosis_value.append(truth)
        logger.info(f"Extracted features from {process_idx} slices.")

    # Stack the extracted features
    extracted_features = np.vstack(extracted_features)
    # convert the ground truth list to a numpy array.
    diagnosis_value = np.array(diagnosis_value)

    return extracted_features, diagnosis_value


def get_extracted_features_and_diagnosis_value() -> tuple[np.ndarray, np.ndarray]:
    """
    Get the extracted features and diagnosis value from the saved file if it
    exists, otherwise extract the features from the slices and save it to file.
    """
    EXTRACTED_FEATURES_NPY = extracted_features_npy_path_with_threshold(
        threshold=empty_slice_threshold
    )
    if EXTRACTED_FEATURES_NPY.exists():
        logger.info("Extracted features already exists, loading from file.")
        with open(EXTRACTED_FEATURES_NPY, "rb") as f:
            concatenated_array = np.load(f, allow_pickle=False)
        extracted_features = concatenated_array[:, :-1]
        diagnosis_value = concatenated_array[:, -1]
        logger.info("Extracted features loaded from file.")
    else:
        (
            extracted_features,
            diagnosis_value,
        ) = extract_feature_from_slice_diagnosis_pairs(load_voi_slice_truth_pairs())
        diagnosis_reshaped = np.reshape(
            diagnosis_value, (-1, 1)
        )  # Reshape to (9016, 1)
        concatenated_array = np.concatenate(
            (extracted_features, diagnosis_reshaped), axis=1
        )
        with open(EXTRACTED_FEATURES_NPY, "wb") as f:
            np.save(f, concatenated_array, allow_pickle=False)
        logger.info("Extracted features saved to file.")
    return extracted_features, diagnosis_value


def split_data_from_features_and_diagnoses(
    extracted_features: np.ndarray,
    diagnosis_value: np.ndarray,
    test_size: float = 0.3,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split the data into a train set (70%) and a test set (30%).

    Args:
        extracted_features (np.ndarray): The extracted features.
        diagnosis_value (np.ndarray): The diagnosis value.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The splitted data.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        extracted_features,
        diagnosis_value,
        test_size=test_size,
        random_state=random_state,
    )
    return X_train, X_test, y_train, y_test


def get_uncalibrated_classifier(X_train: np.ndarray, y_train: np.ndarray) -> svm.SVC:
    # DATA SPLITTING
    if UNCALIBRATED_CLASSIFIER_JOBLIB.exists():
        logger.warning(f"Loading classifier from {UNCALIBRATED_CLASSIFIER_JOBLIB}.")
        return load(UNCALIBRATED_CLASSIFIER_JOBLIB)

    # Create and train a SVM classifier
    uncalibrated_classifier = svm.SVC(probability=True, class_weight="balanced")
    uncalibrated_classifier.fit(X_train, y_train)

    # Save the classifier to file
    dump(uncalibrated_classifier, UNCALIBRATED_CLASSIFIER_JOBLIB)
    logger.info(f"Calibrated classifier saved to {UNCALIBRATED_CLASSIFIER_JOBLIB}")

    # Uncomment the following line to show the probabilities of the prediction
    # logger.info(f"Probabilities of the prediction:\n{classifier.predict_proba(X_train)}")
    return uncalibrated_classifier


def get_calibrated_classifier(
    X_train: np.ndarray, y_train: np.ndarray, uncalibrated_classifier: svm.SVC
):
    if CALIBRATED_CLASSIFIER_JOBLIB.exists():
        logger.warning(f"Loading classifier from {CALIBRATED_CLASSIFIER_JOBLIB}.")
        return load(CALIBRATED_CLASSIFIER_JOBLIB)

    # Use the probabilities to calibrate a new model
    calibrated_classifier = CalibratedClassifierCV(
        uncalibrated_classifier, n_jobs=-1, cv=2
    )
    calibrated_classifier.fit(X_train, y_train)  # not fixable with 2 samples only

    dump(calibrated_classifier, CALIBRATED_CLASSIFIER_JOBLIB)
    logger.info(f"Calibrated classifier saved to {CALIBRATED_CLASSIFIER_JOBLIB}")
    return calibrated_classifier


"""
Training report without data splitting: (train and test with full 9016 data)
                precision   recall   f1-score   support

      benign      0.618     0.983     0.759      4362
      malign      0.965     0.431     0.595      4655

    accuracy                          0.698      9017
   macro avg      0.791     0.707     0.677      9017
weighted avg      0.797     0.698     0.675      9017

How to read the report:
| Name         | Meaning                                                       |
| ------------ | ------------------------------------------------------------- |
| TP           | True Positive                                                 |
| TN           | True Negative                                                 |
| FP           | False Positive                                                |
| FN           | False Negative                                                |
| precision    | TP / (TP + FP)                                                |
| recall or .. | TP / (TP + FN)                                                |
| sensitivity  | TP / (TP + FN)                                                |
| f1-score     | 2 \\* precision \\* recall / (precision + recall)             |
| f1-score     | the harmonic mean of precision and recall                     |
| accuracy     | (TP + TN) / (TP + TN + FP + FN)                               |
| weighted avg | average weighted by the number of instances in each class     |
| macro avg    | average not weighted by the number of instances in each class |

Keynote of the report:
- The model is better at identifying benign cases than malign cases (higher
    recall for benign).
- The model is more precise in predicting malign cases than benign ones.
- The F1 score is moderately good for both classes, but there's room for
    improvement, especially for the malign class.
- The accuracy of 70.1% shows a general effectiveness of the model, but
    considering the weighted metrics and individual class performance is 
    crucial, especially in imbalanced datasets.

Discussion:
- If the model can help exclude benign cases, it can reduce the number of
    unnecessary biopsies. So a high recall for benign cases is good.
"""

X_train, X_test, y_train, y_test = split_data_from_features_and_diagnoses(
    *get_extracted_features_and_diagnosis_value()
)
uncalibrated_classifier = get_uncalibrated_classifier(X_train, y_train)
y_pred_uncalibrated = uncalibrated_classifier.predict(X_train)
logger.info("Training report without calibration:")
logger.info(
    classification_report(
        y_train,
        y_pred_uncalibrated,
        labels=[0, 1],
        target_names=["benign", "malign"],
        digits=3,
        # sample_weight=None, | default value: sample_weight=None,
        # output_dict=False,  | default value: output_dict=False,
        # zero_division=0,    | default value: zero_division="warn",
    )
)

"""
Training report with data splitting: (train with 6311 data, test with 2706 data)
We can see that the accuracy is similar to the one without data splitting.
                precision   recall  f1-score   support
      benign      0.617     0.985     0.758      3038
      malign      0.968     0.432     0.598      3273

    accuracy                          0.698      6311
   macro avg      0.792     0.708     0.678      6311
weighted avg      0.799     0.698     0.675      6311
"""

calibrated_classifier = get_calibrated_classifier(
    X_train, y_train, uncalibrated_classifier
)
y_pred_calibrated = calibrated_classifier.predict(X_train)
logger.info("Training report with calibration:")
logger.info(
    classification_report(
        y_train,
        y_pred_calibrated,
        labels=[0, 1],
        target_names=["benign", "malign"],
        digits=3,
        # sample_weight=None, | default value: sample_weight=None,
        # output_dict=False,  | default value: output_dict=False,
        # zero_division=0,    | default value: zero_division="warn",
    )
)


# Apply the original classifier to the test set
y_pred_test_uncalibrated = uncalibrated_classifier.predict(X_test)

logger.info("Test report for uncalibrated classifier:")
logger.info(
    classification_report(
        y_test,
        y_pred_test_uncalibrated,
        labels=[0, 1],
        target_names=["benign", "malign"],
        digits=3,
    )
)

"""
Test report for uncalibrated classifier:
                precision   recall  f1-score   support

      benign      0.613     0.968     0.750      1324
      malign      0.931     0.413     0.572      1382

    accuracy                          0.685      2706
   macro avg      0.772     0.691     0.661      2706
weighted avg      0.775     0.685     0.659      2706
"""

# Apply the calibrated classifier to the test set
y_pred_test_calibrated = calibrated_classifier.predict(X_test)

logger.info("Test report for calibrated classifier:")
logger.info(
    classification_report(
        y_test,
        y_pred_test_calibrated,
        labels=[0, 1],
        target_names=["benign", "malign"],
        digits=3,
    )
)
"""
Test report for calibrated classifier:
                precision    recall  f1-score   support

      benign      0.637     0.804     0.711      1324
      malign      0.749     0.560     0.641      1382

    accuracy                          0.680      2706
   macro avg      0.693     0.682     0.676      2706
weighted avg      0.694     0.680     0.675      2706

Conclusion:
The higher recall for benign cases is good, so the model can help exclude
benign cases and reduce the number of unnecessary biopsies.
Therefore, the uncalibrated classifier is better than the calibrated one.
"""

# Optional: Probabilities of the predictions on test data (from the original classifier)
# Uncomment the following line if you want to log the probabilities
# logger.info(f"Probabilities of the prediction on test data:\n{classifier.predict_proba(X_test)}")
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
