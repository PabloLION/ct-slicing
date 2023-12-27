__author__ = "Debora Gil, Guillermo Torres, Carles Sanchez, Pau Cano"
__license__ = "GPL"
__email__ = "debora,gtorres,csanchez,pcano@cvc.uab.es"
__year__ = "2023"
__doc__ = """Source code for volume visualization

Computer Vision Center
Universitat Autonoma de Barcelona
"""

# Unit: Segmentation Validation / Advanced Validation

"""
This script is a sample code for k-fold cross validation.
It uses a similar SVC as l3_8_vgg_feature_extraction but on feature vectors of
length 24. In l3_8, the guide was to use the features of the first ReLU layer.
Maybe we can try using the features of the later layers too.
"""
if __name__ != "__main__":
    raise ImportError(f"Script {__file__} should not be imported as a module")

import numpy as np
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import KFold

from ct_slicing.config.data_path import DATA_FOLDER
from ct_slicing.ct_logger import logger

SLICE_FEATURES_PATH = DATA_FOLDER / "py-radiomics" / "slice_glcm1d.npz"
FOLDS = 10


def load_slice_features_diagnosis_as_numpy_array() -> tuple[np.ndarray, np.ndarray]:
    glcm_features = np.load(SLICE_FEATURES_PATH, allow_pickle=True)
    slice_features: np.ndarray = glcm_features["slice_features"]
    # slice_features is an 2d array of shape (7414, 24)
    metadata: np.ndarray = glcm_features["slice_meta"]
    # metadata has shape (7414, 4), with rows like ['LIDC-IDRI-0933_GT1_6' 933 6 'Benign']

    # seems the old code piece was written by someone not familiar with numpy
    valid_cases = (metadata[:, 3] == "Malignant") | (metadata[:, 3] == "Benign")
    features = slice_features[valid_cases]
    diagnoses = np.where(metadata[valid_cases, 3] == "Malignant", 1, 0)

    return features, diagnoses


def k_fold_cross_validation(x: np.ndarray, y: np.ndarray, kf: KFold) -> list[float]:
    scores = []
    for i, (train_index, test_index) in enumerate(kf.split(x)):
        clf = SVC(probability=True, class_weight="balanced")
        calibrated_classifier = CalibratedClassifierCV(clf, n_jobs=-1)
        calibrated_classifier.fit(x[train_index], y[train_index])
        scores.append(calibrated_classifier.score(x[test_index], y[test_index]))
        logger.info(f"Fold {i} score: {scores[-1]}")
    return scores


kf = KFold(n_splits=FOLDS)
features, diagnoses = load_slice_features_diagnosis_as_numpy_array()
scores = k_fold_cross_validation(features, diagnoses, kf)
logger.info(f"average score of all folds: {sum(scores) / FOLDS}")
