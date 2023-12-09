__author__ = "Debora Gil, Guillermo Torres, Carles Sanchez, Pau Cano"
__license__ = "GPL"
__email__ = "debora,gtorres,csanchez,pcano@cvc.uab.es"
__year__ = "2023"
__doc__ = """Source code for volume visualization

Computer Vision Center
Universitat Autonoma de Barcelona
"""

# Unit: Segmentation Validation / Advanced Validation

import numpy as np
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import KFold

from ct_slicing.config.data_path import DATA_FOLDER
from ct_slicing.ct_logger import logger

SLICE_FEATURES_PATH = DATA_FOLDER / "py-radiomics" / "slice_glcm1d.npz"

folds = 10
glcm_features = np.load(SLICE_FEATURES_PATH, allow_pickle=True)
slices = glcm_features["slice_features"]
metadata = glcm_features["slice_meta"]
diagnosis = [meta[3] for meta in metadata]

x = []
y = []
for i in range(len(slices)):
    if diagnosis[i] == "NoNod":
        continue
    x.append(slices[i])
    y.append(1 if diagnosis[i] == "Malignant" else 0)

x = np.asarray(x)
y = np.asarray(y)
kf = KFold(n_splits=folds)
scores = []
for i, (train_index, test_index) in enumerate(kf.split(x)):
    # logger.info(slices[train_index].shape, slices.shape)
    clf = SVC(probability=True, class_weight="balanced")
    calibrated_classifier = CalibratedClassifierCV(clf, n_jobs=-1)
    calibrated_classifier.fit(x[train_index], y[train_index])
    scores.append(calibrated_classifier.score(x[test_index], y[test_index]))
    logger.info(f"Fold {i} score: {scores[-1]}")

accumulative_score = 0
for score in scores:
    accumulative_score += score
average_score = accumulative_score / folds
logger.info(f"average score of all folds: {average_score}")
