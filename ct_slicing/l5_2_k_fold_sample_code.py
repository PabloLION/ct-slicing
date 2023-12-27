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

# glcm means gray level co-occurrence matrix
glcm_features = np.load(SLICE_FEATURES_PATH, allow_pickle=True)
slice_features: list[np.ndarray] = glcm_features["slice_features"]
metadata: np.ndarray[None, np.dtype[np.int8 | np.str_]] = glcm_features["slice_meta"]
# metadata is an 2d array like ['LIDC-IDRI-1011_GT1_1' 1011 1 'Malignant']
diagnosis = [meta[3] for meta in metadata]

_x: list[np.ndarray] = []
_y: list[int] = []
for i in range(len(slice_features)):
    if diagnosis[i] == "NoNod":
        continue
    _x.append(slice_features[i])
    _y.append(1 if diagnosis[i] == "Malignant" else 0)

x = np.vstack(_x)
y = np.asarray(_y)
kf = KFold(n_splits=FOLDS)
scores = []

for i, (train_index, test_index) in enumerate(kf.split(x)):
    # logger.info(slices[train_index].shape, slices.shape)
    clf = SVC(probability=True, class_weight="balanced")
    calibrated_classifier = CalibratedClassifierCV(clf, n_jobs=-1)
    calibrated_classifier.fit(x[train_index], y[train_index])
    scores.append(calibrated_classifier.score(x[test_index], y[test_index]))
    logger.info(f"Fold {i} score: {scores[-1]}")

logger.info(f"average score of all folds: {sum(scores) / FOLDS}")
