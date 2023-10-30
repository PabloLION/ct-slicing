__author__ = "Debora Gil, Guillermo Torres, Carles Sanchez, Pau Cano"
__license__ = "GPL"
__email__ = "debora,gtorres,csanchez,pcano@cvc.uab.es"
__year__ = "2023"
__doc__ = """Source code for volume visualization

Computer Vision Center
Universitat Autonoma de Barcelona
"""

# Import scikit-learn dataset library
from typing import cast
from sklearn import datasets, metrics
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch

from ct_slicing.log import logger


# Load dataset
cancer: Bunch = cast(Bunch, datasets.load_breast_cancer())
logger.info(f"Features: {cancer.feature_names}")  # print 13 features
logger.info(f"Labels: {cancer.target_names}")  # 'malignant' 'benign'

# Split dataset into training set and test set
x_train, x_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.3, random_state=109
)  # 70% training and 30% test

# SVC: Support Vector Classification implementation based on libsvm
classifier = SVC(probability=True, class_weight="balanced")
classifier.fit(x_train, y_train)

#
calibrated_classifier = CalibratedClassifierCV(classifier)
calibrated_classifier.fit(x_train, y_train)


y_pred_fold_tr = calibrated_classifier.predict(x_train)
train_report_dict = metrics.classification_report(
    y_train,
    y_pred_fold_tr,
    labels=[0, 1],
    target_names=["malign", "benign"],
    sample_weight=None,
    digits=3,
    output_dict=False,
    zero_division=0,
)

print(train_report_dict)
