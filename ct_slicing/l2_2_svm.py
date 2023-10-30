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

# Load dataset
cancer: Bunch = cast(Bunch, datasets.load_breast_cancer())
# print the names of the 13 features
print("Features: ", cancer.feature_names)

# print the label type of cancer('malignant' 'benign')
print("Labels: ", cancer.target_names)
# Import train_test_split function

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.3, random_state=109
)  # 70% training and 30% test


clf2 = SVC(probability=True, class_weight="balanced")
clf2.fit(X_train, y_train)


# Use CalibratedClassifierCV to calibrate probabilities
calibrated_classifier = CalibratedClassifierCV(clf2, n_jobs=-1)
calibrated_classifier.fit(X_train, y_train)


y_pred_fold_tr = calibrated_classifier.predict(X_train)
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
