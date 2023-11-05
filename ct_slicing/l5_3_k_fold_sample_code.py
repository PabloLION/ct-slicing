"""
This is a sample source code for kfold splittng

Computer Vision Center
Universitat Autonoma de Barcelona

__author__ = "Debora Gil, Guillermo Torres, Carles Sanchez, Pau Cano"
__license__ = "GPL"
__email__ = "debora,gtorres,csanchez,pcano@cvc.uab.es"
__year__ = "2023"
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import KFold

folds = 10
glcm_features = np.load("slice_glcm1d.npz", allow_pickle=True)
slices = glcm_features["slice_features"]
metadata = glcm_features["slice_meta"]
diagnosis = [meta[3] for meta in metadata]

X = []
Y = []
for i in range(len(slices)):
    if diagnosis[i] == "NoNod":
        continue
    X.append(slices[i])
    Y.append(1 if diagnosis[i] ==  "Malignant" else 0)

X = np.asarray(X)
Y = np.asarray(Y)
kf = KFold(n_splits = folds)
scores = []
for i, (train_index, test_index) in enumerate(kf.split(X)):
    #print(slices[train_index].shape, slices.shape)
    clf = SVC(probability=True,class_weight='balanced')
    calibrated_classifier = CalibratedClassifierCV(clf,n_jobs=-1)
    calibrated_classifier.fit(X[train_index],Y[train_index])
    scores.append(calibrated_classifier.score(X[test_index],Y[test_index]))
    print(f"Fold {i} score: {scores[-1]}")

accumulative_score = 0
for score in scores:
	accumulative_score += score
average_score = accumulative_score / folds
print(f"average score of all folds: {average_score}")