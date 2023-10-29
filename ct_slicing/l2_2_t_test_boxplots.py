__annotations__ = {"working": False, "reason": "File missing"}
__author__ = "Debora Gil, Guillermo Torres, Carles Sanchez, Pau Cano"
__license__ = "GPL"
__email__ = "debora,gtorres,csanchez,pcano@cvc.uab.es"
__year__ = "2023"
__doc__ = """Source code for volume visualization

Computer Vision Center
Universitat Autonoma de Barcelona
"""

# Unit: Features Exploration
# Data: from unit "PyRadiomics" under Section "Features Extraction"
# Resource:
# * T-Test: https://thedatascientist.com/how-to-do-a-t-test-in-python/

import os
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

from ct_slicing.config.data_path import DATA_FOLDER, OUTPUT_FOLDER


def BoxPlotFeatures(feat, y, feature_names, output_dir):
    for c, fn in enumerate(feature_names):
        idx_Malignant = np.nonzero(y == "Malignant")[0]
        idx_Benign = np.nonzero(y == "Benign")[0]
        group = [feat[idx_Malignant, c], feat[idx_Benign, c]]
        plt.figure()
        plt.boxplot(group, labels=["Malignant", "Benign"])
        plt.title(str(c) + " " + fn)
        plt.savefig(os.path.join(output_dir, fn + ".png"))


def t_test(slice_meta, slice_features):
    x = []

    idx = np.nonzero(slice_meta[:, 3] == "Benign")[0]
    x.append(slice_features[idx, :])

    idx = np.nonzero(slice_meta[:, 3] == "Malignant")[0]
    x.append(slice_features[idx, :])

    idx = np.nonzero(slice_meta[:, 3] == "NoNod")[0]
    x.append(slice_features[idx, :])

    p_val = []

    for i in np.arange(x[0].shape[1]):
        aux = stats.ttest_ind(x[0][:, i], x[1][:, i])
        p_val.append(aux[1])

    p_val = np.array(p_val)
    ranking_idx = np.argsort(p_val)
    p_val_sort = p_val[ranking_idx]

    return p_val, ranking_idx, p_val_sort


######################################


FILENAME = DATA_FOLDER / "py-radiomics" / "slice_glcm1d.npz"

data = np.load(FILENAME, allow_pickle=True)
print(data.files)

print(data["slice_meta"].shape)
print(data["slice_meta"][0])

# TODO: slice_flat is missing from the file
print(data["slice_flat"].shape)
print(data["slice_flat"][0])

slice_features = data["slice_flat"]
slice_meta = data["slice_meta"]
p_val, ranking_idx, pval_sort = t_test(slice_meta, slice_features)

columns = [
    "original_glcm_Autocorrelation",
    "original_glcm_ClusterProminence",
    "original_glcm_ClusterShade",
    "original_glcm_ClusterTendency",
    "original_glcm_Contrast",
    "original_glcm_Correlation",
    "original_glcm_DifferenceAverage",
    "original_glcm_DifferenceEntropy",
    "original_glcm_DifferenceVariance",
    "original_glcm_Id",
    "original_glcm_Idm",
    "original_glcm_Idmn",
    "original_glcm_Idn",
    "original_glcm_Imc1",
    "original_glcm_Imc2",
    "original_glcm_InverseVariance",
    "original_glcm_JointAverage",
    "original_glcm_JointEnergy",
    "original_glcm_JointEntropy",
    "original_glcm_MCC",
    "original_glcm_MaximumProbability",
    "original_glcm_SumAverage",
    "original_glcm_SumEntropy",
    "original_glcm_SumSquares",
]

output_dir = OUTPUT_FOLDER
BoxPlotFeatures(
    feat=slice_features,
    y=slice_meta[:, 3],
    feature_names=columns,
    output_dir=output_dir,
)
