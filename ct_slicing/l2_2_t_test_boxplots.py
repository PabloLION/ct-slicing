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
from matplotlib import pyplot as plt
from scipy import stats

from ct_slicing.config.data_path import DATA_FOLDER, OUTPUT_FOLDER
from ct_slicing.log import logger

SAVED_PATH = DATA_FOLDER / "py-radiomics" / "slice_glcm1d.npz"
OUTPUT_DIR = OUTPUT_FOLDER / "t_test_box_plots"

saved_data = np.load(SAVED_PATH, allow_pickle=True)
logger.debug(saved_data.files)
slice_meta = saved_data["slice_meta"]
slice_features = saved_data["slice_features"]  # not "slice_flat"
logger.debug(f"loaded slice_meta with {slice_meta.shape=}, {slice_meta[0]=}")
logger.debug(f"loaded features with {slice_features.shape=}, {slice_features[0]=}")


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


p_val, ranking_idx, pval_sort = t_test(slice_meta, slice_features)


# print("p_val: ", p_val)
# print("ranking_idx: ", ranking_idx)
# print("pval_sort: ", pval_sort)

features = [
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


def save_box_plot_features(feat, y_label, feat_idx):
    feature_name = features[feat_idx]
    c = feat_idx
    logger.debug(f"Plotting for {feature_name=}")
    idx_Malignant = np.nonzero(y_label == "Malignant")[0]
    idx_Benign = np.nonzero(y_label == "Benign")[0]
    group = [feat[idx_Malignant, c], feat[idx_Benign, c]]
    plt.figure()
    plt.boxplot(group, labels=["Malignant", "Benign"])
    plt.title(str(c) + " " + feature_name)
    output_path = OUTPUT_DIR / (feature_name + ".png")
    plt.savefig(output_path)
    plt.close()
    logger.debug(f"Saved {output_path=}")


if __name__ == "__main__":
    assert slice_features.shape[0] == slice_meta.shape[0]
    assert slice_features.shape[1] == len(features)
    for i in range(len(features)):
        save_box_plot_features(
            feat=slice_features,
            y_label=slice_meta[:, 3],
            feat_idx=i,
        )
