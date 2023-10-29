__annotations__ = {"working": False, "reason": "File missing"}
__author__ = "Debora Gil, Guillermo Torres, Carles Sanchez, Pau Cano"
__license__ = "GPL"
__email__ = "debora,gtorres,csanchez,pcano@cvc.uab.es"
__year__ = "2023"
__doc__ = """Source code for volume visualization

Computer Vision Center
Universitat Autonoma de Barcelona
"""

# Not mentioned in class.
# Unit: Data Exploration / Features Exploration
# Data: from "Features Extraction / PyRadiomics"
# Resource:
# * T-Test: https://thedatascientist.com/how-to-do-a-t-test-in-python/

import os

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

from ct_slicing.config.data_path import DATA_FOLDER, OUTPUT_FOLDER
from ct_slicing.log import logger

SAVE_PATH = DATA_FOLDER / "py-radiomics" / "slice_glcm1d.npz"
OUTPUT_DIR = OUTPUT_FOLDER / "t_test_box_plots"

FEATURE_COUNT = 24
SLICE_FEATURE_NAMES = [
    # slice_features in SAVE_PATH have 24 features in the following order
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
if len(SLICE_FEATURE_NAMES) != FEATURE_COUNT:
    raise ValueError("SLICE_FEATURE_NAMES must have 24 features")


def load_saved_data(save_path):
    """
    Load the saved `slice_meta` and `slice_features` data from `save_path`.
    The `slice_meta` is the metadata of each slice. Its third column must be the
    label of "Benign", "Malignant", or "NoNod".
    The `slice_features` is the
    features of each slice. Its shape must be (slice_num, feature_num), and
    the feature_num must be 24 (FEATURE_COUNT). These 24 features are in the order of
    SLICE_FEATURE_NAMES.
    """
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"{save_path=} does not exist")
    saved_data = np.load(save_path, allow_pickle=True)
    logger.debug(f"loaded data with files {saved_data.files=} from {save_path=}")
    slice_meta = saved_data["slice_meta"]
    slice_features = saved_data["slice_features"]  # not "slice_flat"
    logger.debug(f"loaded slice_meta with {slice_meta.shape=}, {slice_meta[0]=}")
    logger.debug(f"loaded features with {slice_features.shape=}, {slice_features[0]=}")

    # in this npz save, there's also a features_rankin_idx, which is
    # same as ranking_idx in t_test (asserted), but we don't need it here.
    # features_rankin_idx = saved_data["features_rankin_idx"]
    # logger.debug(f"loaded features_rankin_idx with {features_rankin_idx.shape=}")

    if slice_meta.shape[1] != 4:
        raise ValueError(f"{slice_meta.shape[1]=} != {4} (expected)")
    if slice_features.shape[0] != slice_meta.shape[0]:
        raise ValueError(
            f"different entries from {slice_features.shape[0]=} != {slice_meta.shape[0]=}"
        )
    if slice_features.shape[1] != len(SLICE_FEATURE_NAMES):
        raise ValueError(
            f"different features from {slice_features.shape[1]=} != {len(SLICE_FEATURE_NAMES)=}"
        )
    return slice_meta, slice_features


def t_test(slice_meta, slice_features):
    """
    This is a two sample t-test is done for each feature, between
    benign and malignant cases. Then we can see which features have
    different mean values between benign and malignant cases.

    Args:
        slice_meta: the metadata of each slice. Its third column must be the
            label of "Benign", "Malignant", or "NoNod"
        slice_features: the features of each slice. Its shape must be
            (slice_num, feature_num), and feature_num==FEATURE_COUNT==24.
            these 24 features are in the order of SLICE_FEATURE_NAMES

    Returns:
        p_val: the p-value of each feature in the t-test
        ranking_idx: the ranking index of the p-value of each feature
        pval_sort: the sorted p-value of each feature
    """

    benign_indices = np.nonzero(slice_meta[:, 3] == "Benign")[0]
    benign_features = slice_features[benign_indices, :]
    mali_indices = np.nonzero(slice_meta[:, 3] == "Malignant")[0]
    malignant_features = slice_features[mali_indices, :]

    # no nodules cases weren't used the in this t-test
    # no_nod_indices = np.nonzero(slice_meta[:, 3] == "NoNod")[0]

    if not (benign_features.shape[1] == malignant_features.shape[1] == FEATURE_COUNT):
        raise ValueError("wrong args `slice_features` in t_test in t_test_box_plots")

    p_val = []
    for i in np.arange(24):
        aux = stats.ttest_ind(benign_features[:, i], malignant_features[:, i])
        p_val.append(aux[1])

    p_val = np.array(p_val)
    ranking_idx = np.argsort(p_val)
    p_val_sort = p_val[ranking_idx]

    return p_val, ranking_idx, p_val_sort


# print("p_val: ", p_val)
# print("ranking_idx: ", ranking_idx)
# print("pval_sort: ", pval_sort)


def save_box_plot_features(feat, y_label, feat_idx):
    feature_name = SLICE_FEATURE_NAMES[feat_idx]
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
    slice_meta, slice_features = load_saved_data(SAVE_PATH)

    # sort the features by its t-value
    p_val, ranking_idx, pval_sort = t_test(slice_meta, slice_features)

    for i in range(len(SLICE_FEATURE_NAMES)):
        save_box_plot_features(
            feat=slice_features,
            y_label=slice_meta[:, 3],
            feat_idx=i,
        )
