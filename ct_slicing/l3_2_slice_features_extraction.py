__author__ = "Debora Gil, Guillermo Torres, Carles Sanchez, Pau Cano"
__license__ = "GPL"
__email__ = "debora,gtorres,csanchez,pcano@cvc.uab.es"
__year__ = "2023"
__doc__ = """Source code for volume visualization

Computer Vision Center
Universitat Autonoma de Barcelona
"""

# TODO: a lot of warnings

import logging
import pandas as pd
from collections import OrderedDict
import SimpleITK as sitk
from radiomics import featureextractor as feature_extractor
from ct_slicing.ct_logger import logger as _logger
from ct_slicing.config.data_path import OUTPUT_FOLDER, REPO_ROOT
from ct_slicing.data_util.nii_file_access import nii_file

from ct_slicing.vis_lib.nifty_io import CoordinateOrder, read_nifty

# choose the case id and nodule id to get the path of the nodule image and mask
IMG, MASK = nii_file("CT", 1, 1)
XLSX_PATH = OUTPUT_FOLDER / "slice_features.xlsx"

# to fix wrong implementation of radiomics.setVerbosity(60)
logging.getLogger("radiomics").setLevel(logging.CRITICAL)  # run radiomics quietly
logging.getLogger("pykwalify").setLevel(logging.CRITICAL)  # pykwalify from radiomics


# TODO: refactor
def saveXLSX(filename, df):
    # write to a .xlsx file.

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(filename)  # default param engine="xlsxwriter"
    # Convert the dataframe to an XlsxWriter Excel object.
    df.to_excel(writer, sheet_name="Sheet1", index=False)
    # Close the Pandas Excel writer and output the Excel file.
    # #TODO: check if saved, due to removal of writer.save()
    writer.close()


def GetFeatures(featureVector, i, patient_id, nodule_id):
    new_row = {}
    # Showing the features and its calculated values
    for featureName in featureVector.keys():
        # print("Computed {}: {}".format(featureName, featureVector[featureName]))
        if (
            ("firstorder" in featureName)
            or ("glszm" in featureName)
            or ("glcm" in featureName)
            or ("glrlm" in featureName)
            or ("gldm" in featureName)
            or ("shape" in featureName)
        ):
            new_row.update({featureName: featureVector[featureName]})
    lst = sorted(new_row.items())  # Ordering the new_row dictionary
    # Adding some columns
    lst.insert(0, ("slice_number", i))
    lst.insert(0, ("nodule_id", nodule_id))
    lst.insert(0, ("patient_id", patient_id))
    od = OrderedDict(lst)
    return od


def SliceMode(
    patient_id, nodule_id, image, mask, meta1, meta2, extractor, maskMinPixels
):
    myList = []
    i = 0

    while i < image.shape[2]:  # X, Y, Z
        # Get the axial cut
        img_slice = image[:, :, i]
        mask_slice = mask[:, :, i]
        try:
            if maskMinPixels < mask_slice.sum():
                # Get back to the format sitk
                img_slice_sitk = sitk.GetImageFromArray(img_slice)
                mask_slice_sitk = sitk.GetImageFromArray(mask_slice)

                # Recover the pixel dimension in X and Y
                (x1, y1, z1) = meta1.spacing
                (x2, y2, z2) = meta2.spacing
                img_slice_sitk.SetSpacing((float(x1), float(y1)))
                mask_slice_sitk.SetSpacing((float(x2), float(y2)))

                # Extract features
                featureVector = extractor.execute(
                    img_slice_sitk, mask_slice_sitk, voxelBased=False
                )
                od = GetFeatures(featureVector, i, patient_id, nodule_id)
                myList.append(od)
            # else:
            #     print("features extraction skipped in slice-i: {}".format(i))
        except Exception as e:
            _logger.error(e)
            print("Exception: skipped in slice-i: {}".format(i))
        i = i + 1

    # #TODO: # df = pd.DataFrame.from_dict(myList)
    df = pd.DataFrame.from_records(myList)
    return df


radiomics_params = str(REPO_ROOT / "ct_slicing" / "config" / "Params.yaml")
# #TODO: extract params
####


# Use a parameter file, this customizes the extraction settings and
# also specifies the input image types to use and
# which features should be extracted.

# Initializing the feature extractor
extractor = feature_extractor.RadiomicsFeatureExtractor(radiomics_params)


# Reading image and mask
image, meta1 = read_nifty(IMG, coordinate_order=CoordinateOrder.xyz)
mask, meta2 = read_nifty(MASK, coordinate_order=CoordinateOrder.xyz)

# #TODO: check what's happening. Do we need a shorthand from CT,3,2 to these values
patient_id = "LIDC-IDRI-0003"
nodule_id = "2"

# Extract features slice by slice.
df = SliceMode(
    patient_id, nodule_id, image, mask, meta1, meta2, extractor, maskMinPixels=200
)

# if you get this message: "ModuleNotFoundError: No module named 'xlsxwriter'"
# then install it doing this: pip install xlsxwriter
saveXLSX(XLSX_PATH, df)

"""
Exercise 2.  Feature extraction from the slices of the image and mask.
a) Execute the code SliceFeaturesExtraction.py and analyse its output. What are the
features extracted?
    Very verbose output, but the features extracted are:
    slice_number
    original_firstorder_10Percentile
    original_firstorder_90Percentile
    original_firstorder_Energy
    original_firstorder_Entropy
    original_firstorder_InterquartileRange
    original_firstorder_Kurtosis
    original_firstorder_Maximum
    original_firstorder_Mean
    original_firstorder_MeanAbsoluteDeviation
    original_firstorder_Median
    original_firstorder_Minimum
    original_firstorder_Range
    original_firstorder_RobustMeanAbsoluteDeviation
    original_firstorder_RootMeanSquared
    original_firstorder_Skewness
    original_firstorder_TotalEnergy
    original_firstorder_Uniformity
    original_firstorder_Variance
    original_glcm_Autocorrelation
    original_glcm_ClusterProminence
    original_glcm_ClusterShade
    original_glcm_ClusterTendency
    original_glcm_Contrast
    original_glcm_Correlation
    original_glcm_DifferenceAverage
    original_glcm_DifferenceEntropy
    original_glcm_DifferenceVariance
    original_glcm_Id
    original_glcm_Idm
    original_glcm_Idmn
    original_glcm_Idn
    original_glcm_Imc1
    original_glcm_Imc2
    original_glcm_InverseVariance
    original_glcm_JointAverage
    original_glcm_JointEnergy
    original_glcm_JointEntropy
    original_glcm_MCC
    original_glcm_MaximumProbability
    original_glcm_SumAverage
    original_glcm_SumEntropy
    original_glcm_SumSquares
    original_gldm_DependenceEntropy
    original_gldm_DependenceNonUniformity
    original_gldm_DependenceNonUniformityNormalized
    original_gldm_DependenceVariance
    original_gldm_GrayLevelNonUniformity
    original_gldm_GrayLevelVariance
    original_gldm_HighGrayLevelEmphasis
    original_gldm_LargeDependenceEmphasis
    original_gldm_LargeDependenceHighGrayLevelEmphasis
    original_gldm_LargeDependenceLowGrayLevelEmphasis
    original_gldm_LowGrayLevelEmphasis
    original_gldm_SmallDependenceEmphasis
    original_gldm_SmallDependenceHighGrayLevelEmphasis
    original_gldm_SmallDependenceLowGrayLevelEmphasis
    original_glrlm_GrayLevelNonUniformity
    original_glrlm_GrayLevelNonUniformityNormalized
    original_glrlm_GrayLevelVariance
    original_glrlm_HighGrayLevelRunEmphasis
    original_glrlm_LongRunEmphasis
    original_glrlm_LongRunHighGrayLevelEmphasis
    original_glrlm_LongRunLowGrayLevelEmphasis
    original_glrlm_LowGrayLevelRunEmphasis
    original_glrlm_RunEntropy
    original_glrlm_RunLengthNonUniformity
    original_glrlm_RunLengthNonUniformityNormalized
    original_glrlm_RunPercentage
    original_glrlm_RunVariance
    original_glrlm_ShortRunEmphasis
    original_glrlm_ShortRunHighGrayLevelEmphasis
    original_glrlm_ShortRunLowGrayLevelEmphasis
    original_glszm_GrayLevelNonUniformity
    original_glszm_GrayLevelNonUniformityNormalized
    original_glszm_GrayLevelVariance
    original_glszm_HighGrayLevelZoneEmphasis
    original_glszm_LargeAreaEmphasis
    original_glszm_LargeAreaHighGrayLevelEmphasis
    original_glszm_LargeAreaLowGrayLevelEmphasis
    original_glszm_LowGrayLevelZoneEmphasis
    original_glszm_SizeZoneNonUniformity
    original_glszm_SizeZoneNonUniformityNormalized
    original_glszm_SmallAreaEmphasis
    original_glszm_SmallAreaHighGrayLevelEmphasis
    original_glszm_SmallAreaLowGrayLevelEmphasis
    original_glszm_ZoneEntropy
    original_glszm_ZonePercentage
    original_glszm_ZoneVariance

b) What are the main differences in the outputs generated by VolumeFeatureExtraction.py 
and SliceFeatureExtraction.py.
    In VolumeFeatureExtraction, the features are extracted from the whole 
    volume, while in SliceFeatureExtraction, the features are extracted from 
    each slice of the volume.
    Also, the name of the features is different: in VolumeFeatureExtraction,
    the features mostly start with "diagnostics_", and only 3 with "original_",
    while in SliceFeatureExtraction, the features all start with "original_",
"""
