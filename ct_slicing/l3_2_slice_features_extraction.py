__author__ = "Debora Gil, Guillermo Torres, Carles Sanchez, Pau Cano"
__license__ = "GPL"
__email__ = "debora,gtorres,csanchez,pcano@cvc.uab.es"
__year__ = "2023"
__doc__ = """Source code for volume visualization

Computer Vision Center
Universitat Autonoma de Barcelona
"""

# TODO: a lot of warnings

import pandas as pd
from collections import OrderedDict
import SimpleITK as sitk
from radiomics import featureextractor
from radiomics import setVerbosity
from ct_slicing.config.data_path import DATA_FOLDER, OUTPUT_FOLDER, REPO_ROOT

from ct_slicing.vis_lib.nifty_io import CoordinateOrder, read_nifty

setVerbosity(60)

XLSX_PATH = OUTPUT_FOLDER / "slice_features.xlsx"


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
        except:
            print("Exception: skipped in slice-i: {}".format(i))
        i = i + 1

    # #TODO: # df = pd.DataFrame.from_dict(myList)
    df = pd.DataFrame.from_records(myList)
    return df


#### Parameters to be configured

IMG_FOLDER = DATA_FOLDER / "CT" / "image"
MASK_FOLDER = DATA_FOLDER / "CT" / "nodule_mask"
IMG = IMG_FOLDER / "LIDC-IDRI-0003.nii.gz"
MASK = MASK_FOLDER / "LIDC-IDRI-0003_R_2.nii.gz"

radiomics_params = str(REPO_ROOT / "ct_slicing" / "pr_config" / "Params.yaml")
# #TODO: extract params
####


# Use a parameter file, this customizes the extraction settings and
# also specifies the input image types to use and
# which features should be extracted.

# Initializing the feature extractor
extractor = featureextractor.RadiomicsFeatureExtractor(radiomics_params)


# Reading image and mask
image, meta1 = read_nifty(IMG, coordinate_order=CoordinateOrder.xyz)
mask, meta2 = read_nifty(MASK, coordinate_order=CoordinateOrder.xyz)

patient_id = "LIDC-IDRI-0003"
nodule_id = "2"

# Extract features slice by slice.
df = SliceMode(
    patient_id, nodule_id, image, mask, meta1, meta2, extractor, maskMinPixels=200
)

# if you get this message: "ModuleNotFoundError: No module named 'xlsxwriter'"
# then install it doing this: pip install xlsxwriter
saveXLSX(XLSX_PATH, df)
