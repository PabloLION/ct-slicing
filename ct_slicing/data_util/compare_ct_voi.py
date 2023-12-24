"""
The result shows that no VOI file is same as the corresponding CT file.
So we cannot remove the CT folder.
"""
from pathlib import Path
from ct_slicing.config.data_path import CT_FOLDER, VOI_FOLDER
import filecmp

from ct_slicing.data_util.nii_file_access import iter_files

EXCLUDED_FILENAMES = (".DS_Store",)


def check_voi_contains_ct():
    for file_rel_path in iter_files(CT_FOLDER):
        if file_rel_path.name in EXCLUDED_FILENAMES:
            continue
        corresponding_voi = VOI_FOLDER / file_rel_path
        if not corresponding_voi.exists():
            print(f"VOI file not found for {file_rel_path}")
            continue
        if not filecmp.cmp(CT_FOLDER / file_rel_path, corresponding_voi):
            print(f"VOI file not equal for {file_rel_path}")
            continue
        print(f"VOI file equal for {file_rel_path}")
        print("")


def check_voi_img_eq_mask():
    for file_rel_path in iter_files(VOI_FOLDER / "nodule_mask"):
        if file_rel_path.name in EXCLUDED_FILENAMES:
            continue
        if (VOI_FOLDER / "image" / file_rel_path).exists():
            print(f"VOI image found for {file_rel_path}")
            continue
        print(f"VOI image not found for {file_rel_path}")
    for file_rel_path in iter_files(VOI_FOLDER / "image"):
        if file_rel_path.name in EXCLUDED_FILENAMES:
            continue
        if (VOI_FOLDER / "nodule_mask" / file_rel_path).exists():
            print(f"VOI nodule mask found for {file_rel_path}")
            continue
        print(f"VOI nodule mask not found for {file_rel_path}")


if __name__ == "__main__":
    check_voi_contains_ct()
    # result: nothing from CT is equal to anything from VOI
    check_voi_img_eq_mask()
    # result: two VOI images are not found
    # VOI image not found for LIDC-IDRI-1002_R_2.nii.gz
    # VOI image not found for LIDC-IDRI-1002_R_1.nii.gz
    # All images have mask
