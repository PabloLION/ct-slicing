"""
Reading file and adjusting the paths is mentally too heavy for me.
Here I use a simpler way to get the file path.
"""

# Every file has a full volume and >=1 mask(s).
# now we have 2 sections: CT and VOI
# I want to write the file name like nii_file("CT", 1, 0)
# nii_file("VOI", 1, 1)
# # 0 means mask


from pathlib import Path
from typing import Literal, NamedTuple
import pickle

from ct_slicing.config.data_path import DATA_FOLDER, REPO_ROOT
from ct_slicing.data_util.compare_ct_voi import CT_FOLDER, VOI_FOLDER, iter_files


class NoduleMaskPair(NamedTuple):
    nodule: Path
    mask: Path


def nii_path(
    section: Literal["CT", "VOI"], case_id: int, mask_id: int
) -> NoduleMaskPair:
    """Return a pair of (image path, mask path) of a given case id and
    nodule index.
    If the file does not exist, raise FileNotFoundError.

    Args:
        section (Literal["CT", "VOI"]): CT or VOI
        case_id (int): case id
        mask_id (int): mask id

    Returns:
        Path: the file path
    """
    if section == "CT":
        # data/CT/image/LIDC-IDRI-0001.nii.gz
        image = DATA_FOLDER / section / "image" / f"LIDC-IDRI-{case_id:04d}.nii.gz"
        # data/CT/nodule_mask/LIDC-IDRI-0003_R_2.nii.gz
        mask = (
            DATA_FOLDER
            / section
            / "nodule_mask"
            / f"LIDC-IDRI-{case_id:04d}_R_{mask_id}.nii.gz"
        )
    elif section == "VOI":
        # data/VOIs/image/LIDC-IDRI-0003_R_2.nii.gz
        image = (
            DATA_FOLDER
            / "VOIs"
            / "image"
            / f"LIDC-IDRI-{case_id:04d}_R_{mask_id}.nii.gz"
        )
        # data/VOIs/nodule_mask/LIDC-IDRI-0001_R_1.nii.gz
        mask = (
            DATA_FOLDER
            / "VOIs"
            / "nodule_mask"
            / f"LIDC-IDRI-{case_id:04d}_R_{mask_id}.nii.gz"
        )
    else:
        raise ValueError(f"section must be CT or VOI, but got {section}")
    return NoduleMaskPair(image, mask)


with open(REPO_ROOT / "ct_slicing" / "data_util" / "nodule_id.pkl", "rb") as f:
    data = pickle.load(f)
    CT_NODULES = data["CT"]
    VOI_NODULES = data["VOI"]


def nii_exist(sections: Literal["CT", "VOI"], case_id: int, mask_id: int) -> bool:
    """
    Check if the nii file exists.
    Created to future proof against more data from other sections.
    And also for indexing and iterator.
    Maybe I'll include the full LUNA16 dataset later.
    """
    if sections == "CT":
        return (case_id, mask_id) in CT_NODULES
    elif sections == "VOI":
        return (case_id, mask_id) in VOI_NODULES
    else:
        raise ValueError(f"sections must be CT or VOI, but got {sections}")


def nii_file(
    sections: Literal["CT", "VOI"], case_id: int, mask_id: int
) -> NoduleMaskPair:
    """
    Return the nii file pair.
    """
    nodule, mask = nii_path(sections, case_id, mask_id)
    if not nii_exist(sections, case_id, mask_id):
        raise FileNotFoundError(f"File not found: {nodule} or {mask}")
    return NoduleMaskPair(nodule, mask)


ct_iter = (nii_file("CT", case, mask) for case, mask in CT_NODULES)
voi_iter = (nii_file("VOI", case, mask) for case, mask in VOI_NODULES)


def nii_path_to_case_id_mask_id(nii_path: Path) -> tuple[int, int]:
    stem = str(nii_path.name).rstrip(".nii.gz").lstrip("LIDC-IDRI-")
    case_id, mask_id = stem.split("_R_")
    return int(case_id), int(mask_id)


# Test if the file path is correct. Assume we won't rename the data files.
def expect_nii_file_not_found_error(
    sections: Literal["CT", "VOI"], case_id: int, mask_id: int
):
    """
    Expect a FileNotFoundError from fn(*args, **kwargs).
    """
    try:
        nii_file(sections, case_id, mask_id)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError(
            f"FileNotFoundError not raised for {sections}, {case_id}, {mask_id}"
        )


def test_nii_file():
    """
    Test if the file path is correct. Assume we won't rename the data files.
    """
    # Known existing file ("CT", 3, 2), ("CT",1,1), (CT,5,2), VOI,1,1; VOI,8,2;
    # VOI,10,3; VOI,11,10; VOI,118,3; VOI235,2; VOI,1011,2;
    # Known not existing ("CT", 3, 1); CT,2,1; CT,5,5; VOI,2,1; VOI,3,1;
    # VOI70,1; VOI224,1, VOI 234,2; VOI235,1; VOI,1011,4;
    nii_file("CT", 3, 2)
    nii_file("CT", 1, 1)
    nii_file("CT", 5, 2)
    nii_file("VOI", 1, 1)
    nii_file("VOI", 8, 2)
    nii_file("VOI", 10, 3)
    nii_file("VOI", 11, 10)
    nii_file("VOI", 118, 3)
    nii_file("VOI", 235, 2)
    nii_file("VOI", 1011, 2)
    # not existing
    expect_nii_file_not_found_error("CT", 3, 1)
    expect_nii_file_not_found_error("CT", 2, 1)
    expect_nii_file_not_found_error("CT", 5, 5)
    expect_nii_file_not_found_error("VOI", 2, 1)
    expect_nii_file_not_found_error("VOI", 3, 1)
    expect_nii_file_not_found_error("VOI", 70, 1)
    expect_nii_file_not_found_error("VOI", 224, 1)
    expect_nii_file_not_found_error("VOI", 234, 2)
    expect_nii_file_not_found_error("VOI", 235, 1)
    expect_nii_file_not_found_error("VOI", 1011, 4)
    print("test_nii_file passed")


def dump_available_nodules():
    """
    Dump all available nodule ids to a file.
    """
    ct_nodules: set[tuple[int, int]] = set()
    voi_nodules: set[tuple[int, int]] = set()
    for file in iter_files(CT_FOLDER / "nodule_mask"):
        ct_nodules.add(nii_path_to_case_id_mask_id(file))
    for file in iter_files(VOI_FOLDER / "image"):
        # Know from `compare_ct_voi.py`, image is subset of nodule_mask.
        voi_nodules.add(nii_path_to_case_id_mask_id(file))

    # dump to file

    with open(REPO_ROOT / "ct_slicing" / "data_util" / "nodule_id.pkl", "wb") as f:
        pickle.dump({"CT": ct_nodules, "VOI": voi_nodules}, f)


if __name__ == "__main__":
    test_nii_file()
    # dump_available_nodules() # use when new data is added
