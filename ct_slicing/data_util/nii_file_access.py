"""
This module is used to access the nii files (CT images and masks, AKA NIfTI files).

Reading file and adjusting the paths is mentally too heavy for me.
Here I use a simpler way to get the file path.
"""

# Every file has a full volume and >=1 mask(s).
# now we have 2 sections: CT and VOI
# I want to write the file name like nii_file("CT", 1, 0)
# nii_file("VOI", 1, 1)
# # 0 means mask


from pathlib import Path
from typing import Iterator, Literal, NamedTuple
import pickle

from ct_slicing.config.data_path import (
    DATA_FOLDER,
    NODULE_ID_PICKLE,
    CT_FOLDER,
    VOI_FOLDER,
)
from ct_slicing.ct_logger import logger

Section = Literal["CT", "VOI"]


class NoduleMaskPair(NamedTuple):
    nodule: Path
    mask: Path


def case_id_to_patient_id(case_id: int) -> str:
    """
    Convert case id to patient id.

    Args:
        case_id (int): case id, e.g. 1 for LIDC-IDRI-0001, 105 for LIDC-IDRI-0105

    Returns:
        str: patient id, e.g. LIDC-IDRI-0001, LIDC-IDRI-0105
    """
    return f"LIDC-IDRI-{case_id:04d}"


def patient_id_to_case_id(patient_id: str) -> int:
    """
    Convert patient id to case id.

    Args:
        patient_id (str): patient id, e.g. LIDC-IDRI-0001, LIDC-IDRI-0105

    Returns:
        int: case id, e.g. 1 for LIDC-IDRI-0001, 105 for LIDC-IDRI-0105
    """
    stem = str(patient_id).rstrip(".nii.gz").lstrip("LIDC-IDRI-")
    if not stem.isnumeric():
        raise ValueError(f"Invalid patient id: {patient_id}")
    return int(stem)


def nii_path_to_case_id_mask_id(nii_path: Path) -> tuple[int, int]:
    stem = str(nii_path.name).rstrip(".nii.gz").lstrip("LIDC-IDRI-")
    case_id, mask_id = stem.split("_R_")
    return int(case_id), int(mask_id)


def _parse_nii_path(section: Section, case_id: int, nodule_id: int) -> NoduleMaskPair:
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
        image = CT_FOLDER / "image" / f"{case_id_to_patient_id(case_id)}.nii.gz"
        # data/CT/nodule_mask/LIDC-IDRI-0003_R_2.nii.gz
        mask = (
            CT_FOLDER
            / "nodule_mask"
            / f"{case_id_to_patient_id(case_id)}_R_{nodule_id}.nii.gz"
        )
    elif section == "VOI":
        # data/VOIs/image/LIDC-IDRI-0003_R_2.nii.gz
        image = (
            VOI_FOLDER
            / "image"
            / f"{case_id_to_patient_id(case_id)}_R_{nodule_id}.nii.gz"
        )
        # data/VOIs/nodule_mask/LIDC-IDRI-0001_R_1.nii.gz
        mask = (
            VOI_FOLDER
            / "nodule_mask"
            / f"{case_id_to_patient_id(case_id)}_R_{nodule_id}.nii.gz"
        )
    else:
        raise ValueError(f"section must be CT or VOI, but got {section}")
    return NoduleMaskPair(image, mask)


def load_nodule_id_pickle() -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """
    Load the nodule id pickle file.
    """
    if not NODULE_ID_PICKLE.exists():
        logger.warning(
            f"Nodule id pickle {NODULE_ID_PICKLE} does not exist. Dumping..."
        )
    with open(NODULE_ID_PICKLE, "rb") as f:
        data = pickle.load(f)
        ct_nodules = data["CT"]
        voi_nodules = data["VOI"]
    return ct_nodules, voi_nodules


def iter_files(base_path: Path):
    """Iterate over all files in the given directory recursively.
    Return the relative path to each file.
    """
    for file_path in base_path.glob("**/*"):
        if file_path.is_file():
            yield file_path.relative_to(base_path)


def dump_nodule_file_path():
    """
    Dump (overwrite) all available nodule id to a file.
    """
    ct_nodules: set[tuple[int, int]] = set()
    voi_nodules: set[tuple[int, int]] = set()
    for file in iter_files(CT_FOLDER / "nodule_mask"):
        ct_nodules.add(nii_path_to_case_id_mask_id(file))
    for file in iter_files(VOI_FOLDER / "image"):
        # Know from `compare_ct_voi.py`, image is subset of nodule_mask.
        voi_nodules.add(nii_path_to_case_id_mask_id(file))
    # dump to file
    with open(NODULE_ID_PICKLE, "wb") as f:
        pickle.dump({"CT": ct_nodules, "VOI": voi_nodules}, f)
    logger.warning(f"Dumped nodule id to {NODULE_ID_PICKLE}")


def get_section_case_id_mask_id_iter() -> (
    tuple[Iterator[tuple[Section, int, int]], Iterator[tuple[Section, int, int]]]
):
    """
    Get the case id and mask id iterator.
    """
    ct_nodules, voi_nodules = load_nodule_id_pickle()
    return (
        (("CT", case_id, mask_id) for case_id, mask_id in ct_nodules),
        (("VOI", case_id, mask_id) for case_id, mask_id in voi_nodules),
    )


def get_nii_path_iter() -> tuple[Iterator[NoduleMaskPair], Iterator[NoduleMaskPair]]:
    """
    Get the nii file path iterator.
    """
    ct_nodules, voi_nodules = load_nodule_id_pickle()
    return (nii_file("CT", case, mask) for case, mask in ct_nodules), (
        nii_file("VOI", case, mask) for case, mask in voi_nodules
    )


def nii_exist(sections: Section, case_id: int, mask_id: int) -> bool:
    """
    Check if the nii file exists.
    Created to future proof against more data from other sections.
    And also for indexing and iterator.
    Maybe I'll include the full LUNA16 dataset later.
    """
    ct_nodules, voi_nodules = load_nodule_id_pickle()
    if sections == "CT":
        return (case_id, mask_id) in ct_nodules
    elif sections == "VOI":
        return (case_id, mask_id) in voi_nodules
    else:
        raise ValueError(f"sections must be CT or VOI, but got {sections}")


def nii_file(sections: Section, case_id: int, nodule_id: int) -> NoduleMaskPair:
    """
    Return the nii file pair.
    """
    nodule, mask = _parse_nii_path(sections, case_id, nodule_id)
    if not nii_exist(sections, case_id, nodule_id):
        raise FileNotFoundError(
            f"File not found for {sections}, {case_id}, {nodule_id}"
        )
    return NoduleMaskPair(nodule, mask)


# Test if the file path is correct. Assume we won't rename the data files.
def _test_nii_file_with_expected_not_found_error(
    sections: Section, case_id: int, mask_id: int
):
    """
    Expect a FileNotFoundError from nii_file.
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
    _test_nii_file_with_expected_not_found_error("CT", 3, 1)
    _test_nii_file_with_expected_not_found_error("CT", 2, 1)
    _test_nii_file_with_expected_not_found_error("CT", 5, 5)
    _test_nii_file_with_expected_not_found_error("VOI", 2, 1)
    _test_nii_file_with_expected_not_found_error("VOI", 3, 1)
    _test_nii_file_with_expected_not_found_error("VOI", 70, 1)
    _test_nii_file_with_expected_not_found_error("VOI", 224, 1)
    _test_nii_file_with_expected_not_found_error("VOI", 234, 2)
    _test_nii_file_with_expected_not_found_error("VOI", 235, 1)
    _test_nii_file_with_expected_not_found_error("VOI", 1011, 4)
    print("test_nii_file passed")


if __name__ == "__main__":
    test_nii_file()
