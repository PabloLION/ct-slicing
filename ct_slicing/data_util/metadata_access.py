"""

"""

from pathlib import Path
import pickle
import pandas as pd

from ct_slicing.config.data_path import (
    META_DATA_PATH,
    METADATA_PICKLE,
    NODULE_METADATA_PICKLE,
)
from ct_slicing.data_util.nii_file_access import nii_file, patient_id_to_case_id
from ct_slicing.ct_logger import logger


def load_metadata_excel_to_data_frame(
    metadata_path: Path = META_DATA_PATH,
) -> pd.DataFrame:
    df_metadata = pd.read_excel(
        metadata_path,
        sheet_name="ML4PM_MetadatabyNoduleMaxVoting",
        engine="openpyxl",
    )
    return df_metadata


from dataclasses import dataclass


@dataclass(frozen=True)
class NoduleMetadata:
    patient_id: str
    nodule_id: int
    series_uid: str
    coord_x: float
    coord_y: float
    coord_z: float
    diameter_mm: float
    bbox_low_x: float
    bbox_low_y: float
    bbox_low_z: float
    bbox_high_x: float
    bbox_high_y: float
    bbox_high_z: float
    diagnosis: str
    diagnosis_value: int
    malignancy: str
    malignancy_value: int
    calcification: str
    calcification_value: int
    internal_structure: str
    internal_structure_value: int
    lobulation: str
    lobulation_value: int
    margin: str
    margin_value: int
    sphericity: str
    sphericity_value: int
    spiculation: str
    spiculation_value: int
    subtlety: str
    subtlety_value: int
    texture: str
    texture_value: int
    len_mal_details: int


def load_all_metadata_as_dataclass(
    df: pd.DataFrame,
) -> dict[tuple[int, int], NoduleMetadata]:
    def to_snake_case(name: str) -> str:
        return "".join(["_" + i.lower() if i.isupper() else i for i in name]).lstrip(
            "_"
        )

    records = {
        (patient_id_to_case_id(row["patient_id"]), row["nodule_id"]): NoduleMetadata(
            **{
                to_snake_case(key) if key != "seriesuid" else "series_uid": value
                for key, value in row.items()
                if isinstance(key, str)
            }
        )
        for _, row in df.iterrows()
    }
    return records


def dump_all_metadata():
    df_metadata = load_metadata_excel_to_data_frame()
    records = load_all_metadata_as_dataclass(df_metadata)
    with open(METADATA_PICKLE, "wb") as f:
        pickle.dump(records, f)
    logger.warning(f"Dumped {len(records)} records to {METADATA_PICKLE}")


def test_load_all_metadata_as_dataclass():
    df_metadata = load_metadata_excel_to_data_frame()
    records = load_all_metadata_as_dataclass(df_metadata)
    assert len(records) == 996, f"expected 996 records, got {len(records)}"
    print("test_load_data_to_dataclass passed")


# #TODO: move this part to a new nodule_data_access.py
# The metadata is only suitable for the nodule data, so we can combine them
# into a single dataclass.


@dataclass(frozen=True)
class NoduleData(NoduleMetadata):
    image_path: Path
    mask_path: Path


def load_nodule_data_as_dataclass(
    df: pd.DataFrame,
) -> dict[tuple[int, int], NoduleData]:
    def to_snake_case(name: str) -> str:
        return "".join(["_" + i.lower() if i.isupper() else i for i in name]).lstrip(
            "_"
        )

    records = {}
    for _, row in df.iterrows():
        case_id = patient_id_to_case_id(row["patient_id"])
        nodule_id = row["nodule_id"]
        try:
            nodule_mask_pair = nii_file("VOI", case_id, nodule_id)
        except FileNotFoundError:
            logger.warning(f"Skipped {case_id=}, {nodule_id=} due to missing file")
            continue

        records[(case_id, nodule_id)] = NoduleData(
            **{
                to_snake_case(key) if key != "seriesuid" else "series_uid": value
                for key, value in row.items()
                if isinstance(key, str)
            },
            image_path=nodule_mask_pair.nodule,
            mask_path=nodule_mask_pair.mask,
        )

    return records


def dump_nodule_data():
    """
    Dump (overwrite) all available VOI nodule metadata with path info to a file.
    """
    # when running this function, we there are two known missing files for
    # case_id=1002, nodule_id=1 and case_id=1002, nodule_id=2
    df_metadata = load_metadata_excel_to_data_frame()
    records = load_nodule_data_as_dataclass(df_metadata)
    with open(NODULE_METADATA_PICKLE, "wb") as f:
        pickle.dump(records, f)
    logger.warning(f"Dumped {len(records)} records to {NODULE_METADATA_PICKLE}")


if __name__ == "__main__":
    test_load_all_metadata_as_dataclass()
