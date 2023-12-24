"""
This module provides functions to load metadata from excel file to dataclass
"""

from pathlib import Path
import pickle
import pandas as pd

from ct_slicing.config.data_path import (
    META_DATA_PATH,
    METADATA_PICKLE,
)
from ct_slicing.data_util.nii_file_access import patient_id_to_case_id
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


def load_metadata(case_id: int, nodule_id: int) -> NoduleMetadata:
    if not METADATA_PICKLE.exists():
        logger.warning(f"Metadata pickle {METADATA_PICKLE} does not exist. Dumping...")
        dump_all_metadata()
    with open(METADATA_PICKLE, "rb") as f:
        records = pickle.load(f)
    return records[case_id, nodule_id]


def test_dump_all_metadata():
    dump_all_metadata()
    expected_records = load_all_metadata_as_dataclass(
        load_metadata_excel_to_data_frame()
    )
    with open(METADATA_PICKLE, "rb") as f:
        records = pickle.load(f)
    assert records == expected_records, "records not equal"
    print("test_dump_all_metadata passed")


def test_load_metadata():
    metadata = load_metadata(1, 1)
    assert metadata.patient_id == "LIDC-IDRI-0001"
    assert metadata.nodule_id == 1
    assert (
        metadata.series_uid
        == "1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192"
    ), f"expected 1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192, but got {metadata.series_uid}"
    print("test_load_metadata passed")


if __name__ == "__main__":
    test_load_all_metadata_as_dataclass()
    test_dump_all_metadata()
    test_load_metadata()
