import pickle
import pandas as pd

from ct_slicing.config.data_path import DATA_FOLDER, REPO_ROOT
from ct_slicing.data_util.data_access import patient_id_to_case_id
from ct_slicing.ct_logger import logger

META_DATA_PATH = DATA_FOLDER / "MetadatabyNoduleMaxVoting.xlsx"

df_metadata = pd.read_excel(
    META_DATA_PATH,
    sheet_name="ML4PM_MetadatabyNoduleMaxVoting",
    engine="openpyxl",
)


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


def load_data_to_dataclass(df: pd.DataFrame) -> dict[tuple[int, int], NoduleMetadata]:
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


META_DATA_PICKLE = REPO_ROOT / "data" / "metadata.pickle"


def dump_available_metadata():
    records = load_data_to_dataclass(df_metadata)
    with open(META_DATA_PICKLE, "wb") as f:
        pickle.dump(records, f)
    logger.warning(f"Dumped {len(records)} records to {META_DATA_PICKLE}")


def test_load_data_to_dataclass():
    records = load_data_to_dataclass(df_metadata)
    assert len(records) == 996, f"expected 996 records, got {len(records)}"
    print("test_load_data_to_dataclass passed")


if __name__ == "__main__":
    test_load_data_to_dataclass()
    dump_available_metadata()
