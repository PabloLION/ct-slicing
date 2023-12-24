# The metadata is mostly for the VOI nodule data, so we can combine them into a
# single dataclass.

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing_extensions import deprecated
import pandas as pd

from ct_slicing.data_util.metadata_access import (
    NoduleMetadata,
    load_metadata_excel_to_data_frame,
)
from ct_slicing.data_util.nii_file_access import nii_file, patient_id_to_case_id
from ct_slicing.ct_logger import logger
from ct_slicing.config.data_path import NODULE_METADATA_PICKLE


@dataclass(frozen=True)
class NoduleData(NoduleMetadata):
    image_path: Path
    mask_path: Path


# #TODO: repetition of code
# #FIX: this is wrong because the path is not the same on different machines
@deprecated("Use load_metadata instead")
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
