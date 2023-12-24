from ct_slicing.ct_logger import logger
from ct_slicing.data_util.metadata_access import (
    dump_nodule_data,
)
from ct_slicing.data_util.nii_file_access import dump_available_nodules


def update_intermediate_results():
    logger.info("Updating intermediate results...")
    # dump_available_metadata is kind of outdated dump_nodule_metadata is enough
    dump_nodule_data()  # updates `nodule_metadata.pkl` (NODULE_METADATA_PICKLE)
    dump_available_nodules()  # updates `nodule_id.pkl` (NODULE_ID_PICKLE)

    logger.info("Done updating intermediate results.")
