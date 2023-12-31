from ct_slicing.ct_logger import logger
from ct_slicing.data_util.metadata_access import dump_all_metadata
from ct_slicing.data_util.nii_file_access import dump_nodule_file_path


def update_intermediate_results():
    logger.info("Updating intermediate results...")
    dump_all_metadata()  # updates `nodule_metadata.pkl` (NODULE_METADATA_PICKLE)
    dump_nodule_file_path()  # updates `nodule_id.pkl` (NODULE_ID_PICKLE)
    logger.info("Done updating intermediate results.")


if __name__ == "__main__":
    update_intermediate_results()
