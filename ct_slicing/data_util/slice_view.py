from math import prod
import numpy as np
from typing import Iterator
from ct_slicing.data_util.metadata_access import load_all_metadata

from ct_slicing.data_util.nii_file_access import load_nodule_id_pickle, nii_file
from ct_slicing.image_process import process_image
from ct_slicing.vis_lib.nifty_io import CoordinateOrder, read_nifty


EMPTY_SLICE_THRESHOLD = 0  # 0.1 (==10%) has the same effect as 0


def load_voi_slice_truth_pairs(
    empty_slice_threshold: float = EMPTY_SLICE_THRESHOLD,
) -> Iterator[tuple[np.ndarray, int]]:
    _ct_data, voi_data = load_nodule_id_pickle()
    all_metadata = load_all_metadata()

    for case_id, nodule_id in voi_data:
        metadata = all_metadata[(case_id, nodule_id)]
        diagnosis = metadata.diagnosis_value
        voi_path, _mask_path = nii_file("VOI", case_id, nodule_id)
        voi_image, _voi_meta = read_nifty(voi_path, CoordinateOrder.zyx)
        for voi_slice in voi_image:  # voi_slice is a np.ndarray of shape (H,W)
            if empty_slice_threshold > 0 and (
                np.count_nonzero(voi_slice)
                < prod(voi_slice.shape) * empty_slice_threshold
            ):
                # In the given VOI data, all slices have >10% non-zero voxels
                print(f"skipping empty slice in {case_id=} {nodule_id=}")
                continue  # skip empty slices
            # process_image: same pre-process used in featuresExtraction.py
            # #TODO: maybe we should apply the process_image to the 224x224 image
            yield process_image(voi_slice), diagnosis
