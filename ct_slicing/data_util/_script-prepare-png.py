if __name__ != "__main__":
    raise ImportError(f"Script {__file__} should not be imported as a module")

import logging
from venv import logger
import numpy
from torchvision import transforms
from ct_slicing.config.data_path import SLICE_IMAGE_FOLDER
from ct_slicing.data_util.metadata_access import load_all_metadata

from ct_slicing.data_util.nii_file_access import load_nodule_id_pickle, nii_file
from ct_slicing.vis_lib.nifty_io import CoordinateOrder, read_nifty

# Parameters
EMPTY_SLICE_THRESHOLD = 0  # 0.1 (==10%) has the same effect as 0


logger.setLevel(logging.INFO)

_ct_data, voi_data = load_nodule_id_pickle()

max_height_or_width = 0  # cannot use height: some images have very small height
for case_id, nodule_id in voi_data:
    voi_path, _mask_path = nii_file("VOI", case_id, nodule_id)
    voi_image, _voi_meta = read_nifty(voi_path, CoordinateOrder.zyx)
    # get H,W of voi_image
    max_height_or_width = max(
        max_height_or_width, voi_image.shape[1], voi_image.shape[2]
    )
print(max_height_or_width)  # 109

# the transform used in featuresExtraction.py was flawed, this is the correction
default_transform = transforms.Compose(
    [
        # transforms.Lambda(process_image), this process makes images meaningless
        transforms.ToPILImage(),
        # transforms.Resize(224), not working, because the image is not square
        # transforms.Resize(256),
        # transforms.Normalize(mean=[0.5], std=[0.5]),  # should not normalize
        # if we have to normalize, we should do it on the whole scan instead of
        # only this small region
        transforms.CenterCrop(max_height_or_width),  # correct way to resize
    ]
)


def convert_nodule_to_png_batch(
    case_id,
    nodule_id,
    diagnosis,
    coordinate_order=CoordinateOrder.zyx,
    empty_slice_threshold: float = EMPTY_SLICE_THRESHOLD,
    transform=default_transform,
):
    """
    Convert the nodule to png images. The images will be saved in the same folder as
    the nodule data.
    """

    voi_path, _mask_path = nii_file("VOI", case_id, nodule_id)
    voi_image, _voi_meta = read_nifty(voi_path, coordinate_order)
    for slice_idx, voi_slice in enumerate(voi_image):
        # voi_slice is a np.ndarray of shape (H,W)
        if empty_slice_threshold > 0 and (
            numpy.count_nonzero(voi_slice)
            < numpy.prod(voi_slice.shape) * empty_slice_threshold
        ):
            # In the given VOI data, all slices have >10% non-zero voxels
            print(f"skipping empty slice in {case_id=} {nodule_id=}")
            continue  # skip empty slices
        # process_image: same pre-process used in featuresExtraction.py

        pil_image = transform(voi_slice.astype(numpy.uint8))
        image_name = f"{case_id:04}-{nodule_id:02}-{slice_idx:02}-{diagnosis}.png"
        pil_image.save(image_path := SLICE_IMAGE_FOLDER / image_name)
        logger.debug(f"Saved image to {image_path}")


all_metadata = load_all_metadata()

_ct_data, voi_data = load_nodule_id_pickle()
for case_id, nodule_id in voi_data:
    diagnosis = all_metadata[case_id, nodule_id].diagnosis_value
    convert_nodule_to_png_batch(case_id, nodule_id, diagnosis=diagnosis)
    logger.info(f"Converted {case_id=} {nodule_id=}, {diagnosis=}")
logger.info("PNG image generation finished")
