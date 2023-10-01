"""
This is the source code for volume visualization

Computer Vision Center
Universitat Autonoma de Barcelona

__author__ = "Debora Gil, Guillermo Torres, Carles Sanchez, Pau Cano"
__license__ = "GPL"
__email__ = "debora,gtorres,csanchez,pcano@cvc.uab.es"
__year__ = "2023"
"""

from enum import Enum
from pathlib import Path
from typing import NamedTuple
import numpy as np
import SimpleITK as sitk  # to read .nii Files


class NiiMetadata(NamedTuple):
    """
    Metadata of NIfTI `.nii` files.
    Tuple of (Origin, Resolution, Orientation)
    """

    origin: sitk.VectorDouble
    spacing: sitk.VectorDouble
    direction: sitk.VectorDouble


class CoordinateOrder(Enum):
    """
    Order of dimensions in array.
    """

    xyz = "xyz"  # sets x,y,z as first, second and third dimension of volume
    zyx = "zyx"  # swaps x and z to set z as first dimension


class CoordinateOrderError(ValueError):
    def __init__(self, received_order: CoordinateOrder):
        self.received_order = received_order

    def __str__(self):
        return f"Coordinate order {self.received_order} not supported. Supported orders are: xyz, zyx"


def read_nifty(
    file_path: Path, coordinate_order: CoordinateOrder = CoordinateOrder.xyz
) -> tuple[np.ndarray, NiiMetadata]:
    """
    Reads a NIfTI file and returns the volume and metadata.

    Args:
    - file_path (Path): Full path to the file, e.g. "/home/user/Desktop/BD/LIDC-IDRI-0001_GT1.nii.gz"
    - coordinate_order (str): Order of dimensions in array:
        - 'xyz' (Default) sets z as volume third dimension
        - 'zyx' swaps x and z to set z as first dimension

    Returns:
    - volume_xyz (np.ndarray): Numpy array containing .nii volume
    - metadata (NiiMetadata): NIfTI metadata

    Example:
    1. Skip metadata output argument
    import os
    file_path=os.path.join("Data_Session1","LIDC-IDRI-0001_GT1.nii.gz")
    vol,_=read_nifty(file_path)
    """
    image = sitk.ReadImage(file_path)
    metadata = NiiMetadata(image.GetOrigin(), image.GetSpacing(), image.GetDirection())

    print(f"Successfully read Nifty file with {metadata=} from {file_path=}")

    # Convert SimpleITK image to Numpy array. By default, the dimension is in (z,y,x) order.
    volume_zyx = sitk.GetArrayFromImage(image)

    if coordinate_order == CoordinateOrder.xyz:
        volume = np.transpose(volume_zyx, (2, 1, 0))  # to xyz coordinate system.
    elif coordinate_order == CoordinateOrder.zyx:
        volume = volume_zyx
    else:
        raise CoordinateOrderError(coordinate_order)

    print("Volume shape: {}".format(volume.shape))
    print("Minimum value: {}".format(np.min(volume)))
    print("Maximum value: {}".format(np.max(volume)))

    return volume, metadata  # return two items.


def save_nifty(
    volume: np.ndarray,
    metadata: NiiMetadata | None,
    file_path: Path,
    coordinate_order: CoordinateOrder = CoordinateOrder.xyz,
) -> None:
    """
    Save a NIfTI file from a numpy array.

    Args:
    - volume (np.ndarray): The numpy array containing the NIfTI volume.
    - metadata (NiiMetadata): The NIfTI metadata (optional). If omitted, default (identity) values are used.
    - file_path (Path): The full path to the output file, e.g. Path("/home/user/LIDC-IDRI-0001_GT1.nii.gz").
    - coordinate_order (str): The order of dimensions in the array. Default is "xyz", which sets z as the volume's third dimension. If set to "zyx", x and z are swapped to set z as the first dimension.

    Returns:
    - None
    """
    # Converting from Numpy array to SimpleITK image.
    if coordinate_order == CoordinateOrder.xyz:
        volume = np.transpose(volume, (2, 1, 0))  # from (x,y,z) to (z,y,x)
    elif coordinate_order == CoordinateOrder.zyx:
        pass  # volume is already in (z,y,x) order
    else:
        raise CoordinateOrderError(coordinate_order)

    image = sitk.GetImageFromArray(volume)

    if metadata is not None:
        # Setting some properties to the new image
        image.SetOrigin(metadata.origin)
        image.SetSpacing(metadata.spacing)
        image.SetDirection(metadata.direction)

    sitk.WriteImage(image, file_path)
