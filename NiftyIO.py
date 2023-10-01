"""
This is the source code for volume visualization

Computer Vision Center
Universitat Autonoma de Barcelona

__author__ = "Debora Gil, Guillermo Torres, Carles Sanchez, Pau Cano"
__license__ = "GPL"
__email__ = "debora,gtorres,csanchez,pcano@cvc.uab.es"
__year__ = "2023"
"""

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


def read_nifty(
    file_path: str, coordinate_order: str = "xyz"
) -> tuple[np.ndarray, NiiMetadata]:
    """
    Reads a NIfTI file and returns the volume and metadata.

    Args:
    - file_path (str): Full path to the file, e.g. "/home/user/Desktop/BD/LIDC-IDRI-0001_GT1.nii.gz"
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
    print("Reading Nifty format from {}".format(file_path))
    print("Image size: {}".format(image.GetSize()))

    metadata = NiiMetadata(image.GetOrigin(), image.GetSpacing(), image.GetDirection())

    # Converting from SimpleITK image to Numpy array. But also is changed the coordinate systems
    # from the image which use (x,y,z) to the array using (z,y,x).
    volume_zyx = sitk.GetArrayFromImage(image)
    if coordinate_order == "xyz":
        volume_xyz = np.transpose(
            volume_zyx, (2, 1, 0)
        )  # to back to the initial xyz coordinate system.
    else:
        volume_xyz = volume_zyx

    print("Volume shape: {}".format(volume_xyz.shape))
    print("Minimum value: {}".format(np.min(volume_xyz)))
    print("Maximum value: {}".format(np.max(volume_xyz)))

    return volume_xyz, metadata  # return two items.


def save_nifty(
    volume: np.ndarray, metadata: NiiMetadata, file_path: str, coordinate_order="xyz"
) -> None:
    """
    Save a NIfTI file from a numpy array.

    Args:
    - volume (np.ndarray): The numpy array containing the NIfTI volume.
    - metadata (NiiMetadata): The NIfTI metadata (optional). If omitted, default (identity) values are used.
    - file_path (str): The full path to the output file, e.g. "/home/user/Desktop/BD/LIDC-IDRI-0001_GT1.nii.gz".
    - coordinate_order (str): The order of dimensions in the array. Default is "xyz", which sets z as the volume's third dimension. If set to "zyx", x and z are swapped to set z as the first dimension.

    Returns:
    - None

    #TODO:
    case with metadata==None
    """
    # Converting from Numpy array to SimpleITK image.
    if coordinate_order == "xyz":
        volume = np.transpose(volume, (2, 1, 0))  # from (x,y,z) to (z,y,x)
    image = sitk.GetImageFromArray(volume)

    if metadata is not None:
        # Setting some properties to the new image
        image.SetOrigin(metadata.origin)
        image.SetSpacing(metadata.spacing)
        image.SetDirection(metadata.direction)

    sitk.WriteImage(image, file_path)
