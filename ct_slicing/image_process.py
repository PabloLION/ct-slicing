"""
This module contains functions to process the image before feature extraction.
"""

import numpy as np

from ct_slicing.ct_logger import logger


def process_image(
    image: np.ndarray,
    shift_value: int = 1024,
    scaled_range: tuple[int, int] = (0, 4000),
    gray_levels: int = 24,
) -> np.ndarray:
    """
    Process the image before feature extraction with three steps:
    1. shift_values: add shift_value to all values
    2. set_range: scale the range to scaled_range
    3. set_gray_level: quantize the gray intensity to gray_levels discrete levels

    Args:
        image (np.ndarray): an image with values between 0 and 1
        shift_value (int): the value to shift the image by
        scaled_range (tuple[int, int]): the range to scale the image to
        gray_levels (int): the number of gray levels to use

    Returns:
        np.ndarray: the preprocessed image
    """
    image = shift_values(image, shift_value)
    image = set_range(image, scaled_range[0], scaled_range[1])
    image = set_gray_level(image, gray_levels)
    return image


def shift_values(image: np.ndarray, value: int) -> np.ndarray:
    """
    Move the values of the image to the right by a value.
    Was called ShiftValues
    """

    image = image + value
    logger.debug(f"Range after Shift: {image.min()} - {image.max()}")
    return image


def set_range(image: np.ndarray, in_min: int, in_max: int) -> np.ndarray:
    """
    Set the range of the image to the specified values.
    Was called SetRange.
    """
    image = (image - image.min()) / (image.max() - image.min())
    image = image * (in_max - in_min) + in_min

    image[image < 0] = 0
    image[image > image.max()] = image.max()
    logger.debug(f"Range after SetRange: {image.min():.2f} - {image.max():.2f}")
    return image


def set_gray_level(image: np.ndarray, levels: int) -> np.ndarray:
    """
    Set the number of gray levels of the image to the specified value.
    Was called SetGrayLevel

    Args:
        image (np.ndarray): an image with values between 0 and 1
        levels (int): the number of gray levels to use
    """
    image = (image * levels).astype(np.uint8)  # get into integer values
    logger.debug(
        f"Range after SetGrayLevel: {image.min():.2f} - {image.max():.2f} levels={levels}"
    )
    return image
