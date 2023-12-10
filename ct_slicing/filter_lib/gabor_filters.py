__author__ = "Debora Gil"
__license__ = "GPL"
__email__ = "debora@cvc.uab.es"
__year__ = "2023"
__doc__ = """Source code for volume visualization

Computer Vision Center
Universitat Autonoma de Barcelona
"""

# Unit: Feature Extraction / Local Descriptors


### IMPORT PY LIBRARIES
# Python Library 2 manage volumetric data
from itertools import product
from typing import Collection
import numpy as np

# 2D GaborKernels
from skimage.filters import gabor_kernel


def gabor_2d_bank(
    theta: Collection[float] = np.arange(0, 4, 1) / 4.0 * np.pi,
    sigma: Collection[float] = [1, 3],
    frequency: Collection[float] = [0.05, 0.25],
) -> tuple[list[np.ndarray], list[np.ndarray], list[list[float]]]:
    """2D Gabor filter factory.
    By default, it creates 16 filters with 4 orientations, 2 frequencies and 2 sigma

    Args:
        theta (Collection[float], optional): Defaults to [0, pi/4, pi/2, 3*pi/4]
        sigma (Collection[float], optional): Defaults to [1, 3].
        frequency (Collection[float], optional): Defaults to [0.05, 0.25].
    """

    kernel_re, kernel_im, params = [], [], []
    for t, s, f in product(theta, sigma, frequency):
        kernel = gabor_kernel(f, theta=t, sigma_x=s, sigma_y=s)
        kernel_re.append(np.real(kernel))
        kernel_im.append(np.imag(kernel))
        params.append([t, s, f])

    return kernel_re, kernel_im, params


def gabor_3d_bank(
    angles: np.ndarray = np.tile(np.arange(0, 4) / 4.0 * np.pi, (2, 1)),
    sigma: Collection[float] = [1, 3],
    frequency: Collection[float] = [0.05, 0.25],
) -> tuple[list[np.ndarray], list[list[float]]]:
    """3D Gabor filter factory.
    By default, it creates 64 filters with 16 orientations, 2 frequencies and 2 sigma

    Args:
        - theta (np.ndarray, optional): Defaults to
            np.tile(np.arange(0, 4) / 4.0 * np.pi, (2, 1)).
        - sigma (Collection[float], optional): Defaults to [1, 3].
        - frequency (Collection[float], optional): Defaults to [0.05, 0.25].

    Returns:
        - tuple[list[np.ndarray], list[list[float]]]:
            A tuple containing the list of kernels and the list of parameters.
    """
    assert angles.shape[0] == 2, "theta must be a 2xN matrix"
    kernels, params = [], []

    for t0, t1, s, f in product(angles[0, :], angles[1, :], sigma, frequency):
        kernel = gabor_3d_filter((s, s, s), (t0, t1), f)
        kernels.append(kernel)
        params.append((t0, t1, s, f))

    return kernels, params


def gabor_3d_filter(
    sigma: tuple[float, float, float], angles: tuple[float, float], frequency: float
) -> np.ndarray:
    """
    Creates a 3D Gabor filter.

    Args:
        - sigma (Tuple[float, float, float]): Standard deviations
            (sigma_x, sigma_y, sigma_z) of the Gaussian envelope.
        - angles (Tuple[float, float]): Orientation angles (theta, phi)
            of the filter.
        - frequency (float): Frequency of the sinusoidal component.

    Returns:
        - np.ndarray: 3D Gabor filter.
    """
    (sigma_x, sigma_y, sigma_z), (theta, phi) = sigma, angles
    # bounding box for the filter
    x, y, z = gabor_3d_mesh_grid(sigma, angles)

    # Rotation
    x_theta = (
        x * np.cos(theta) * np.cos(phi)
        + y * np.sin(theta) * np.cos(phi)
        + z * np.sin(phi)
    )
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    z_theta = (
        -x * np.cos(theta) * np.sin(phi)
        - y * np.sin(theta) * np.sin(phi)
        + z * np.cos(phi)
    )

    # Gaussian Envelope
    gaussian_envelope = np.exp(
        -0.5
        * (
            x_theta**2 / sigma_x**2
            + y_theta**2 / sigma_y**2
            + z_theta**2 / sigma_z**2
        )
    )

    # Complex Gabor Filter
    gabor_filter = gaussian_envelope * np.cos(2 * np.pi * frequency * x_theta)

    return gabor_filter


def gabor_3d_bounding_box(
    sigma: tuple[float, float, float], angles: tuple[float, float]
) -> tuple[int, int, int, int, int, int]:
    """
    Calculate the (bounding box) for a 3D Gabor filter.

    Args:
        - sigma (Tuple[float, float, float]): Standard deviations
            (sigma_x, sigma_y, sigma_z) of the Gaussian envelope.
        - theta (Tuple[float, float]): Orientation angles (theta, phi)
            of the filter.

    Returns:
        - Tuple[float, float, float, float, float, float]:
            The bounding box (x_min, x_max, y_min, y_max, z_min, z_max).
    """

    (sigma_x, sigma_y, sigma_z), (theta, phi) = sigma, angles

    # Define the number of standard deviations to include in the support
    n_std, inv_std = 3, 1 / 3

    # Calculate the maximum extent of the filter in each dimension
    x_max = n_std * max(
        inv_std,
        abs(sigma_x * np.cos(theta) * np.cos(phi)),
        abs(sigma_y * np.sin(theta)),
        abs(sigma_z * np.cos(theta) * np.sin(phi)),
    )
    y_max = n_std * max(
        inv_std,
        abs(sigma_x * np.sin(theta) * np.cos(phi)),
        abs(sigma_y * np.cos(theta)),
        abs(sigma_z * np.sin(theta) * np.sin(phi)),
    )
    z_max = n_std * max(inv_std, abs(sigma_x * np.sin(phi)), abs(sigma_z * np.cos(phi)))

    # the minimum extent is negative of the maximum
    x_max, y_max, z_max = np.ceil([x_max, y_max, z_max]).astype(int)
    x_min, y_min, z_min = -x_max, -y_max, -z_max

    return x_min, x_max, y_min, y_max, z_min, z_max


def gabor_3d_mesh_grid(
    sigma: tuple[float, float, float], angles: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the mesh grid for a 3D Gabor filter.

    Args:
        - sigma (Tuple[float, float, float]): Standard deviations
            (sigma_x, sigma_y, sigma_z) of the Gaussian envelope.
        - angles (Tuple[float, float]): Orientation angles (theta, phi)
            of the filter.

    Returns:
        - Tuple[np.ndarray, np.ndarray, np.ndarray]:
            The mesh grid (x, y, z).
    """
    # bounding box for the filter
    x_min, x_max, y_min, y_max, z_min, z_max = gabor_3d_bounding_box(sigma, angles)
    grid = np.meshgrid(
        np.arange(x_min, x_max + 1),
        np.arange(y_min, y_max + 1),
        np.arange(z_min, z_max + 1),
    )  # assert the list length is 3
    assert len(grid) == 3, "The grid must be a 3D mesh grid"
    x, y, z = grid
    return x, y, z


# to check if the old and new gabor filter banks are the same
# we should check into every kernel, like this:
# old, new = GaborFilterBank2D(), gabor_2d_bank()
# for i in range(3):
#     for o, n in zip(old[i], new[i]):
#         assert np.array_equal(o, n)
