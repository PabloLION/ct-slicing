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
import numpy as np

# 2D GaborKernels
from skimage.filters import gabor_kernel


def GaborFilterBank2D(theta=None, sigma=None, frequency=None):
    # Default values
    if theta is None:
        theta = np.arange(0, 4, 1)
        theta = theta / 4.0 * np.pi

    if sigma is None:
        sigma = [1, 3]

    if frequency is None:
        frequency = [0.05, 0.25]

    kernelsReal = []
    kernelsImag = []
    Params = []
    for theta0 in theta:
        for sigma0 in sigma:
            for frequency0 in frequency:
                kernel = gabor_kernel(
                    frequency0, theta=theta0, sigma_x=sigma0, sigma_y=sigma0
                )
                kernelsReal.append(np.real(kernel))
                kernelsImag.append(np.imag(kernel))
                Params.append([theta0, sigma0, frequency0])

    return kernelsReal, kernelsImag, Params


###################################################################


def GaborFilterBank3D(theta=None, sigma=None, frequency=None):
    # Default values
    if theta is None:
        theta = np.empty([2, 4])
        theta[0, :] = np.arange(0, 4, 1)
        theta[0, :] = theta[0, :] / 4.0 * np.pi
        theta[1, :] = theta[0, :]

    if sigma is None:
        sigma = [1, 3]

    if frequency is None:
        frequency = [0.05, 0.25]

    kernels = []

    Params = []
    for theta0 in theta[0, :]:
        for theta1 in theta[1, :]:
            for sigma0 in sigma:
                for frequency0 in frequency:
                    kernel = gabor_fn3D(
                        [sigma0, sigma0, sigma0], [theta0, theta1], frequency0
                    )
                    kernels.append(kernel)
                    Params.append([theta0, theta1, sigma0, frequency0])

    return kernels, Params


def gabor_fn3D(sigma, theta, frequency):
    sigma_x = sigma[0]
    sigma_y = sigma[1]
    sigma_z = sigma[2]
    theta1 = theta[0]
    theta2 = theta[1]
    # Bounding box
    xmin, xmax, ymin, ymax, zmin, zmax = gabor_support(sigma, theta)
    (x, y, z) = np.meshgrid(
        np.arange(xmin, xmax + 1), np.arange(ymin, ymax + 1), np.arange(zmin, zmax + 1)
    )

    # Rotation
    x_theta = (
        x * np.cos(theta1) * np.cos(theta2)
        + y * np.sin(theta1) * np.cos(theta2)
        + z * np.sin(theta2)
    )
    y_theta = -x * np.sin(theta1) + y * np.cos(theta1)
    z_theta = (
        -x * np.cos(theta1) * np.sin(theta2)
        - y * np.sin(theta1) * np.sin(theta2)
        + z * np.cos(theta2)
    )

    gb_exp = np.exp(
        -0.5
        * (
            x_theta**2 / sigma_x**2
            + y_theta**2 / sigma_y**2
            + z_theta**2 / sigma_z**2
        )
    )
    gb = gb_exp * np.cos(2 * np.pi * frequency * x_theta)

    return gb


###########
def gabor_support(sigma, theta):
    sigma_x = sigma[0]
    sigma_y = sigma[1]
    sigma_z = sigma[2]
    theta1 = theta[0]
    theta2 = theta[1]

    nstds = 3  # Number of standard deviation sigma
    xmax = max(
        abs(nstds * sigma_x * np.cos(theta1) * np.cos(theta2)),
        abs(nstds * sigma_y * np.sin(theta1)),
    )
    xmax = max(xmax, abs(nstds * sigma_z * np.cos(theta1) * np.sin(theta2)))
    xmax = np.ceil(max(1, xmax))

    ymax = max(
        abs(nstds * sigma_x * np.sin(theta1) * np.cos(theta2)),
        abs(nstds * sigma_y * np.cos(theta1)),
    )
    ymax = max(ymax, abs(nstds * sigma_z * np.sin(theta1) * np.sin(theta2)))
    ymax = np.ceil(max(1, ymax))

    zmax = max(
        abs(nstds * sigma_x * np.sin(theta2)), abs(nstds * sigma_z * np.cos(theta2))
    )
    zmax = np.ceil(max(1, zmax))

    xmin = -xmax
    ymin = -ymax
    zmin = -zmax

    return xmin, xmax, ymin, ymax, zmin, zmax
