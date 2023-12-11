__author__ = "Debora Gil, Guillermo Torres, Carles Sanchez, Pau Cano"
__license__ = "GPL"
__email__ = "debora,gtorres,csanchez,pcano@cvc.uab.es"
__year__ = "2023"
__doc__ = """Source code for volume visualization

Computer Vision Center
Universitat Autonoma de Barcelona
"""

# Unit: Feature Extraction / Local Descriptors
### REFERENCES
# www.scipy-lectures.org/advanced/image_processing/#edge-detection

"""
# Renaming

Old Name | New Name
--- | ---
sig | gaussian_sigma
Medsze | median_size

"""

if __name__ != "__main__":
    raise ImportError(f"Cannot import a script file. {__file__} is not a module.")

import numpy as np
from mayavi import mlab
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.ndimage import sobel, gaussian_filter, median_filter
from skimage.filters import threshold_otsu
from sklearn.cluster import KMeans
from ct_slicing.config.dev_config import DEFAULT_PLOT_BLOCK

from ct_slicing.data_util.data_access import nii_file
from ct_slicing.vis_lib.nifty_io import read_nifty
from ct_slicing.vis_lib.volume_cut_browser import CutDirection, VolumeCutBrowser
from ct_slicing.filter_lib.gabor_filters import gabor_2d_bank
from ct_slicing.filter_lib.browse_gabor_filt_bank import VisualizeGaborFilterBank
from ct_slicing.ct_logger import logger

# Parameters
# choose the case id and nodule id to get the path of the nodule image and mask
IMAGE_PATH, MASK_PATH = nii_file("CT", 1, 1)  # also tried with 5,2 (benign)
ROI_PATH, _ = nii_file("VOI", 1, 1)
gabor_filter_index_for_3d_vis = 1


image_name = "filled_square"  # filled_square, square, SA, SA_Mask, SAROI
image_name = "SAROI"  # filled_square, square, SA, SA_Mask, SAROI
gaussian_sigma = 10  # sigma of gaussian filter
median_size = 3  # size of median filter
filter_image = "gaussian"  # none, gaussian, median
gabor_params = "default"  # default, non_default

# Constant and processed parameters
nii_roi, _ = read_nifty(ROI_PATH)
nii_mask, _ = read_nifty(MASK_PATH)
nii_vol, _ = read_nifty(IMAGE_PATH)
LESION_LABEL = "Lesion Values"


# Define Use Case Image
def get_im(image_name: str) -> np.ndarray:
    if image_name == "filled_square":
        # Synthetic Filled Square
        im = np.zeros((256, 256))
        im[64:-64, 64:-64] = 1
        Angle = 15
        im = ndi.rotate(im, Angle, mode="constant")  # 2D rotation
    elif image_name == "square":
        # Synthetic Square
        im = np.zeros((256, 256))
        im[64:-64, 64:-64] = 1
        Angle = 15
        im = ndi.rotate(im, Angle, mode="constant")  # 2D rotation
        im = ndi.gaussian_gradient_magnitude(im, sigma=2)
    elif image_name == "SA":
        k = int(nii_vol.shape[2] / 2)  # Cut at the middle of the volume
        im = nii_vol[:, :, k]
    elif image_name == "SAROI":
        k = int(nii_roi.shape[2] / 2)  # Cut at the middle of the volume
        im = nii_roi[:, :, k]
    elif image_name == "SA_Mask":
        k = int(nii_mask.shape[2] / 2)  # Cut at the middle of the volume
        im = nii_mask[:, :, k]
    else:
        raise ValueError("Incorrect image_name name.")
    return im


def get_img(
    filter_image: str, im: np.ndarray, median_size: int, gaussian_sigma
) -> np.ndarray:
    # Image Filtering
    if filter_image == "gaussian":
        img = gaussian_filter(im, sigma=gaussian_sigma)
    elif filter_image == "median":
        img = median_filter(im, median_size)
    else:
        img = im
    return img


im = get_im(image_name)
img = get_img(filter_image, im, median_size, gaussian_sigma)

# Exercise 1
for plt_idx, ex1_sig in enumerate([2, 4, 8]):
    img = gaussian_filter(im, sigma=ex1_sig)
    plt.subplot(1, 3, plt_idx + 1)
    plt.imshow(img, cmap="gray")
    plt.title(f"sigma={ex1_sig}")
plt.suptitle("Exercise 1. Gaussian filter with different sigmas")
plt.show(block=DEFAULT_PLOT_BLOCK)
plt.close()

for plt_idx, ex1_med in enumerate([3, 5, 7, 1500]):
    img = median_filter(im, median_size)
    plt.subplot(1, 4, plt_idx + 1)
    plt.imshow(img, cmap="gray")
    plt.title(f"median_size={ex1_med}")
plt.suptitle("Exercise 1. Median filter with different sizes")
plt.show(block=DEFAULT_PLOT_BLOCK)
plt.close()


# IMAGE DESCRIPTORS

# 1. GRADIENT
# 1.1 2D Images
# EX1. Compute gradient of 'filled_square', 'SAROI'
#      for different filters and analyze results

# Derivative along x-axis
sx: np.ndarray = sobel(img, axis=1, mode="constant")
# Derivative along y-axis
sy: np.ndarray = sobel(img, axis=0, mode="constant")
# Image Gradient (Sobel Edge Detector)
EdgeSob = np.sqrt(sx**2 + sy**2)

# Show Results
fig1 = plt.figure()
ax = fig1.add_subplot(141)
ax.imshow(img, cmap="gray")
ax = fig1.add_subplot(142)
ax.imshow(sx, cmap="gray")
ax.set_title("vertical edges")
ax = fig1.add_subplot(143)
ax.imshow(sy, cmap="gray")
ax.set_title("horizontal edges")
ax = fig1.add_subplot(144)
ax.imshow(EdgeSob, cmap="gray")
ax.set_title("Gradient Magnitude (EdgeDetector)")
plt.suptitle("Vertical and horizontal Sobel Edge Detector")
plt.show(block=DEFAULT_PLOT_BLOCK)
plt.close()

# 1.2 3D Volumes
# EX2. Compute the gradient of a 3D volume
# Hint: Add the derivatives in the z-axis

# 2. RIDGES, VALLEYS
# 2.1 2D Images
# EX3. Analyze results for ridges/valleys of 'square',
#      the opposite 1-'square' and 'filled_square'

# Second Derivative along x-axis
sx = sobel(img, axis=1, mode="constant")
sxx: np.ndarray = sobel(sx, axis=1, mode="constant")
# Second Derivative along y-axis
sy = sobel(img, axis=0, mode="constant")
syy: np.ndarray = sobel(sy, axis=0, mode="constant")
# Laplacian (Ridge/Valley Detector)
laplace = sxx + syy

# Show Results of Ridge/Valley Detector
fig1 = plt.figure()
ax = fig1.add_subplot(141)
ax.imshow(img, cmap="gray")
ax = fig1.add_subplot(142)
ax.imshow(sxx, cmap="gray")
ax.set_title("vertical ridges/valleys")
ax = fig1.add_subplot(143)
ax.imshow(syy, cmap="gray")
ax.set_title("horizontal ridges/valleys")
ax = fig1.add_subplot(144)
ax.imshow(laplace, cmap="gray")
ax.set_title("Laplacian")
plt.suptitle("Vertical and horizontal Laplacian (Ridge/Valley Detector)")
plt.show(block=DEFAULT_PLOT_BLOCK)
plt.close()

fig1 = plt.figure()
ax = fig1.add_subplot(131)
ax.imshow(img, cmap="gray")
ax = fig1.add_subplot(132)
ax.imshow(abs(laplace) * (laplace > 0), cmap="gray")
ax.set_title("valleys (Positive Laplacian)")
ax = fig1.add_subplot(133)
ax.imshow(abs(laplace) * (laplace < 0), cmap="gray")
ax.set_title("ridges (Negative Laplacian)")
plt.suptitle("Abs of Laplacian (Ridge/Valley Detector)")
plt.show(block=DEFAULT_PLOT_BLOCK)
plt.close()

# 3. GABOR FILTERS
# 3.1 2D Images
# EX4. Compare the response to each bank of Gabor filters
#      to the edge detector and Laplacian (ridge/valley detector)
#      of EX1 and EX3

# 3.2 Filter Bank
if gabor_params == "default":
    gabor_2d_re, gabor_2d_im, gabor_2d_params = gabor_2d_bank()
elif gabor_params == "non_default":
    gabor_2d_re, gabor_2d_im, gabor_2d_params = gabor_2d_bank(
        sigma=[2, 4], frequency=[0.25, 0.5]
    )
else:
    raise ValueError("Incorrect gabor_params name.")
# gabor_2d_re, gabor_2d_im, params has the same length, where
# _re is the real part of the filter, and _im is the imaginary part.


# 3.3 Visualize Filters
# visualize the two banks of filters in a set of 2D images
# use z and x to navigate through the filters #TODO: change this
VisualizeGaborFilterBank(
    gabor_2d_re,
    gabor_2d_params,
    title="2D Gabor Filter Bank (Real Part)",
)
VisualizeGaborFilterBank(
    gabor_2d_im,
    gabor_2d_params,
    title="2D Gabor Filter Bank (Imaginary Part)",
)

mlab.figure()
mlab.surf(gabor_2d_re[gabor_filter_index_for_3d_vis], warp_scale="auto")
mlab.title("3D Gabor Filter (Real Part)")
if not DEFAULT_PLOT_BLOCK:
    mlab.show()  # cannot make this non-block. calling it optionally.

# 3.4 Apply Filters
n_filter = len(gabor_2d_re)  # count of the filters
logger.info(f"Number of Filters: {n_filter}")

# apply the filters to the image by convolution
# TODO: not sure why split the real/imaginary part the Gabor filter
result_size = im.shape + (n_filter,)  # one more dimension for the result
gabor_2d_re_img, gabor_2d_im_img = np.empty(result_size), np.empty(result_size)
# gabor_2d_PART_img is the image after applying the filter of PART part.
for k in range(n_filter):
    gabor_2d_re_img[:, :, k] = ndi.convolve(im, gabor_2d_re[k], mode="wrap")
    gabor_2d_im_img[:, :, k] = ndi.convolve(im, gabor_2d_im[k], mode="wrap")

# show the result after applying the filters
# #TODO: not sure how to interpret the result. maybe I did something wrong?
VolumeCutBrowser(
    gabor_2d_re_img,
    cut_dir=CutDirection.Sagittal,
    title="2D Gabor Filter Bank (Real Part)",
)
VolumeCutBrowser(
    gabor_2d_im_img,
    cut_dir=CutDirection.Sagittal,
    title="2D Gabor Filter Bank (Imaginary Part)",
)


# 4. FEATURE SPACES
# EX5: Compare Otsu binarization that only takes into account
#  image intensity to using the values of intensity together with
#  some of the local descriptors: (intensity,Laplacian)

# 4.0 Data
# we are using new im, new sx,sxx,sy,syy,laplace
# SA Cut
k = int(nii_roi.shape[2] / 2)  # Cut at the middle of the volume
im: np.ndarray = nii_roi[:, :, k]
im: np.ndarray = (im - im.min()) / (im.max() - im.min())
im_mask = nii_mask[:, :, k]

# SA cut Laplacian
sx = sobel(im, axis=1, mode="constant")
sxx = sobel(sx, axis=1, mode="constant")
sy = sobel(im, axis=0, mode="constant")
syy = sobel(sy, axis=0, mode="constant")
laplace = sxx + syy

# 4.1 Intensity thresholding
otsu_threshold = threshold_otsu(im)
im_seg = im > otsu_threshold

# Show Segmentation
plt.figure()
plt.imshow(im, cmap="gray")
plt.contour(im, [otsu_threshold], colors="red")
plt.title("Otsu Thresholding")
plt.show(block=DEFAULT_PLOT_BLOCK)
plt.close()

# Show Intensity histogram and Otsu Threshold
plt.figure()
plt.hist(im.flatten(), edgecolor="k", bins=5)
plt.hist(
    im[np.nonzero(im_mask)],
    bins=5,
    edgecolor="k",
    alpha=0.5,
    facecolor="r",
    label=LESION_LABEL,
)
plt.title("Intensity Histogram")
plt.plot([otsu_threshold, otsu_threshold], [0, 4000], "k", lw=2, label="Otsu Threshold")
plt.legend()
plt.show(block=DEFAULT_PLOT_BLOCK)
plt.close()

# 4.2 Feature Space Partition
# Pixel Representation in a 2D space. In the plot
# each pixel it is assigned a x-coordinate given by its intensity
# and a y-coordinate given by its Laplacian
#              Pixel(i,j) ---> (im(i,j),Lap(i,j))
#
# EX6: Try to divide the plane with a line splitting (discriminating)
#      the red (lesion) and blue points (background)


plt.figure()
plt.plot(im.flatten(), laplace.flatten(), ".")
plt.plot(
    im[np.nonzero(im_mask)], laplace[np.nonzero(im_mask)], "r.", label=LESION_LABEL
)
plt.xlabel("Intensity")
plt.ylabel("Laplacian")
plt.title("Pixel Distribution in the Space of Values given by Intensity-Laplacian")
plt.legend()
plt.show(block=DEFAULT_PLOT_BLOCK)
plt.close()

# 4.3 Kmeans Clustering
# EX7: Run k-means with and without normalization of the feature space

laplace = (laplace - laplace.min()) / (laplace.max() - laplace.min())
X = np.array((im.flatten(), laplace.flatten()))
k_means = KMeans(n_clusters=2, n_init=10)
k_means = k_means.fit(X.transpose())
labels = k_means.predict(X.transpose())

plt.figure()
plt.plot(im.flatten(), laplace.flatten(), ".")
plt.plot(
    im[np.nonzero(im_mask)], laplace[np.nonzero(im_mask)], "r.", label=LESION_LABEL
)
plt.xlabel("Intensity")
plt.ylabel("Laplacian")
plt.title("Pixel Distribution in the Space of Values given by (Intensity,Laplacian).")
x = X[0, np.nonzero(labels == 1)]
y = X[1, np.nonzero(labels == 1)]
plt.plot(x.flatten(), y.flatten(), "k.", label="k-means Clustering")
plt.legend()
plt.show(block=DEFAULT_PLOT_BLOCK)
plt.close()

"""
# Exercise

Exercise 1. Edges. Compute the gradient of the use images labelled 'filled_square',
'SAROI' without any filtering (filter_image =‘none’) and compare the results to the
gradient obtained using a gaussian of sig=2, 4, 8 and a median filter of Medsze=3,5,7.
    The code for gaussian_filter of different sigmas is shown under the 
    commented line `# Exercise 1`. We can see that the edges are more blurred
    with higher sigmas.
    The code for median_filter of different sizes is shown under the
    commented line `# Exercise 1`. For all median sizes, the images are the 
    same. This is because the median filter is a non-linear filter.

Exercise 2. Ridges and Valleys. Compute the Laplacian, Ridges (negative
Laplacian) and Valleys (positive Laplacian) of the use images labelled 'filled_square',
its opposite (1-'filled_square') and ‘square’. Compare results across use cases.
    See the code under the commented line ``# EX2. Compute the gradient of a 3D
    volume`` The plots are with super-titles `Vertical and horizontal Laplacian
    (Ridge/Valley Detector)` and `Abs of Laplacian (Ridge/Valley Detector)`.
    
    We can see that the Laplacian of the square is the same as the Laplacian of
    the filled square.

Exercise 3. Gabor Filters. Compute the default Gabor filter bank and:
a) Visualize the two banks of filters.
    See code under `# 3.3 Visualize Filters`. The plots are with titles.

b) Apply the two filter banks to Compute the Laplacian, Ridges (negative
Laplacian) and Valleys (positive Laplacian) of the use images labelled
'filled_square', its opposite (1-'filled_square') and ‘square’.
    Not sure how to interpret the results after applying two filter banks.
    See code under `# 3.4 Apply Filters`. The plots are with titles.

c) Visualize responses to each filter for the two Gabor filter banks and compare
results to the ones obtained in Ex1 and Ex2. Would you use any of the Gabor
filters to detect edges, valleys or ridges.
    I cannot do it because I don't know how to interpret the results.

d) Repeat a)-c) using the alternative Gabor filter bank. What is the difference with
the default Gabor filter bank? Would you use any of the Gabor filters to detect
edges, valleys or ridges.
    I cannot do it because I don't know how to interpret the results.

Exercise 4. Feature Spaces. Compute Otsu thresholding for the use image
‘SAROI’ and visualize the binarization. Analyse the histogram of SAROI intensity
showing the lesion values in red and discuss if there exists a threshold (vertical line)
able to perfectly separate lesion from other structures.
    There's none threshold that can perfectly separate lesion from other 
    structures. See the code under `# 4.1 Intensity thresholding`.

Consider for each pixel the pair of values given by SAROI intensity (im) and its
Laplacian (Lap). Visualize the point cloud defined by assigning for each pixel a x-
coordinate given by its intensity and a y-coordinate given by its Laplacian. Try to divide
the plane with a line splitting (discriminating/classifying) the red (lesion) and blue points
(background). Do you think you could achieve a more accurate separation than using
only one feature (intensity)?
    No it there shouldn't be. There's no lesion in the image. 
    Even if we change some blue points to red, there's no separating line.
    See the code under `# 4.2 Feature Space Partition`.
"""
