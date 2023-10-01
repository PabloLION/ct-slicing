"""
This is the source code for volume visualization

Computer Vision Center
Universitat Autonoma de Barcelona

__author__ = "Debora Gil, Guillermo Torres, Carles Sanchez, Pau Cano"
__license__ = "GPL"
__email__ = "debora,gtorres,csanchez,pcano@cvc.uab.es"
__year__ = "2023"
"""

import sys
from enum import Enum
from typing import Callable, cast

import matplotlib.axes as axes
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import Event, KeyEvent


class VolumeCutDirection(Enum):
    """
    Order of dimensions in array.
    """

    ShortAxis = "SA"  # cut in the short axis
    Sagittal = "Sag"  # cut in the sagittal axis
    Coronal = "Cor"  # cut in the coronal axis

    @staticmethod
    def get_median_index(volume: np.ndarray, direction: "VolumeCutDirection") -> int:
        """
        Returns the median index of the volume in the given direction.
        """
        if direction == VolumeCutDirection.ShortAxis:
            return round(volume.shape[2] / 2)
        elif direction == VolumeCutDirection.Sagittal:
            return round(volume.shape[0] / 2)
        elif direction == VolumeCutDirection.Coronal:
            return round(volume.shape[1] / 2)
        else:
            raise VolumeCutDirectionError(direction)

    @staticmethod
    def get_max_index(volume: np.ndarray, direction: "VolumeCutDirection") -> int:
        """
        Returns the maximum index of the volume in the given direction.
        """
        if direction == VolumeCutDirection.ShortAxis:
            return volume.shape[2]
        elif direction == VolumeCutDirection.Sagittal:
            return volume.shape[0]
        elif direction == VolumeCutDirection.Coronal:
            return volume.shape[1]
        else:
            raise VolumeCutDirectionError(direction)

    @staticmethod
    def get_cut_img(
        volume: np.ndarray, direction: "VolumeCutDirection", index: int
    ) -> np.ndarray:
        """
        Returns the cut of the volume in the given direction at the given index.
        """
        if direction == VolumeCutDirection.ShortAxis:
            return volume[:, :, index]
        elif direction == VolumeCutDirection.Sagittal:
            return np.squeeze(volume[index, :, :])
        elif direction == VolumeCutDirection.Coronal:
            return np.squeeze(volume[:, index, :])
        else:
            raise VolumeCutDirectionError(direction)


class VolumeCutDirectionError(ValueError):
    def __init__(self, cut: VolumeCutDirection):
        self.cut = cut

    def __str__(self):
        return f"Coordinate order {self.cut} not supported. Supported orders are: short axis, sagittal, coronal"


################################################################################
#
# EXAMPLE:
# ServerDir='Y:\Shared\Guille'; NIIFile='LIDC-IDRI-0305_GT1_1.nii.gz'
# nii_vol,_=NiftyIO.readNifty(os.path.join(ServerDir,NIIFile))
#
# VolumeCutBrowser(nii_vol)


class VolumeCutBrowser:
    """
    # EXAMPLE:
    # DataDir='C://Data_Session1//Case0016';
    # NIIFile='LIDC-IDRI-0016_GT1.nii.gz'
    # niivol,_=NiftyIO.readNifty(os.path.join(ServerDir,NIIFile))
    # VolumeCutBrowser(nii_vol)
    """

    idx: int
    ax: axes.Axes
    img_stack: np.ndarray  # Image Stack
    contour_stack: np.ndarray | None  # Segmentation Stack

    def __init__(
        self,
        img_stack: np.ndarray,
        contour_stack: np.ndarray | None = None,
        cut_dir: VolumeCutDirection = VolumeCutDirection.ShortAxis,
    ):
        self.img_stack = img_stack
        self.cut = cut_dir
        self.contour_stack = contour_stack
        self.idx = VolumeCutDirection.get_median_index(img_stack, cut_dir)

        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_connect(
            "key_press_event", cast(Callable[[Event], None], self.press)
        )
        self.DrawScene()

    def press(self, event: KeyEvent):
        sys.stdout.flush()
        if event.key == "x":
            self.idx -= 1
            self.idx = max(0, self.idx)
            self.DrawScene()
        elif event.key == "z":
            self.idx += 1
            max_idx = VolumeCutDirection.get_max_index(self.img_stack, self.cut)
            self.idx = min(max_idx, self.idx)
            self.DrawScene()
        else:  # no reaction on other keys
            pass

    def DrawScene(self):
        self.ax.cla()

        image = VolumeCutDirection.get_cut_img(self.img_stack, self.cut, self.idx)

        self.ax.imshow(image, cmap="gray")
        self.ax.set_title(f"cut: {self.idx}. Press 'x' to decrease; 'z' to increase")

        if self.contour_stack is not None:  # Draw segmentation contour
            image = VolumeCutDirection.get_cut_img(
                self.contour_stack, self.cut, self.idx
            )
            self.ax.contour(image, [0.5], colors="r")

        self.fig.canvas.draw()
        plt.show()


#########################################################
def ShowMosaic(images: np.ndarray, NRow=4, NCol=4) -> None:
    # #TODO: check n row and n col default values
    fig = plt.figure(figsize=(14, 14))
    NIm = images.shape[2]

    for cnt in np.arange(NIm):
        y = fig.add_subplot(NRow, NCol, cnt + 1)
        img = images[:, :, cnt]
        y.imshow(img, cmap="gray")

        x = y.axes  # Axis object
        assert x

        x.get_xaxis().set_visible(False)
        x.get_yaxis().set_visible(False)
