__author__ = "Debora Gil, Guillermo Torres, Carles Sanchez, Pau Cano"
__license__ = "GPL"
__email__ = "debora,gtorres,csanchez,pcano@cvc.uab.es"
__year__ = "2023"
__doc__ = """Source code for volume visualization

Computer Vision Center
Universitat Autonoma de Barcelona
"""

import sys
from enum import Enum
from typing import Callable, cast

import matplotlib.axes as axes
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import Event, KeyEvent


class CutDirection(Enum):
    """
    View direction of the volume. Should be one of: short axis, sagittal, coronal.
    See more at https://en.wikipedia.org/wiki/Sagittal_plane
    """

    ShortAxis = "SA"  # cut in the short axis
    Sagittal = "Sag"  # cut in the sagittal axis
    Coronal = "Cor"  # cut in the coronal axis

    @staticmethod
    def to_idx(cut_dir: "CutDirection") -> int:
        """
        Returns the index of the volume in the given direction.
        """
        if cut_dir == CutDirection.Sagittal:
            return 0
        elif cut_dir == CutDirection.Coronal:
            return 1
        elif cut_dir == CutDirection.ShortAxis:
            return 2
        else:
            raise VolumeCutDirectionError(cut_dir)

    @staticmethod
    def get_median_index(volume: np.ndarray, direction: "CutDirection") -> int:
        """Returns the median index of the volume in the given direction."""
        return volume.shape[CutDirection.to_idx(direction)] // 2

    @staticmethod
    def get_max_index(volume: np.ndarray, direction: "CutDirection") -> int:
        """Returns the maximum index of the volume in the given direction."""
        return volume.shape[CutDirection.to_idx(direction)] - 1

    @staticmethod
    def get_cut_img(
        volume: np.ndarray, direction: "CutDirection", index: int
    ) -> np.ndarray:
        """
        Returns the cut of the volume in the given direction at the given index.
        """
        return volume.take(index, axis=CutDirection.to_idx(direction))
        # volume.take(index, axis=i) returns the index-th slice of the volume in the first dimension
        # e.g. volume.take(0, axis=0) returns np.squeeze(volume[:, :, index])


class VolumeCutDirectionError(ValueError):
    def __init__(self, cut: CutDirection):
        self.cut = cut

    def __str__(self):
        return f"Coordinate order {self.cut} not supported. Supported orders are: short axis, sagittal, coronal"


class VolumeCutBrowser:
    """
    Visualization of a volume in a given direction.

    # EXAMPLE:
    # DataDir='C://Data_Session1//Case0016';
    # NIIFile='LIDC-IDRI-0016_GT1.nii.gz'
    # nii_vol,_=NiftyIO.readNifty(os.path.join(ServerDir,NIIFile))
    # VolumeCutBrowser(nii_vol)
    """

    idx: int
    ax: axes.Axes
    img_stack: np.ndarray  # Image Stack
    contour_stack: np.ndarray | None  # Segmentation Stack

    def __init__(
        self,
        img_stack: np.ndarray,
        cut_dir: CutDirection,
        contour_stack: np.ndarray | None = None,
    ):
        self.img_stack = img_stack
        self.cut = cut_dir
        self.contour_stack = contour_stack
        self.idx = CutDirection.get_median_index(img_stack, cut_dir)

        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_connect(
            "key_press_event", cast(Callable[[Event], None], self.press)
        )
        self.draw_scene()

    def press(self, event: KeyEvent):
        sys.stdout.flush()
        if event.key == "x":
            self.idx -= 1
            self.idx = max(0, self.idx)
            self.draw_scene()
        elif event.key == "z":
            self.idx += 1
            max_idx = CutDirection.get_max_index(self.img_stack, self.cut)
            self.idx = min(max_idx, self.idx)
            self.draw_scene()
        else:  # no reaction on other keys
            pass

    def draw_scene(self):
        self.ax.cla()
        image = CutDirection.get_cut_img(self.img_stack, self.cut, self.idx)
        self.ax.imshow(image, cmap="gray")
        self.ax.set_title(f"cut: {self.idx}. Press 'x' to decrease; 'z' to increase")

        if self.contour_stack is not None:  # Draw segmentation contour
            image = CutDirection.get_cut_img(self.contour_stack, self.cut, self.idx)
            self.ax.contour(image, [0.5], colors="r")

        self.fig.canvas.draw()
        plt.show()


def show_mosaic(images: np.ndarray, NRow=4, NCol=4) -> None:
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
