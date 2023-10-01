"""
This is the source code for volume visualization

Computer Vision Center
Universitat Autonoma de Barcelona

__author__ = "Debora Gil, Guillermo Torres, Carles Sanchez, Pau Cano"
__license__ = "GPL"
__email__ = "debora,gtorres,csanchez,pcano@cvc.uab.es"
__year__ = "2023"
"""

from typing import Callable, cast
from matplotlib.backend_bases import Event, KeyEvent
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as axes

import sys


################################################################################
#
# EXAMPLE:
# # Load NII Volume
# from BasicVisualization.DICOMViewer import VolumeSlicer
# import BasicIO.NiftyIO
# import os
# ServerDir='Y:\Shared\Guille'; NIIFile='LIDC-IDRI-0305_GT1_1.nii.gz'
# niivol,_=NiftyIO.readNifty(os.path.join(ServerDir,NIIFile))
#
# VolumeCutBrowser(niivol)


class VolumeCutBrowser:
    """

    # EXAMPLE:
    # # Load NII Volume
    # from BasicVisualization.DICOMViewer import VolumeSlicer
    # import BasicIO.NiftyIO
    # import os
    # DataDir='C://Data_Session1//Case0016';
    # NIIFile='LIDC-IDRI-0016_GT1.nii.gz'
    # niivol,_=NiftyIO.readNifty(os.path.join(ServerDir,NIIFile))
    #
    # VolumeCutBrowser(niivol)
    """

    idx: int
    ax: axes.Axes
    img_stack: np.ndarray  # Image Stack
    segmentation_stack: np.ndarray | None  # Segmentation Stack

    def __init__(
        self, img_stack: np.ndarray, IMSSeg: np.ndarray | None = None, cut="SA"
    ):
        self.img_stack = img_stack
        self.idx = 0
        self.cut = cut
        self.segmentation_stack = IMSSeg

        if self.cut == "SA":
            self.idx = round(self.img_stack.shape[2] / 2)
        elif self.cut == "Sag":
            self.idx = round(self.img_stack.shape[0] / 2)
        elif self.cut == "Cor":
            self.idx = round(self.img_stack.shape[1] / 2)
        else:
            raise ValueError("cut must be SA, Sag or Cor")

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
            if self.cut == "SA":
                max_idx = self.img_stack.shape[2] - 1
            elif self.cut == "Sag":
                max_idx = self.img_stack.shape[0] - 1
            elif self.cut == "Cor":
                max_idx = self.img_stack.shape[1] - 1
            else:
                raise ValueError("cut must be SA, Sag or Cor")

            self.idx = min(max_idx, self.idx)
            self.DrawScene()
        else:  # no reaction on other keys
            pass

    def DrawScene(self):
        self.ax.cla()

        if self.cut == "SA":
            image = self.img_stack[:, :, self.idx]
        elif self.cut == "Sag":
            image = np.squeeze(self.img_stack[self.idx, :, :])
        elif self.cut == "Cor":
            image = np.squeeze(self.img_stack[:, self.idx, :])
        else:
            raise ValueError("cut must be SA, Sag or Cor")

        self.ax.imshow(image, cmap="gray")
        self.ax.set_title(
            "cut: " + str(self.idx) + ' Press "x" to decrease; "z" to increase'
        )

        if self.segmentation_stack is not None:  # Draw segmentation contour
            if self.cut == "SA":
                image = self.segmentation_stack[:, :, self.idx]
            elif self.cut == "Sag":
                image = np.squeeze(self.segmentation_stack[self.idx, :, :])
            elif self.cut == "Cor":
                image = np.squeeze(self.segmentation_stack[:, self.idx, :])
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
