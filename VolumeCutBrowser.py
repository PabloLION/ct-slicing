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
    IMS: np.ndarray  # IMS: Image Stack
    IMSSeg: np.ndarray | None  # IMS segmentation

    def __init__(self, IMS: np.ndarray, IMSSeg: np.ndarray | None = None, Cut="SA"):
        self.IMS = IMS
        self.idx = 0
        self.Cut = Cut
        self.IMSSeg = IMSSeg

        if self.Cut == "SA":
            self.idx = round(self.IMS.shape[2] / 2)
        elif self.Cut == "Sag":
            self.idx = round(self.IMS.shape[0] / 2)
        elif self.Cut == "Cor":
            self.idx = round(self.IMS.shape[1] / 2)
        else:
            raise ValueError("Cut must be SA, Sag or Cor")

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
            if self.Cut == "SA":
                Mx = self.IMS.shape[2] - 1
            elif self.Cut == "Sag":
                Mx = self.IMS.shape[0] - 1
            elif self.Cut == "Cor":
                Mx = self.IMS.shape[1] - 1
            else:
                raise ValueError("Cut must be SA, Sag or Cor")

            self.idx = min(Mx, self.idx)
            self.DrawScene()
        else:  # no reaction on other keys
            pass

    def DrawScene(self):
        self.ax.cla()

        if self.Cut == "SA":
            Im = self.IMS[:, :, self.idx]
        elif self.Cut == "Sag":
            Im = np.squeeze(self.IMS[self.idx, :, :])
        elif self.Cut == "Cor":
            Im = np.squeeze(self.IMS[:, self.idx, :])
        else:
            raise ValueError("Cut must be SA, Sag or Cor")

        self.ax.imshow(Im, cmap="gray")
        self.ax.set_title(
            "Cut: " + str(self.idx) + ' Press "x" to decrease; "z" to increase'
        )

        if self.IMSSeg is not None:  # Draw segmentation contour
            if self.Cut == "SA":
                Im = self.IMSSeg[:, :, self.idx]
            elif self.Cut == "Sag":
                Im = np.squeeze(self.IMSSeg[self.idx, :, :])
            elif self.Cut == "Cor":
                Im = np.squeeze(self.IMSSeg[:, self.idx, :])
            self.ax.contour(Im, [0.5], colors="r")

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
