__author__ = "Debora Gil"
__license__ = "GPL"
__email__ = "debora@cvc.uab.es"
__year__ = "2023"
__doc__ = """Source code for volume visualization

Computer Vision Center
Universitat Autonoma de Barcelona
"""

# Unit: Feature Extraction / Local Descriptors


import numpy as np
import matplotlib.pyplot as plt
import sys


class BrowseGaborFiltBank:
    def __init__(self, GaborBank2D, params):
        self.GaborBank2D = GaborBank2D
        self.NFilt = np.shape(self.GaborBank2D)
        self.NFilt = self.NFilt[0]
        self.params = params
        self.idx = 0
        self.fig, self.ax1 = plt.subplots(1, 1)
        self.fig.canvas.mpl_connect("key_press_event", self.press)
        self.DrawScene()

    def press(self, event):
        sys.stdout.flush()
        if event.key == "x":
            self.idx -= 1
            self.idx = max(0, self.idx)
            self.DrawScene()
        if event.key == "z":
            self.idx += 1
            self.idx = min(self.NFilt - 1, self.idx)
            self.DrawScene()

    def DrawScene(self):
        theta0, sigma0, frequency0 = self.params[self.idx]
        theta0 = theta0 * 180 / np.pi
        GaborFilt = self.GaborBank2D[self.idx]
        self.ax1.imshow(GaborFilt, cmap="gray")
        self.ax1.set_title(
            "FilterNumber: "
            + str(self.idx)
            + " Theta: "
            + str(theta0)
            + " Sig: "
            + str(sigma0)
            + " Freq: "
            + str(frequency0)
        )
        self.fig.canvas.draw()
