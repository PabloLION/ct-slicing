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

from ct_slicing.config.dev_config import DEFAULT_PLOT_BLOCK


# #TODO: ref: move to vis_lib
# #LONG: I don't feel the "show on init" is a good idea. Maybe we should split
# #LONG+ the show from the init, and change the init to only create an instance
class VisualizeGaborFilterBank:
    n_filter: int

    def __init__(
        self,
        gabor_2d_bank,
        params,
        *,
        plot_block: bool = DEFAULT_PLOT_BLOCK,
        title: str = "Visualize Gabor Filter Bank",
    ):
        self.gabor_2d_bank = gabor_2d_bank
        self.n_filter = (
            len(gabor_2d_bank)
            if isinstance(gabor_2d_bank, list)
            else np.shape(gabor_2d_bank)[0]
        )
        self.params = params
        self.idx = 0
        self.fig, self.ax = plt.subplots(1, 1)
        self.fig.canvas.mpl_connect("key_press_event", self.press)
        self.draw_scene()
        plt.suptitle(title)
        plt.show(block=plot_block)
        plt.close()

    def press(self, event):
        sys.stdout.flush()
        if event.key == "x":
            self.idx -= 1
            self.idx = max(0, self.idx)
            self.draw_scene()
        if event.key == "z":
            self.idx += 1
            self.idx = min(self.n_filter - 1, self.idx)
            self.draw_scene()

    def draw_scene(self):
        theta0, sigma0, frequency0 = self.params[self.idx]
        theta0 = theta0 * 180 / np.pi
        GaborFilt = self.gabor_2d_bank[self.idx]
        self.ax.imshow(GaborFilt, cmap="gray")
        self.ax.set_title(
            f"FilterNumber: {self.idx} Theta: {theta0} Sig: {sigma0} Freq: {frequency0}"
        )
        self.fig.canvas.draw()
