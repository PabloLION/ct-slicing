"""
U-Net model cannot be trained on the whole dataset of different sizes. Discarded.
To run this model, package `segmentation-models-pytorch` is required.
"""
if __name__ != "__main__":
    raise ImportError(f"Script {__file__} should not be imported as a module")

import torch
import torch.nn as nn
import segmentation_models_pytorch

from ct_slicing.ct_logger import logger
from ct_slicing.data_util.nii_file_access import load_nodule_id_pickle, nii_file
from ct_slicing.vis_lib.nifty_io import CoordinateOrder, read_nifty

num_epochs = 10

# Initialize U-Net model
unet_model = segmentation_models_pytorch.Unet(
    encoder_name="resnet34",  # choose encoder, e.g., resnet34, mobilenet_v2
    encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
    in_channels=1,  # model input channels (1 for grayscale images, 3 for RGB, etc.)
    classes=1,  # model output channels (number of classes in your dataset)
)

from torch.utils.data import Dataset, DataLoader, IterableDataset


# Define your dataset class
class MyDataset(IterableDataset):
    _voi_case_id_nodule_id: set[tuple[int, int]]

    def __init__(self):
        _ct_nodules, _voi_nodules = load_nodule_id_pickle()
        self._voi_case_id_nodule_id = set(_voi_nodules)  # copy for safety

    def __getitem__(self, idx):
        voi_path, _mask_path = nii_file("VOI", *idx)
        voi_image, _voi_meta = read_nifty(voi_path, CoordinateOrder.zyx)
        mask_image, _mask_meta = read_nifty(_mask_path, CoordinateOrder.zyx)
        print(voi_image.shape, mask_image.shape)
        return voi_image, mask_image

    def __iter__(self):
        yield from (
            self[case_id, nodule_id]
            for (case_id, nodule_id) in self._voi_case_id_nodule_id
        )

    def __len__(self):
        return len(self._voi_case_id_nodule_id)


# Create the dataset and dataloader
dataset = MyDataset()
dataloader = DataLoader(dataset, batch_size=32)

optimizer = torch.optim.Adam(unet_model.parameters(), lr=1e-4)
criterion = nn.BCELoss()  # Binary Cross Entropy Loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device {device}")
unet_model.to(device)

for epoch in range(num_epochs):
    unet_model.train()
    for batch in dataloader:
        images, masks = batch
        images, masks = images.to(device), masks.to(device)
        print(images.shape, masks.shape)

        optimizer.zero_grad()
        outputs = unet_model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        # Additional code to track loss, accuracy, etc.

# Switch model to evaluation mode
unet_model.eval()
# Use the model for prediction or evaluation
# ...

# Save
torch.save(unet_model.state_dict(), "unet_model.pth")

# Load
unet_model.load_state_dict(torch.load("unet_model.pth"))
unet_model.to(device)
