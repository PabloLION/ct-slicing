if __name__ != "__main__":
    raise ImportError(f"Script {__file__} should not be imported as a module")

import logging
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from ct_slicing.config.data_path import (
    MODEL_OPTIMIZER_PATH,
    SLICE_IMAGE_FOLDER,
)

from ct_slicing.ct_logger import logger


# Parameters
criterion = nn.CrossEntropyLoss()
n_epoch = 10  # number of training epochs
# #TODO: not implemented. Now we only use the default parameters
# model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Prepare the model and optimizer
def load_or_create_model_and_optimizer():
    """
    Load the model and optimizer from file if they exist, otherwise create them.
    For training the model faster, we should use the same optimizer as before.

    Returns:
        model, optimizer
    """
    # try also:
    # criterion = nn.NLLLoss()
    # criterion = nn.BCELoss()
    # criterion = nn.MSELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    if MODEL_OPTIMIZER_PATH.exists():
        logger.info("Model and optimizer found, loading...")
        checkpoint = torch.load(str(MODEL_OPTIMIZER_PATH))
    else:
        logger.info("Model and optimizer not found, creating...")
        checkpoint = None

    # Initialize the model structure
    model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
    # set the output layer to 2 classes: benign and malign, keeping in_features same
    model.fc = nn.Linear(model.fc.in_features, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if checkpoint:
        logger.info("Using checkpoint for model and optimizer")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model, optimizer


logger.setLevel(logging.INFO)
model, optimizer = load_or_create_model_and_optimizer()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

## prepare the data

train_dataset = datasets.ImageFolder(
    root=str(SLICE_IMAGE_FOLDER), transform=transforms.ToTensor()
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# Training
logger.info("Start training")
for epoch in range(n_epoch):
    model.train()
    for batch, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)  # Move data, label to `device`
        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(data)  # Forward pass
        loss = criterion(outputs, label)
        loss.backward()  # Backward pass
        optimizer.step()  # and then optimize

        logger.info(f"Epoch: {epoch}, Batch: {batch:03}, Loss: {loss.item()}")
    logger.info(f"Epoch: {epoch} finished")

# Saving the model and optimizer, maybe should use try except KeyboardInterrupt?
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    },
    str(MODEL_OPTIMIZER_PATH),
)
