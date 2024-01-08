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
    DATA_SPLIT_INDICES_PATH,
    MODEL_OPTIMIZER_PATH,
    SLICE_IMAGE_FOLDER,
)

from ct_slicing.ct_logger import logger


# Parameters
criterion = nn.CrossEntropyLoss()
n_epoch = 1  # number of training epochs
# note on n_epoch: on my mac M1U, 100% data * 10 epochs took 5h23m;
# so 70% data * 16 epochs should take about 6h2m

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
        checkpoint = torch.load(str(MODEL_OPTIMIZER_PATH))
        logger.warning("Model and optimizer found, checkpoint loaded")
    else:
        logger.warning("Model and optimizer not found, creating new model...")
        checkpoint = None

    # Initialize the model structure
    model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
    # set the output layer to 2 classes: benign and malign, keeping in_features same
    model.fc = nn.Linear(model.fc.in_features, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if checkpoint:
        logger.warning("Using checkpoint for model and optimizer")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model, optimizer


logger.setLevel(logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, optimizer = load_or_create_model_and_optimizer()
model.to(device)


## prepare the data
def split_data_or_restore_split(training_split_ratio=0.7):
    """
    Split the data into train and test set, or restore the split if it exists.
    """

    from torch.utils.data import random_split
    import pickle

    whole_dataset = datasets.ImageFolder(
        root=str(SLICE_IMAGE_FOLDER), transform=transforms.ToTensor()
    )

    if DATA_SPLIT_INDICES_PATH.exists():
        from torch.utils.data import Subset

        with open(DATA_SPLIT_INDICES_PATH, "rb") as f:
            indices = pickle.load(f)
        train_indices, test_indices = indices["train_indices"], indices["test_indices"]
        logger.warning("Loaded data split indices from file.")
        return Subset(whole_dataset, train_indices), Subset(whole_dataset, test_indices)

    # If the split does not exist, create it
    total_size = len(whole_dataset)
    train_size = int(training_split_ratio * total_size)
    test_size = total_size - train_size
    train_dataset, test_dataset = random_split(whole_dataset, [train_size, test_size])

    # Save the split
    train_indices, test_indices = train_dataset.indices, test_dataset.indices
    with open(DATA_SPLIT_INDICES_PATH, "wb") as f:
        pickle.dump({"train_indices": train_indices, "test_indices": test_indices}, f)
    logger.warning("New data split indices is created and saved.")

    return train_dataset, test_dataset


train_dataset, _test_dataset = split_data_or_restore_split()
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
