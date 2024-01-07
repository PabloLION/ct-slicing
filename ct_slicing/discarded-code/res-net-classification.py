if __name__ != "__main__":
    raise ImportError(f"Script {__file__} should not be imported as a module")

import logging
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from ct_slicing.config.data_path import SLICE_IMAGE_FOLDER

from ct_slicing.ct_logger import logger

logger.setLevel(logging.INFO)

model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# try also:
# criterion = nn.NLLLoss()
# criterion = nn.BCELoss()
# criterion = nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


## prepare the data
from torch.utils.data import DataLoader

train_dataset = datasets.ImageFolder(
    root=str(SLICE_IMAGE_FOLDER), transform=transforms.ToTensor()
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# setup for training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_epochs = 10  # number of training epochs
# set the output layer to 2 classes: benign and malign, keeping in_features same
model.fc = nn.Linear(model.fc.in_features, 2)
model.to(device)

logger.info("Start training")
for epoch in range(n_epochs):
    for batch, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)  # Move data, label to `device`
        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(data)  # Forward pass
        loss = criterion(outputs, label)
        loss.backward()  # Backward pass
        optimizer.step()  # and then optimize

        logger.info(f"Epoch: {epoch}, Batch: {batch:03}, Loss: {loss.item()}")
    logger.info(f"Epoch: {epoch} finished")

torch.save(model.state_dict(), "output/trained-model/resnet152_model.pth")


# X_train, X_test, y_train, y_test = split_data_from_features_and_diagnoses()
# classification_report(
#     y_train,
#     y_pred,
#     labels=[0, 1],
#     target_names=["benign", "malign"],
#     digits=3,
# )
