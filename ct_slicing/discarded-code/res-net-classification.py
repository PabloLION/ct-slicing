if __name__ != "__main__":
    raise ImportError(f"Script {__file__} should not be imported as a module")

import logging
import pandas

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report

from ct_slicing.config.data_path import (
    DATA_SPLIT_INDICES_PATH,
    MODEL_OPTIMIZER_PATH,
    OUTPUT_FOLDER,
    SLICE_IMAGE_FOLDER,
)
from ct_slicing.ct_logger import logger
from ct_slicing.data_util.slice_view import is_first_or_last_k_image


# Parameters
criterion = nn.CrossEntropyLoss()
n_epoch = 0  # number of training epochs
# note on n_epoch: on my mac M1U, 100% data * 10 epochs took 5h23m;
# so 70% data * 16 epochs should take about 6h2m
run_test = True  # whether to run test after training
# Parameters END

# #TODO: not implemented. Now we only use the default parameters
# model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

n_epoch_trained = 0  # number of epochs the model has been trained


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

        global n_epoch_trained
        n_epoch_trained = checkpoint.get("n_epoch_trained", 0)
    return model, optimizer


def split_data_or_restore_split(training_split_ratio=0.7):
    """
    Split the data into train and test set, or restore the split if it exists.

    Args:
        training_split_ratio: the ratio of training set to the whole dataset

    Returns:
        train_dataset, test_dataset

    #TODO:
        - store multiple files for different train_test_split_ratio
        - option to force create a new split
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


def train_model(
    train_dataset: Dataset,
    n_epoch: int,
    n_epoch_trained: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module = criterion,
):
    """
    Train the model on the training set.

    Args:
        train_dataset: the training set
        n_epoch: number of epochs to train
        n_epoch_trained: number of epochs the model has been trained
        model: the model to train
        optimizer: the optimizer to use
        criterion: the loss function to use

    Returns:
        None

    #TODO:
        - add validation set
        - add early stopping
        - option to not save the model
    """
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Training
    logger.info("Start training")
    for epoch in range(n_epoch_trained, n_epoch_trained + n_epoch):
        model.train()
        for batch, (data, label) in enumerate(train_loader):
            data, label = data.to(device), label.to(
                device
            )  # Move data, label to `device`
            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = model(data)  # Forward pass
            loss = criterion(outputs, label)
            loss.backward()  # Backward pass
            optimizer.step()  # and then optimize

            logger.info(f"Epoch: {epoch}, Batch: {batch:03}, Loss: {loss.item()}")
        logger.info(f"Epoch: {epoch} finished")
    logger.info("Finished training")

    # Saving the model and optimizer, maybe should use try except KeyboardInterrupt?
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "n_epoch_trained": n_epoch_trained + n_epoch,
        },
        str(MODEL_OPTIMIZER_PATH),
    )
    logger.warning(f"Model and optimizer saved to {MODEL_OPTIMIZER_PATH}")


def test_model(
    test_dataset: Dataset,
    model: nn.Module,
) -> tuple[list[int], list[int], list[str]]:
    logger.info("Started testing")

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    model.eval()  # Set the model to evaluation mode

    all_predictions = []
    all_labels = []
    image_names = []
    batch_size = test_loader.batch_size or 0  # should be 32 as defined above

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            logger.info(f"Batch: {batch_idx:03}")
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Identify and store paths of wrong predictions
            for idx in range(len(labels)):
                absolute_idx = batch_idx * batch_size + idx
                original_idx = test_loader.dataset.indices[absolute_idx]  # type: ignore
                img_path = test_loader.dataset.dataset.samples[original_idx][0]  # type: ignore
                *_dir, img_name = img_path.split("/")
                image_names.append(img_name)

    logger.info("Finished testing")
    return all_predictions, all_labels, image_names


# Prepare the model and optimizer
logger.setLevel(logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, optimizer = load_or_create_model_and_optimizer()
model.to(device)

## prepare the data
train_dataset, test_dataset = split_data_or_restore_split()

# Train the model
if n_epoch > 0:
    train_model(train_dataset, n_epoch, n_epoch_trained, model, optimizer)

# Or test the model
if run_test:
    predictions, labels, image_names = test_model(test_dataset, model)

    # save the predictions, labels, image_names to a excel file
    test_result = list(zip(predictions, labels, image_names, strict=True))
    test_result_data_frame = pandas.DataFrame(  # type: ignore
        [predictions, labels, image_names],
        columns=["predictions", "labels", " image_names"],
    )
    test_result_data_frame.to_excel(OUTPUT_FOLDER / "test_result.xlsx")

    # Generate classification report
    logger.info("Raw classification report without filtering")
    report0 = classification_report(
        labels, predictions, target_names=["benign", "malign"], digits=4
    )
    logger.info(report0)

    # the image_names shows that most of the wrong predictions are due to the
    # quality of the input, because they are mostly the first or last few
    # slices of a nodule that contains almost no voxels of the nodule.

    # we should try to do the classification report for only the middle slices.
    # And even only train with the middle slices.

    filtered_predictions_1, filtered_labels_1, filtered_image_names_1 = zip(
        *filter(
            lambda x: not is_first_or_last_k_image(x[2], 1),
            zip(predictions, labels, image_names),
        )
    )
    logger.info("Classification report with filtering out first and last 1 slice")
    report1 = classification_report(
        filtered_labels_1,
        filtered_predictions_1,
        target_names=["benign", "malign"],
        digits=4,
    )
    logger.info(report1)

    filtered_predictions_2, filtered_labels_2, filtered_image_names_2 = zip(
        *filter(
            lambda x: not is_first_or_last_k_image(x[2], 2),
            zip(predictions, labels, image_names),
        )
    )
    logger.info("Classification report with filtering out first and last 2 slices")
    report2 = classification_report(
        filtered_labels_2,
        filtered_predictions_2,
        target_names=["benign", "malign"],
        digits=4,
    )
    logger.info(report2)

    filtered_predictions_3, filtered_labels_3, filtered_image_names_3 = zip(
        *filter(
            lambda x: not is_first_or_last_k_image(x[2], 3),
            zip(predictions, labels, image_names),
        )
    )
    logger.info("Classification report with filtering out first and last 3 slices")
    report3 = classification_report(
        filtered_labels_3,
        filtered_predictions_3,
        target_names=["benign", "malign"],
        digits=4,
    )
    logger.info(report3)
