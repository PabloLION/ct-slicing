from torchvision import models, transforms, datasets

import torch
import torch.nn as nn

from sklearn.metrics import classification_report

from ct_slicing.config.data_path import MODEL_PATH, SLICE_IMAGE_FOLDER


model = models.resnet152()  # Initialize the model structure
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_ftrs = model.fc.in_features
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH))  # this is new
model.to(device)


test_dataset = datasets.ImageFolder(
    root=str(SLICE_IMAGE_FOLDER), transform=transforms.ToTensor()
)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

model.eval()  # Set the model to evaluation mode
all_preds = []
all_labels = []
wrong_preds_paths = []

with torch.no_grad():
    for batch_idx, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Identify and store paths of wrong predictions
        wrong_indices = (preds != labels).nonzero(as_tuple=False).view(-1)
        for index in wrong_indices:
            img_path = test_loader.dataset.samples[index][0]
            wrong_preds_paths.append(img_path)

        for idx, (pred, label) in enumerate(zip(preds, labels)):
            absolute_idx = batch_idx * test_loader.batch_size + idx
            img_path = test_loader.dataset.samples[absolute_idx][0]
            if pred != label:
                wrong_preds_paths.append(img_path)
                print(
                    f"Wrong Prediction: {pred.item()=}, Label: {label.item()=}, Path: {img_path}"
                )
            # else:
            #     print(f"        Correct prediction {pred=} for {img_path=}")


# Generate classification report
report = classification_report(
    all_labels, all_preds, target_names=["benign", "malign"], digits=4
)
print(report)

"""
The result I got:

              precision    recall  avgscore   support
weighted avg       0.99      0.99      0.99      4362
      malign       0.99      0.99      0.99      4655

    accuracy                           0.99      9017
   macro avg       0.99      0.99      0.99      9017
weighted avg       0.99      0.99      0.99      9017
"""


for path in wrong_preds_paths:
    print(path)
