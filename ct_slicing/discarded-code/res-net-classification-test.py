from torchvision import models, transforms, datasets

import torch
import torch.nn as nn

from sklearn.metrics import classification_report

from ct_slicing.config.data_path import MODEL_PATH, SLICE_IMAGE_FOLDER


model = models.resnet152()  # Initialize the model structure
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
The result I got from training and testing with 100% data:

Wrong Prediction=1, Label=0, Path=output/png-slice-images/benign/0017-01-04.png
Wrong Prediction=1, Label=0, Path=output/png-slice-images/benign/0021-03-00.png
Wrong Prediction=1, Label=0, Path=output/png-slice-images/benign/0027-01-01.png
Wrong Prediction=1, Label=0, Path=output/png-slice-images/benign/0045-11-01.png
Wrong Prediction=1, Label=0, Path=output/png-slice-images/benign/0046-05-00.png
Wrong Prediction=1, Label=0, Path=output/png-slice-images/benign/0068-02-04.png
Wrong Prediction=1, Label=0, Path=output/png-slice-images/benign/0091-03-08.png
Wrong Prediction=1, Label=0, Path=output/png-slice-images/benign/0129-11-04.png
Wrong Prediction=1, Label=0, Path=output/png-slice-images/benign/0144-02-01.png
Wrong Prediction=1, Label=0, Path=output/png-slice-images/benign/0144-02-02.png
Wrong Prediction=1, Label=0, Path=output/png-slice-images/benign/0179-09-03.png
Wrong Prediction=1, Label=0, Path=output/png-slice-images/benign/0182-03-03.png
Wrong Prediction=1, Label=0, Path=output/png-slice-images/benign/0194-01-00.png
Wrong Prediction=1, Label=0, Path=output/png-slice-images/benign/0194-01-02.png
Wrong Prediction=1, Label=0, Path=output/png-slice-images/benign/0194-01-05.png
Wrong Prediction=1, Label=0, Path=output/png-slice-images/benign/0194-01-09.png
Wrong Prediction=1, Label=0, Path=output/png-slice-images/benign/0289-02-00.png
Wrong Prediction=1, Label=0, Path=output/png-slice-images/benign/0300-02-18.png
Wrong Prediction=1, Label=0, Path=output/png-slice-images/benign/0308-02-11.png
Wrong Prediction=1, Label=0, Path=output/png-slice-images/benign/0403-01-00.png
Wrong Prediction=1, Label=0, Path=output/png-slice-images/benign/0416-02-00.png
Wrong Prediction=1, Label=0, Path=output/png-slice-images/benign/0420-02-02.png
Wrong Prediction=1, Label=0, Path=output/png-slice-images/benign/0429-02-01.png
Wrong Prediction=1, Label=0, Path=output/png-slice-images/benign/0456-02-00.png
Wrong Prediction=1, Label=0, Path=output/png-slice-images/benign/0456-02-09.png
Wrong Prediction=1, Label=0, Path=output/png-slice-images/benign/0486-02-08.png
Wrong Prediction=1, Label=0, Path=output/png-slice-images/benign/0500-01-03.png
Wrong Prediction=1, Label=0, Path=output/png-slice-images/benign/0503-02-02.png
Wrong Prediction=1, Label=0, Path=output/png-slice-images/benign/0510-05-10.png
Wrong Prediction=1, Label=0, Path=output/png-slice-images/benign/0524-01-01.png
Wrong Prediction=1, Label=0, Path=output/png-slice-images/benign/0542-03-01.png
Wrong Prediction=1, Label=0, Path=output/png-slice-images/benign/0557-02-08.png
Wrong Prediction=1, Label=0, Path=output/png-slice-images/benign/0569-03-02.png
Wrong Prediction=1, Label=0, Path=output/png-slice-images/benign/0579-01-03.png
Wrong Prediction=1, Label=0, Path=output/png-slice-images/benign/0579-01-04.png
Wrong Prediction=1, Label=0, Path=output/png-slice-images/benign/0615-01-01.png
Wrong Prediction=1, Label=0, Path=output/png-slice-images/benign/0651-03-31.png
Wrong Prediction=1, Label=0, Path=output/png-slice-images/benign/0655-08-05.png
Wrong Prediction=1, Label=0, Path=output/png-slice-images/benign/0659-04-00.png
Wrong Prediction=1, Label=0, Path=output/png-slice-images/benign/0673-01-00.png
Wrong Prediction=1, Label=0, Path=output/png-slice-images/benign/0728-01-04.png
Wrong Prediction=1, Label=0, Path=output/png-slice-images/benign/0748-03-00.png
Wrong Prediction=1, Label=0, Path=output/png-slice-images/benign/0775-10-00.png
Wrong Prediction=1, Label=0, Path=output/png-slice-images/benign/0807-04-09.png
Wrong Prediction=1, Label=0, Path=output/png-slice-images/benign/0837-03-28.png
Wrong Prediction=1, Label=0, Path=output/png-slice-images/benign/0837-03-31.png
Wrong Prediction=1, Label=0, Path=output/png-slice-images/benign/0843-02-04.png
Wrong Prediction=1, Label=0, Path=output/png-slice-images/benign/0843-02-05.png
Wrong Prediction=1, Label=0, Path=output/png-slice-images/benign/0852-02-03.png
Wrong Prediction=1, Label=0, Path=output/png-slice-images/benign/0911-01-00.png
Wrong Prediction=1, Label=0, Path=output/png-slice-images/benign/0912-05-12.png
Wrong Prediction=1, Label=0, Path=output/png-slice-images/benign/0924-01-00.png
Wrong Prediction=0, Label=1, Path=output/png-slice-images/malign/0031-03-01.png
Wrong Prediction=0, Label=1, Path=output/png-slice-images/malign/0037-01-04.png
Wrong Prediction=0, Label=1, Path=output/png-slice-images/malign/0066-01-06.png
Wrong Prediction=0, Label=1, Path=output/png-slice-images/malign/0130-01-02.png
Wrong Prediction=0, Label=1, Path=output/png-slice-images/malign/0148-03-00.png
Wrong Prediction=0, Label=1, Path=output/png-slice-images/malign/0164-03-07.png
Wrong Prediction=0, Label=1, Path=output/png-slice-images/malign/0181-08-05.png
Wrong Prediction=0, Label=1, Path=output/png-slice-images/malign/0209-01-08.png
Wrong Prediction=0, Label=1, Path=output/png-slice-images/malign/0220-01-10.png
Wrong Prediction=0, Label=1, Path=output/png-slice-images/malign/0223-03-05.png
Wrong Prediction=0, Label=1, Path=output/png-slice-images/malign/0249-01-08.png
Wrong Prediction=0, Label=1, Path=output/png-slice-images/malign/0259-01-04.png
Wrong Prediction=0, Label=1, Path=output/png-slice-images/malign/0262-01-03.png
Wrong Prediction=0, Label=1, Path=output/png-slice-images/malign/0262-01-07.png
Wrong Prediction=0, Label=1, Path=output/png-slice-images/malign/0305-08-02.png
Wrong Prediction=0, Label=1, Path=output/png-slice-images/malign/0305-08-10.png
Wrong Prediction=0, Label=1, Path=output/png-slice-images/malign/0312-01-04.png
Wrong Prediction=0, Label=1, Path=output/png-slice-images/malign/0312-01-05.png
Wrong Prediction=0, Label=1, Path=output/png-slice-images/malign/0312-02-03.png
Wrong Prediction=0, Label=1, Path=output/png-slice-images/malign/0312-03-00.png
Wrong Prediction=0, Label=1, Path=output/png-slice-images/malign/0312-03-05.png
Wrong Prediction=0, Label=1, Path=output/png-slice-images/malign/0353-03-02.png
Wrong Prediction=0, Label=1, Path=output/png-slice-images/malign/0415-03-01.png
Wrong Prediction=0, Label=1, Path=output/png-slice-images/malign/0450-07-00.png
Wrong Prediction=0, Label=1, Path=output/png-slice-images/malign/0450-07-05.png
Wrong Prediction=0, Label=1, Path=output/png-slice-images/malign/0476-02-05.png
Wrong Prediction=0, Label=1, Path=output/png-slice-images/malign/0476-03-02.png
Wrong Prediction=0, Label=1, Path=output/png-slice-images/malign/0476-03-04.png
Wrong Prediction=0, Label=1, Path=output/png-slice-images/malign/0476-03-07.png
Wrong Prediction=0, Label=1, Path=output/png-slice-images/malign/0660-05-02.png
Wrong Prediction=0, Label=1, Path=output/png-slice-images/malign/0671-03-04.png
Wrong Prediction=0, Label=1, Path=output/png-slice-images/malign/0686-04-00.png
Wrong Prediction=0, Label=1, Path=output/png-slice-images/malign/0686-05-04.png
Wrong Prediction=0, Label=1, Path=output/png-slice-images/malign/0686-15-00.png
Wrong Prediction=0, Label=1, Path=output/png-slice-images/malign/0686-15-02.png
Wrong Prediction=0, Label=1, Path=output/png-slice-images/malign/0705-01-00.png
Wrong Prediction=0, Label=1, Path=output/png-slice-images/malign/0799-01-01.png
Wrong Prediction=0, Label=1, Path=output/png-slice-images/malign/0799-01-13.png
Wrong Prediction=0, Label=1, Path=output/png-slice-images/malign/0810-03-18.png

              precision    recall  f1-score   support

      benign     0.9910    0.9881    0.9896      4362
      malign     0.9889    0.9916    0.9902      4655

    accuracy                         0.9899      9017
   macro avg     0.9899    0.9899    0.9899      9017
weighted avg     0.9899    0.9899    0.9899      9017
"""


for path in wrong_preds_paths:
    print(path)
