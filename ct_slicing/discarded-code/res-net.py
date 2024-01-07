"""
ResNet model is better for 224x224 images. With arbitrary size images,
I'm afraid the padding will affect the model's performance.
"""
from torchvision import models

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
# print(*model.named_children())
for name, child in model.named_children():
    print(f"{name}: {child}")
exit()
