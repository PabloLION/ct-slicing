__author__ = "Debora Gil, Guillermo Torres, Carles Sanchez, Pau Cano"
__license__ = "GPL"
__email__ = "debora,gtorres,csanchez,pcano@cvc.uab.es"
__year__ = "2023"
__doc__ = """Source code for volume visualization

Computer Vision Center
Universitat Autonoma de Barcelona
"""

import torchvision.models as models

# Load the pre-trained model VGG16
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
# pretrained=True is deprecated, use weights=models.VGG16_Weights.IMAGENET1K_V1 instead

# Show the architecture of the VGG16 model
print(model)  # similar to next line
print(model.named_parameters)

# Show the layer's weights of the VGG16 model
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)
