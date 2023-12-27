__author__ = "Debora Gil, Guillermo Torres, Carles Sanchez, Pau Cano"
__license__ = "GPL"
__email__ = "debora,gtorres,csanchez,pcano@cvc.uab.es"
__year__ = "2023"
__doc__ = """Source code for volume visualization

Computer Vision Center
Universitat Autonoma de Barcelona
"""

if __name__ != "__main__":
    raise ImportError(f"Script {__file__} should not be imported as a module")

import torchvision.models as models

# Load the pre-trained model VGG16
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
# pretrained=True is deprecated, use weights=models.VGG16_Weights.IMAGENET1K_V1 instead

# Show the architecture of the VGG16 model
print(model)  # Answer of exercise 2.b
# print(model.named_parameters)

# Show the layer's weights of the VGG16 model
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)

"""
# Exercise

## Exercise 2. Understanding a pre-trained network.
b) To gain a better understanding of the VGG16 architecture, identify the layers responsible
that perform features extraction and classification, see the vgg_architecture.py code.
What is the input and output shape of the VGG16 architecture? What database was
VGG16 trained with?
    - layers responsible for features extraction: 
        - Conv2d, ReLU, MaxPool2d
        - See the code in with comment `# Answer of exercise 2.b`
    - layers responsible for classification:
        - Linear, ReLU, Dropout, Linear, ReLU, Dropout, Linear
        - See the code in with comment `# Answer of exercise 2.b`
    - The VGG16 model typically expects input images with the following shape:
    
        - Input Channels: 3 (RGB color channels)
        - Height: 224 pixels
        - Width: 224 pixels

        and the output shape is (1000, 1), because the VGG16 model was trained to
        classify images into 1000 different categories.
        The VGG16 architecture was trained with the ImageNet database.

See the next part of the exercise in `l3_8_vgg_features_extraction.py`
"""
