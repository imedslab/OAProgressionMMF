from torchvision import models
from ._torchvision import resnext50_32x4d


# See http://pytorch.org/vision/stable/models.html#classification for the more models
dict_fes = {
    "squeezenet1_0": models.squeezenet1_0,
    "vgg16": models.vgg16,
    "densenet161": models.densenet161,
    "inception_v3": models.inception_v3,
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnext50_32x4d": resnext50_32x4d,
}
