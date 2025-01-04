import torch

from FedUnlearner.models import AllCNN, ResNet18


def get_model(model: str,
              num_classes: int,
              device: str,
              dataset: str,
              pretrained=False) -> torch.nn.Module:
    """
    model: str, Name of the model to be created
    num_classes: int, Number of classes of dataset
    pretrained: bool, Whether to load pretrained model

    Returns Model:torch.nn.Module
    """
    new_model = None
    if model == 'allcnn':
        if dataset == 'mnist':
            new_model = AllCNN(num_classes=num_classes, num_channels=1)
        else:
            new_model = AllCNN(num_classes=num_classes)
    elif model == 'resnet18':
        if dataset == 'mnist':
            new_model = ResNet18(num_classes=num_classes,
                                 pretrained=pretrained, num_channels=1)
        else:
            new_model = ResNet18(num_classes=num_classes,
                                 pretrained=pretrained)
    else:
        raise "Invalid model name"
    new_model = new_model.to(device)
    return new_model
