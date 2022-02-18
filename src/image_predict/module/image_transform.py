import torch
import torchvision.transforms as T


def train_transform():
    transform = T.Compose(
        [
            T.Resize([224, 224]),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            # T.ToTensor(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    return transform


def test_transform():
    transform = T.Compose(
        [
            T.Resize([224, 224]),
            # T.ToTensor(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    return transform
