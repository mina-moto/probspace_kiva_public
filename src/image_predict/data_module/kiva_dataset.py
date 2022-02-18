from torch.utils.data import Dataset
from torchvision.io import read_image

from image_predict.module.image_transform import test_transform, train_transform


class KivaDataset(Dataset):
    def __init__(self, image_path_list, loan_id_list=None, is_train=True):
        self._X = image_path_list
        self._y = loan_id_list
        self._is_train = is_train

    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):
        image_path = self._X[idx]
        image = read_image(image_path)
        if self._is_train:
            transform = train_transform()
        else:
            transform = test_transform()
        image = transform(image)
        if self._y is not None:
            target = self._y[idx]
            return image, target
        return image
