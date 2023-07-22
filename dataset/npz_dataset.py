from torch.utils import data
import torch

from config import PROJECT_PATH
import numpy as np

class NpzDataset(data.Dataset):
    def __init__(self, dataset, use_image_id=False):
        file_path = "{}/attacked_images/{}/{}_images.npz".format(PROJECT_PATH, dataset, dataset)
        file_data = np.load(file_path)
        self.dataset = dataset
        self.images = file_data["images"]
        self.labels = file_data["labels"]
        self.use_image_id = use_image_id
        if self.use_image_id:
            self.image_id = file_data["image_id"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        if self.use_image_id:
            image_id = self.image_id[index]
            return image_id, torch.from_numpy(image), label
        return torch.from_numpy(image),label


class NpzExtraDataset(NpzDataset):
    def __init__(self, dataset):
        super(NpzExtraDataset, self).__init__(dataset)
        file_path = "{}/attacked_images/{}/{}_images_for_candidate.npz".format(PROJECT_PATH, dataset, dataset)
        file_data = np.load(file_path)
        self.dataset = dataset
        self.images = file_data["images"]
        self.labels = file_data["labels"]

