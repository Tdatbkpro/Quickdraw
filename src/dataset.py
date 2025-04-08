from torch.utils.data import Dataset
import numpy as np
from src.config import CLASSES

class QuickDrawDataset(Dataset):
    def __init__(self, root_path="data", total_images_per_class=10000, ratio=0.8, mode="train", transform=None):
        self.root_path = root_path
        self.transform = transform
        self.num_classes = len(CLASSES)

        self.num_per_class = int(total_images_per_class * ratio) if mode == "train" \
            else int(total_images_per_class * (1 - ratio))
        self.offset = 0 if mode == "train" else int(total_images_per_class * ratio)
        self.total_samples = self.num_per_class * self.num_classes

    def __len__(self):
        return self.total_samples

    def __getitem__(self, index):
        class_idx = index // self.num_per_class
        image_idx = self.offset + (index % self.num_per_class)

        file_path = f"{self.root_path}/full_numpy_bitmap_{CLASSES[class_idx]}.npy"
        data = np.load(file_path, mmap_mode='r')
        image = data[image_idx].reshape((1, 28, 28)).copy()  # (1, 28, 28)
        image = image.transpose((1, 2, 0))  # To (28, 28, 1)

        if self.transform:
            image = self.transform(image)

        return image, class_idx
