import pandas as pd
from torch.utils.data import Dataset
import os
from PIL import Image
import torch
import torchvision.transforms as transforms

def split_imgs(img_root):
    file_list = [f for f in os.listdir(img_root) if f.endswith('.jpg') and f.startswith('Train')]

    # train:val:test = 16:4:5
    file_num = len(file_list)
    train_img_num = int(file_num*(16/25))
    val_img_num = int(file_num*(4/25))
    train_imgs_path = [os.path.join(img_root, f) for f in file_list[:train_img_num]]
    val_imgs_path = [os.path.join(img_root, f) for f in file_list[train_img_num:train_img_num+val_img_num]]
    test_imgs_path = [os.path.join(img_root, f) for f in file_list[train_img_num+val_img_num:]]
    return train_imgs_path, val_imgs_path, test_imgs_path


class CustomTrainDataset(Dataset):
    def __init__(self, root, csv_file, img_paths, transform=None):
        self.transform = transform
        self.labels = pd.read_csv(csv_file)
        self.image_paths = img_paths
        # ラベル名を整数値にマッピング
        self.label_to_int = {'healthy': 0, 'multiple_diseases': 1, 'rust': 2, 'scab': 3}

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        image_id = os.path.basename(image_path).split('.')[0]
        row = self.labels[self.labels['image_id'] == image_id].iloc[0, 1:]
        # ラベルを整数値に変換
        label = torch.tensor([self.label_to_int[key] for key, value in row.items() if value == 1][0], dtype=torch.long)

        return image, label

    def __len__(self):
        return len(self.image_paths)

class CustomValDataset(Dataset):
    def __init__(self, root, csv_file, img_paths, transform=None):
        self.transform = transform
        self.labels = pd.read_csv(csv_file)
        self.image_paths = img_paths
        self.label_to_int = {'healthy': 0, 'multiple_diseases': 1, 'rust': 2, 'scab': 3}

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        image_id = os.path.basename(image_path).split('.')[0]
        row = self.labels[self.labels['image_id'] == image_id].iloc[0, 1:]
        # ラベルを整数値に変換
        label = torch.tensor([self.label_to_int[key] for key, value in row.items() if value == 1][0], dtype=torch.long)

        return image, label

    def __len__(self):
        return len(self.image_paths)

class CustomTestDataset(Dataset):
    def __init__(self, root, csv_file, img_paths, transform=None):
        self.transform = transform
        self.labels = pd.read_csv(csv_file)
        self.image_paths = img_paths
        self.label_to_int = {'healthy': 0, 'multiple_diseases': 1, 'rust': 2, 'scab': 3}

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        image_id = os.path.basename(image_path).split('.')[0]
        row = self.labels[self.labels['image_id'] == image_id].iloc[0, 1:]
        # ラベルを整数値に変換
        label = torch.tensor([self.label_to_int[key] for key, value in row.items() if value == 1][0], dtype=torch.long)

        return image, label

    def __len__(self):
        return len(self.image_paths)


def get_datasets(root, train_csv_file, train_transform, val_transform):
    train_img_paths, val_img_paths, test_img_paths = split_imgs(root)
    train_dataset = CustomTrainDataset(root, train_csv_file, train_img_paths, transform=train_transform)
    val_dataset = CustomValDataset(root, train_csv_file, val_img_paths, transform=val_transform)
    test_dataset = CustomTestDataset(root, train_csv_file, test_img_paths, transform=val_transform)
    return train_dataset, val_dataset, test_dataset
