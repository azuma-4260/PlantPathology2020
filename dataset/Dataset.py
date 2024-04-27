import pandas as pd
from torch.utils.data import Dataset
import os
from PIL import Image
import torch
import torchvision.transforms as transforms

def count_images(data_root, mode):
    # 指定されたディレクトリ内のファイルリストを取得
    file_list = os.listdir(data_root)
    # modeで指定された文字列で始まり、拡張子が.jpgのファイルの数をカウント
    count = sum(1 for file in file_list if file.startswith(mode) and file.endswith('.jpg'))
    return count


class CustomTrainDataset(Dataset):
    def __init__(self, root, csv_file, train_imgs_count,  transform=None, train_ratio=0.8):
        self.root = root
        self.transform = transform
        self.labels = pd.read_csv(csv_file)
        self.image_paths = [os.path.join(root, f) for f in os.listdir(root) \
                                if f.endswith('.jpg') and f.startswith('Train') \
                                and int(f.split('.')[0].split('_')[1]) < int(train_imgs_count*train_ratio)]
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
    def __init__(self, root, csv_file, train_imgs_count, transform=None, train_ratio=0.8):
        self.root = root
        self.transform = transform
        self.labels = pd.read_csv(csv_file)
        self.image_paths = [os.path.join(root, f) for f in os.listdir(root) \
                                if f.endswith('.jpg') and f.startswith('Train')\
                                and int(f.split('.')[0].split('_')[1]) >= int(train_imgs_count*train_ratio)]
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
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_paths = [os.path.join(root, f) for f in os.listdir(root) \
                                if f.endswith('.jpg') and f.startswith('Test')]

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.image_paths)


def get_datasets(root, train_csv_file, test_csv_file, train_transform, val_transform, train_ratio=0.8):
    train_imgs_count = count_images(root, mode='Train')
    train_dataset = CustomTrainDataset(root, train_csv_file, train_imgs_count, transform=train_transform, train_ratio=train_ratio)
    val_dataset = CustomValDataset(root, train_csv_file, train_imgs_count, transform=val_transform, train_ratio=train_ratio)
    test_dataset = CustomTestDataset(root, val_transform)
    return train_dataset, val_dataset, test_dataset
