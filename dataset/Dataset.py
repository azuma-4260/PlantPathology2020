import pandas as pd
from torch.utils.data import Dataset
import os
from PIL import Image
import torch
from collections import defaultdict

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


class CustomDataset(Dataset):
    def __init__(self, labels: pd.DataFrame, img_paths, transform=None):
        self.transform = transform
        self.labels = labels
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


def get_datasets(root, train_csv_file, train_transform, val_transform):
    train_img_paths, val_img_paths, test_img_paths = split_imgs(root)
    labels = pd.read_csv(train_csv_file)
    train_dataset = CustomDataset(labels, train_img_paths, transform=train_transform)
    val_dataset = CustomDataset(labels, val_img_paths, transform=val_transform)
    test_dataset = CustomDataset(labels, test_img_paths, transform=val_transform)
    return train_dataset, val_dataset, test_dataset

def partition_imgs_by_label(img_path_list, labels):
    d = defaultdict(list)

    for img_path in img_path_list:
        img_name = os.path.basename(img_path)
        img_id = img_name.split('.')[0]

        row = labels[labels['image_id'] == img_id].iloc[0, 1:]
        label = [k for k, v in row.items() if v == 1][0]
        d[label].append(img_path)

    datas = [[], [], [], []]
    for k, v in d.items():
        if k == 'multiple_diseases':
            for i, data in enumerate(datas):
                one_third = len(v)//3
                if i != 3:
                    data.extend(v[:one_third*2])
                    data.extend(v[:one_third*2])
                    data.extend(v[:one_third*2])
                else:
                    data.extend(v[one_third*2:])
        else:
            quarter = len(v)//4
            datas[0].extend(v[:quarter])
            datas[1].extend(v[quarter:quarter*2])
            datas[2].extend(v[quarter*2:quarter*3])
            datas[3].extend(v[quarter*3:])

    return datas

def get_ensemble_datasets(img_root, train_csv, train_transform, val_transform):
    train_imgs, val_imgs, test_imgs = split_imgs(img_root)
    labels: pd.DataFrame = pd.read_csv(train_csv)
    datas = partition_imgs_by_label(train_imgs, labels)
    train_dataset_A = CustomDataset(labels, datas[0], train_transform)
    train_dataset_B = CustomDataset(labels, datas[1], train_transform)
    train_dataset_C = CustomDataset(labels, datas[2], train_transform)
    ensemble_datset = CustomDataset(labels, datas[3], train_transform)
    val_dataset = CustomDataset(labels, val_imgs, val_transform)
    test_dataset = CustomDataset(labels, test_imgs, val_transform)
    return (train_dataset_A, train_dataset_B, train_dataset_C), ensemble_datset, val_dataset, test_dataset
