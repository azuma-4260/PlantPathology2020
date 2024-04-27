import argparse
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from dataset.Dataset import get_datasets
from torchvision.models import resnet50
from train import train
from dataset.Sampler import get_sampler
import os
from utils.logger import setup_logger

def main(is_train, img_root, train_csv_file, test_csv_file, use_sampler, save_dir, logger):
    train_transform = T.Compose([
        T.Resize(256),
        T.RandomCrop(224),
        T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset, val_dataset, test_dataset = get_datasets(img_root, train_csv_file, test_csv_file,
                                                train_transform, val_transform, train_ratio=0.8)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if is_train:
        if args.use_sampler:
            train_loader = DataLoader(train_dataset, batch_size=64, sampler=get_sampler(train_dataset), num_workers=16)
        else:
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=16)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=16)

        model = resnet50(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(2048, 4)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.fc.parameters(), lr=0.001, weight_decay=5e-4)

        train(model, criterion, optimizer, train_loader,
                val_loader, device, save_dir, logger, num_epochs=15)
    else:
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        model = torch.load(os.path.join(work_dir, 'best_model.pth'))
        model.to(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or evaluate the model.')
    parser.add_argument('--is_train', type=bool, default=True, help='Train mode if True, else evaluate.')
    parser.add_argument('--img_root', type=str, required=True, help='Root directory for the images.')
    parser.add_argument('--train_csv_file', type=str, required=True, help='Train CSV file with annotations.')
    parser.add_argument('--test_csv_file', type=str, required=True, help='Test CSV file with annotations.')
    parser.add_argument('--use_sampler', type=bool, default=False, help='Use sampler if True.')
    parser.add_argument('--work_dir', type=str, required=True, help='Directory for saving results.')
    args = parser.parse_args()

    current_time = datetime.datetime.now()
    folder_name = current_time.strftime('%Y%m%d_%H%M')
    save_dir = os.path.join(args.work_dir, folder_name)
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)
    os.makedirs(save_dir, exist_ok=True)

    logger = setup_logger(save_dir, args.use_sampler)

    main(args.is_train, args.img_root, args.train_csv_file,
        args.test_csv_file, args.use_sampler, save_dir, logger)
