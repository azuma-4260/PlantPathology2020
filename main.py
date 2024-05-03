import argparse
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from dataset.Dataset import get_datasets
from torchvision.models import resnet50, vit_h_14, ViT_H_14_Weights
from train import train
from test import test
from dataset.Sampler import get_sampler
import os
from utils.logger import setup_logger

def main(is_train, model, criterion, optimizer, model_input_size, img_root, train_csv_file, patient, use_sampler, save_dir, test_weight_path, logger):
    train_transform = T.Compose([
        T.Resize((model_input_size, model_input_size)),
        T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomAdjustSharpness(sharpness_factor=5, p=0.5),
        T.RandomRotation(10),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = T.Compose([
        T.Resize((model_input_size, model_input_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset, val_dataset, test_dataset = get_datasets(img_root, train_csv_file, train_transform, val_transform)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    if is_train:
        if use_sampler:
            train_loader = DataLoader(train_dataset, batch_size=64, sampler=get_sampler(train_dataset), num_workers=16)
        else:
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=16)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=16)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        train(model, criterion, optimizer, train_loader,
                val_loader, patient, device, save_dir, logger, num_epochs=20)

        logger.info('Test Start')
        model = torch.load(os.path.join(save_dir, 'best_model.pth'))
        model.to(device)
        test(model, test_loader, device, logger, save_dir)

    else:
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        model = torch.load(test_weight_path)
        model.to(device)

        test(model, test_loader, device, logger, save_dir)

def get_model(model_name: str):
    assert model_name in ['resnet50', 'vit'], 'incorrect model name'

    if model_name == 'resnet50':
        model = resnet50(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(2048, 4)
        optimizer = optim.Adam(model.fc.parameters(), lr=0.001, weight_decay=5e-4)
        model_input_size = 224

    elif model_name == 'vit':
        model = vit_h_14(ViT_H_14_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        model.heads[0] = nn.Linear(1280, 4)
        optimizer = optim.Adam(model.heads[0].parameters(), lr=0.001, weight_decay=5e-4)
        model_input_size = 518

    criterion = nn.CrossEntropyLoss()

    return model, criterion, optimizer, model_input_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or evaluate the model.')
    parser.add_argument('--is_train', type=bool, default=False, help='Train mode if True, else evaluate.')
    parser.add_argument('--model_name', type=str, default='resnet50', help='Train mode if True, else evaluate.')
    parser.add_argument('--img_root', type=str, required=True, help='Root directory for the images.')
    parser.add_argument('--train_csv_file', type=str, required=True, help='Train CSV file with annotations.')
    parser.add_argument('--patient', type=int, required=False, default=20)
    parser.add_argument('--use_sampler', type=bool, default=False, help='Use sampler if True.')
    parser.add_argument('--work_dir', type=str, required=True, help='Root Directory for saving results.')
    parser.add_argument('--test_weight', type=str, required=False, help='Path of model weight for test.')
    args = parser.parse_args()

    model, criterion, optimizer, model_input_size = get_model(args.model_name)

    current_time = datetime.datetime.now()
    folder_name = current_time.strftime('%Y%m%d_%H%M')
    save_dir = os.path.join(args.work_dir, folder_name)
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)
    os.makedirs(save_dir, exist_ok=True)

    logger = setup_logger(save_dir, args.use_sampler, args.is_train)

    main(args.is_train, model, criterion, optimizer, model_input_size, args.img_root, args.train_csv_file, args.patient, args.use_sampler, save_dir, args.test_weight, logger)
