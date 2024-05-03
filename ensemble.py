import argparse
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from dataset.Dataset import get_ensemble_datasets
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights
from train import train, train_ensemble
from test import test_ensemble
import os
from utils.logger import setup_logger
from ensemble.Ensembler import Ensembler

def main(img_root, csv, is_train, patient, save_dir, test_root, logger):
    train_transform = T.Compose([
        T.Resize(256),
        T.RandomCrop(224),
        T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(10),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_datasets, ensemble_datset, val_dataset, test_dataset = get_ensemble_datasets(img_root=img_root, train_csv=csv,
                                                                      train_transform=train_transform, val_transform=val_transform)

    model1 = efficientnet_v2_l(EfficientNet_V2_L_Weights.DEFAULT)
    model2 = efficientnet_v2_l(EfficientNet_V2_L_Weights.DEFAULT)
    model3 = efficientnet_v2_l(EfficientNet_V2_L_Weights.DEFAULT)
    models = [model1, model2, model3]
    for model in models:
        model.classifier[1] = nn.Linear(1280, 4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if is_train:
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=16)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        criterion = nn.CrossEntropyLoss()

        # train individual model
        for i, model in enumerate(models):
            model.to(device)
            train_loader = DataLoader(train_datasets[i], batch_size=64, shuffle=True, num_workers=16)
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
            logger.info(f'\nmodel{i} Train Start')
            train(model, criterion, optimizer, train_loader, val_loader, patient,
                  device, save_dir, logger, num_epochs=50, model_number=i, scheduler=scheduler)

        # train ensemble model
        logger.info('\nEnsemble Train Start')
        ensembler = Ensembler()
        ensembler.to(device)
        train_loader = DataLoader(ensemble_datset, batch_size=64, shuffle=True, num_workers=16)
        optimizer = optim.Adam(ensembler.parameters(), lr=0.001, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        train_ensemble(models, ensembler, criterion, optimizer, scheduler, train_loader, val_loader,
                        device, save_dir, logger, num_epochs=50)

        # test
        logger.info('\nTest Start')
        models = [
            torch.load(os.path.join(save_dir, 'best_model0.pth')),
            torch.load(os.path.join(save_dir, 'best_model1.pth')),
            torch.load(os.path.join(save_dir, 'best_model2.pth'))
        ]
        for model in models:
            model.to(device)
        ensembler = torch.load(os.path.join(save_dir, 'ensembler_best_model.pth'))
        ensembler.to(device)
        test_ensemble(models, ensembler, test_loader, device, logger, save_dir)
    else:
        logger.info(f'Test root: {test_root}')
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        models = [
            torch.load(os.path.join(test_root, 'best_model0.pth')),
            torch.load(os.path.join(test_root, 'best_model1.pth')),
            torch.load(os.path.join(test_root, 'best_model2.pth'))
        ]
        for model in models:
            model.to(device)
        ensembler = torch.load(os.path.join(test_root, 'ensembler_best_model.pth'))
        ensembler.to(device)
        test_ensemble(models, ensembler, test_loader, device, logger, save_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or evaluate the model.')
    parser.add_argument('--is_train', type=bool, default=False, help='Train mode if True, else evaluate.')
    parser.add_argument('--img_root', type=str, required=False, default='data/images', help='Root directory for the images.')
    parser.add_argument('--csv', type=str, required=False, default='data/train.csv', help='Train CSV file with annotations.')
    parser.add_argument('--patient', type=int, required=False, default=20)
    parser.add_argument('--work_dir', type=str, required=False, default='./work_dir', help='Root Directory for saving results.')
    parser.add_argument('--test_root', type=str, required=False, help='Root Directory for test weight.')
    args = parser.parse_args()

    current_time = datetime.datetime.now()
    folder_name = current_time.strftime('%Y%m%d_%H%M')
    save_dir = os.path.join(args.work_dir, folder_name)
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)
    os.makedirs(save_dir, exist_ok=True)

    logger = setup_logger(save_dir, None, args.is_train)

    main(args.img_root, args.csv, args.is_train, args.patient, save_dir, args.test_root, logger)
