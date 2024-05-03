from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch

def test(model, test_loader, device, logger, save_dir):
    model.eval()
    all_labels = []
    all_preds = []

    for images, labels in tqdm(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    calc_score(all_labels, all_preds, save_dir, logger)

def test_ensemble(models, ensembler, test_loader, device, logger, save_dir):
    for model in models:
        model.eval()
    ensembler.eval()

    all_labels = []
    all_preds = []

    for images, labels in tqdm(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        output1 = models[0](images)
        output2 = models[1](images)
        output3 = models[2](images)
        outputs = torch.cat((output1, output2, output3), dim=1)
        outputs.to(device)

        ensemble_outputs = ensembler(outputs)
        preds = torch.argmax(ensemble_outputs, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    calc_score(all_labels, all_preds, save_dir, logger)


def calc_score(all_labels, all_preds, save_dir, logger):
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    conf_matrix = confusion_matrix(all_labels, all_preds)

    logger.info(f'正解率: {accuracy}')
    logger.info(f'適合率: {precision}')
    logger.info(f'再現率: {recall}')
    logger.info(f'F1値: {f1}')
    logger.info(f'混同行列:\n{conf_matrix}')

    plot_confusion_matrix(conf_matrix, save_dir)

def plot_confusion_matrix(conf_matrix, save_dir):
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('predict')
    plt.ylabel('label')
    plt.title('confusion matrix')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()
