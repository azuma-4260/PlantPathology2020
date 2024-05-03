import torch
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import os
join = os.path.join

def train(model, criterion, optimizer, train_loader, val_loader, patient, device, save_dir, logger, num_epochs):
    losses = []
    accuracies = []
    val_losses = []
    val_accs = []
    best_val_acc = 0.0
    non_updated_count = 0

    for epoch in trange(num_epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            running_acc += torch.mean(preds.eq(labels).float())

            loss.backward()
            optimizer.step()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = running_acc / len(train_loader)

        losses.append(epoch_loss)
        accuracies.append(epoch_acc)

        val_loss, val_acc = validate(model, criterion, val_loader, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # 最高の検証精度でのモデルの重みを保存
            torch.save(model, join(save_dir,'best_model.pth'))
            non_updated_count = 0
        else:
            non_updated_count += 1

        logger.info(f'Epoch {epoch+1}/{num_epochs} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} ValLoss: {val_loss:.4f} ValAcc: {val_acc:.4f}')
        if non_updated_count > patient:
                logger.info('Patient Limit')
                break

    # Lossのグラフを描画して保存
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(join(save_dir, 'loss_plot.png'))
    plt.close()

    # Accuracyのグラフを描画して保存
    plt.figure(figsize=(10, 5))
    plt.plot([_acc.cpu() for _acc in accuracies], label='Train Accuracy')
    plt.plot([_acc.cpu() for _acc in val_accs], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(join(save_dir,'accuracy_plot.png'))
    plt.close()

def validate(model, criterion, val_loader, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            preds = torch.argmax(outputs, dim=1)
            running_loss += loss.item()
            running_acc += torch.mean(preds.eq(labels).float())

    val_loss = running_loss / len(val_loader)
    val_acc = running_acc / len(val_loader)

    return val_loss, val_acc


