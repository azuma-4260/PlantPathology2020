import torch
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import os
join = os.path.join

def train(model, criterion, optimizer, train_loader, val_loader,
          patient, device, save_dir, logger, num_epochs, model_number=-1, scheduler=None):
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
        for images, labels in train_loader:
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
            # ensembleの場合はmodelの番号を含める
            save_model_name = 'best_model.pth' if model_number == -1 else f'best_model{model_number}.pth'
            torch.save(model, join(save_dir, save_model_name))
            non_updated_count = 0
        else:
            non_updated_count += 1

        logger.info(f'Epoch {epoch+1}/{num_epochs} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} ValLoss: {val_loss:.4f} ValAcc: {val_acc:.4f}')
        if scheduler is not None:
            scheduler.step(val_loss)
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
    plt.savefig(join(save_dir, f'loss_plot{"" if model_number == -1 else model_number}.png'))
    plt.close()

    # Accuracyのグラフを描画して保存
    plt.figure(figsize=(10, 5))
    plt.plot([_acc.cpu() for _acc in accuracies], label='Train Accuracy')
    plt.plot([_acc.cpu() for _acc in val_accs], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(join(save_dir, f'accuracy_plot{"" if model_number == -1 else model_number}.png'))
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

def train_ensemble(models, ensembler, criterion, optimizer, scheduler, train_loader, val_loader,
                   device, save_dir, logger, num_epochs):
    losses = []
    accuracies = []
    val_losses = []
    val_accs = []
    best_val_acc = 0.0

    for model in models:
        model.eval()

    for epoch in trange(num_epochs):
        ensembler.train()
        running_loss = 0.0
        running_acc = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                output1 = models[0](images) # [b, 4]
                output2 = models[1](images) # [b, 4]
                output3 = models[2](images) # [b, 4]
                outputs = torch.cat((output1, output2, output3), dim=1) # [b, 12]
                outputs.to(device)

            ensemble_outputs = ensembler(outputs)
            preds = torch.argmax(ensemble_outputs, dim=1)
            loss = criterion(ensemble_outputs, labels)

            running_loss += loss.item()
            running_acc += torch.mean(preds.eq(labels).float())

            loss.backward()
            optimizer.step()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = running_acc / len(train_loader)

        losses.append(epoch_loss)
        accuracies.append(epoch_acc)

        val_loss, val_acc = validate_ensemble(models, ensembler, criterion, val_loader, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # 最高の検証精度でのモデルの重みを保存
            torch.save(ensembler, join(save_dir,'ensembler_best_model.pth'))

        logger.info(f'Epoch {epoch+1}/{num_epochs} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} ValLoss: {val_loss:.4f} ValAcc: {val_acc:.4f}')
        scheduler.step(val_loss)

    # Lossのグラフを描画して保存
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Ensemble Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(join(save_dir, 'ensemble_loss_plot.png'))
    plt.close()

    # Accuracyのグラフを描画して保存
    plt.figure(figsize=(10, 5))
    plt.plot([_acc.cpu() for _acc in accuracies], label='Train Accuracy')
    plt.plot([_acc.cpu() for _acc in val_accs], label='Validation Accuracy')
    plt.title('Ensemble Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(join(save_dir,'ensemble_accuracy_plot.png'))
    plt.close()

def validate_ensemble(models, ensembler, criterion, val_loader, device):
    ensembler.eval()
    running_loss = 0.0
    running_acc = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            output1 = models[0](images)
            output2 = models[1](images)
            output3 = models[2](images)
            outputs = torch.cat((output1, output2, output3), dim=1)
            outputs.to(device)

            ensemble_outputs = ensembler(outputs)
            loss = criterion(ensemble_outputs, labels)

            preds = torch.argmax(ensemble_outputs, dim=1)
            running_loss += loss.item()
            running_acc += torch.mean(preds.eq(labels).float())

    val_loss = running_loss / len(val_loader)
    val_acc = running_acc / len(val_loader)

    return val_loss, val_acc
