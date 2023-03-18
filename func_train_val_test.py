import torch
import numpy as np

autocast = torch.cuda.amp.autocast
scaler = torch.cuda.amp.GradScaler()

def train(model, train_loader, device, loss_fn, optimizer, train_total):
    model.train()

    train_loss = 0.
    train_acc = 0.
    total = 0
    for i, (images, labels) in enumerate(train_loader):
        # Forward
        images = images.to(device)
        labels = labels.to(device)
        with autocast():
            preds = model(images)
            loss = loss_fn(preds, labels)
        #preds = model(images)
        #loss = loss_fn(preds, labels)
        train_loss += loss.item()

        # Backward
        optimizer.zero_grad()
        # scale loss(FP16)，backward得到scaled的梯度(FP16)【放大loss，防止梯度消失】
        scaler.scale(loss).backward()
        #loss.backward()

        # Update weights
        #optimizer.step()
        # scaler 更新参数，会先自动unscale梯度
        # 如果有nan或inf，自动跳过
        scaler.step(optimizer)
        # scaler factor更新
        scaler.update()

        # Prediction -> acc
        _, pred_labels = torch.max(preds, 1)
        # pred_labels = preds.squeeze()
        batch_correct = (pred_labels == labels).squeeze().sum().item()
        train_acc += batch_correct

        batch_size = labels.size(0)
        total += batch_size



    train_acc = train_acc / train_total
    train_loss = train_loss / len(train_loader)

    return train_acc, train_loss


def valid(model, val_loader, device, loss_fn, optimizer, valid_total):
    valid_acc = 0.
    valid_loss = 0.
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            preds = model(images)

            loss = loss_fn(preds, labels)
            valid_loss += loss.item()

            _, pred_labels = torch.max(preds, 1)
            batch_correct = (pred_labels == labels).squeeze().sum().item()
            valid_acc += batch_correct

    valid_acc = valid_acc / valid_total
    valid_loss = valid_loss / len(val_loader)

    return valid_acc, valid_loss



def test(dataloader, model, device, loss_function):
    """
    Test on batches in dataloader and save misclassified samples.
    """

    # Set model in evaluation mode
    model.eval()

    # Initalize result metrics and lists for misclassifications
    total_loss = 0.0
    total_accuracy = 0.0
    num_samples = 0
    misclassified_images = []
    misclassified_labels = []
    correct_labels = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):

            # Load images and labels and predict labels
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images)

            # Compute loss
            loss = loss_function(preds, labels)
            total_loss += loss.item()

            # Predict labels and compute number of correctly predicted labels in batch
            _, pred_labels = torch.max(preds, 1)
            batch_correct = (pred_labels==labels).squeeze().sum().item()
            total_accuracy += batch_correct

            # Find misclassifications and save
            # Find misclassifications and save
            try:
                misclassified_idxs = np.where(pred_labels!=labels)
            except:
                misclassified_idxs = np.where(pred_labels.cpu()!=labels.cpu())
            #Correct labels of misclassified images
            correct_label = labels[misclassified_idxs]
            #Predicted labels of misclassified images
            misclassified_label = pred_labels[misclassified_idxs]

            if misclassified_label.numel():
                for i in range(len(misclassified_label)):
                    misclassified_image = images[misclassified_idxs[0][i],:,:,:].squeeze()
                    misclassified_images.append(misclassified_image)
                    misclassified_labels.append(int(misclassified_label[i]))
                    correct_labels.append(int(correct_label[i]))

            # Compute batch size and increase number of total samples
            batch_size = labels.size(0)
            num_samples += batch_size

    # Compute mean accuracy and loss
    mean_accuracy = total_accuracy/num_samples
    mean_loss = total_loss/len(dataloader)

    return mean_accuracy, mean_loss, misclassified_images, misclassified_labels, correct_labels
