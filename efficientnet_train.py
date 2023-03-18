import torch
import torchvision
import torch.optim as optim
import argparse
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from cvae import LinearCVAE
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
import time
import os
from torch_dataset import DatasetFromCSV
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from models.efficientnet import EfficientNet
from func_train_val_test import train,valid
import albumentations as album
from albumentations.pytorch import ToTensorV2
import gc
from datetime import datetime
import nvidia_smi
import csv


def create_save_path(save_path,timestamp):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    #保存模型参数的地址
    checking_point_path = os.path.join(save_path, 'weights',timestamp)
    if not os.path.exists(checking_point_path):
        os.makedirs(checking_point_path)
    #保存训练日志的地址
    log_path = os.path.join(save_path, 'train_log',timestamp)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    return log_path, checking_point_path


def save_best_model(model,epoch, current_acc, best, checking_point_path):

    if best < current_acc:
        best = current_acc
        model_path = f'{checking_point_path}/best_{epoch}.pt'
        torch.save(model.state_dict(), model_path)
        print('best model saved')
    return best


def save_csv(save_csv_path, model_name, epochs, train_time, used_memory, used_memory_peak):
    if not os.path.exists(save_csv_path):
        with open(save_csv_path, mode='w', newline='') as f:
            csv_head = ['model name', 'epochs','training time(s)','used memory(Gb)','used memory_peak(Gb)']
            writer = csv.DictWriter(f, fieldnames=csv_head)
            writer.writeheader()

    with open(save_csv_path, mode='a', newline='') as f:
        csv_write = csv.writer(f)
        data_row = [model_name,epochs,train_time,used_memory,used_memory_peak]
        csv_write.writerow(data_row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type = int, default=16)
    parser.add_argument('--init_lr', type=float, default=0.001)
    parser.add_argument('--n_class', type=int, default=3)
    parser.add_argument('--model_name',type=str,default='efficientnet_b0')
    parser.add_argument('--save_path', type=str, default='./checkpoints/efficientnet_b0')
    args = parser.parse_args()

    epochs = args.epochs
    init_lr = args.init_lr
    batch_size = args.batch_size
    n_class = args.n_class

    img_path = "./data/medical_data/data_org_image"

    #数据集分割
    csv_root = './data/medical_data/medical_csv'
    csv_name = 'data_label.csv'
    #csv_train_path, csv_val_path, csv_test_path = data_partition(csv_root, csv_name, valid_size=0.15, test_size=0.15)
    data_mean = torch.tensor([0.7750, 0.5888, 0.7629])
    data_std = torch.tensor([0.2129, 0.2971, 0.1774])

    transforms = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_mean, std=data_std),
    ])

    #数据准备
    csv_train_path = './data/medical_data/medical_csv/train_data.csv'
    csv_val_path = './data/medical_data/medical_csv/valid_data.csv'
    train_dataset = DatasetFromCSV(img_path=img_path, csv_path=csv_train_path,transforms=transforms)
    valid_dataset = DatasetFromCSV(img_path=img_path, csv_path=csv_val_path,transforms=transforms)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)

    #导入模型
    #model
    model = EfficientNet.from_name('efficientnet-b0', num_classes=n_class)
    #model = torchvision.models.resnet18(weights=None, num_classes = n_class)
    #model = torchvision.models.mobilenet_v2(weights=None, num_classes = n_class)
    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)

    # Loss function
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.Adam(lr=init_lr, params=model.parameters(), betas=(0.9, 0.99))

    optimizer_step = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # Weight and log save path
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    save_path = args.save_path
    log_path, checkpoint_path = create_save_path(save_path,TIMESTAMP)

    log_writer = SummaryWriter(log_path)

    # -----------Training and validation-------

    train_total = len(train_dataset)
    valid_total = len(valid_dataset)

    start = time.time()
    best = 0.0

    for epoch in range(epochs):
        print('Training......')
        train_acc, train_loss = train(model, train_loader, device, loss_fn, optimizer, train_total)
        print(f'Train -> Epoch: {epoch:>03d}, train_acc: {train_acc:.4f}, train_loss: {train_loss:.4f}')

        valid_acc, valid_loss = valid(model, valid_loader, device, loss_fn, optimizer, valid_total)
        print(f'Valid -> Epoch: {epoch:>03d}, valid_acc: {valid_acc:.4f}, valid_loss: {valid_loss:.4f}')

        optimizer_step.step()  # update learning rate
        lr = optimizer.param_groups[0]['lr']

        # Save best model only
        best = save_best_model(model, epoch, valid_acc, best, checkpoint_path)

        # Write log
        log_writer.add_scalar("Train/Train_Acc", train_acc, epoch)
        log_writer.add_scalar("Train/Val_Acc", valid_acc, epoch)
        log_writer.add_scalar("Train/Train_Loss", train_loss, epoch)
        log_writer.add_scalar("Train/Val_Loss", valid_loss, epoch)
        log_writer.add_scalar("Train/LR", lr, epoch)

    log_writer.close()
    total_train_time = time.time() - start
    print(f'Total training time: {total_train_time}s')

    #计算显存消耗
    used_memory = torch.cuda.memory_allocated(device) / (1024 * 1024 * 1024)
    used_memory_peak = torch.cuda.max_memory_allocated(device) / (1024 * 1024 * 1024)
    print(f'memory consumption:{used_memory}')
    print(f'peak memory consumption:{used_memory_peak}')
    model_name = args.model_name
    save_csv('./checkpoints/model_train_info.csv', model_name, epochs, total_train_time,used_memory,used_memory_peak)









