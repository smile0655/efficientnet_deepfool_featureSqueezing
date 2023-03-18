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
from func_train_val_test import test
import albumentations as album
from albumentations.pytorch import ToTensorV2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--init_lr', type=float, default=0.001)
    parser.add_argument('--n_class', type=int, default=3)
    parser.add_argument('--data_path', type=str, default='./data/medical_data/data_pert_image')
    parser.add_argument('--weights_path', type=str, default='./checkpoints/resnet18/weights/2023-02-13T03-25-38/best_24.pt')
    args = parser.parse_args()

    epochs = args.epochs
    init_lr = args.init_lr
    batch_size = args.batch_size
    n_class = args.n_class
    weights_path = args.weights_path
    data_path = args.data_path



    #数据集分割
    #csv_train_path, csv_val_path, csv_test_path = data_partition(csv_root, csv_name, valid_size=0.15, test_size=0.15)
    data_mean = torch.tensor([0.7750, 0.5888, 0.7629])
    data_std = torch.tensor([0.2129, 0.2971, 0.1774])
    transforms = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_mean, std=data_std),
    ])

    csv_test_path = './data/medical_data/medical_csv/attacked_data_label.csv'
    test_dataset = DatasetFromCSV(img_path=data_path, csv_path=csv_test_path, transforms=transforms)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    #model

    #model = EfficientNet.from_name('efficientnet-b0', num_classes=n_class)
    model = torchvision.models.resnet18(weights=None, num_classes=n_class)
    model.load_state_dict(torch.load(weights_path))
    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # Loss function
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    test_acc, test_loss,_,_,_ = test(test_loader, model, device, loss_fn)
    print(f'test -> test_acc: {test_acc:.4f}, test_loss: {test_loss:.4f}')