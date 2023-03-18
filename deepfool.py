from torchvision.utils import save_image
from torch.autograd import Variable
import torch as torch
import copy
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
from torch.utils.tensorboard import SummaryWriter
import time
import os
from torch_dataset import DatasetFromCSV
import pandas as pd
import numpy as np
from models.efficientnet import EfficientNet
from func_train_val_test import train,valid
from albumentations.pytorch import ToTensorV2
import gc
from datetime import datetime
import nvidia_smi
import collections
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import csv

def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, collections.abc.Iterable):
        for elem in x:
            zero_gradients(elem)

def clip_tensor(A, minv, maxv):
    # 用于限制tensor的值在min和max之间。如果tensor中元素的值小于min，则设置该元素值为min，如果tensor中元素的值大于max，则设置该元素值为max。
    # 这里用来把图片的值限制在0-255之间
    A = torch.max(A, minv*torch.ones(A.shape))
    A = torch.min(A, maxv*torch.ones(A.shape))
    return A

def similarity(images, attacked_images,mean,std):
    # images:（b,c,w,h),attacked_images:(c,w,h) or (b,c,w,h)

    if images.shape != attacked_images.shape:
        attacked_images = attacked_images[None, :, :, :]
    similarities = np.zeros(images.shape[0])
    data_min = np.divide(-mean, std)
    data_max = np.divide(mean, std)
    data_range = (data_max- data_min).max()

    for i, (image, attacked_image) in enumerate(zip(images, attacked_images)):

        similarities[i] = ssim(im1=attacked_image.permute(1,2,0).numpy(),
                                im2=image.permute(1,2,0).numpy(),
                                channel_axis=2,
                                data_range=data_range.item())
    return similarities


def deepfool(image, model,num_classes=3, overshoot=0.02, max_iter=10):

    data_mean = torch.tensor([0.7750, 0.5888, 0.7629])
    data_std = torch.tensor([0.2129, 0.2971, 0.1774])
    #输进模型的transform
    in_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_mean, std=data_std),
    ])
    clip = lambda x: clip_tensor(x, 0, 255)
    #输出图像的transform
    out_transform = transforms.Compose([
        transforms.Normalize(mean=[0, 0, 0], std=list(map(lambda x: 1 / x, data_std))),
        transforms.Normalize(mean=list(map(lambda x: -x, data_mean)), std=[1, 1, 1]),
        transforms.Lambda(clip),
        transforms.ToPILImage(),
    ])

    image = torch.squeeze(image) # squeeze to 3*224*224
    f_image = model.forward(Variable(image[None, :, :, :], requires_grad=True)).data.numpy().flatten() #模型的输出
    I = (np.array(f_image)).flatten().argsort()[::-1]
    I = I[0:num_classes]
    pred_label = I[0] #原图的预测标签

    input_shape = image.numpy().shape # (C,W,H)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    pert_image = copy.deepcopy(image)
    pert_image = Variable(pert_image[None, :, :, :], requires_grad=True)
    pred = model(pert_image)

    loop_i = 0
    pert_label = pred_label

    while pert_label == pred_label and loop_i < max_iter:
        pert = np.inf
        # 求出输出向量中预测标签对应的值对图片的梯度
        pred[0, I[0]].backward(retain_graph=True)
        grad_f_0 = pert_image.grad.data.numpy().copy()

        for k in range(1,num_classes):
            zero_gradients(pert_image)

            # 求出输出向量的各个值对图片的梯度（除了预测的标签对应的值）
            pred[0, I[k]].backward(retain_graph=True)
            grad_f_k = pert_image.grad.data.numpy().copy()

            grad_diff = grad_f_k - grad_f_0
            f_diff=(pred[0, I[k]] - pred[0, I[0]]).data.numpy()
            pert_k = abs(f_diff) / np.linalg.norm(grad_diff.flatten())
            if pert_k < pert:
                pert = pert_k
                w = grad_diff

        r_i = (pert + 1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)  # r_tot是每次循环累积的
        pert_image_tmp = image + (1 + overshoot) * torch.from_numpy(r_tot)
        '''
        pert_image_tmp = out_transform(pert_image_tmp[0]) # 第一维是batch
        pert_image_tmp = Image.fromarray((np.array(pert_image_tmp)), 'RGB') # array to image format
        pert_image_quantized = in_transform(pert_image_tmp)
        pert_image = Variable(pert_image_quantized[None, :], requires_grad=True)
        '''
        pert_image_quantized = pert_image_tmp[0]
        pert_image = Variable(pert_image_quantized[None, :], requires_grad=True)
        pred = model(pert_image)
        pert_label = np.argmax(pred.data.numpy().flatten())

        loop_i += 1

    r_tot = (1 + overshoot) * r_tot
    return r_tot, loop_i, pred_label.item(), pert_label.item(), pert_image_quantized

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type = int, default=1)
    parser.add_argument('--n_class', type=int, default=3)
    parser.add_argument('--overshoot', type=int, default=0.02, help='overshoot')
    parser.add_argument('--max_iter', type=int, default=10, help='max_iter')
    args = parser.parse_args()

    batch_size = args.batch_size
    n_class = args.n_class
    overshoot = args.overshoot
    max_iter = args.max_iter

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #model = EfficientNet.from_name('efficientnet-b0', num_classes=n_class)
    model = torchvision.models.resnet18(weights=None, num_classes=n_class)
    model.load_state_dict(torch.load('./checkpoints/resnet18/weights/2023-02-13T03-25-38/best_24.pt'))
    #model.to(device)

    # Switch to evaluation mode
    model.eval()

    data_mean = torch.tensor([0.7750, 0.5888, 0.7629])
    data_std = torch.tensor([0.2129, 0.2971, 0.1774])
    in_transforms = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
        transforms.Normalize(mean=data_mean, std=data_std),
    ])
    clip = lambda x: clip_tensor(x, 0, 255)
    out_transform = transforms.Compose([
        transforms.Normalize(mean=[0, 0, 0], std=list(map(lambda x: 1 / x, data_std))),
        transforms.Normalize(mean=list(map(lambda x: -x, data_mean)), std=[1, 1, 1]),
        transforms.Lambda(clip),
        transforms.ToPILImage(),
    ])
    img_path = "./data/medical_data/data_org_image"
    csv_test_path = './data/medical_data/medical_csv/test_data.csv'
    test_dataset = DatasetFromCSV(img_path=img_path, csv_path=csv_test_path, transforms=in_transforms)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    img_names = test_dataset.img_names

    fool_count = 0
    loop_sum = 0
    simi_sum = 0
    sim = []

    save_csv_path = './data/medical_data/medical_csv/attacked_data_label_resnet18.csv'
    with open(save_csv_path,mode='w', newline='') as f:
        csv_head = ['image_name','label']
        writer = csv.DictWriter(f, fieldnames=csv_head)
        writer.writeheader()

    for idx, (image, label) in enumerate(test_loader):
        #image = image.to(device)
        r, loop_i, label_orig, label_pert, pert_image = deepfool(image, model)

        with open(save_csv_path, mode='a', newline='') as f:
            csv_write = csv.writer(f)
            data_row = [str(img_names[idx])+'pert.png', label_pert]
            csv_write.writerow(data_row)

        simi = similarity(image.detach(), pert_image.detach(),data_mean,data_std)
        print(f'similarities: {simi}')
        sim.append(simi)
        simi_sum += simi.sum()
        pert_image = out_transform(pert_image)

        pert_file = './data/medical_data/data_pert_image_resnet18/'+str(img_names[idx])+'pert.png'
        Image.fromarray((np.array(pert_image)), 'RGB').save(pert_file)
        loop_sum += loop_i
        if label_pert != label_orig:
            fool_count = fool_count + 1
        print(idx)
        print(f'predicted label:{label_orig}------attacked label:{label_pert}')
        print(f'loop number:{loop_i}')

    fool_ratio = fool_count/len(test_dataset)
    mean_loop = loop_sum/len(test_dataset)
    mean_similarity = simi_sum/len(test_dataset)
    print(f'fool ratio:{fool_ratio}')
    print(f'mean loop number:{mean_loop}')
    print(f'mean similarity:{mean_similarity}')

