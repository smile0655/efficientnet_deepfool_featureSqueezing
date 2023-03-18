from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from collections import Counter
import torchvision.transforms

def default_loader(path):
    # (C, H, W)格式
    return Image.open(path).convert('RGB')

class DatasetFromCSV(Dataset):
    def __init__(self, img_path, csv_path, header=0, transforms=None, loader=default_loader):
        # header=0表示读进来的是第一行是表头的数据
        self.dataframe = pd.read_csv(csv_path, header=header)
        # 读取第2列作为标签
        self.labels = np.array(self.dataframe.iloc[:, 1])
        # 读取图片名
        self.image_path = img_path
        imgs = []
        self.img_names = np.array(self.dataframe.iloc[:,0])
        for img in self.img_names:
            imgs.append(os.path.join(self.image_path, str(img)))
        self.images = np.array(imgs)

        self.transforms = transforms
        self.loader = loader

    def __getitem__(self, index):
        label = self.labels[index]
        img = self.images[index] #这里只是路径+图片名，还不是图片数据本身
        img = self.loader(img) #读取图片数据
        if self.transforms is not None:
            img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.images)

def compute_mean_std(img_path,csv_train_path):

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    dataset = DatasetFromCSV(img_path, csv_train_path, transforms=transforms)
    # 图片数量
    num_img = len(dataset)
    # 计算RGB3个颜色位各自的mean和std
    mean = [0.0, 0.0, 0.0]
    std = [0.0, 0.0, 0.0]
    for data in dataset:
        img = data[0]
        for i in range(3):
            mean[i] += img[i, :, :].mean()
            std[i] += img[i, :, :].std()

    mean = np.asarray(mean) / num_img
    std = np.asarray(std) / num_img
    return mean, std

def partition_data(input_file, ratios, out_path, uniform=False, num_seed=None,
    shuffle=True, skip_header=1):
    """
    Partition data into training, validation and test data.
    Saves partitioned data to three .csv files.
    """

    # Define ratios for traning, validation and test
    ratio_train = ratios[0]
    ratio_validation = ratios[1]
    ratio_test = ratios[2]

    # Import data
    data = np.genfromtxt(input_file, skip_header=skip_header, dtype='str', delimiter=',')

    # Divide by image and label
    imgs = data[:,0]
    lbls = data[:,1]

    if shuffle:
        np.random.seed(num_seed)
        idxs_shuffle = np.random.permutation(len(lbls))
    else:
        idxs_shuffle = np.arange(len(lbls))
    imgs = imgs[idxs_shuffle]
    lbls = lbls[idxs_shuffle]

    # Print info
    print('NUMBER OF SAMPLES PER CLASS')
    samples_per_class(lbls)
    print('')

    # If uniformly many classes (do not use all data however)
    if uniform:
        # Find indices of each class
        idxs0 = np.where(lbls=='0')
        idxs1 = np.where(lbls=='1')
        idxs2 = np.where(lbls=='2')

        # Find number of samples for each class
        N0 = len(idxs0[0])
        N1 = len(idxs1[0])
        N2 = len(idxs2[0])
        N = min(N0, N1, N2)

        # Divide data by classes
        imgs0 = imgs[idxs0]
        lbls0 = lbls[idxs0]
        imgs1 = imgs[idxs1]
        lbls1 = lbls[idxs1]
        imgs2 = imgs[idxs2]
        lbls2 = lbls[idxs2]

        # Shuffle order
        idxs_shuffle0 = np.random.permutation(N0)
        idxs_shuffle1 = np.random.permutation(N1)
        idxs_shuffle2 = np.random.permutation(N2)
        imgs0 = imgs0[idxs_shuffle0]
        lbls0 = lbls0[idxs_shuffle0]
        imgs1 = imgs1[idxs_shuffle1]
        lbls1 = lbls1[idxs_shuffle1]
        imgs2 = imgs2[idxs_shuffle2]
        lbls2 = lbls2[idxs_shuffle2]

        # Take samples from each class
        imgs = []
        lbls = []
        for n in range(N):
            imgs.append(imgs0[n])
            lbls.append(lbls0[n])
            imgs.append(imgs1[n])
            lbls.append(lbls1[n])
            imgs.append(imgs2[n])
            lbls.append(lbls2[n])

    # Divide into training, validation and test data sets
    #end index of training data
    idxs_train = int(ratio_train*len(imgs))
    #end index of validation data
    idxs_validation = int((ratio_train+ratio_validation)*len(imgs))
    #training images
    imgs_train = imgs[0:idxs_train]
    #validation images
    imgs_validation = imgs[idxs_train:idxs_validation]
    #test images starting from end of validation index
    imgs_test = imgs[idxs_validation:]
    #training labels
    lbls_train = lbls[0:idxs_train]
    #validation labels
    lbls_validation = lbls[idxs_train:idxs_validation]
    #test labels
    lbls_test = lbls[idxs_validation:]

    # If path does not exist, create it
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Save to files
    np.savetxt(out_path+'data_labels_train.csv', np.vstack((imgs_train, lbls_train)).T, delimiter=',', fmt='%s')
    np.savetxt(out_path+'data_labels_validation.csv', np.vstack((imgs_validation, lbls_validation)).T, delimiter=',', fmt='%s')
    np.savetxt(out_path+'data_labels_test.csv', np.vstack((imgs_test, lbls_test)).T, delimiter=',', fmt='%s')

    # Print information
    print('PARTITIONED DATA')
    print('  Training:    ' + str(int(100*ratios[0]))+'%, ' + str(len(lbls_train)) + ' samples')
    samples_per_class(lbls_train)
    print('')
    print('  Validation:  ' + str(int(100*ratios[1]))+'%, ' + str(len(lbls_validation)) + ' samples')
    samples_per_class(lbls_validation)
    print('')
    print('  Testing:     ' + str(int(100*ratios[2]))+'%, ' + str(len(lbls_test)) + ' samples')
    samples_per_class(lbls_test)
    print('')
    print('  Total:       ' + str(len(lbls_train)+len(lbls_validation)+len(lbls_test)))

def samples_per_class(lbls):
    """
    Helper function to plot samples in each class for specific array of labels.
    """
    counter = Counter(lbls)
    keys = counter.keys()
    values = counter.values()
    pairs = sorted(zip(keys, values))
    sum = 0
    output_format = '    Class {:<1}: {:>5}   {:>3}%'
    for pair in pairs:
        sum += pair[1]
    for pair in pairs:
        print(output_format.format(pair[0], pair[1], int(100*pair[1]/sum)))
    print('    Total:   {:>5}'.format(sum))


def data_partition(in_csv_root,csv_name,out_csv_root,valid_size,test_size):
    csv_path = os.path.join(in_csv_root,csv_name)
    df = pd.read_csv(csv_path)
    img = df.iloc[:,0].values
    label = df.iloc[:,1].values
    if valid_size ==0:
        train_img, test_img, train_label, test_label = train_test_split(img, label, test_size=test_size,
                                                                       random_state=25)

    else:
        train_img, temp_img,train_label,tmp_label = train_test_split(img,label, test_size=test_size+valid_size, random_state=25)
        val_img, test_img, val_label, test_label = train_test_split(temp_img, tmp_label,test_size=test_size/(test_size+valid_size), random_state=25)

        val_data = np.vstack((val_img, val_label)).T
        val_data = pd.DataFrame(val_data, columns=['image_name', 'label'])
        val_path = os.path.join(out_csv_root, 'valid_data.csv')
        val_data.to_csv(val_path, index=False)  # 保存为csv的时候不要最左边的列索引

    train_data = np.vstack((train_img, train_label)).T
    train_data = pd.DataFrame(train_data, columns=['image_name', 'label'])
    train_path = os.path.join(out_csv_root, 'train_data.csv')
    train_data.to_csv(train_path, index=False)  # 保存为csv的时候不要最左边的列索引

    test_data = np.vstack((test_img, test_label)).T
    test_data = pd.DataFrame(test_data, columns=['image_name', 'label'])
    test_path = os.path.join(out_csv_root, 'test_data.csv')
    test_data.to_csv(test_path, index=False)  # 保存为csv的时候不要最左边的列索引


