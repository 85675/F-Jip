from __future__ import print_function
import os
import random
from typing import Tuple, Type
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn.metrics
import torch
import torchvision.models
from PIL import Image
from scipy.spatial.distance import cdist
from torch import autograd
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
import cv2
import math
from tqdm import tqdm
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import copy
from mdgcu118factutilsdense import *
import itertools

class DCNNstu(nn.Module):
    def __init__(self, model1):
        super(DCNNstu, self).__init__()
        self.resnet1 = model1
        self.linear1 = torch.nn.Linear(256, 2)
        self.linear3 = torch.nn.Linear(256, 30)
    def forward(self, x1):
        xc1, xc3 = self.resnet1(x1)

        return F.leaky_relu(self.linear1(xc1)),F.leaky_relu(self.linear3(xc3)),xc1


def AMP(path1,path2,path3,path4,path5):
    path = [path1,path2,path3,path4,path5]
    amp = []
    for p in path:
        filelist1 = os.listdir(p)
        amp1 = []
        sum = 0
        for i in range(0, len(filelist1)):
            img_path = p + '/' + str(filelist1[i])
            img = os.listdir(img_path)
            img.sort()
            if str(filelist1[i]) == "benign":
                for ii in range(0, 50):
                    trg_img1 = cv2.imread(img_path+'/'+str(img[ii]))
                    # trg_img1 = cv2.resize(trg_img1, (512, 512), interpolation=cv2.INTER_CUBIC)
                    fft_trg_np1 = np.fft.fftn(trg_img1)
                    amp_trg1, pha_trg1 = np.abs(fft_trg_np1), np.angle(fft_trg_np1)
                    a_trg1 = np.fft.fftshift(amp_trg1, axes=(0, 1))
                    amp1.append(a_trg1)
            else:
                for ii in range(0, 50):
                    trg_img1 = cv2.imread(img_path+'/'+str(img[ii]))
                    # trg_img1 = cv2.resize(trg_img1, (512, 512), interpolation=cv2.INTER_CUBIC)
                    fft_trg_np1 = np.fft.fftn(trg_img1)
                    amp_trg1, pha_trg1 = np.abs(fft_trg_np1), np.angle(fft_trg_np1)
                    a_trg1 = np.fft.fftshift(amp_trg1, axes=(0, 1))
                    amp1.append(a_trg1)

        for n in range(100):
            sum = sum + amp1[n]
        mamp = sum/100
        amp.append(mamp)

    return amp

def jigsaw_generator(aug, n,bool,pset):
    l = []
    for a in range(n):
        for b in range(n):
            l.append([a, b, n*a+b])
    block_size = 512 // n
    rounds = n ** 2
    ord = l.copy()
    # random.shuffle(l)
    temp1 = np.zeros([512,512])
    style = []
    order = np.random.randint(0, len(pset), 1)
    random.shuffle(aug)
    for i in range(rounds):
        x, y, o = l[i]
        m = np.random.randint(0,3,1)
        style.append(m)
        augz = aug[int(m)][:,:,0]
        temp1[..., x * block_size:(x + 1) * block_size, y * block_size:(y + 1) * block_size] = augz[..., x * block_size:(x + 1) * block_size, y * block_size:(y + 1) * block_size]
    jigsaws = temp1.copy()
    style = np.unique(style)
    if bool == True:
        for i in range(rounds):
            x, y, o = l[i]
            x1, y1, o1 = ord[pset[int(order)][i]]
            jigsaws[..., x * block_size:(x + 1) * block_size, y * block_size:(y + 1) * block_size] = temp1[..., x1 * block_size:(x1 + 1) * block_size,
                                                       y1 * block_size:(y1 + 1) * block_size]
    return jigsaws,len(style)-1,int(order)

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=0.25):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +  # calmp夹断用法
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive



class MyDatasettrainw(Dataset):
    def __init__(self, data_dir_train_E,data_dir_train_E1,data_dir_train_E2,data_dir_train_E3,data_dir_train_E4, transform=None):
        super(MyDatasettrainw, self).__init__()
        all_dir_E = list()
        for path in [data_dir_train_E,data_dir_train_E1,data_dir_train_E2,data_dir_train_E3,data_dir_train_E4]:
            filelist1 = os.listdir(path)
            for i in range(0,len(filelist1)):
                img_path = path + '/' + str(filelist1[i])
                img = os.listdir(img_path)
                img.sort()
                number = len(img)
                if str(filelist1[i]) == "benign":
                    for ii in range(0, number):
                        pic_path_b1 = img_path + '/' + str(img[ii])
                        all_dir_E.append((pic_path_b1, 0))
                else:
                    for ii in range(0, number):
                        pic_path_b1 = img_path + '/' + str(img[ii])
                        all_dir_E.append((pic_path_b1, 1))
        self.sample_list_E = all_dir_E
        self.transform = transform

    def __len__(self):
        return len(self.sample_list_E)

    def __getitem__(self, index):

        item_E1,  label = self.sample_list_E[index]
        image_E1 = cv2.imread(item_E1)
        image_E1 = Image.fromarray(image_E1)
        return image_E1,label,item_E1


class MyDatasettrain(Dataset):
    def __init__(self,dataset,transform2):
        self.dataset = dataset
        self.transform2 = transform2

    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        img,label,name=self.dataset[idx]
        img2,style,order = self.transform2(img)
        return img2,style,order,label,name


class MAug(object):
    def __init__(self, amp,pset):
        self.amp = amp
        self.pset = pset

    def __call__(self, x):
        h1 = 246
        h2 = 266
        w1 = 246
        w2 = 266
        xp = np.array(x)
        x = Image.fromarray(xp)
        fft_src_np = np.fft.fftn(x)
        amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
        a_src = np.fft.fftshift(amp_src, axes=(0, 1))
        aug = []
        for n in range(0, len(self.amp)):
            change_src = a_src.copy()
            sample = np.random.beta(0.5, 0.5)
            change_src[h1:h2, w1:w2] = sample * change_src[h1:h2, w1:w2] + (1 - sample) * self.amp[n][h1:h2, w1:w2]
            change_src = np.fft.ifftshift(change_src,  axes=(0, 1))
            aug_img0 = np.abs(change_src) * np.exp((1j) * pha_src)
            aug_img0 = np.fft.ifftn(aug_img0)
            aug_img0 = np.real(aug_img0)
            norm_image = cv2.normalize(aug_img0, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            aug_img0 = norm_image * 255
            aug_img0 = aug_img0.astype(np.uint8)
            aug_img0 = np.where(np.array(x) == (np.array(x)).min(), 0, aug_img0)
            aug.append(aug_img0)
        x1, style1, order = jigsaw_generator(aug, 3, True, self.pset)
        x1 = np.expand_dims(x1, 2).repeat(3, axis=2)
        x1 = x1.astype(np.uint8)
        x = [xp, x1]
        style = [0, style1]
        order = [0, order]
        for m in range(len(x)):
            # x[m] = x[m].transpose((2,0,1))
            x[m] = Image.fromarray(x[m])
            x[m] = transforms.ToTensor()(x[m])

        return x,style,order


def hamming_loss(preds, targets):
    # 计算Hamming Loss
    hamming_loss = 1 - (preds == targets).float().mean()
    return hamming_loss



def loadData_train(train_dataset, batch_size, shuffle=False):
    data_transform_trainaug = transforms.Compose([
        transforms.ColorJitter(brightness=[0, 2], contrast=[0, 2],
                               saturation=[0, 2], hue=[-0.5,0.5]),
        MAug(amp,pset),
    ])
    dataset = MyDatasettrain(train_dataset, transform2=data_transform_trainaug)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16,pin_memory=True)

def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
seed_it(2028)

pset = [[1 ,3 ,6 ,7 ,0 ,2 ,8 ,5 ,4] ,[2 ,0 ,1 ,4 ,3 ,6 ,5 ,8 ,7] ,[3 ,2 ,0 ,1 ,5 ,8 ,7 ,4 ,6] ,[4 ,5 ,3 ,0 ,8 ,7 ,1 ,6 ,2] ,[5 ,4 ,7 ,8 ,6 ,0 ,2 ,1 ,3] ,[6 ,7 ,8 ,2 ,1 ,3 ,4 ,0 ,5] ,[7 ,8 ,5 ,6 ,2 ,4 ,0 ,3 ,1] ,[8 ,6 ,4 ,5 ,7 ,1 ,3 ,2 ,0] ,[0 ,1 ,2 ,3 ,6 ,4 ,7 ,8 ,5] ,[1 ,0 ,3 ,6 ,7 ,5 ,4 ,2 ,8] ,[2 ,3 ,4 ,5 ,0 ,8 ,6 ,7 ,1] ,[3 ,2 ,8 ,4 ,5 ,7 ,0 ,1 ,6] ,[4 ,5 ,0 ,8 ,2 ,1 ,3 ,6 ,7] ,[5 ,4 ,7 ,2 ,8 ,6 ,1 ,3 ,0] ,[8 ,6 ,1 ,7 ,4 ,2 ,5 ,0 ,3] ,[6 ,7 ,5 ,0 ,1 ,3 ,8 ,4 ,2] ,[7 ,8 ,6 ,1 ,3 ,0 ,2 ,5 ,4] ,[0 ,1 ,4 ,2 ,5 ,6 ,3 ,7 ,8] ,[1 ,0 ,7 ,4 ,2 ,3 ,5 ,8 ,6] ,[2 ,3 ,0 ,6 ,7 ,1 ,8 ,4 ,5] ,[3 ,4 ,2 ,5 ,1 ,8 ,0 ,6 ,7] ,[4 ,5 ,8 ,3 ,6 ,7 ,1 ,2 ,0] ,[5 ,2 ,3 ,7 ,8 ,0 ,6 ,1 ,4] ,[6 ,7 ,1 ,8 ,0 ,4 ,2 ,5 ,3] ,[7 ,8 ,6 ,1 ,3 ,5 ,4 ,0 ,2] ,[8 ,6 ,5 ,0 ,4 ,2 ,7 ,3 ,1] ,[0 ,1 ,4 ,2 ,7 ,5 ,8 ,3 ,6] ,[1 ,0 ,7 ,5 ,8 ,4 ,3 ,6 ,2] ,[2 ,3 ,5 ,0 ,6 ,1 ,4 ,8 ,7],[0 ,1 ,2 ,3 ,4 ,5 ,6 ,7 ,8]]


data_dir_train_E4 = 'source domain4'
data_dir_train_E2 = 'source domain2'
data_dir_train_E = 'source domain'
data_dir_train_E1 = 'source domain1'
data_dir_train_E3 = 'source domain3'

amp = AMP("source domain","source domain1","source domain2","source domain3","source domain4")
dataset = MyDatasettrainw(data_dir_train_E,data_dir_train_E1,data_dir_train_E2,data_dir_train_E3,data_dir_train_E4)

train_size = int(len(dataset) * 0.8)
validate_size = int(len(dataset))-train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, validate_size],generator=torch.Generator().manual_seed(6))
train_loader = loadData_train(train_dataset, 15, shuffle=True)


dataset_sizes_train = train_loader.dataset.__len__()
print('dataset_sizes_train',dataset_sizes_train)

def train():
    use_cuda = torch.cuda.is_available()
    print(use_cuda)

    # Data
    print('==> Preparing data..')

    net2 = load_model(model_name='resnet50_pmg', pretrain=True, require_grad=True)
    netstu = DCNNstu(net2)
    device = torch.device("cuda:3")
    netstu.to(device)

    CELoss = nn.CrossEntropyLoss()

    optimizertea5 = optim.Adam([
        {'params': netstu.parameters(), 'lr': 1e-5},
    ], weight_decay=1e-4)

    max_stu_acc = 0
    e = 0
    nepochs = []
    scaler1 = GradScaler()

    for epoch in range(20):
        print('\nEpoch: %d' % epoch)
        netstu.train()
        train_loss = 0
        test_target= []
        order_target = []
        test_data_predict = []
        order_predict = []
        test_data_predict_proba = []
        for inputs, style, order, targets, name in tqdm(train_loader):
            inputs1 = inputs[0]
            inputs2 = inputs[1]
            styleori = style[0]
            styleaug = style[1]
            orderori = order[0]
            orderaug = order[1]

            inputs1,inputs2,styleori,styleaug,orderori,orderaug,targets = inputs1.to(device),inputs2.to(device),styleori.to(device),styleaug.to(device),orderori.to(device),orderaug.to(device),targets.to(device)
            inputs = torch.cat([inputs1,inputs2], dim=0)

            optimizertea5.zero_grad()
            with autocast():
                output,oporder, output_feature = netstu(inputs)
                output_ori, output_aug = torch.split(output, int(output.shape[0]/2))
                output_orderori, output_orderaug = torch.split(oporder, int(oporder.shape[0] / 2))

                oriceloss = CELoss(output_ori, targets)
                augceloss = CELoss(output_aug, targets)
                oraugloss = CELoss(output_orderaug, orderaug)

            scaler1.scale(oriceloss + augceloss + oraugloss).backward()
            scaler1.step(optimizertea5)
            scaler1.update()


            #  training log
            data1 = F.softmax(output_ori, dim=1)
            _, predicted = torch.max(output_ori.data, 1)
            train_loss = train_loss + (oriceloss).item()
            test_data_predict.extend(predicted.cpu().numpy())
            test_data_predict_proba.extend(data1[:, 1].detach().cpu().numpy())
            test_target.extend(targets.data.cpu().numpy())
        # scheduler.step()
        epoch_loss = train_loss / train_loader.dataset.__len__()
        print('loss:',epoch_loss)
        test_data_confusion_matrix = confusion_matrix(test_target, test_data_predict)
        M = test_data_confusion_matrix
        print(test_data_confusion_matrix)

        test_data_accuracy_score = accuracy_score(test_target, test_data_predict)
        print(test_data_accuracy_score)
        order_accuracy_score = accuracy_score(order_target, order_predict)
        print(order_accuracy_score)


        sensitity = M[1, 1] / (M[1, 1] + M[1, 0])
        print(sensitity)

        specificity = M[0, 0] / (M[0, 1] + M[0, 0])
        print(specificity)

        FPR_test_data, TPR_test_data, threshold_test_data = roc_curve(test_target, test_data_predict_proba)
        test_data_roc_auc = auc(FPR_test_data, TPR_test_data)
        print(test_data_roc_auc)

        nepochs.append(epoch)




train()
