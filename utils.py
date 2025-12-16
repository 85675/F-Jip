import numpy as np
import random

import torchvision.models
from tqdm import tqdm
import pandas as pd
import torch
from torchvision import datasets, models, transforms
import torch, gc
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score, roc_curve, recall_score, accuracy_score, auc, f1_score, \
    classification_report, precision_score, confusion_matrix
import torch.nn.functional as F
from mdgcu118factmodeldense import *
from Resnet import *
from sklearn.metrics import roc_auc_score, roc_curve, recall_score, accuracy_score, auc
from torchvision import models




def load_model(model_name, pretrain=True, require_grad=True):
    print('==> Building model..')
    if model_name == 'resnet50_pmg':
        net = resnet50(pretrained=True)
        for param in net.parameters():
            param.requires_grad = require_grad
        net = PMG(net, 256, 200)

    return net


def test(test_loader,net,criterion):
    net.eval()
    use_cuda = torch.cuda.is_available()
    test_target = []
    test_data_predict = []
    test_data_predict_proba = []
    # outfeatures = []
    device = torch.device("cuda:3")
    with torch.no_grad():
        for (inputs1, targets, name) in tqdm(test_loader):
            # print(name)
            if use_cuda:
                inputs1, targets = inputs1.to(device), targets.to(device)
            output_concat,_,_ = net(inputs1)
            data1 = F.softmax(output_concat, dim=1)
            test_data_predict_proba.extend(data1[:, 1].detach().cpu().numpy())
            _, predicted = torch.max(output_concat.data, 1)
            test_target.extend(targets.data.cpu().numpy())
            test_data_predict.extend(predicted.cpu().numpy())

        test_data_confusion_matrix = confusion_matrix(test_target, test_data_predict)

        M = test_data_confusion_matrix
        print(test_data_confusion_matrix)

        test_data_accuracy_score = accuracy_score(test_target, test_data_predict)
        print(test_data_accuracy_score)

        sensitity = M[1, 1] / (M[1, 1] + M[1, 0])
        print(sensitity)

        specificity = M[0, 0] / (M[0, 1] + M[0, 0])
        print(specificity)

        test_data_f1_score = f1_score(test_target, test_data_predict)
        print(test_data_f1_score)

        test_data_precision_score = precision_score(test_target, test_data_predict)
        print(test_data_precision_score)

        FPR_test_data, TPR_test_data, threshold_test_data = roc_curve(test_target, test_data_predict_proba)
        test_data_roc_auc = auc(FPR_test_data, TPR_test_data)
        print(test_data_roc_auc)
        y = TPR_test_data - FPR_test_data
        Youden_index = np.argmax(y)  # Only the first occurrence is returned.
        optimal_threshold = threshold_test_data[Youden_index]

        return test_data_roc_auc

