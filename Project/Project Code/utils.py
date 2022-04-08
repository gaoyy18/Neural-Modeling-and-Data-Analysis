
import torch
import numpy as np
import time,os
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from torch import nn


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


#计算F1score
def calc_f1(y_true, y_pre, threshold=0.5):
    y_true = y_true.view(-1).cpu().detach().numpy().astype(np.int)
    y_pre = y_pre.view(-1).cpu().detach().numpy() > threshold
    return f1_score(y_true, y_pre)


#计算Accuracy, precision, Recall, F1score
def calc_score(y_true, y_pre, threshold=0.5):
    size=y_true.size()
    label_size = size[1]
    batch_size = size[0]

    y_true1 = y_true.cpu().detach().numpy().astype(np.int)
    y_pre1 = y_pre.cpu().detach().numpy() > threshold
    acc = np.sum(np.sum(y_pre1==y_true1, axis=1)==label_size)/batch_size

    y_true = y_true.view(-1).cpu().detach().numpy().astype(np.int)
    y_pre = y_pre.view(-1).cpu().detach().numpy() > threshold
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pre,
                                                 labels=None,
                                                 pos_label=1,
                                                 average='binary',
                                                 warn_for=('precision','recall','f-score'),
                                                 sample_weight=None,
                                                 zero_division="warn")
    # acc = accuracy_score(y_true, y_pre)
    
    return acc, p, r, f1


#打印时间
def print_time_cost(since):
    time_elapsed = time.time() - since
    return '{:.0f}m{:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60)


# 调整学习率
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


