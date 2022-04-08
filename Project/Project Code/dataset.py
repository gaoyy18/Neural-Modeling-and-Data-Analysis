import pywt, os, copy
import torch
import numpy as np
import pandas as pd
from config import config
from torch.utils.data import Dataset
from sklearn.preprocessing import scale
from scipy import signal




def scaling(X, sigma=0.1):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1, X.shape[1]))
    myNoise = np.matmul(np.ones((X.shape[0], 1)), scalingFactor)
    return X * myNoise

def verflip(sig):
    '''
    信号竖直翻转
    :param sig:
    :return:
    '''
    return sig[::-1, :]

def shift(sig, interval=20):
    '''
    上下平移
    :param sig:
    :return:
    '''
    for col in range(sig.shape[1]):
        offset = np.random.choice(range(-interval, interval))
        sig[:, col] += offset
    return sig


def transform(sig, train=False):
    # # 数据增强
    if train:
        if np.random.randn() > 0.5: sig = scaling(sig)
        # if np.random.randn() > 0.5: sig = verflip(sig)
        if np.random.randn() > 0.5: sig = shift(sig)
    # 后置不可或缺的步骤
    sig = sig.transpose()
    sig = torch.tensor(sig.copy(), dtype=torch.float)
    return sig

def transform_age(age):
    if age == 'SECOND':
        return -1.5
    elif age == 'THIRD':
        return -0.5
    elif age == 'FOURTH':
        return 0.5
    elif age == 'FIFTH':
        return 1.5
    else:
        return 0


def transform_gender(gender):
    if gender == 'FEMALE':
        return -1
    elif gender == 'MALE':
        return 1
    else:
        return 0


class ECGDataset(Dataset):
    """
    A generic data loader where the samples are arranged in this way:
    dd = {'train': train, 'val': val, 'test': test, 'idx2name': idx2name, 'file2idx': file2idx, 'file2ag': file2ag, 'file2gen': file2gen, 'wc':wc_data}
    Initiate the dataset with the following bool parameters: train, test
    """

    def __init__(self, train=True, test=False):
        super(ECGDataset, self).__init__()
        dd = torch.load(config.data)
        self.train = train
        self.test = test
        if test:
            self.data = dd['test']
        else:
            self.data = dd['train'] if train else dd['val']
        self.idx2name = dd['idx2name']
        self.file2idx = dd['file2idx']
        self.file2ag = dd['file2ag']
        self.file2gen = dd['file2gen']
        self.wc = 1. / np.log(dd['wc'])

    def __getitem__(self, index):
        fid = self.data[index]
        file_path = os.path.join(config.data_dir, fid)
        df = pd.read_csv(file_path, sep=' ').values
        x = transform(df, self.train)

        target = np.zeros(config.num_classes)
        target[self.file2idx[fid]] = 1
        target = torch.tensor(target, dtype=torch.float)

        age = self.file2ag[fid]
        age = transform_age(age)
        age = torch.tensor(age, dtype=torch.float)

        gender = self.file2gen[fid]
        gender = transform_gender(gender)
        gender = torch.tensor(gender, dtype=torch.float)
        return x, age, gender, target

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    d = ECGDataset(config.data)
    # x(8*5000), age(1), gender(1), target(34, in one-hot)
    print(d[2])
    # 各个标签用于计算loss时的weights，标签对应的数据量越小，weight应该越大
    print(d.wc)
