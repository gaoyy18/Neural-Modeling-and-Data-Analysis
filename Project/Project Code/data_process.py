import os, torch, math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from config import config

# 保证每次划分数据一致，请不要更改
np.random.seed(28)


def name2index(path):
    '''
    把类别名称转换为index索引
    :param path: 文件路径
    :return: 字典
    '''
    list_name = []
    for line in open(path, encoding='utf-8'):
        list_name.append(line.strip())
    name2indx = {name: i for i, name in enumerate(list_name)}
    # print(name2indx)
    return name2indx


def file2index(path, name2idx):
    '''
    获取文件id对应的标签类别
    :param path:文件路径
    :return:文件id对应label列表的字段
    '''
    file2index = dict()
    for line in open(path, encoding='utf-8'):
        arr = line.strip().split('\t')
        id = arr[0]
        labels = [name2idx[name] for name in arr[3:]]
        # print(id, labels)
        file2index[id] = labels
    # print(file2index)
    return file2index

def file2age(path, name2idx):
    
    file2age = dict()
    for line in open(path, encoding='utf-8'):
        arr = line.strip().split('\t')
        id = arr[0]
        ages = arr[1]
        
        file2age[id] = ages
    # print(file2age)
    return file2age

def file2gender(path, name2idx):
    
    file2gender = dict()
    for line in open(path, encoding='utf-8'):
        arr = line.strip().split('\t')
        id = arr[0]
        genders = arr[2]
        
        file2gender[id] = genders
    # print(file2gender)
    return file2gender


def index2file(file2idx):
    '''
    标签类别对应的文件id
    :return:label列表的字段对应的所有文件id
    '''
    idx2file = [[] for _ in range(config.num_classes)]  # list
    
    for file, list_idx in file2idx.items():
        for idx in list_idx:
            idx2file[idx].append(file)

    print(idx2file)  
    return idx2file


def split_data(file2idx, val_ratio=0.1, test_ratio=0.2):
    '''
    划分数据集,train,val,test需保证每类至少有1个样本
    :param file2idx:
    :param val_ratio, test_ratio:验证集，测试集占总数据的比例
    :return:所有数据，训练集，验证集，测试集路径
    '''
    data = set(os.listdir(config.data_dir))
    val = set()
    test = set()

    item_data = list(data)
    item_data.sort()
    item_test = np.random.choice(item_data, math.ceil(len(data) * test_ratio), replace=False)
    item1 = list(set(item_data) - set(item_test))
    item1.sort()
    item_val = np.random.choice(item1, math.ceil(len(data) * val_ratio), replace=False)

    val = val.union(item_val)
    test = test.union(item_test)

    train = data.difference(val)
    train = train.difference(test)
    
    print('len of total dataset:', len(data))
    print('len of train dataset:', len(train))
    print('len of val. dataset:', len(val))
    print('len of test dataset:', len(test))
    

    return list(data), list(train), list(val), list(test)

    # 另一种划分方法：对每类数据分别按照比例提取至训练集，测试集
    # idx2file = [[] for _ in range(config.num_classes)]  # list
    
    # for file, list_idx in file2idx.items():
    #     for idx in list_idx:
    #         idx2file[idx].append(file)
    # # print(idx2file)

    # for item in idx2file:

    #     num_val = math.ceil(len(item) * (val_ratio))
    #     num_test = math.ceil(len(item) * (test_ratio))

    #     item_val = np.random.choice(item, num_val, replace=False)
    #     item1 = list(set(item) - set(item_val))

    #     item_test = np.random.choice(item1, num_test, replace=False)

    #     val = val.union(item_val)
    #     test = test.union(item_test)

    # train = data.difference(val)
    # train = train.difference(test)
    # print(len(val))
    # print(len(test))
    # print(len(train))
    # return list(train), list(val), list(test)

    
def count_labels(data, file2idx):
    '''
    统计每个类别的样本数
    :param data:
    :param file2idx:
    :return:
    '''
    cc = [0] * config.num_classes
    for fp in data:
        for i in file2idx[fp]:
            cc[i] += 1
    return np.array(cc)


def load(name2idx, idx2name):
    '''
    可视化，装载数据
    '''
    file2idx = file2index(config.data_label, name2idx)
    file2ag = file2age(config.data_label, name2idx)
    file2gen = file2gender(config.data_label, name2idx)
    data, train, val, test = split_data(file2idx)

    # print('label:\n', idx2name)
    # print(file2idx)
    # print(file2ag)
    # print(file2gen)

    # print and plot the label distribution
    
    wc_data = count_labels(data, file2idx)
    wc_train = count_labels(train, file2idx)
    wc_val = count_labels(val, file2idx)
    wc_test = count_labels(test, file2idx)
    # print('label distribution of val. set:\n', wc_val)
    # print('label distribution of training set:\n', wc_train)
    # print('label distribution of the total dataset:\n', wc_data)
    # print('label distribution of test set:\n', wc_test)

    # fig = plt.figure()
    # names = []
    # for k, v in idx2name.items():
    #     names.append(k)
    
    # x = list(range(len(names)))
    # width = 0.2
    # plt.bar(x, wc_data, width=width, label='total dataset')
    
    # for i in range(len(x)):
    #     x[i] = x[i] + width
    # plt.bar(x, wc_train, width=width, tick_label=names, label='training set')

    # for i in range(len(x)):
    #     x[i] = x[i] + width
    # plt.bar(x, wc_val, width=width, label='val. set')

    # for i in range(len(x)):
    #     x[i] = x[i] + width
    # plt.bar(x, wc_test, width=width, label='test set')
    # plt.legend()
    # plt.title('Dataset distribution')
    # plt.xlabel('label')
    # plt.ylabel('number')
    # plt.show()

    dd = {'train': train, 'val': val, 'test': test, 'idx2name': idx2name, 'file2idx': file2idx, 'file2ag': file2ag, 'file2gen': file2gen, 'wc':wc_data}
    torch.save(dd, config.data)


def vis():
    '''
    随机挑选一个心电样本可视化；
    8导联数据各自图像，以及同一坐标轴下的图像。
    '''
    data = set(os.listdir(config.data_dir))
    item_data = list(data)
    data_randchoice = np.random.choice(item_data, 1)

    df = pd.read_csv(os.path.join('data/ECG_Signal/', data_randchoice.item()), sep=" ")
    plt.figure()
    plt.suptitle(data_randchoice.item())

    ax1 = plt.subplot(4, 2, 1)
    plt.plot(df.index, df['I'])
    plt.title('I')

    ax1 = plt.subplot(4, 2, 2)
    plt.plot(df.index, df['II'])
    plt.title('II')

    ax1 = plt.subplot(4, 2, 3)
    plt.plot(df.index, df['V1'])
    plt.title('V1')

    ax1 = plt.subplot(4, 2, 4)
    plt.plot(df.index, df['V2'])
    plt.title('V2')

    ax1 = plt.subplot(4, 2, 5)
    plt.plot(df.index, df['V3'])
    plt.title('V3')

    ax1 = plt.subplot(4, 2, 6)
    plt.plot(df.index, df['V4'])
    plt.title('V4')

    ax1 = plt.subplot(4, 2, 7)
    plt.plot(df.index, df['V5'])
    plt.title('V5')

    ax1 = plt.subplot(4, 2, 8)
    plt.plot(df.index, df['V6'])
    plt.title('V6')
   
    plt.show()

    plt.figure()
    plt.plot(df.index, df['I'], label='I', linewidth=0.8)
    plt.plot(df.index, df['II'], label='II', linewidth=0.8)
    plt.plot(df.index, df['V1'], label='V1', linewidth=0.8)
    plt.plot(df.index, df['V2'], label='V2', linewidth=0.8)
    plt.plot(df.index, df['V3'], label='V3', linewidth=0.8)
    plt.plot(df.index, df['V4'], label='V4', linewidth=0.8)
    plt.plot(df.index, df['V5'], label='V5', linewidth=0.8)
    plt.plot(df.index, df['V6'], label='V6', linewidth=0.8)
    plt.legend()
    plt.title(data_randchoice.item())
    plt.show()


if __name__ == '__main__':
    pass
    name2idx = name2index(config.arrythmia)
    idx2name = {idx: name for name, idx in name2idx.items()}
    load(name2idx, idx2name)
    # vis()