import os


class Config:
    # for data_process.py
    root = r'data'
    data_dir = os.path.join(root, 'ECG_Signal')

    data_label = os.path.join(root, 'Label.txt')
    
    arrythmia = os.path.join(root, 'Arrythmia.txt')
    
    data = os.path.join(root, 'data.pth')

    # for train
    #训练的模型名称
    model_name = 'ECGNet_v3'
    #在第几个epoch进行到下一个state,调整lr
    stage_epoch = [50,100,150]
    #训练时的batch大小
    batch_size = 128
    #label的类别数
    num_classes = 34
    #最大训练多少个epoch
    max_epoch = 200
    
    #保存模型的文件夹
    ckpt = 'ckpt'
    #保存提交文件的文件夹
    sub_dir = 'submit'
    #初始的学习率
    lr = 5e-5
    #保存模型当前epoch的权重
    current_w = 'current_w.pth'
    #保存最佳的权重
    best_w = 'best_w.pth'
    # 学习率衰减 lr/=lr_decay
    lr_decay = 10

    #for test
    temp_dir=os.path.join(root,'temp')


config = Config()
