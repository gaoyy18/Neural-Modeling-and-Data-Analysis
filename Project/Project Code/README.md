## 基于AI信号的心电辅助诊断


## Folder structure
```shell
└─project
  ├─README.md                        
  ├─data
    ├─ECG_Signal
      ├─100001.txt
      ├─100003.txt
      ...
      └─149898.txt
    ├─Arrythmia.txt
    ├─data.pth
    └─Label.txt
  ├─models
    ├─__init__.py
    ├─ResNet.py
    ├─ResNext.py
    ...
    └─ECGNet_V3.py
  ├─config.py                
  ├─data_process.py
  ├─dataset.py
  ├─utils.py
  └─main.py
  ```

## 数据预处理
```shell
python data_process.py
```

## 模型训练，验证，测试
注意：

1. 确定config.py里面的参数以开始训练

2. 使用包括age和gender的数据与否，需要在main.py中train_epoch()和val_epoch()中的前向传递部分进行修改。


### 训练及验证
```shell
python main.py train --device_ids=56  #从零开始，使用编号为5和6的两块显卡训练，如果用单卡或者更多卡，需要修改一下main.py里面的train函数中的设置，测试同理
```
```shell
python main.py train --ckpt=./ckpt/ECGNet_v1_202201102329 --resume --device_ids=56  #从上次保存的断点，继续训练
```

### 测试
```shell
python main.py test --ckpt=./ckpt/ECGNet_v3_202201121515/best_w.pth --device_ids=56  #测试
```

### 评价指标说明
1. accuracy: 每个样本的34维都是对的才算成是true
2. precision, recall, F1-score: 使用的是micro-F1，如果y_pred与y_true维度大小是(batch_size, label_size)，调用f1_score时，参数average='micro'；
将y_pred与y_true展平到1维时，调用f1_score的参数average='binary'。