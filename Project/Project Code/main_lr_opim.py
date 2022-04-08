import torch, time, os, shutil
import models, utils
import numpy as np
import pandas as pd
from tensorboard_logger import Logger
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import ECGDataset
from config import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


torch.manual_seed(41)
torch.cuda.manual_seed(41)


# 保存当前模型的权重，并且更新最佳的模型权重
def save_ckpt(state, is_best, model_save_dir):
    current_w = os.path.join(model_save_dir, config.current_w)
    best_w = os.path.join(model_save_dir, config.best_w)
    torch.save(state, current_w)
    if is_best: shutil.copyfile(current_w, best_w)


def train_epoch(model, optimizer, criterion, train_dataloader, show_interval=10):
    model.train()
    f1_meter, loss_meter, it_count = 0, 0, 0
    acc_meter, precision_meter, recall_meter = 0, 0, 0

    # for inputs, target in train_dataloader:
    # for i, (inputs, target) in enumerate(train_dataloader):
    for i, data in enumerate(train_dataloader):
        # print(i)
        # print(data)
        inputs, age, sex, target=data
        # solution 1
        #data0= [age, sex,0x 4998]
        #torch.tile(sig,data0)
        # sol1 end
        # exit(0)
        # print(target)
        inputs = inputs.to(device)
        target = target.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        output = model(inputs, age, sex)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_meter += loss.item()
        it_count += 1
        # f1 = utils.calc_f1(target, torch.sigmoid(output))
        acc, p, r, f1 = utils.calc_score(target, torch.sigmoid(output))
        f1_meter += f1
        acc_meter += acc
        precision_meter += p
        recall_meter += r
        if it_count != 0 and it_count % show_interval == 0:
            # print("%d,loss:%.3e f1:%.3f acc:%.3f precision:%.3f recall:%.3f " % (it_count, loss.item(), f1, acc, p, r))
            print("%d, f1:%.3f acc:%.3f precision:%.3f recall:%.3f " % (it_count, f1, acc, p, r))
    
    return loss_meter / it_count, f1_meter / it_count, acc_meter / it_count, precision_meter / it_count, recall_meter / it_count


def val_epoch(model, criterion, val_dataloader, threshold=0.5):
    model.eval()
    f1_meter, loss_meter, it_count = 0, 0, 0
    acc_meter, precision_meter, recall_meter = 0, 0, 0
    with torch.no_grad():
        # for inputs, target in val_dataloader:
        for i, data in enumerate(val_dataloader):
            inputs, age, sex, target=data
            
            inputs = inputs.to(device)
            target = target.to(device)
            output = model(inputs, age, sex)
            #print(output.size())
            #print(output)
            #print(target)
            #exit(0)
            loss = criterion(output, target)
            loss_meter += loss.item()
            it_count += 1
            output = torch.sigmoid(output)
            # f1 = utils.calc_f1(target, output, threshold)
            acc, p, r, f1 = utils.calc_score(target, output, threshold)
            f1_meter += f1
            acc_meter += acc
            precision_meter += p
            recall_meter += r
    return loss_meter / it_count, f1_meter / it_count, acc_meter / it_count, precision_meter / it_count, recall_meter / it_count


def train(args):
    # model
    model = getattr(models, config.model_name)()
    if args.ckpt and not args.resume:
        state = torch.load(args.ckpt, map_location='cpu')
        model.load_state_dict(state['state_dict'])
        print('train with pretrained weight val_f1', state['f1'])
    
    # model = model.to(device)
    # 多卡训练
    model = torch.nn.DataParallel(model, device_ids= [0,1,2,3]).cuda()
    
    # data
    train_dataset = ECGDataset(train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=6, drop_last = True)
    val_dataset = ECGDataset(train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=4, drop_last = True)
    
    test_dataset = ECGDataset(test=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=4, drop_last = True)
    
    print("train_datasize", len(train_dataset), "val_datasize", len(val_dataset))
    # optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler_1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, verbose=True)
    # scheduler_2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=True)
    w = torch.tensor(train_dataset.wc, dtype=torch.float).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=w)
    # 模型保存文件夹
    model_save_dir = '%s/%s_%s' % (config.ckpt, config.model_name, time.strftime("%Y%m%d%H%M"))
    if args.ex: model_save_dir += args.ex
    best_f1 = -1
    lr = config.lr
    start_epoch = 1
    stage = 1
    # 从上一个断点，继续训练
    if args.resume:
        if os.path.exists(args.ckpt):  # 这里是存放权重的目录
            model_save_dir = args.ckpt
            current_w = torch.load(os.path.join(args.ckpt, config.current_w))
            best_w = torch.load(os.path.join(model_save_dir, config.best_w))
            best_f1 = best_w['loss']
            start_epoch = current_w['epoch'] + 1
            lr = current_w['lr']
            stage = current_w['stage']
            model.load_state_dict(current_w['state_dict'])
            # 如果中断点恰好为转换stage的点
            if start_epoch - 1 in config.stage_epoch:
                stage += 1
                lr /= config.lr_decay
                utils.adjust_learning_rate(optimizer, lr)
                model.load_state_dict(best_w['state_dict'])
            print("=> loaded checkpoint (epoch {})".format(start_epoch - 1))
    logger = Logger(logdir=model_save_dir, flush_secs=2)
    # =========>开始训练<=========
    for epoch in range(start_epoch, config.max_epoch + 1):
        since = time.time()
        train_loss, train_f1, train_acc, train_precision, train_recall = train_epoch(model, optimizer, criterion, train_dataloader, show_interval=100)
        val_loss, val_f1, val_acc, val_precision, val_recall = val_epoch(model, criterion, val_dataloader)
        print('#epoch:%02d lr:%s train_loss:%.3e train_f1:%.3f train_acc:%.3f train_precision:%.3f train_recall:%.3f\n\t\t'
                                 '  val_loss  :%.3e val_f1  :%.3f val_acc  :%.3f val_precision  :%.3f val_recall  :%.3f time:%s\n'
              % (epoch, optimizer.param_groups[0]['lr'], train_loss, train_f1, train_acc, train_precision, train_recall,
                 val_loss, val_f1, val_acc, val_precision, val_recall, utils.print_time_cost(since)))
        lr = optimizer.param_groups[0]['lr']
        scheduler_1.step(val_f1)
        # scheduler_2.step(val_loss)
        
        test_loss, test_f1, test_acc, test_precision, test_recall = val_epoch(model, criterion, test_dataloader)
        print('\t\t  test_loss :%0.3e test_f1 :%.3f test_acc :%.3f test_precision:%.3f test_recall :%.3f\n'
              % (test_loss, test_f1, test_acc, test_precision, test_recall))
        
        logger.log_value('train_loss', train_loss, step=epoch)
        logger.log_value('train_f1', train_f1, step=epoch)
        logger.log_value('train_acc', train_acc, step=epoch)
        logger.log_value('train_precision', train_precision, step=epoch)
        logger.log_value('train_recall', train_recall, step=epoch)
        logger.log_value('val_loss', val_loss, step=epoch)
        logger.log_value('val_f1', val_f1, step=epoch)
        logger.log_value('val_acc', val_acc, step=epoch)
        logger.log_value('val_precision', val_precision, step=epoch)
        logger.log_value('val_recall', val_recall, step=epoch)
        state = {"state_dict": model.state_dict(), "epoch": epoch, "loss": val_loss, 'f1': val_f1, 'lr': lr,
                 'stage': stage}
        save_ckpt(state, best_f1 < val_f1, model_save_dir)
        best_f1 = max(best_f1, val_f1)
        '''
        if epoch in config.stage_epoch:
            stage += 1
            lr /= config.lr_decay
            best_w = os.path.join(model_save_dir, config.best_w)
            model.load_state_dict(torch.load(best_w)['state_dict'])
            print("*" * 10, "step into stage%02d lr %.3ef" % (stage, lr))
            utils.adjust_learning_rate(optimizer, lr)
        '''

# 验证集验证
def val(args):
    list_threhold = [0.5]
    model = getattr(models, config.model_name)()
    if args.ckpt: 
        model = torch.nn.DataParallel(model, device_ids= [0,1,2,3]).cuda()
        model.load_state_dict(torch.load(args.ckpt, map_location='cpu')['state_dict'])
    
    val_dataset = ECGDataset(train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=4, drop_last = True)

    w = torch.tensor(val_dataset.wc, dtype=torch.float).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=w)

    for threshold in list_threhold:
        val_loss, val_f1, val_acc, val_precision, val_recall = val_epoch(model, criterion, val_dataloader, threshold)
        print('threshold %.2f val_loss:%0.3e val_f1:%.3f val_acc:%.3f val_precision:%.3f val_recall:%.3f\n' % (threshold, val_loss, val_f1, val_acc, val_precision, val_recall))

# 测试，与验证几乎相同，所以直接调用了val_epoch()
def test(args):
    list_threhold = [0.5]
    model = getattr(models, config.model_name)()
    if args.ckpt: 
        model = torch.nn.DataParallel(model, device_ids= [0,1,2,3]).cuda()
        model.load_state_dict(torch.load(args.ckpt, map_location='cpu')['state_dict'])

    test_dataset = ECGDataset(test=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=4, drop_last = True)
    print("test_datasize", len(test_dataset))
    
    # test_dataset = ECGDataset(train=False)
    # test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=4, drop_last = True)

    w = torch.tensor(test_dataset.wc, dtype=torch.float).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=w)

    for threshold in list_threhold:
        test_loss, test_f1, test_acc, test_precision, test_recall = val_epoch(model, criterion, test_dataloader, threshold)
        print('threshold %.2f test_loss:%0.3e test_f1:%.3f test_acc:%.3f test_precision:%.3f test_recall:%.3f\n'
              % (threshold, test_loss, test_f1, test_acc, test_precision, test_recall))


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("command", metavar="<command>", help="train or infer")
    parser.add_argument("--ckpt", type=str, help="the path of model weight file")
    parser.add_argument("--ex", type=str, help="experience name")
    parser.add_argument("--resume", action='store_true', default=False)
    parser.add_argument("--device_ids", type=list, help='GPU ids used, eg. --device_ids=56')
    args = parser.parse_args()

    # 多卡训练还应更改train里面的设置，测试同理
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.device_ids)

    import torch
    from torch import nn, optim
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
      print ("cuda")
    else:
      print ("cpu")

    torch.manual_seed(41)
    torch.cuda.manual_seed(41)

    if (args.command == "train"):
        since_all = time.time()
        train(args)
        print ("training time: %s" %utils.print_time_cost(since_all))
    if (args.command == "test"):
        test(args)
    if (args.command == "val"):
        val(args)
