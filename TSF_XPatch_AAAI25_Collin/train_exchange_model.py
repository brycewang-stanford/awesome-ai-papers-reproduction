import argparse
import os
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
import time

# 从项目模块中导入必要的组件
from models.model import XPatch  # 导入您重构的XPatch模型
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

def train_model():
    # -- 参数解析 --
    parser = argparse.ArgumentParser(description='[XPatch] Training script for Exchange Rate Dataset')

    # 基本配置 (针对Exchange Rate数据集的默认值)
    parser.add_argument('--model', type=str, required=False, default='XPatch', help='model name')
    parser.add_argument('--train_only', type=bool, required=False, default=False, help='perform training on full input dataset without validation and testing')
    parser.add_argument('--data', type=str, required=False, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='exchange_rate/exchange_rate.csv', help='data file path')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')

    # 模型核心参数
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length for decoder')
    parser.add_argument('--features', type=str, default='M', help='forecasting task, M for multivariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s,t,h,d,b,w,m]')
    parser.add_argument('--enc_in', type=int, default=8, help='encoder input size (number of features for exchange rate)')
    
    # Patching 相关参数
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')

    # 分解和 RevIN 参数
    parser.add_argument('--ma_type', type=str, default='ema', help='moving average type: reg, ema, dema')
    parser.add_argument('--alpha', type=float, default=0.3, help='alpha for EMA')
    parser.add_argument('--beta', type=float, default=0.3, help='beta for DEMA')
    parser.add_argument('--revin', type=int, default=1, help='whether to use RevIN, 1 for True, 0 for False')

    # 训练超参数
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate strategy')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    
    # GPU / 设备配置
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')

    args = parser.parse_args()

    # -- 设备检测与设置 --
    if args.use_gpu and torch.cuda.is_available():
        print("Using GPU")
        device = torch.device('cuda:{}'.format(args.gpu))
    elif torch.backends.mps.is_available():
        print("Using MPS")
        device = torch.device('mps')
    else:
        print("Using CPU")
        device = torch.device('cpu')

    # -- 固定随机种子以保证可复现性 --
    fix_seed = 2023
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    for ii in range(args.itr):
        # -- 设置实验标识 --
        setting = 'XPatch_Exchange_sl{}_pl{}_itr{}'.format(args.seq_len, args.pred_len, ii)
        
        # -- 数据加载 --
        print("Loading data...")
        train_data, train_loader = data_provider(args, 'train')
        vali_data, vali_loader = data_provider(args, 'val')
        test_data, test_loader = data_provider(args, 'test')
        
        # -- 模型初始化 --
        print("Initializing model...")
        model = XPatch(
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            enc_in=args.enc_in,
            patch_len=args.patch_len,
            stride=args.stride,
            alpha=args.alpha,
            ma_type=args.ma_type,
            beta=args.beta,
            revin=bool(args.revin)
        ).float().to(device)

        # -- 优化器和损失函数 --
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        if args.loss == 'mse':
            criterion = nn.MSELoss()
        else:
            criterion = nn.L1Loss()
            
        # -- 训练过程 --
        print('>>>>>>> Start Training : {} >>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        
        # 提前终止工具
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)
        
        # 模型检查点保存路径
        path = os.path.join(args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        for epoch in range(args.train_epochs):
            epoch_time = time.time()
            iter_count = 0
            train_loss = []
            
            model.train()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                optimizer.zero_grad()
                
                # 将数据移动到指定设备
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                
                # 模型前向传播
                outputs = model(batch_x)
                
                # 对齐输出和标签
                outputs = outputs[:, -args.pred_len:, :]
                batch_y = batch_y[:, -args.pred_len:, :]
                
                # 计算损失
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())
                
                # 反向传播和优化
                loss.backward()
                optimizer.step()
            
            # -- 验证过程 --
            vali_loss = []
            model.eval()
            with torch.no_grad():
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                    batch_x = batch_x.float().to(device)
                    batch_y = batch_y.float().to(device)
                    
                    outputs = model(batch_x)
                    
                    outputs = outputs[:, -args.pred_len:, :]
                    batch_y = batch_y[:, -args.pred_len:, :]
                    
                    pred = outputs.detach()
                    true = batch_y.detach()
                    
                    loss = criterion(pred, true)
                    vali_loss.append(loss.item())

            train_loss_avg = np.average(train_loss)
            vali_loss_avg = np.average(vali_loss)
            
            print("Epoch: {0}, | Train Loss: {1:.7f} | Vali Loss: {2:.7f} | Cost Time: {3:.2f}s".format(
                epoch + 1, train_loss_avg, vali_loss_avg, time.time() - epoch_time))
                
            # 提前终止
            early_stopping(vali_loss_avg, model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            # 调整学习率
            adjust_learning_rate(optimizer, epoch + 1, args)
            
        # -- 测试过程 --
        print('>>>>>>> Start Testing : {} >>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        
        # 加载最佳模型
        best_model_path = os.path.join(path, 'checkpoint.pth')
        model.load_state_dict(torch.load(best_model_path))
        
        preds = []
        trues = []
        
        model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                
                outputs = model(batch_x)
                
                outputs = outputs[:, -args.pred_len:, :]
                batch_y = batch_y[:, -args.pred_len:, :]
                
                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                
                preds.append(pred)
                trues.append(true)
                
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        
        mae, mse = metric(preds, trues)
        print('======= Final Test Results =======')
        print('MSE: {}, MAE: {}'.format(mse, mae))
        
        # 将结果保存到文件
        result_path = './results'
        if not os.path.exists(result_path):
            os.makedirs(result_path)
            
        with open(os.path.join(result_path, "result_exchange_custom.txt"), 'a') as f:
            f.write(setting + "  \n")
            f.write('MSE:{}, MAE:{}'.format(mse, mae))
            f.write('\n\n')

if __name__ == '__main__':
    train_model() 