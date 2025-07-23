from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Solar, Dataset_Pred
from torch.utils.data import DataLoader

# 通过命令行参数选择数据集 --data ETTH1 则选择Dataset_ETT_hour
# 通过命令行参数选择数据集 --data ETTm1 则选择Dataset_ETT_minute
# 通过命令行参数选择数据集 --data Solar 则选择Dataset_Solar

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Solar': Dataset_Solar,
    'custom': Dataset_Custom,
}


def data_provider(args, flag):
    Data = data_dict[args.data] # 根据命令行参数选择相应的数据集
    timeenc = 0 if args.embed != 'timeF' else 1 # 如果embed不是timeF，则timeenc为0，否则为1
    train_only = args.train_only # 如果train_only为True，则只使用训练集，否则按比例划分训练/验证/测试集

    if flag == 'test':
        shuffle_flag = False # 测试集不需要打乱
        drop_last = True # 丢弃最后一个不完整的批次。
        # drop_last = False # without the "drop-last" trick
        batch_size = args.batch_size # 测试集的batch_size 时间序列数据中每个时间步之间的时间间隔是多长
        freq = args.freq # 测试集的频率
    elif flag == 'pred':
        shuffle_flag = False # 预测集不需要打乱
        drop_last = False # 预测集不需要drop_last
        batch_size = 1 # 预测集的batch_size为1
        freq = args.freq # 预测集的频率
        Data = Dataset_Pred # 预测集的数据集 这里会强制覆盖之前从 data_dict 中选择的类，转而使用专门为预测任务设计的 Dataset_Pred 类。
    else:
        shuffle_flag = True 
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    # if flag == 'train':
    #     drop_last = False

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        train_only=train_only
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
