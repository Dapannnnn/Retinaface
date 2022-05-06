# config.py

cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],  # 用于生成prior box，16,32对应80*80特征图，64*128对应40*40
    'steps': [8, 16, 32],       # 用于生成prior box，用于计算feature map的size，这样才能对应的生成prior bbox
    'variance': [0.1, 0.2],     # 用于生成prior box
    'clip': False,              # 用于生成prior box
    'loc_weight': 2.0,          # loc loss放大倍数
    'gpu_train': True,
    'batch_size': 28,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,          # 图像统一缩放尺度
    'pretrain': True,           # 是否预训练
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},  # 设置backbone返回的feature map， key是backbone的layer name， value是返会字典的key
    'in_channel': 32,           # 设置backbone的通道
    'out_channel': 64           # 设置backbone的通道
}

cfg_re50 = {
    'name': 'Resnet50',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 1,
    'ngpu': 4,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'image_size': 840,
    'pretrain': True,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,
    'out_channel': 256
}

