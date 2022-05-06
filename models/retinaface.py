import torch
import torch.nn as nn
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision.models._utils as _utils
import torch.nn.functional as F
from collections import OrderedDict

from models.net import MobileNetV1 as MobileNetV1
from models.net import FPN as FPN
from models.net import SSH as SSH



class ClassHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(ClassHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors*2, kernel_size=(1,1),stride=1,padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 2)

class BboxHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous() # b*c*h*w --> b*h*w*c

        return out.view(out.shape[0], -1, 4)  # 第二个维度等于 h*w，即把宽和高给整合起来

class LandmarkHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(LandmarkHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*10,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 10)


class RetinaFace(nn.Module):
    def __init__(self, cfg=None, phase='train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFace, self).__init__()
        self.phase = phase
        backbone = None
        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()  # 接收img, 输出1000维的向量，当前为imagenet模型形式
            if cfg['pretrain']:
                checkpoint = torch.load("./weights/mobilenetV1X0.25_pretrain.tar", map_location=torch.device('cpu'))
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]  # remove module. 因为采用并行训练，dict前会自动加上module.所以这里要去掉它
                    new_state_dict[name] = v
                # load params
                backbone.load_state_dict(new_state_dict)
        elif cfg['name'] == 'Resnet50':
            import torchvision.models as models
            backbone = models.resnet50(pretrained=cfg['pretrain'])

        self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])  # body才是真正的backbone
        in_channels_stage2 = cfg['in_channel']
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = cfg['out_channel']
        self.fpn = FPN(in_channels_list, out_channels)  # 特征金字塔网络，用于生成3种尺度特征
        self.ssh1 = SSH(out_channels, out_channels)     # ssh网络是用了3阶段卷积，然后拼接3阶段的feature map
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'])  # 为每个anchor生成二分类概率
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])    # 为每个anchor生成bbox框的修正参数
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg['out_channel']) # 同上，生成landmark

    def _make_class_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels, anchor_num))  # anchor_num表示1个点配几个anchors，这里是2
        return classhead
    
    def _make_bbox_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num))
        return bboxhead

    def _make_landmark_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels,anchor_num))
        return landmarkhead

    def forward(self, inputs):
        out = self.body(inputs)  # out是dict，key是config所配置的1,2,3，分别对应backbone的stage1,stage2,stage3
        # torch.Size([28, 64, 80, 80])     下降8倍，输入是640
        # torch.Size([28, 128, 40, 40])    下降16倍
        # torch.Size([28, 256, 20, 20])    下降32倍
        # FPN
        fpn = self.fpn(out)
        # torch.Size([28, 64, 80, 80])     通道统一变为64
        # torch.Size([28, 64, 40, 40])
        # torch.Size([28, 64, 20, 20])

        # SSH
        feature1 = self.ssh1(fpn[0])  # 输入和输出的shape不变，依旧是 [28, 64, 80, 80]
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        # anchor计算：80*80 + 40*40 + 20*20 = 8400， 8400*anchor_num = 16800 anchor_num=2
        # bbox_regressions torch.Size([28, 16800, 4])
        # classifications torch.Size([28, 16800, 2])
        # ldm_regressions torch.Size([28, 16800, 10])
        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)


        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        return output