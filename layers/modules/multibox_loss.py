import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.box_utils import match, log_sum_exp
from data import cfg_mnet
GPU = cfg_mnet['gpu_train']

class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh  # 0.35
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]

    def forward(self, predictions, priors, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """

        loc_data, conf_data, landm_data = predictions  # (bs, 16800, 4), (bs, 16800, 2), (bs, 16800, 10)
        priors = priors    # anchors (16800, 4)
        num = loc_data.size(0) # bs
        num_priors = (priors.size(0))  # 16800 = ( 80*80 + 40*40 + 20*20 ) * 2

        # step1: 计算模型输出的目标，分别是 loc_t, conf_t, landm_t；  注：模型预测的是offset
        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)   # (28, 16800, 4)     # bbox，检测框
        landm_t = torch.Tensor(num, num_priors, 10)  # (28, 16800, 10)  landmark，关键点定位
        conf_t = torch.LongTensor(num, num_priors)   # (28, 16800)     分类
        for idx in range(num):
            truths = targets[idx][:, :4].data  # 边界框
            labels = targets[idx][:, -1].data  # 是否为人脸
            landms = targets[idx][:, 4:14].data   # 关键点
            defaults = priors.data

            # 在match函数里计算模型输出所需要优化的目标，分别是 loc_t, conf_t, landm_t
            match(self.threshold, truths, defaults, self.variance, labels, landms, loc_t, conf_t, landm_t, idx)
        if GPU:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            landm_t = landm_t.cuda()

        # step2： 计算关键点的loss
        zeros = torch.tensor(0).cuda()
        # landm Loss (Smooth L1)
        # Shape: [batch,num_priors,10]
        pos1 = conf_t > zeros   # conf_t是分类标签，pos1.shape==(28,16800)，标记anchor是否为正例
        num_pos_landm = pos1.long().sum(1, keepdim=True)  # 各个样本上正例anchor的数量
        N1 = max(num_pos_landm.data.sum().float(), 1)  # 1852: 一个batch的样本里，被标记为正样本anchor的数量，用于求取landm平均值
        pos_idx1 = pos1.unsqueeze(pos1.dim()).expand_as(landm_data) # 28,16800,10
        landm_p = landm_data[pos_idx1].view(-1, 10)  # landm_data为模型输出
        landm_t = landm_t[pos_idx1].view(-1, 10)     # landm_t 为 match函数计算得到的 目标值
        loss_landm = F.smooth_l1_loss(landm_p, landm_t, reduction='sum')

        # step3：计算回归框的损失
        pos = conf_t != zeros  # shape is [28, 16800], binary, 正样本为True，负样本为False
        conf_t[pos] = 1   # 计算bbox损失不关心是什么类别，所以类别标签全部设置为1
        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)  # expand to（28,16800,4），每个点计算损失值
        loc_p = loc_data[pos_idx].view(-1, 4)  # 整个batch，有 1961 个anchor框， loc_p shape(1961, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)     # 同 loc_p
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # step4: 计算分类损失
        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)    # (28,16800,2) --> (470400,2)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        # gather实现获取分类标签所对应的模型输出概率。如conf_t.view(-1, 1)[0]是0，表示第0个样本的分类标签是第0类
        # 则其目标为 batch_conf[0] 当中的对应第0类的那个元素，即 batch_conf[0][0]

        # Hard Negative Mining，通过loss_c排序实现样本挑选
        loss_c[pos.view(-1, 1)] = 0  # filter out pos boxes for now; pos（28, 16800），它标记anchor是正样本还是负样本
        loss_c = loss_c.view(num, -1)  # num = 28
        _, loss_idx = loss_c.sort(1, descending=True)  # 降序后，idx，[0, 0]是6829，则loss_c[0, 6829]这个loss是最高的
        _, idx_rank = loss_idx.sort(1)  # 标号的排序，例如 [0, 6829]值为0，[0, 13751]值为16799，用于挑选是否被选中为负样本
        # 逻辑为先计算需要N个负样本，然后idx_rank来判断，当前anchors的分类loss排名情况，如果排在N以内，被选中，设置为True
        # 如果排名在N以外，则不被选中，设置为False； 详情可见neg那行（往下3行）代码

        num_pos = pos.long().sum(1, keepdim=True)  # 各个样本中，正样本数量; shape == (28,1)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)  # 计算各样本的负样本个数；shape == （28,1）
        # 标记各样本的anchor是否为负样本，是的话为True；(28, 16800), neg.sum() == num_pos.sum() * 7
        neg = idx_rank < num_neg.expand_as(idx_rank)    # idx_rank 是anchor产生的loss的降序排名，排名靠前loss大，被选中

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)    #  pos_idx.sum() = 3986; (28, 16800, 2), 3922的原因是正样本有1993, anchor对应的分类向量有2个元素，所以是2倍
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)    #  neg_idx.sum() = 27902; (28, 16800, 2), 13951 = 1993*7
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)  # 挑出选中的anchor对应的分类分数；(15944, 2)
        targets_weighted = conf_t[(pos+neg).gt(0)]  # (15944) 15944 =  1993 + 13951
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')  # conf_p

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = max(num_pos.data.sum().float(), 1)   # N是正样本个数， 1993
        loss_l /= N
        loss_c /= N
        loss_landm /= N1

        return loss_l, loss_c, loss_landm
