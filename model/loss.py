# _*_ coding: utf-8 _*_
# @Time : 2020/10/12 下午2:15 
# @Author : yanqiuxia
# @Version：V 0.1
# @File : loss.py
import torch
import torch.nn as nn
from torch import tensor
from torch.autograd import Variable

#单标签分类
class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()

        self.gamma = gamma

        if alpha is None:
            alpha = torch.ones(class_num, 1) * 0.5

        if isinstance(alpha, Variable):
            self.alpha_t = alpha
        else:
            self.alpha_t = Variable(alpha)

    def forward(self, probs, y_trues):
        '''

        :param probs: [N,C]
        :param y_trues: [N,C]
        :return:
        '''
        probs_t = (probs * y_trues.float()).sum(1).view(-1, 1)  # [N,1]
        probs_t = probs_t + 1e-32
        log_p = probs_t.log()
        ce_loss = log_p

        ids = y_trues.max(1)[1]
        if probs.is_cuda and not self.alpha_t.is_cuda:
            self.alpha_t = self.alpha_t.cuda()
        alpha_t = self.alpha_t[ids.data.view(-1)]  # [batch_size,1]

        focal_matrix = -alpha_t * torch.pow(abs(1.0 - probs_t), self.gamma)  # [batch_size,1]
        loss = focal_matrix * ce_loss  # [batch_size]
        loss = torch.mean(loss)
        return loss

#多标签分类
class BinaryFocalLoss(torch.nn.Module):

    def __init__(self, class_num, alpha=None, gamma=2):
        super(BinaryFocalLoss, self).__init__()

        if alpha is None:
            alpha = torch.ones(1, class_num)

        if isinstance(alpha, Variable):
            self.alpha_t = alpha
        else:
            self.alpha_t = Variable(alpha)

        self.gamma = gamma
        # self.bceloss = torch.nn.BCELoss(self.alpha_t, reduce=False)
        self.bceloss = torch.nn.BCELoss(reduction='none')

    def forward(self, probs, y_trues, masks=None):
        '''

        :param probs: [N,C]
        :param y_trues: [N,C]
        :param masks: [N]
        :return:
        '''
        if probs.is_cuda and not self.alpha_t.is_cuda:
            self.alpha_t = self.alpha_t.cuda()
        bce_loss = self.bceloss(probs, y_trues.float()) #[N,C]

        pt = torch.exp(-bce_loss)

        focal_loss = torch.pow(abs(1-pt), self.gamma) * bce_loss * self.alpha_t #[N,C]

        focal_loss = torch.sum(focal_loss, dim=1)
        if masks is not None:
            focal_loss = focal_loss * masks
        loss = torch.mean(focal_loss)
        return loss