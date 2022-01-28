# _*_ coding: utf-8 _*_
# @File : event_extract.py
# @Time : 2021/11/2 21:56
# @Author : Yan Qiuxia
from typing import List, Dict
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from model.bert_encoder import BertEncoder
from model.loss import BinaryFocalLoss


class EventExtract(nn.Module):
    def __init__(self,
                 num_event_entity_labels,
                 drop_rate=0.1,
                 cnn_hidden_size=128,
                 K=3,
                 bert_model_path=None,
                 gpu=False,
                 ):
        ''
        super(EventExtract, self).__init__()
        self.cnn_hidden_size = cnn_hidden_size
        self.K = K

        self.num_event_entity_labels = num_event_entity_labels

        self.bert_encoder = BertEncoder(drop_rate=drop_rate, bert_model_path=bert_model_path)

        # self.fc = nn.Linear(self.hidden_size * 4, num_event_entity_labels)

        self.fc = nn.Linear(self.cnn_hidden_size * 3, num_event_entity_labels)

        self.loss_fuc = BinaryFocalLoss(class_num=num_event_entity_labels)

        self.span_conv = nn.Conv2d(
            in_channels=self.bert_encoder.bert_config.hidden_size,
            out_channels=self.cnn_hidden_size,
            kernel_size=(self.K, 1)
        )
        self.prefix_conv = nn.Conv2d(
            in_channels=self.bert_encoder.bert_config.hidden_size,
            out_channels=self.cnn_hidden_size,
            kernel_size=(self.K, 1)

        )

        self.suffix_conv = nn.Conv2d(
            in_channels=self.bert_encoder.bert_config.hidden_size,
            out_channels=self.cnn_hidden_size,
            kernel_size=(self.K, 1)
        )

        self.gpu = gpu

        if self.gpu:
            self.fc.cuda()
            self.loss_fuc.cuda()

    def get_doc_emb(self, lstm_feats):
        '''

        :param lstm_feats: [batch_size, seq_len, hidden_size]
        :return:
        '''
        doc_emb = lstm_feats.max(dim=1, keepdim=False)[0]  # [batch_size, hidden_size]
        return doc_emb

    def get_span_emb(self, spans_dranges, tokens_emb, device=None):
        '''
        :param spans_dranges: [batch_size,num_spans,2]
        :param tokens_emb: [batch_size,seq_len,hidden_size]
        :return:
        '''
        spans_emb_list = []
        for i, span_dranges in enumerate(spans_dranges):
            spans_emb = []
            for span_drange in span_dranges:
                span_token_emb = tokens_emb[i, span_drange[0]: span_drange[1], :]  # [num_mention_tokens, hidden_size]
                if span_token_emb.size()[0] > 0:
                    span_emb = span_token_emb.max(dim=0)[0]  # [hidden_size]
                else:
                    span_emb = torch.zeros([span_token_emb.size()[1]], device=device)

                spans_emb.append(span_emb)
            if len(spans_emb) > 0:
                spans_emb = torch.stack(spans_emb, dim=0)  # [num_spans, hidden_size]
            else:
                spans_emb = torch.zeros([tokens_emb.size()[2]], device=device)
            spans_emb_list.append(spans_emb)
        batch_spans_embs = torch.stack(spans_emb_list, dim=0)  # [batch_size, num_spans, hidden_size]

        return batch_spans_embs

    def get_conv_pool_feat(self, conv:nn.Conv2d, conv_inputs, num_tokens, device=None):
        '''

        @param conv:
        @param conv_inputs: [batch_size*num_spans, num_tokens, hidden_size]
        @return:
        '''
        if num_tokens < conv.kernel_size[0]:
            pad_num = conv.kernel_size[0] - conv_inputs.size()[1]
            num_tokens = conv.kernel_size[0]
            pad_tensor = torch.zeros((conv_inputs.size()[0], pad_num, conv_inputs.size()[-1]), device=device)
            conv_inputs = torch.cat((conv_inputs, pad_tensor), dim=1)

        conv_inputs = conv_inputs.permute(0,2,1).unsqueeze(dim=-1) #[batch_size*num_spans, hidden_size,num_tokens,1]
        conv_outs = conv(conv_inputs).squeeze(3)#[batch_size*num_spans, cnn_hidden_size,num_tokens]
        pool_filter_size = num_tokens - conv.kernel_size[0] +1
        # [batch_size*num_spans, cnn_hidden_size,1]
        pool_out = F.max_pool1d(conv_outs, pool_filter_size)
        # [batch_size*num_spans, cnn_hidden_size]
        pool_out = pool_out.squeeze(2)
        return pool_out

    def get_span_emb_byconv(self, spans_dranges, tokens_emb, word_seq_lengths, device=None):
        '''
        :param spans_dranges: [batch_size,num_spans,2]
        :param tokens_emb: [batch_size,seq_len,hidden_size]
        :param tokens_masks : [batch_size, seq_len, hidden_size]
        :return:
        '''

        max_token_length = word_seq_lengths.max().item()

        spans_starts = spans_dranges[:,:,0]
        spans_ends = spans_dranges[:,:,1]
        spans_widths = spans_ends-spans_starts
        max_batch_span_width = spans_widths.max().item()
        max_batch_prefix_width = spans_starts.max().item()
        max_batch_suffix_width = max_token_length-spans_ends.min().item()
        spans_emb_list = []
        prefix_emb_list = []
        suffix_emb_list = []
        for i, span_dranges in enumerate(spans_dranges):
            spans_emb = []
            prefix_emb = []
            suffix_emb = []
            token_length = word_seq_lengths[i].item()
            for span_drange in span_dranges:
                # 获取span的向量
                cur_span_width = (span_drange[1]-span_drange[0]).item()

                cur_span_token_emb = tokens_emb[i, span_drange[0]: span_drange[1], :]  # [num_mention_tokens, hidden_size]

                span_token_emb = torch.zeros([max_batch_span_width, tokens_emb.size()[-1]], device=device)
                if cur_span_width > 0:
                    span_token_emb[:cur_span_width, :] = cur_span_token_emb #[span_width, hidden_size]
                spans_emb.append(span_token_emb)

                # 获取span 前文的向量
                cur_prefix_width = span_drange[0].item()
                cur_prefix_token_emb = tokens_emb[i, :cur_prefix_width, :]# [num_tokens, hidden_size]
                prefix_token_emb = torch.zeros([max_batch_prefix_width, tokens_emb.size()[-1]], device=device)
                if cur_prefix_width > 0:
                    prefix_token_emb[:cur_prefix_width, :] = cur_prefix_token_emb
                prefix_emb.append(prefix_token_emb)

                #获取 span后文的向量
                cur_suffix_width = token_length-span_drange[1].item()
                cur_suffix_token_emb = tokens_emb[i, span_drange[1]:token_length,:]
                suffix_token_emb = torch.zeros([max_batch_suffix_width,tokens_emb.size()[-1]], device=device)
                if cur_suffix_width>0:
                    suffix_token_emb[:cur_suffix_width,:] = cur_suffix_token_emb
                suffix_emb.append(suffix_token_emb)

            spans_emb = torch.stack(spans_emb, dim=0)  # [num_spans, span_width, hidden_size]
            spans_emb_list.append(spans_emb)

            prefix_emb = torch.stack(prefix_emb, dim=0)# [num_spans, num_tokens, hidden_size]
            prefix_emb_list.append(prefix_emb)

            suffix_emb = torch.stack(suffix_emb, dim=0)  # [num_spans, num_tokens, hidden_size]
            suffix_emb_list.append(suffix_emb)

        batch_spans_embs = torch.stack(spans_emb_list, dim=0)  # [batch_size, num_spans, num_tokens, hidden_size]
        batch_prefix_embs = torch.stack(prefix_emb_list, dim=0) # [batch_size, num_spans, num_tokens, hidden_size]
        batch_suffix_embs = torch.stack(suffix_emb_list, dim=0) # [batch_size, num_spans, num_tokens, hidden_size]

        batch_spans_embs = batch_spans_embs.reshape(-1,max_batch_span_width,tokens_emb.size()[-1])
        # [batch_size*num_spans, cnn_hidden_size]
        spans_feats = self.get_conv_pool_feat(self.span_conv, batch_spans_embs, max_batch_span_width,device=device)

        '''
        使用卷积和maxpooling 得到span 的特征以及上文和下文的特征
        '''

        if max_batch_prefix_width >0:
            batch_prefix_embs = batch_prefix_embs.reshape(-1, max_batch_prefix_width, tokens_emb.size()[-1])
            prefix_feats = self.get_conv_pool_feat(self.prefix_conv, batch_prefix_embs, max_batch_prefix_width,device=device)
        else:
            prefix_feats = torch.zeros(spans_feats.size(), device=device)

        if max_batch_suffix_width >0:
            batch_suffix_embs = batch_suffix_embs.reshape(-1, max_batch_suffix_width, tokens_emb.size()[-1])
            suffix_feats = self.get_conv_pool_feat(self.suffix_conv, batch_suffix_embs, max_batch_suffix_width,device=device)
        else:
            suffix_feats = torch.zeros(spans_feats.size(), device=device)

        out_feats = torch.cat([prefix_feats, spans_feats, suffix_feats], dim=-1)

        return out_feats # [batch_size*num_spans, cnn_hidden_size*3]

    def spans_cls(self, docs_emb, spans_emb):
        '''

        :param docs_emb: [batch_size, hidden_size]
        :param spans_emb: [batch_size*num_spans, hidden_size]
        :return:
        '''
        # docs_emb = docs_emb.unsqueeze(dim=1)
        # # [batch_size, num_spans, hidden_size]
        # docs_emb = docs_emb.expand(spans_emb.size())
        # # [batch_size, num_spans, hidden_size*2]
        # spans_context_emb = torch.cat((docs_emb, spans_emb), dim=2)
        #
        # # [batch_size*num_spans, hidden_size*2]
        # spans_context_emb = spans_context_emb.reshape((-1, self.hidden_size * 4))

        logits = self.fc(spans_emb)  # (N, C)
        probs = F.sigmoid(logits)  # (N,C)
        pred_ids = torch.where(probs > 0.5, torch.ones_like(probs), torch.zeros_like(probs)).int()  # (N,C)

        return probs, pred_ids

    def get_cls_loss(self, spans_probs, spans_labels, spans_masks):
        '''

        :param spans_probs:[batch_size*num_spans,C]
        :param spans_labels:[batch_size, num_spans,C]
        :return:
        '''
        spans_labels = spans_labels.reshape((-1, self.num_event_entity_labels))
        spans_masks = spans_masks.reshape(-1)
        loss = self.loss_fuc(spans_probs, spans_labels, spans_masks)
        return loss

    def forward(self, word_inputs,
                word_seq_lengths,
                seq_token_masks,
                spans_dranges,
                spans_masks,
                spans_labels=None
                ):
        '''

        :param word_inputs: [batch_size, seq_len]
        :param word_seq_lengths: [batch_size]
        :param seq_token_masks: [batch_size, seq_len]
        :param spans_dranges [batch_size, max_spans,2]
        :param spans_masks [batch_size, max_spans]
        :param spans_labels [batch_size, max_spans, entity_num]
        :return:
        '''
        batch_size = word_inputs.size()[0]

        spans_dranges_to_model = spans_dranges
        spans_labels_to_model = spans_labels
        spans_masks_to_model = spans_masks

        #得到bert最后一层的输出
        bert_seq_out = self.bert_encoder(word_inputs, seq_token_masks)

        # [batch_size, hidden_size]
        docs_emb = self.get_doc_emb(bert_seq_out)

        # [batch_size, num_spans, hidden_size]
        max_span_num = spans_dranges_to_model.size()[1]
        if max_span_num > 0:
            # spans_emb = self.get_span_emb(spans_dranges_to_model, lstm_feats, device=word_inputs.device)
            '''
            使用卷积，pooling,提取span,前文，后文特征
            '''
            #[batch_size*num_spans, cnn_hidden_size*3]
            spans_emb = self.get_span_emb_byconv(spans_dranges_to_model,bert_seq_out,word_seq_lengths,device=word_inputs.device)
            if self.gpu:
                spans_emb = spans_emb.cuda()

            spans_probs, spans_pred_ids = self.spans_cls(docs_emb, spans_emb)
            if spans_labels_to_model is not None:
                cls_loss = self.get_cls_loss(spans_probs, spans_labels_to_model, spans_masks_to_model)
            else:
                cls_loss = None

            '''
            spans_pred_ids (batch_size,max_num_spans,C)
            '''
            spans_pred_ids = spans_pred_ids.reshape((batch_size, -1, self.num_event_entity_labels))

        else:
            # loss = ner_loss
            spans_pred_ids = []
            spans_probs = []
            for i in range(batch_size):
                spans_pred_ids.append([])
                spans_probs.append([])
            cls_loss = torch.tensor(0.0).float()

        return cls_loss, spans_pred_ids, spans_probs


if __name__ == '__main__':
    ''
    x = torch.tensor([[1, 2], [2, 9], [3, 0]])

    x = x.unsqueeze(dim=1)
    print(x.size())
    y = x.expand([3, 2, 2])
    print(y)
    print(y.size())
