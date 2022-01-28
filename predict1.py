# _*_ coding: utf-8 _*_
# @Time : 2020/10/19 下午5:58 
# @Author : yanqiuxia
# @Version：V 0.1
# @File : predict.py
import argparse
import json
import os
import sys
from typing import List, Dict
# from tkinter import _flatten
from itertools import chain
from collections import defaultdict

import numpy as np
import torch
from torch.autograd import Variable

from data_helper import sents2bert_inputs
from parameter import Parameter


class Predict(object):
    def __init__(self, tokenizer, param:Parameter, model, event_entity_id2label):
        self.tokenizer = tokenizer
        self.param = param
        self.model = model
        self.event_entity_id2label = event_entity_id2label

    def get_max_span_num(self, spans_list):
        max_span_num = 0
        for spans in spans_list:
            if len(spans) > max_span_num:
                max_span_num = len(spans)
        return max_span_num

    def get_span_dranges_masks(self, spans):
        '''

        :param spans:
        :return:
        '''

        span_dranges = []
        span_masks = []

        #使用bert 加入[cls]和[sep] offset需要加1

        for span in spans:
            span_name = span['span_name']
            start_offset = span['start_offset']+1
            end_offset = span['end_offset']+1
            drange = [start_offset, end_offset]
            span_dranges.append(drange)
            span_masks.append(1)

        return span_dranges, span_masks

    def _generate_inputbatch(self, text_list, spans_list):

        batch_token_inputs = []
        batch_token_length = []
        batch_token_masks = []

        batch_spans_dranges = []
        batch_spans_masks = []

        max_token_length = max([len("".join(text)) for text in text_list])
        if self.param.use_bert:
            '''
            如果使用Bert  最大长度不能超过512
            '''
            max_token_length = min(max_token_length+2, 512)

        max_span_num = self.get_max_span_num(spans_list)

        for text,spans in zip(text_list,spans_list):

            '''
            使用bert加[CLS]和[SEP],超过512直接进行截断
            '''
            sent_words = sents2bert_inputs(text, self.tokenizer, add_special_tokens=True)
            # words = list(_flatten(sent_words))
            words = list(chain.from_iterable(sent_words))

            token_length = len(words)

            batch_token_length.append(token_length)
            sequence_id = words

            pad_token_inputs = self.pad_seq(sequence_id, max_token_length)
            batch_token_inputs.append(pad_token_inputs)
            token_masks = np.ones([token_length])
            pad_token_masks = self.pad_seq(token_masks, max_token_length)
            batch_token_masks.append(pad_token_masks)

            #offset 加1
            span_dranges, span_masks = self.get_span_dranges_masks(spans)

            pad_span_dranges = np.zeros([max_span_num, 2], dtype=np.int64)
            if len(span_dranges) > 0:
                pad_span_dranges[:len(span_dranges), :] = span_dranges

            pad_span_masks = np.zeros([max_span_num], dtype=np.float32)
            if len(span_masks) > 0:
                pad_span_masks[:len(span_masks)] = span_masks

            batch_spans_dranges.append(pad_span_dranges)
            batch_spans_masks.append(pad_span_masks)

        batch_token_inputs = np.array(batch_token_inputs)
        batch_token_length = np.array(batch_token_length)
        batch_token_masks = np.array(batch_token_masks)
        batch_spans_dranges = np.array(batch_spans_dranges)
        batch_spans_masks = np.array(batch_spans_masks)

        return Variable(torch.from_numpy(batch_token_inputs)), \
               Variable(torch.from_numpy(batch_token_length)), \
               Variable(torch.from_numpy(batch_token_masks).float()), \
               Variable(torch.from_numpy(batch_spans_dranges)), \
               Variable(torch.from_numpy(batch_spans_masks))

    def pad_seq(self, sequence, max_tokens_size):
        pad_sequence = np.zeros([max_tokens_size], dtype=np.int64)
        pad_sequence[:len(sequence)] = sequence

        return pad_sequence

    def _model_predict(self, text_list,spans_list):
        '''
        支持批量预测
        :param text_list: 二维list
        :return:
        '''
        self.model.eval()

        batch_token_inputs, batch_token_length, batch_token_masks,\
        batch_spans_dranges, batch_spans_masks = self._generate_inputbatch(
            text_list,spans_list)
        if self.param.gpu:
            batch_token_inputs = batch_token_inputs.cuda()
            batch_token_length = batch_token_length.cuda()
            batch_token_masks = batch_token_masks.cuda()

        loss, spans_pred_ids,spans_probs= self.model(
            word_inputs=batch_token_inputs,
            word_seq_lengths=batch_token_length,
            seq_token_masks=batch_token_masks,
            spans_dranges=batch_spans_dranges,
            spans_masks=batch_spans_masks,
            spans_labels=None,
        )


        '''
        spans_dranges [batch_size,max_num_span,2]
        spans_masks [batch_size, max_num_span]
        spans_pred_ids [batch_size, max_num_span, C]
        spans_list [batch_size, num_span] 元素为dict
        '''
        result_spans_list = self._generate_label(spans_list, spans_pred_ids,spans_probs)

        return result_spans_list

    def _generate_label(self, spans_list, spans_pred_ids,spans_probs):
        '''

        @param spans_list:
        @param spans_pred_ids:
        @return:
        '''

        if not isinstance(spans_pred_ids, list):
            spans_pred_ids = spans_pred_ids.cpu().numpy().tolist()
            spans_probs = spans_probs.cpu().detach().numpy().tolist()

        result_spans_list = []

        for spans, span_pred_ids,span_prob in zip(spans_list,spans_pred_ids,spans_probs ):

            spans_result = []
            for i,span in enumerate(spans):
                # span_pred_id = span_pred_ids[i]
                # span_pred_id = np.array(span_pred_id)
                # pred_idxs = np.where(span_pred_id == 1)[0]
                #
                # if len(pred_idxs) > 0:
                #     for pred_idx in pred_idxs:
                #         span_label = self.event_entity_id2label.get(pred_idx)
                #         result = {
                #             "span_name": span['span_name'],
                #             "start_offset": span['start_offset'],
                #             "end_offset": span['end_offset'],
                #             "label": span_label
                #         }
                #         spans_result.append(result)
                result = {
                    "span_name":span['span_name'],
                    "start_offset":span['start_offset'],
                    "end_offset": span['end_offset'],
                    "prob":span_prob
                }
                spans_result.append(result)
            result_spans_list.append(spans_result)
        return result_spans_list

    def predict(self, sentence_list,spans):
        '''
        :param text:
        :return:
        '''
        entities_list = self._model_predict([sentence_list],[spans])
        if len(entities_list) > 0:
            entities = entities_list[0]
        else:
            entities = []

        return entities

    def batch_predict(self, text_list, spans_list):
        '''
        :param text_list:
        :return:
        '''

        entities_list = self._model_predict(text_list,spans_list)

        return entities_list
