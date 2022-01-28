# _*_ coding: utf-8 _*_
# @Time : 2021/2/8 下午3:34 
# @Author : yanqiuxia
# @Version：V 0.1
# @File : data_loader.py
import numpy as np
import torch
from torch.autograd import Variable


def iterate_minibatches_order(data, batchsize):
    token_inputs = data['tokens']
    tokens_length = data['tokens_length']

    spans_dranges_list = data["spans_dranges"]
    spans_labels_list = data["spans_labels"]

    data_num = len(token_inputs)

    indices = np.argsort([-len(doc) for doc in token_inputs])
    start_idx = 0
    if data_num < batchsize:
        excerpts = indices[start_idx:]
        out_token_inputs = []
        out_tokens_length = []
        out_spans_dranges = []
        out_spans_labels = []

        for excerpt in excerpts:
            out_token_inputs.append(token_inputs[excerpt])
            out_tokens_length.append(tokens_length[excerpt])
            out_spans_dranges.append(spans_dranges_list[excerpt])
            out_spans_labels.append(spans_labels_list[excerpt])

        yield out_token_inputs, out_tokens_length, out_spans_dranges, out_spans_labels

    else:
        for start_idx in range(0, data_num - batchsize + 1, batchsize):
            excerpts = indices[start_idx:start_idx + batchsize]
            out_token_inputs = []
            out_tokens_length = []
            out_spans_dranges = []
            out_spans_labels = []

            for excerpt in excerpts:
                out_token_inputs.append(token_inputs[excerpt])
                out_tokens_length.append(tokens_length[excerpt])
                out_spans_dranges.append(spans_dranges_list[excerpt])
                out_spans_labels.append(spans_labels_list[excerpt])

            yield out_token_inputs, out_tokens_length, out_spans_dranges, out_spans_labels

        if start_idx + batchsize < data_num:
            excerpts = indices[start_idx + batchsize:]
            out_token_inputs = []
            out_tokens_length = []
            out_spans_dranges = []
            out_spans_labels = []

            for excerpt in excerpts:
                out_token_inputs.append(token_inputs[excerpt])
                out_tokens_length.append(tokens_length[excerpt])
                out_spans_dranges.append(spans_dranges_list[excerpt])
                out_spans_labels.append(spans_labels_list[excerpt])

            yield out_token_inputs, out_tokens_length, out_spans_dranges, out_spans_labels


# def iterate_minibatches(inputs, targets, batchsize, shuffle):
#     assert inputs.shape[0] == targets.shape[0]
#
#     if shuffle:
#         indices = np.arange(inputs.shape[0])
#         np.random.shuffle(indices)
#     for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
#         if shuffle:
#             excerpt = indices[start_idx:start_idx + batchsize]
#         else:
#             excerpt = slice(start_idx, start_idx + batchsize)
#         yield inputs[excerpt], targets[excerpt]
#     if start_idx + batchsize < inputs.shape[0]:
#         if shuffle:
#             excerpt = indices[start_idx + batchsize:]
#         else:
#             excerpt = slice(start_idx + batchsize, start_idx + batchsize * 2)
#         yield inputs[excerpt], targets[excerpt]


def gen_minibatch(param, data, event_entity_label_num):
    for token_inputs, tokens_length, spans_dranges, spans_labels in \
            iterate_minibatches_order(data, param.batch_size):

        token_inputs_mat, token_masks_mat, tokens_length, \
        spans_dranges_mat, spans_masks_mat, spans_labels_mat = \
            data2mat(token_inputs, tokens_length, spans_dranges, spans_labels, event_entity_label_num)

        if param.gpu:
            yield token_inputs_mat.cuda(), token_masks_mat.cuda(), \
                  tokens_length.cuda(), spans_dranges_mat.cuda(), \
                  spans_masks_mat.cuda(), spans_labels_mat.cuda()
        else:
            yield token_inputs_mat, token_masks_mat, \
                  tokens_length, spans_dranges_mat, \
                  spans_masks_mat, spans_labels_mat


def get_max_token_length(tokens_length):
    max_token_length = np.max(tokens_length)

    return max_token_length


def get_max_span_num(spans_labels_list):
    max_span_num = 0
    for spans_labels in spans_labels_list:
        if len(spans_labels) > max_span_num:
            max_span_num = len(spans_labels)
    return max_span_num


def data2mat(token_inputs, tokens_length, spans_dranges, spans_labels, event_entity_label_num):
    tokens_length = np.array(tokens_length)
    max_token_length = get_max_token_length(tokens_length)
    token_inputs_mat = np.zeros([len(token_inputs), max_token_length], dtype=np.int64)
    token_masks_mat = np.zeros([len(token_inputs), max_token_length], dtype=np.float32)
    max_span_num = get_max_span_num(spans_labels)

    for i, token_input in enumerate(token_inputs):
        if len(token_input) > 0:
            token_inputs_mat[i, :len(token_input)] = token_input
            token_masks_mat[i, :len(token_input)] = 1

    spans_masks_mat = np.zeros([len(spans_dranges), max_span_num], dtype=np.float32)
    spans_dranges_mat = np.zeros([len(spans_dranges), max_span_num, 2], dtype=np.int64)

    for i, span_dranges in enumerate(spans_dranges):
        if len(span_dranges) > 0:
            spans_dranges_mat[i, :len(span_dranges), :] = span_dranges
            spans_masks_mat[i, :len(span_dranges)] = 1

    spans_labels_mat = np.zeros([len(spans_labels), max_span_num, event_entity_label_num], dtype=np.int64)

    for i, span_labels in enumerate(spans_labels):
        if len(span_labels) > 0:
            spans_labels_mat[i, :len(span_labels), :] = span_labels

    token_inputs_mat = Variable(torch.from_numpy(token_inputs_mat))
    token_masks_mat = Variable(torch.from_numpy(token_masks_mat))
    tokens_length = Variable(torch.from_numpy(tokens_length))
    spans_dranges_mat = Variable(torch.from_numpy(spans_dranges_mat))
    spans_masks_mat = Variable(torch.from_numpy(spans_masks_mat))
    spans_labels_mat = Variable(torch.from_numpy(spans_labels_mat))

    return token_inputs_mat, token_masks_mat, tokens_length, \
           spans_dranges_mat, spans_masks_mat, spans_labels_mat


if __name__ == '__main__':
    ''
    # data_num = 10
    # batchsize = 4
    # indices = np.arange(0, data_num)
    #
    # for start_idx in range(0, data_num - batchsize + 1, batchsize):
    #     excerpts = indices[start_idx:start_idx + batchsize]
    #     yield excerpts
    #
    #
    # if start_idx + batchsize < data_num:
    #     excerpts = indices[start_idx + batchsize:]
    #     yield excerpts
