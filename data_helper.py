# _*_ coding: utf-8 _*_
# @Time : 2020/12/9 上午10:16 
# @Author : yanqiuxia
# @Version：V 0.1
# @File : utils.py
import json

import numpy as np
# from tkinter import _flatten
from itertools import chain


def read_word_embed(pre_trained_path, biovec=False):
    logger.info('loading pre-trained embedding from {}'.format(pre_trained_path))
    if not biovec:
        with open(pre_trained_path, encoding='utf-8') as f:
            words, vectors = zip(*[line.strip().split(' ', 1) for line in f])
            wv = np.loadtxt(vectors)
    else:
        with open(pre_trained_path + 'types.txt', encoding='utf-8') as f:
            words = [line.strip() for line in f]
        wv = np.loadtxt(pre_trained_path + 'vectors.txt')
    return words, wv


def load_vocab(vocab_file):
    fp = open(vocab_file, 'r', encoding='utf-8')
    lines = fp.readlines()
    id_ = 0
    word2id = {}
    for line in lines:
        word = line.strip()
        word2id[word] = id_
        id_ += 1
    return word2id


def load_word_vectors(embedding_file, norm=True, biovec=False):
    # 读取word2vec

    words, wv = read_word_embed(embedding_file, biovec=biovec)
    pad_unk_vector = np.random.random_sample((2, wv.shape[1])) - .5
    wv = np.vstack((pad_unk_vector, wv))

    word2id = {}
    word2id.update({'PAD': 0})
    word2id.update({'UNK': 1})
    word2id.update({w: i + 2 for i, w in enumerate(words)})

    logger.info('loading pre-trained embedding vocab_size is %d, dim is %d' % (wv.shape[0], wv.shape[1]))

    # Normalize each row (word vector) in the matrix to sum-up to 1
    if norm:
        row_norm = np.sum(np.abs(wv) ** 2, axis=-1) ** (1. / 2)
        wv /= row_norm[:, np.newaxis]

    return wv, word2id


def load_corpus_file(file_in):
    with open(file_in, 'r', encoding='utf-8') as fp:
        json_data = json.load(fp)
        data_list = json_data['result']
    return data_list


def build_entitylabel2id(entity_label_file):
    """
    schema BIEOS
    :param entity_label_file:
    :return:
    """

    op = open(entity_label_file, 'r', encoding='utf-8')
    load_dict = json.load(op)
    entitylabel2id = {}
    id_ = 0
    entitylabel2id["PAD"] = id_
    id_ = 1
    entitylabel2id["O"] = id_  # 从1开始编码， 0是pad
    id_ = 2
    schema = "BIES"
    for key, value in load_dict.items():
        for s in schema:
            entitylabel = s + "-" + value
            entitylabel2id[entitylabel] = id_
            id_ += 1

    op.close()

    return entitylabel2id


def build_event_entitylabel2id(event_entity_label_file):
    '''
    创建事件主体标签
    :param event_entity_label_file:
    :return:
    '''
    op = open(event_entity_label_file, 'r', encoding='utf-8')
    event_entitylabel2id = json.load(op)
    op.close()
    return event_entitylabel2id


def build_BIOES2id():
    '''
    创建实体标签
    :return:
    '''
    BIOES2id = {
        "PAD": 0,
        "O": 1,
        "B-ORG": 2,
        "I-ORG": 3,
        "E-ORG": 4,
        "S-ORG": 5
    }
    return BIOES2id


def seq2number(sequence, vocab2id, defaultid=None):
    sequence_id = []
    for item in sequence:
        id_ = vocab2id.get(item, defaultid)
        if id_ is not None:
            sequence_id.append(id_)
        else:
            print('the %s not in library' % item)

    sequence_id = np.array(sequence_id)

    return sequence_id


def sents2bert_inputs(sent_list, tokenizer, add_special_tokens=False):
    """

    :param add_special_tokens:
    :return:
    """
    bert_length = 512
    sent_words = []
    if not add_special_tokens:
        pre_word_length = 0
        for content in sent_list:
            if pre_word_length + len(content) <= bert_length:
                sent_words.append(tokenizer.convert_tokens_to_ids(list(content)))
            else:
                sent_words.append(tokenizer.convert_tokens_to_ids(list(content)[:bert_length - pre_word_length]))
                break
            pre_word_length += len(content)
    else:
        '''
        后续实现加[CLS]和[SEP] 暂时只实现单句的,多句后续再实现
        '''
        sent_words = ["[CLS]"]
        content = list("".join(sent_list))
        if len(content) > bert_length - 2:
            content = content[:bert_length - 2]
        sent_words.extend(content)
        sent_words.append("[SEP]")
        sent_words_id = tokenizer.convert_tokens_to_ids(sent_words)
        sent_words = [sent_words_id]

    return sent_words


def get_bert_tokens_label(spans, words_length):
    '''
    使用bert预训练语言模型，暂时不在句子开头和结尾增加[CLS] [SEP]
    其中label 都使用ORG
    :param entities:
    :param words_length:
    :return:
    '''
    tokens_label = ["O"] * words_length
    for span in spans:
        start_offset = span['start_offset']
        end_offset = span['end_offset']
        # label = span['label']
        # 所有的label作为ORG实体
        label = "ORG"
        if end_offset >= words_length:
            print(span['span_name'] + "超出文章范围，直接丢弃！")
        else:
            if end_offset - start_offset > 1:
                tokens_label[start_offset] = "B-" + label
                tokens_label[end_offset - 1] = "E-" + label
                for index in range(start_offset + 1, end_offset - 1):
                    tokens_label[index] = "I-" + label
            else:
                tokens_label[start_offset] = "S-" + label
    return tokens_label


def get_span_offsets_dict(spans):
    offsets = {}  # key (start_offset, end_offset) value: list [label]
    for span in spans:
        if offsets.__contains__((span['start_offset'], span['end_offset'])):
            var = offsets.get((span['start_offset'], span['end_offset']))
            var.append(span['label'])
            offsets[(span['start_offset'], span['end_offset'])] = var
        else:
            offsets[(span['start_offset'], span['end_offset'])] = [span['label']]
    return offsets


def get_bert_span_info(spans, event_entitylabel2id, words_length):
    '''
    使用bert预训练语言模型，句子开头和结尾增加[CLS] [SEP]  start_offset 和end_offset加1
    :param spans:
    :param entity2id:
    :return:
    '''
    spans_dranges = []  # [num_spans, 2]
    spans_labels = []  # [num_spans, entity_num]

    # key (start_offset, end_offset) value: list [label]
    spans_offsets_dict = get_span_offsets_dict(spans)

    for key, value in spans_offsets_dict.items():
        '''句子开头和结尾增加[CLS][SEP]
        start_offset
        和end_offset加1
        '''
        start_offset = key[0]+1
        end_offset = key[1]+1
        labels = value
        drange = [start_offset, end_offset]
        if end_offset < words_length-1:
            span_label_one_hot = [0] * len(event_entitylabel2id)

            for label in labels:
                if label != "NA":
                    span_label_id = event_entitylabel2id.get(label)
                    if span_label_id is not None:
                        span_label_one_hot[int(span_label_id)] = 1
                    else:
                        print("标签 %s 不在字典里面，请检查" % label)
                else:
                    '''
                    如果 label 为NA,则标签 One hot 全为 0
                    '''

            spans_dranges.append(drange)
            spans_labels.append(span_label_one_hot)
        else:
            ''
            print("span offset 超出文章范围！")

    return spans_dranges, spans_labels,


def get_max_token_length(tokens_length):
    max_token_length = np.max(tokens_length)

    return max_token_length


def get_max_span_num(spans_dranges_list):
    max_span_num = 0
    for spans_dranges in spans_dranges_list:
        if len(spans_dranges) > max_span_num:
            max_span_num = len(spans_dranges)
    return max_span_num


def data2bert_number(data, event_entity_label2id, tokenizer):
    tokens_index = []
    tokens_length = []

    spans_dranges_list = []
    spans_labels_list = []

    for example in data:
        content_list = example['content']
        '''在句子开头加[CLS]和[SEP]
        '''
        sent_words = sents2bert_inputs(content_list, tokenizer, add_special_tokens=True)

        # words = list(_flatten(sent_words))
        words = list(chain.from_iterable(sent_words))

        token_length = len(words)
        tokens_length.append(token_length)

        sequence_id = words
        tokens_index.append(sequence_id)

        spans = example['spans']
        '''
        bert 在句子开头加[CLS]和[SEP]， start_offset和 end_offset要加1
        '''
        spans_dranges, spans_labels = get_bert_span_info(spans, event_entity_label2id, len(words))
        spans_dranges_list.append(spans_dranges)
        spans_labels_list.append(spans_labels)

    out_data = {
        "tokens": tokens_index,
        "tokens_length": tokens_length,
        "spans_dranges": spans_dranges_list,
        "spans_labels": spans_labels_list
    }
    max_token_length = get_max_token_length(tokens_length)
    print('最大句子长度：', max_token_length)
    max_span_num = get_max_span_num(spans_dranges_list)
    print('最大span个数：', max_span_num)

    return out_data


if __name__ == '__main__':
    ''

    # entitylabel2id = build_entitylabel2id("D:/PycharmProjects/yuqing_event_extract/data/entity_label.json")
    # print(entitylabel2id)
    #
    # train_data = load_corpus_file('D:/PycharmProjects/yuqing_event_extract/data/test.json')
    # print(train_data)
    #
    # from transformers import BertTokenizer
    # tokenizer = BertTokenizer.from_pretrained(
    #     "D:/PycharmProjects/yuqing_event_extract/data/bert-base-chinese-pytorch")
    #
    # train_data = data2bert_number(train_data, entitylabel2id, tokenizer)
    entitylabel2id = {
        "a": 0,
        "b": 1,
        "c": 2,
        "d": 3,
        "e": 4
    }
    spans = [
        {
            "start_offset": 0,
            "end_offset": 1,
            "label": "d"
        },
        {
            "start_offset": 2,
            "end_offset": 3,
            "label": "c"
        },
        {
            "start_offset": 2,
            "end_offset": 3,
            "label": "e"
        },
        {
            "start_offset": 2,
            "end_offset": 3,
            "label": "a"
        }
    ]

    spans_dranges, spans_labels = get_bert_span_info(spans, entitylabel2id)

    spans_dranges = [
        [[0, 1], [1, 2], [2, 3]],
        [[4, 5], [8, 9]]
    ]
    print(spans_dranges)
    max_span_num = get_max_span_num(spans_dranges)
    print(max_span_num)
