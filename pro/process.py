# _*_ coding: utf-8 _*_
# @File : process.py
# @Time : 2021/11/6 11:42
# @Author : Yan Qiuxia
import json
import numpy as np
np.random.seed(2021)

def read_json(path):
    with open(path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    return data


def save_json(data, file_out):
    op = open(file_out, 'w', encoding="utf-8")

    json.dump(data, op, ensure_ascii=False, indent=1)
    op.close()


def combine_label(spans):
    span_offset_dict = {}
    for span in spans:
        if span.__contains__("label"):
            label = span['label']
        else:
            label = "ORG"

        start_offset = span['start_offset']
        end_offset = span['end_offset']
        if not span.__contains__((start_offset, end_offset)):
            span_offset_dict[(start_offset, end_offset)] = label
        else:
            cur_label = span[(start_offset, end_offset)] + "_" + label
            span_offset_dict[(start_offset, end_offset)] = cur_label
    return span_offset_dict


def get_tokens_label(spans, words_length):
    '''
    使用bert预训练语言模型，暂时不在句子开头和结尾增加[CLS] [SEP]
    其中label 都使用ORG
    :param entities:
    :param words_length:
    :return:
    '''
    span_offset_dict = combine_label(spans)

    tokens_label = ["O"] * words_length
    for key, value in span_offset_dict.items():
        start_offset = key[0]
        end_offset = key[1]
        label = value

        if end_offset - start_offset > 1:
            tokens_label[start_offset] = "B-" + label
            tokens_label[end_offset - 1] = "E-" + label
            for index in range(start_offset + 1, end_offset - 1):
                tokens_label[index] = "I-" + label
        else:
            tokens_label[start_offset] = "S-" + label
    return tokens_label


def merge_true_pred_file(true_file, pred_file):
    json_true_data = read_json(true_file)
    json_pred_data = read_json(pred_file)

    true_data_list = json_true_data['result']
    pred_data_list = json_pred_data['result']

    if len(true_data_list) != len(pred_data_list):
        print("真实数据个数和预测数据个数不一致，请检查，其中真实数据数目为%d，预测数据的数目为%d" % (len(true_data_list), len(pred_data_list)))

    data_list = []

    for true_data, pred_data in zip(true_data_list, pred_data_list):
        data = true_data
        data['pred_spans'] = pred_data['spans']
        data_list.append(data)

    return data_list

def get_dir_data(dir_path):
    json_data = read_json(dir_path)
    data_list = json_data['result']
    return data_list


def print_result(data_list, file_out, only_print_true=False):
    op = open(file_out, 'w', encoding="utf-8")

    for data in data_list:
        doc_name = data['doc_name']
        data_set_name = data['data_set_name']

        spans = data['spans']
        content = data['content']
        if isinstance(content,list):
            content = "".join(data['content'])

        true_tokens_labels = get_tokens_label(spans, len(content))
        if only_print_true:
            if len(spans) > 0:

                print("doc_name:{0}/t data_set_name:{1}".format(doc_name, data_set_name), file=op)
                for w, true_token in zip(list(content), true_tokens_labels):
                    print("{0}/t{1}".format(w, true_token), file=op)

        else:
            pred_spans = data['pred_spans']
            pred_tokens_labels = get_tokens_label(pred_spans, len(content))
            if len(spans) > 0 or len(pred_spans) > 0:
                print("doc_name:{0}/t data_set_name:{1}".format(doc_name, data_set_name), file=op)
                for w, true_token, pred_token in zip(list(content), true_tokens_labels, pred_tokens_labels):
                    print("{0}/t{1}/t{2}".format(w, true_token, pred_token), file=op)

    op.close()


def get_offsets(spans):
    offsets = set()
    for span in spans:
        offsets.add((span['start_offset'], span['end_offset']))
    return list(offsets)


def is_intersection(s1, e1, s2, e2):
    flag = True
    if ((e1 > s2 and e1 < e2) or (s1 < s2 and e1 > e2) or (s1 > s2 and s1 < e2)):
        flag = True
    else:
        flag = False
    return flag


def is_intersection_list(adrange, dranges):
    flag = False
    for drange in dranges:
        flag = is_intersection(adrange[0], adrange[1], drange[0], drange[1])
        if flag:
            break
    return flag


def statis_label_num(file_path):
    ''
    label_num = {}
    json_data = read_json(file_path)
    data_list = json_data['result']

    for data in data_list:
        spans = data['spans']
        for span in spans:
            label = span['label']
            if label_num.__contains__(label):
                num = label_num.get(label)+1
                label_num[label] = num
            else:
                label_num[label] = 1

    print(label_num)


def combine_true_pred_spans(file_path, file_out):
    json_data = read_json(file_path)
    data_list = json_data['result']

    new_data_list = []
    for data in data_list:
        spans = data['spans']
        true_offsets = get_offsets(spans)
        pred_spans = data['pred_spans']
        extra_spans = []
        for pred_span in pred_spans:
            start_offset = pred_span['start_offset']
            end_offset = pred_span['end_offset']
            if (start_offset, end_offset) not in true_offsets:
                flag = is_intersection_list((start_offset, end_offset), true_offsets)
                if not flag:
                    pred_span['label'] = "NA"
                    extra_spans.append(pred_span)

        spans += extra_spans
        if len(spans)>0:
            new_data = {
                "doc_name": data['doc_name'],
                "data_set_name": data['data_set_name'],
                "content": data['content'],
                "spans":spans
            }

            new_data_list.append(new_data)
    result = {
        "result":new_data_list
    }

    print("数据总数%d"%len(new_data_list))

    save_json(result,file_out)


if __name__ == '__main__':
    ''

    # true_file = "D:/PycharmProjects/event-extract-join/data/train.json"
    # json_true_data = read_json(true_file)
    # data_list = json_true_data['result']
    # file_out = "D:/PycharmProjects/event-extract-join/data/predict_result.txt"
    # print_result(data_list, file_out, only_print_true=True)

    file_path = "/home/aipf/work/single-cls/data/v0_0_6/dev_ner_predict.json"
    file_out =  "/home/aipf/work/single-cls/data/v0_0_6/dev.json"
    combine_true_pred_spans(file_path, file_out)

    file_path = "/home/aipf/work/single-cls/data/v0_0_6/dev_spans.json"
    statis_label_num(file_path)

    file_path = "/home/aipf/work/single-cls/data/v0_0_6/train_ner_predict_aug.json"
    file_out = "/home/aipf/work/single-cls/data/v0_0_6/train.json"
    combine_true_pred_spans(file_path, file_out)

    file_path = "/home/aipf/work/single-cls/data/v0_0_6/train.json"
    statis_label_num(file_path)







