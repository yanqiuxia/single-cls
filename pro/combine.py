# _*_ coding: utf-8 _*_
# @File : combine.py
# @Time : 2021/11/15 19:35
# @Author : Yan Qiuxia
import json
import os
import numpy as np


def read_json(path):
    with open(path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    return data


def save_json(data, file_out):
    op = open(file_out, 'w', encoding="utf-8")

    json.dump(data, op, ensure_ascii=False, indent=1)
    op.close()


def get_prob_list(data_list):
    data_prob_list = []
    for data in data_list:
        spans = data['pred_spans']
        data_probs = []
        if len(spans) > 0:
            for span in spans:
                span_prob = span['prob']
                data_probs.append(span_prob)
        else:
            ''
        data_prob_list.append(data_probs)
    return data_prob_list


def get_multi_files_prob(files):
    data_spans_prob = []  # [data_num, num_spans, C]

    for i, fname in enumerate(files):

        cur_file = os.path.join(dir_path, fname)
        print(cur_file)
        if os.path.isfile(cur_file):
            json_data = read_json(cur_file)
            data_list = json_data['result']

            for j, data in enumerate(data_list):
                spans = data['pred_spans']

                if i == 0:
                    spans_prob = []
                else:
                    spans_prob = data_spans_prob[j]  # [num_spans,C]

                if len(spans) > 0:

                    for k, span in enumerate(spans):
                        cur_span_prob = np.asarray(span['prob'])
                        if i == 0:
                            spans_prob.append(cur_span_prob)
                        else:
                            cur_span_prob += spans_prob[k]
                            spans_prob[k] = cur_span_prob

                else:
                    ''
                if i == 0:
                    data_spans_prob.append(spans_prob)

    return data_list, data_spans_prob


def combine_result(dir_path, entitylabel2id_file,file_out):

    files = os.listdir(dir_path)

    # 得到所有文件的概率值
    data_list, data_spans_prob = get_multi_files_prob(files)

    event_entity_label2id = read_json(entitylabel2id_file)
    event_entity_id2label = {int(v): k for k, v in event_entity_label2id.items()}


    # 根据概率计算结果,得到标签
    for data, spans_prob in zip(data_list,data_spans_prob):
        spans_prob = spans_prob
        spans = data['pred_spans']
        spans_result = []

        for span, span_prob in zip(spans, spans_prob):
            span_prob= span_prob/len(files)
            pred_idxs = np.where(span_prob>0.5)[0]
            for pred_idx in pred_idxs:
                span_label = event_entity_id2label.get(pred_idx)
                result = {
                    "span_name": span['span_name'],
                    "start_offset": span['start_offset'],
                    "end_offset": span['end_offset'],
                    "label": span_label
                }
                spans_result.append(result)

        data['pred_spans'] = spans_result

    result = {
        "result": data_list
    }
    save_json(result, file_out)


if __name__ == '__main__':
    ''
    dir_path = "/home/aipf/work/single-cls/data/dev_result"
    entitylabel2id_file = "/home/aipf/work/single-cls/data/event_entity_label2id.json"
    file_out = "/home/aipf/work/single-cls/data/comb_result/dev_result.json"
    combine_result(dir_path, entitylabel2id_file, file_out)