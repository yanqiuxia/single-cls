# _*_ coding: utf-8 _*_
# @File : vote.py
# @Time : 2021/11/19 10:56
# @Author : Yan Qiuxia
import json
import os
from collections import defaultdict


def read_json(path):
    with open(path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    return data


def save_json(data, file_out):
    op = open(file_out, 'w', encoding="utf-8")

    json.dump(data, op, ensure_ascii=False, indent=1)
    op.close()


def get_nan_num(spans_list):
    nan_num = 0
    for spans in spans_list:

        if len(spans) == 0:
            nan_num += 1

    return nan_num


def get_drange_labels(spans):
    drange_label_map = defaultdict(list)
    for span in spans:
        start_offset = span['start_offset']
        end_offset = span['end_offset']
        drange = (start_offset, end_offset)
        label = span['label']
        label_list = drange_label_map[drange]
        label_list.append(label)
        drange_label_map[drange] = label_list
    return drange_label_map


def get_drange_name(spans):
    drange_name_map = defaultdict(str)
    for span in spans:
        start_offset = span['start_offset']
        end_offset = span['end_offset']
        drange = (start_offset, end_offset)
        name = span['span_name']
        if not drange_name_map.__contains__(drange):
            drange_name_map[drange] = name
    return drange_name_map


def is_intersection(s1, e1, s2, e2):
    flag = True
    if ((e1 > s2 and e1 < e2) or (s1 < s2 and e1 > e2) or (s1 > s2 and s1 < e2)):
        flag = True
    else:
        flag = False
    return flag


def get_spans_by_vote(spans_list, k=2):
    '''

    @param: spans_list
    @return:
    '''

    '''
    首先union 所有的drange 
    '''

    all_drange_name_map = {}
    all_drange_set = set()
    drange_label_map_list = []

    for spans in spans_list:
        '''
        获取drange 的标签
        '''
        drange_label_map = get_drange_labels(spans)
        drange_label_map_list.append(drange_label_map)

        '''
        union 所有的drange
        '''
        all_drange_set = all_drange_set | set(drange_label_map.keys())

        '''
        获取drange 的name
        '''
        drange_name_map = get_drange_name(spans)
        all_drange_name_map.update(drange_name_map)


    '''
    遍历所有的drange ，获取其投票结果
    '''
    spans_result = []

    for drange in all_drange_set:

        '''
        统计每个文件标签预测次数
        '''
        label_num = defaultdict(int)
        for drange_label_map in drange_label_map_list:
            '''
            获取当前文件drange 的标签,是多标签
            '''
            labels = drange_label_map[drange]
            '''
            将标签次数加1
            '''
            for label in labels:
                label_num[label] = label_num[label] + 1

        '''
        获取标签次数出现大于K次，则认为他是属于当前标签
        '''
        label_list = []
        for label, num in label_num.items():
            if num >= k:
                label_list.append(label)
        '''
        输出标准结果
        '''
        for label in label_list:
            span_name = all_drange_name_map.get(drange)
            if span_name is not None:
                span_result = {
                    "span_name": span_name,
                    "start_offset": drange[0],
                    "end_offset": drange[1],
                    "label": label
                }
                spans_result.append(span_result)
            else:
                ''
                print("drange (%d, %d)的 span name is None" % (drange[0], drange[1]))

    return spans_result


def multi_vote(files_in, file_out, k=2):
    files = os.listdir(files_in)
    data_map = defaultdict(list)  # 每条样本存储多个list,多份文件包含这个样本

    file_num = 0
    for file in files:
        if file == ".ipynb_checkpoints":
            continue
        print(file)
        file_num += 1
        file_path = os.path.join(files_in, file)
        json_data = read_json(file_path)
        data_list = json_data['result']
        for i, data in enumerate(data_list):
            cur_datas = data_map[i]
            cur_datas.append(data)
            data_map[i] = cur_datas

    print("文件总个数为%d" % file_num)

    new_data_list = []

    for key, value in data_map.items():

        if len(value) != file_num:
            print("当前样本个数和文件个数不一致")

        spans_list = []
        for data in value:
            spans = data['spans']
            spans_list.append(spans)

        nan_num = get_nan_num(spans_list)

        if nan_num >= k:
            data = {
                "doc_name": data['doc_name'],
                "data_set_name": data['data_set_name'],
                "content": data['content'],
                "spans": data['spans'],
                # "pred_spans": []

            }
            new_data_list.append(data)
        else:
            '''
            统计spans 里面每个类别出现
            '''
            vote_spans = get_spans_by_vote(spans_list, k)
            vote_spans = sorted(vote_spans, key=lambda keys: keys.get("start_offset"), reverse=False)
            data = {
                "doc_name": data['doc_name'],
                "data_set_name": data['data_set_name'],
                "content": data['content'],
                # "spans": data['spans'],
                "spans": vote_spans

            }
            new_data_list.append(data)

    result = {
        "result": new_data_list
    }
    save_json(result, file_out)


if __name__ == '__main__':
    ''
    files_in = "../data/test"
    file_out = "../data/test_vote.json"
    multi_vote(files_in, file_out, k=3)
