# _*_ coding: utf-8 _*_
# @File : vote.py
# @Time : 2021/11/19 10:56
# @Author : Yan Qiuxia
import json
from collections import defaultdict

def read_json(path):
    with open(path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    return data


def save_json(data, file_out):
    op = open(file_out, 'w', encoding="utf-8")

    json.dump(data, op, ensure_ascii=False, indent=1)
    op.close()


def get_nan_num(spans1,spans2,spans3,spans4,spans5, spans6, spans7):
    nan_num =0
    if len(spans1)==0:
        nan_num += 1

    if len(spans2)==0:
        nan_num += 1

    if len(spans3)==0:
        nan_num += 1

    if len(spans4)==0:
        nan_num += 1

    if len(spans5)==0:
        nan_num += 1

    if len(spans6) == 0:
        nan_num += 1

    if len(spans7) == 0:
        nan_num += 1

    return nan_num


def get_drange_labels(spans):
    drange_label_map = defaultdict(list)
    for span in spans:
        start_offset = span['start_offset']
        end_offset = span['end_offset']
        drange = (start_offset,end_offset)
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


def get_spans_by_vote(spans1,spans2,spans3,spans4,spans5,spans6,spans7,k=4):
    '''

    @param spans1:
    @param spans2:
    @param spans3:
    @param spans4:
    @param spans5:
    @param spans6
    @param spans7
    @return:
    '''

    drange_label_map1 = get_drange_labels(spans1)
    drange_label_map2 = get_drange_labels(spans2)
    drange_label_map3 = get_drange_labels(spans3)
    drange_label_map4 = get_drange_labels(spans4)
    drange_label_map5 = get_drange_labels(spans5)
    drange_label_map6 = get_drange_labels(spans6)
    drange_label_map7 = get_drange_labels(spans7)

    drange_name_map1 = get_drange_name(spans1)
    drange_name_map2 = get_drange_name(spans2)
    drange_name_map3 = get_drange_name(spans3)
    drange_name_map4 = get_drange_name(spans4)
    drange_name_map5 = get_drange_name(spans5)
    drange_name_map6 = get_drange_name(spans6)
    drange_name_map7 = get_drange_name(spans7)

    all_drange_name_map = drange_name_map1
    all_drange_name_map.update(drange_name_map2)
    all_drange_name_map.update(drange_name_map3)
    all_drange_name_map.update(drange_name_map4)
    all_drange_name_map.update(drange_name_map5)
    all_drange_name_map.update(drange_name_map6)
    all_drange_name_map.update(drange_name_map7)

    all_drange_set = set(drange_label_map1.keys()) | set(drange_label_map2.keys()) \
                     | set(drange_label_map3.keys() | set(drange_label_map4.keys())) \
                     | set(drange_label_map5.keys()) | set(drange_label_map6.keys()) \
                     |set(drange_label_map7.keys())

    spans_result = []

    for drange in all_drange_set:
        labels1 = drange_label_map1[drange]
        labels2 = drange_label_map2[drange]
        labels3 = drange_label_map3[drange]
        labels4 = drange_label_map4[drange]
        labels5 = drange_label_map5[drange]
        labels6 = drange_label_map6[drange]
        labels7 = drange_label_map7[drange]

        label_num = defaultdict(int)

        for label in labels1:
            label_num[label] = label_num[label] + 1

        for label in labels2:
            label_num[label] = label_num[label] + 1

        for label in labels3:
            label_num[label] = label_num[label] + 1

        for label in labels4:
            label_num[label] = label_num[label] + 1

        for label in labels5:
            label_num[label] = label_num[label] + 1

        for label in labels6:
            label_num[label] = label_num[label] + 1

        for label in labels7:
            label_num[label] = label_num[label] + 1

        label_list = []
        for label, num in label_num.items():
            if num>=k:
                label_list.append(label)
        for label in label_list:
            span_name = all_drange_name_map.get(drange)
            if span_name is not None:
                span_result = {
                    "span_name":span_name,
                    "start_offset":drange[0],
                    "end_offset":drange[1],
                    "label":label
                }
                spans_result.append(span_result)
            else:
                ''
                print("drange (%d, %d)的 span name is None"%(drange[0],drange[1]))

    return spans_result


def multi_vote(file1, file2, file3, file4, file5, file6, file7, file_out,k=3):
    data_list1 =None
    data_list2 = None
    data_list3 = None
    data_list4 = None
    data_list5 = None
    data_list6 = None
    data_list7 = None

    if file1 is not None:
        json_data1 = read_json(file1)
        data_list1 = json_data1['result']

    if file2 is not None:
        json_data2 = read_json(file2)
        data_list2 = json_data2['result']

    if file3 is not None:
        json_data3 = read_json(file3)
        data_list3 = json_data3['result']

    if file4 is not None:
        json_data4 = read_json(file4)
        data_list4 = json_data4['result']

    if file5 is not None:
        json_data5 = read_json(file5)
        data_list5 = json_data5['result']

    if file6 is not None:
        json_data6 = read_json(file6)
        data_list6 = json_data6['result']

    if file7 is not None:
        json_data7 = read_json(file7)
        data_list7 = json_data7['result']

    if data_list1 is not None and data_list2 is not None:
        if len(data_list1)!=len(data_list2):
            print("数据个数不一致")

    if data_list1 is not None and data_list3 is not None:
        if len(data_list1)!=len(data_list3):
            print("数据个数不一致")

    if data_list1 is not None and data_list4 is not None:
        if len(data_list1) != len(data_list4):
            print("数据个数不一致")

    if data_list1 is not None and data_list5 is not None:
        if len(data_list1) != len(data_list5):
            print("数据个数不一致")

    if data_list1 is not None and data_list6 is not None:
        if len(data_list1) != len(data_list6):
            print("数据个数不一致")

    if data_list1 is not None and data_list7 is not None:
        if len(data_list1) != len(data_list7):
            print("数据个数不一致")

    new_data_list = []
    for data1, data2,data3,data4,data5,data6,data7 in \
            zip(data_list1,data_list2,data_list3,data_list4,data_list5,data_list6, data_list7):
        spans1 = data1["pred_spans"]
        spans2 = data2["pred_spans"]
        spans3 = data3["pred_spans"]
        spans4 = data4["pred_spans"]
        spans5 = data5["pred_spans"]
        spans6 = data6['pred_spans']
        spans7 = data7['pred_spans']
        nan_num = get_nan_num(spans1, spans2, spans3, spans4, spans5, spans6, spans7)
        if nan_num>=k:
            data = {
                "doc_name": data1['doc_name'],
                "data_set_name": data1['data_set_name'],
                "content": data1['content'],
                # "spans":data1['spans'],
                "pred_spans": []

            }
            new_data_list.append(data)
        else:
            '''
            统计spans 里面每个类别出现
            '''
            vote_spans = get_spans_by_vote(spans1,spans2,spans3,spans4,spans5,spans6,spans7, k)
            vote_spans = sorted(vote_spans, key=lambda keys: keys.get("start_offset"), reverse=False)
            data = {
                "doc_name": data1['doc_name'],
                "data_set_name": data1['data_set_name'],
                "content": data1['content'],
                # "spans": data1['spans'],
                "pred_spans": vote_spans

            }
            new_data_list.append(data)
    result = {
        "result":new_data_list
    }
    save_json(result,file_out)


if __name__ == '__main__':
    ''
    file1 = "../data/test/test1.json"
    file2 = "../data/test/test2.json"
    file3 = "../data/test/test3.json"
    file4 = "../data/test/test4.json"
    file5 = "../data/test/test5.json"
    file6 = ""
    file7 = ""
    file_out = "../data/test/test_vote.json"
    multi_vote(file1, file2, file3, file4, file5, file6, file7, file_out, k=4)


