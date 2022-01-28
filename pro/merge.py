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


def get_spans_by_label(spans_list, k=2,  need_label="破产事件主体", doc_id=None):
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
            将标签次数加1,只统计需要标签的次数
            '''
            for label in labels:
                if label == need_label :
                    label_num[label] = label_num[label] + 1

        '''
        获取标签属于当前标签的
        '''
        label_list = []
        for label, num in label_num.items():
            if num >= k:
                label_list.append(label)

        if len(label_list)>0:
            print("标签出现次数",label_num)

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
                    "label": label,
                    "num": label_num.get(label),
                    "doc_id": doc_id,
                }
                spans_result.append(span_result)
            else:
                ''
                print("drange (%d, %d)的 span name is None" % (drange[0], drange[1]))

    return spans_result


def get_multi_file_span(files_in, file_out, k=2, label="破产事件主体"):
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
            spans = data['pred_spans']
            spans_list.append(spans)

        '''
        统计spans 里面每个类别出现
        '''
        if label == "董监高成员异常事件主体" and key==130:
            print("文章130 标签为董监高成员异常事件主体")
            data = {
                "doc_name": data['doc_name'],
                "data_set_name": data['data_set_name'],
                "content": data['content'],
                "spans": []
            }
        else:
            vote_spans = get_spans_by_label(spans_list, k, label, key)
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


def get_non_label_spans(spans,need_label="破产事件主体"):
    new_spans = []
    for span in spans:
        label = span['label']
        if label!=need_label:
            new_spans.append(span)
    return new_spans


def get_label_spans(spans,need_label="破产事件主体"):
    new_spans = []
    for span in spans:
        label = span['label']
        if label==need_label:
            new_span = {
                "span_name":span['span_name'],
                "start_offset":span['start_offset'],
                "end_offset":span['end_offset'],
                "label":span['label']
            }
            new_spans.append(new_span)
    return new_spans


def merge_two_file(file1, file2, file_out, label="破产事件主体"):
    json_data = read_json(file1)
    data_list1 = json_data['result']

    json_data = read_json(file2)
    data_list2 = json_data['result']
    new_data_list = []
    for data1, data2 in zip(data_list1, data_list2):
        spans1 = data1['spans']
        spans2 = data2['spans']
        print("spans 长度%d" % len(spans1))
        need_spans1 = get_non_label_spans(spans1, label)
        need_spans2 = get_label_spans(spans2, label)
        need_spans1 += need_spans2
        print("new spans 长度%d" % len(need_spans1))
        print("*************************************")

        data1['spans'] = need_spans1

    result = {
        "result": data_list1
    }
    save_json(result, file_out)





if __name__ == '__main__':
    ''

    files_in = "/home/aipf/work/single-cls/data/testB/cls2"
    file_out = "/home/aipf/work/single-cls/data/testB/test_董监高成员异常事件主体.json"
    k = 1
    label = "董监高成员异常事件主体"

    get_multi_file_span(files_in, file_out, k=k, label=label)

    # file1 = "/home/aipf/work/single-cls/data/testB/vote_5_1125.json"
    # file2 = "/home/aipf/work/single-cls/data/testB/test_破产事件主体.json"
    # file_out = "/home/aipf/work/single-cls/data/testB/vote_5_1125_破产.json"
    #
    # merge_two_file(file1, file2, file_out, label="破产事件主体")


