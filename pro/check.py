# _*_ coding: utf-8 _*_
# @File : check.py
# @Time : 2021/11/9 9:35
# @Author : Yan Qiuxia
import json


def read_json(path):
    with open(path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    return data


def check_two_file(file_path1,file_path2):
    json_data1 = read_json(file_path1)
    json_data2 = read_json(file_path2)
    data_list1 = json_data1['result']
    data_list2 = json_data2['result']
    flag = True
    count = 0
    for i, (data1, data2) in enumerate(zip(data_list1,data_list2)):
        spans1 = data1['spans']
        spans2 = data2['spans']
        if len(spans1)!=len(spans2):
            print("第%d数据不一致"%i)
            flag = False
            count += 1
        else:
            sorted_spans1 = sorted(spans1, key=lambda keys: keys.get("start_offset"), reverse=False)
            sorted_spans2 = sorted(spans2, key=lambda keys: keys.get("start_offset"), reverse=False)
            for span1,span2 in zip(sorted_spans1,sorted_spans2):
                start_offset1 = span1['start_offset']
                end_offset1 = span1['end_offset']
                label1 = span1['label']

                start_offset2 = span2['start_offset']
                end_offset2 = span2['end_offset']
                label2 = span2['label']

                if start_offset1==start_offset2 and end_offset1==end_offset2 and label1==label2:
                   ''
                else:
                    print(" doc id %d span 不一致"%i)
                    flag = False

                    count += 1

    if flag:
        print("两个文件完全一致")
    else:
        print("两个文件不一致")

    print("不一致的文档个数%d"%count)


def check_file_max_length(file_path, max_length=510):
    json_data = read_json(file_path)
    data_list = json_data['result']
    count = 0
    for data in data_list:
        content = data['content']
        if len(content) > max_length:
            count += 1

    print("数据超过最大长度的数据为%d" % count)


def check_span(file_path):
    fp = open(file_path, 'r', encoding='utf-8')
    json_data = json.load(fp)
    data_list = json_data['result']
    count = 0
    for data in data_list:
        spans = data['spans']
        content = data['content']
        if isinstance(content, list):
            content = "".join(content)
        if len(spans) > 0:
            ''
            for span in spans:
                start_offset = span['start_offset']
                end_offset = span['end_offset']
                span_name = span['span_name']
                if span_name != content[start_offset:end_offset]:
                    print('span 在文章的位置不对！')
                    count += 1
    print("span 在文章的位置不对的数目为%d" % count)
    fp.close()


if __name__ == '__main__':
    ''
    # file_path = "D:/PycharmProjects/single-cls/data/dev_sentences.json"
    # check_file_max_length(file_path, 510)
    # check_span(file_path)
    #
    # file_path = "D:/PycharmProjects/single-cls/data/train_sentences.json"
    # check_file_max_length(file_path, 510)
    # check_span(file_path)

    file_path1 = "/home/aipf/work/团队共享目录/初赛/single-cls/data/testA/predA_cls_vote5_1121.json"
    file_path2 = "/home/aipf/work/团队共享目录/初赛/single-cls/data/testA/vote_5_1126.json"
    check_two_file(file_path1, file_path2)