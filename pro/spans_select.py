# _*_ coding: utf-8 _*_
# @File : spans_select.py
# @Time : 2021/11/8 16:05
# @Author : Yan Qiuxia
import json

def read_json(path):
    with open(path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    return data

def save_to_json(data, path):

    with open(path, 'w', encoding='utf-8') as op:
        json.dump(data,op,ensure_ascii=False,indent=1)


def get_span_data(file_path, file_out):
    json_data = read_json(file_path)
    data_list = json_data['result']
    print("data num is %d"%len(data_list))
    new_data_list = []
    for data in data_list:
        spans = data['spans']
        if len(spans)>0:
            new_data_list.append(data)
    print("span data num is %d"%len(data_list))
    
    result = {
        "result":new_data_list
    }
    save_to_json(result,file_out)



if __name__ == '__main__':
    ''
    file_path = "D:/PycharmProjects/single-cls/data/train_sentences.json"
    file_out = "D:/PycharmProjects/single-cls/data/train_spans.json"
    get_span_data(file_path, file_out)

    file_path = "D:/PycharmProjects/single-cls/data/dev_sentences.json"
    file_out = "D:/PycharmProjects/single-cls/data/dev_spans.json"
    get_span_data(file_path, file_out)