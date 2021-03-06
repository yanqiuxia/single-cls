# _*_ coding: utf-8 _*_
# @File : lac_seg.py
# @Time : 2021/11/1 19:24
# @Author : Yan Qiuxia
from LAC import LAC
import json
from sentence_split import cut_sent
lac = LAC(mode="lac")

def segment(file_path, file_out):

    fp = open(file_path, 'r', encoding="utf-8")
    json_data = json.load(fp)
    fp.close()
    data_list = json_data['result']
    for data in data_list:
        content = data['content']
        sentences = cut_sent(content)
        seg_results = lac.run(sentences)
        sentence_start_offset = 0
        spans = []
        for sentence, seg_result in zip(sentences,seg_results):

            start_offset = sentence_start_offset
            end_offset = sentence_start_offset
            for word, tag in zip(seg_result[0],seg_result[1]):
                end_offset += len(word)
                if tag=="ORG":
                    span = {
                        "span_name":word,
                        "label":"ORG",
                        "start_offset":start_offset,
                        "end_offset":end_offset
                    }
                    spans.append(span)
                start_offset = end_offset
            
            sentence_start_offset = end_offset
        data['lac_spans'] = spans
    op = open(file_out, 'w',encoding='utf-8')
    result = {
        "result":data_list
    }
    json.dump(result,op,ensure_ascii=False,indent=1)
    op.close()


def get_offsets(spans):
    offsets = set()
    for span in spans:
        offsets.add((span['start_offset'], span['end_offset']))
    return list(offsets)

def is_intersection(a1,b1,a2,b2):
    flag = True
    if ((b1>a2 and b1<b2) or (a1<a2 and b1>b2) or (a1>a2 and a1<b2)):
        flag = True
    else:
        flag = False

def compute_recall_prec(data_list):

    TP, true_count = 0, 0
    FN, pred_count = 0, 0
    for data in data_list:
        spans = data.get('spans')

        cut_spans = data.get('lac_spans')
        true_offsets = get_offsets(spans)
        true_count += len(true_offsets)
        pred_offsets = get_offsets(cut_spans)
        pred_count += len(pred_offsets)
        for pred_off in pred_offsets:
            if pred_off in true_offsets:
                TP += 1
            for true_off in true_offsets:
                flag = is_intersection(pred_off[0],pred_off[1],true_off[0],true_off[1])
                if flag:
                    FN += 1

    recall = 0.0
    pred = 0.0
    if true_count == 0:
        recall = 1.0
    else:
        recall = round(TP / true_count, 5)

    if pred_count == 0:
        pred = 1.0
    else:
        pred = round(1- FN/ pred_count, 5)

    print("?????????%f, ?????????%f, ???????????????%d?????????????????????%d, ?????????????????????%d, ??????????????????%d"%(recall,pred,TP,true_count,FN,pred_count))



if __name__ == '__main__':
    ''
    text = """?????????????????????????????????????????????????????????????????????????????????\t11???4??????????????????????????????????????????????????????????????????????????????5%???????????????????????????????????????????????????3%???????????????????????????????????????????????????
????????????????????????????????????????????????????????????????????????8???????????????2019??????????????????????????????????????????????????????????????????1166.65?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
???17.19??????????????????21.7%????????????????????????????????????3.66?????????????????????85.05%?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????2019?????????????????????????????????????????????????????????????????????
?????????????????????8???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????4.4%???17.2%???9.8%???26.1%???-19.5%?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????>????????????????????????????????????????????????5.7%???????????????????????????????????????737MAX?????????R5??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
???????????????????????????????????????????????????????????????????????????????????????11??????????????????????????????????????????????????????????????????19????????????74%??????????????????3.7%???????????????????????????????????????????????????????????????????????????????????????????????????>????????????????????????????????????????????????S1?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????50%?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????>?????????????????????????????????????????????????????????2019??????9????????????????????????????????????????????????5.0????????????????????????8.7%??????2018?????????????????????11.6%????????????????????????????????????????????????????????????????????????????????????????????????????????????>?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
"""
    seg_result = lac.run(text)
    for w,tag in zip(seg_result[0],seg_result[1]):
        print(w,tag)

