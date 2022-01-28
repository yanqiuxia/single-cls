"""
Score the predictions with gold labels, using precision, recall and F1 metrics.
"""

import sys
from collections import Counter
import json


def read_json(path):
    with open(path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    return data


def get_dir_data(dir_path):
    json_data = read_json(dir_path)
    data_list = json_data['result']
    return data_list

def get_offsets_dict(spans):
    offsets_dict = {}
    for span in spans:
        offset = (span['start_offset'], span['end_offset'])
        if offset not in offsets_dict:
            offsets_dict[offset] = [span['label']]
        else:
            offsets_dict[offset].append(span['label'])

    return offsets_dict

def get_gold_dict(spans):
    gold_dict = {}
    for span in spans:
        label = span['label']
        start = span['start_offset']
        end = span['end_offset']
        if label in gold_dict:
            gold_dict[label].append((start, end))
        else:
            gold_dict[label] = [(start, end)]
    return gold_dict


def online_score(data_list):
    correct_by_subject = Counter()    # 预测出来是正确的
    guessed_by_subject = Counter()    # 预测出来的总数
    gold_by_subject = Counter()       # 标准答案的总数

    ner_correct = Counter()

    # Loop over the data to compute a score
    for data in data_list:
        key_spans = data['spans']
        pred_spans = data['pred_spans']
        if len(key_spans) > 0:
            if len(pred_spans) == 0:
                # 一个都没预测出来
                for k_span in key_spans:
                    label = k_span['label']
                    gold_by_subject[label] += 1

            else:
                for k_span in key_spans:
                    k_label = k_span['label']
                    gold_by_subject[k_label] += 1     # 实际为某一类的计数

                gold_dict = get_gold_dict(key_spans)
                gold_offset = get_offsets_dict(key_spans)
                for p_span in pred_spans:
                    p_label = p_span['label']
                    guessed_by_subject[p_label] += 1    # 预测为某一类的计数

                    start = p_span['start_offset']
                    end = p_span['end_offset']

                    if (start, end) in gold_offset:
                        ner_correct[p_label] += 1

                    if p_label in gold_dict:
                        if (start, end) in gold_dict[p_label]:
                            correct_by_subject[p_label] += 1    # 统计预测正确的数量（位置+类别）

        else:
            if len(pred_spans) > 0:
                # 全都预测错了
                for p_span in pred_spans:
                    label = p_span['label']
                    guessed_by_subject[label] += 1

    # Print information
    print("========================================")
    print("Per event subject statistics:")
    subjects = gold_by_subject.keys()
    longest_subject = 0
    precisions, reaclls, f1s = [], [], []
    cls_reaclls, cls_f1s = [], []
    for subject in sorted(subjects):
        longest_subject = max(len(subject), longest_subject)
    for subject in sorted(subjects):
        # (compute the score)
        correct = correct_by_subject[subject]
        guessed = guessed_by_subject[subject]
        gold = gold_by_subject[subject]
        prec = 1.0
        if guessed > 0:
            prec = float(correct) / float(guessed)
        recall = 0.0
        if gold > 0:
            recall = float(correct) / float(gold)
        f1 = 0.0
        if prec + recall > 0:
            f1 = 2.0 * prec * recall / (prec + recall)
        # (print the score)
        sys.stdout.write(("{:<" + str(longest_subject) + "}").format(subject))
        sys.stdout.write("  P(分类): ")
        if prec < 0.1: sys.stdout.write(' ')
        if prec < 1.0: sys.stdout.write(' ')
        sys.stdout.write("{:.2%}".format(prec))
        sys.stdout.write("  R: ")
        if recall < 0.1: sys.stdout.write(' ')
        if recall < 1.0: sys.stdout.write(' ')
        sys.stdout.write("{:.2%}".format(recall))
        sys.stdout.write("  F1: ")
        if f1 < 0.1: sys.stdout.write(' ')
        if f1 < 1.0: sys.stdout.write(' ')
        sys.stdout.write("{:.2%}".format(f1))
        sys.stdout.write("  事件总数: %d" % gold)
        sys.stdout.write("  预测: %d" % guessed)
        sys.stdout.write("  实体: %d" % ner_correct[subject])
        sys.stdout.write("  实体+分类: %d" % correct)
        sys.stdout.write("  分类召回: ")
        cls_r = 0.0
        if ner_correct[subject] > 0:
            cls_r = float(correct) / float(ner_correct[subject])
        sys.stdout.write("{:.2%}".format(cls_r))

        sys.stdout.write("  实体精确率: ")
        ner_p = 0.0
        if guessed > 0:
            ner_p = float(ner_correct[subject]) / float(guessed)
        sys.stdout.write("{:.2%}".format(ner_p))
        sys.stdout.write("  实体召回率: ")
        sys.stdout.write("{:.2%}".format(float(ner_correct[subject]) / float(gold)))
        sys.stdout.write("\n")

        precisions.append(prec)
        reaclls.append(recall)
        f1s.append(f1)
        cls_reaclls.append(cls_r)
        cls_f1 = 0.0
        if prec + recall > 0:
            cls_f1 = 2.0 * prec * cls_r / (prec + cls_r)
        cls_f1s.append(cls_f1)
    print("========================================")
    cls_macro_r = sum(cls_reaclls) / len(subjects)
    cls_macro_f1 = sum(cls_f1s) / len(subjects)

    prec_macro = sum(precisions) / len(subjects)
    recall_macro = sum(reaclls) / len(subjects)
    f1_macro = sum(f1s) / len(subjects)
    # Print the aggregate score
    print("分类指标:")
    print("Precision (macro-avg): {:.3%}".format(prec_macro))
    print("   Recall (macro-avg): {:.3%}".format(cls_macro_r))
    print("       F1 (macro-avg): {:.3%}".format(cls_macro_f1))
    print("（实体+分类）Final Score:")
    print("Precision (macro-avg): {:.3%}".format(prec_macro))
    print("   Recall (macro-avg): {:.3%}".format(recall_macro))
    print("       F1 (macro-avg): {:.3%}".format(f1_macro))

    return prec_macro, recall_macro, f1_macro


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

if __name__ == "__main__":
    file_path = "/home/aipf/work/single-cls/data/dev_cls_predict.json"
    data_list = get_dir_data(file_path)
    online_score(data_list)

    #
    # true_file = "D:/PycharmProjects/event-extract-join/data/train.json"
    # pred_file = "D:/PycharmProjects/event-extract-join/data/train.json"
    # data_list = merge_true_pred_file(true_file, pred_file)
    # online_score(data_list)

