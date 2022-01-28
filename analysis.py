# _*_ coding: utf-8 _*_
# @Time : 2020/10/10 下午2:48 
# @Author : yanqiuxia
# @Version：V 0.1
# @File : analysis.py
import re
from collections import OrderedDict

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer
from util.log import logger


def safe_divide(x, y):
    if y == 0.0:
        return 0.0
    return x / y


def calc_metrics(y_trues, y_preds, is_train=True, id2label=None):

    if is_train:
        result = OrderedDict({
            "准确率": 0.0,
            "平均精确率": 0.0,
            "平均召回率": 0.0,
            "平均F1": 0.0,
            "微精确率": 0.0,
            "微召回率": 0.0,
            "微F1": 0.0,
            "宏精确率": 0.0,
            "宏召回率": 0.0,
            "宏F1": 0.0,
        })
    else:
        result = OrderedDict({
            "准确率": 0.0,
            "平均精确率": 0.0,
            "平均召回率": 0.0,
            "平均F1": 0.0,
            "微精确率": 0.0,
            "微召回率": 0.0,
            "微F1": 0.0,
            "宏精确率": 0.0,
            "宏召回率": 0.0,
            "宏F1": 0.0,
        })


    acc = accuracy_score(y_trues, y_preds)

    p_micro = precision_score(y_trues, y_preds, average='micro')
    r_micro = recall_score(y_trues, y_preds, average='micro')
    f1_micro = f1_score(y_trues, y_preds, average='micro')

    result["准确率"] = round(acc, 4)
    result["微精确率"] =  round(p_micro, 4)
    result["微召回率"] = round(r_micro, 4)
    result["微F1"] =round(f1_micro, 4)

    p_macro = precision_score(y_trues, y_preds, average='macro')
    r_macro = recall_score(y_trues, y_preds, average='macro')
    f1_macro = f1_score(y_trues, y_preds, average='macro')

    result["宏精确率"] = round(p_macro, 4)
    result["宏召回率"] = round(r_macro, 4)
    result["宏F1"] = round(f1_macro, 4)

    result['平均精确率'] = round((p_micro + p_macro) / 2, 4)
    result['平均召回率'] = round((r_micro + r_macro) / 2, 4)
    avg_f1 = (f1_micro + f1_macro) / 2
    result['平均F1'] = round(avg_f1, 4)

    details = {}
    if not is_train:
        classify_report = classification_report(y_trues, y_preds)
        classify_report = format_classify_report(classify_report,id2label)
        print(classify_report)
        # details = get_detail_from_report(classify_report, id2label)

        # try:
        #     auc_micro = roc_auc_score(y_trues, y_preds, average='micro')
        #     auc_macro = roc_auc_score(y_trues, y_preds, average='macro')
        # except Exception as e:
        #     logger.exception(e)
        #     auc_micro = 0.0
        #     auc_macro = 0.0
        # finally:
        #     result['微auc'] = round(auc_micro, 4)
        #     result['宏auc'] = round(auc_macro, 4)
        result['details'] = details

    return result




def recalc_macro(y_trues, y_preds, id_beta):
    '''

    :param y_trues:
    :param y_preds:
    :param id_beta:
    :return:
    '''
    # 用于计算宏平均
    true_label_num = np.sum(y_trues, axis=0) + 1e-32
    pred_label_num = np.sum(y_preds, axis=0) + 1e-32
    correct_label_num = np.sum(np.multiply(y_trues, y_preds), axis=0)

    p_macro_arr = correct_label_num / pred_label_num
    r_macro_arr = correct_label_num / true_label_num
    f_beta_macro_arr = 2 * p_macro_arr * r_macro_arr / (p_macro_arr + r_macro_arr + 1e-32)

    for id_, beta in id_beta.items():
        f_beta_macro_arr[id_] = safe_divide((1 + beta ** 2) * p_macro_arr[id_] * r_macro_arr[id_],
                                            beta ** 2 * p_macro_arr[id_] + r_macro_arr[id_])

    p_macro = np.mean(p_macro_arr)
    r_macro = np.mean(r_macro_arr)
    f_beta_macro = np.mean(f_beta_macro_arr)
    return p_macro, r_macro, f_beta_macro


def format_classify_report(classify_report, id2label):
    splits = re.split('\n', classify_report)
    new_classify_report = 'classify_report: \n'
    for i, var in enumerate(splits):

        if (i >= 2 and i < 2 + len(id2label)):
            temps = var.split(' ')
            if i < 12:
                id_ = int(temps[11])
                label = id2label.get(id_)
                temps[11] = label
            else:
                id_ = int(temps[10])
                label = id2label.get(id_)
                temps[10] = label

            var2 = ' '.join(temps)
            new_classify_report += var2
            new_classify_report += '\n'
        else:
            new_classify_report += var
            new_classify_report += '\n'
    return new_classify_report


def get_detail_from_report(classify_report, id2label):
    splits = re.split('\n', classify_report)

    details = {}
    index_values = []
    for i, var in enumerate(splits):
        if i == 0:
            ''
            temps = var.split()
            for temp in temps:
                if temp == 'f1-score':
                    temp = 'F1'
                if temp == "precision":
                    temp = '精确率'
                if temp == "recall":
                    temp = "召回率"
                if temp == "support":
                    temp = "数据总数"
                index_values.append(temp)
            continue

        if i == 1:
            continue

        temps = var.split()
        detail = OrderedDict({
            "精确率":0.0,
            "召回率":0.0,
            "F1":0.0,
            "数据总数":0.0,
        })
        if len(temps) == 0:
            break
        for j, temp in enumerate(temps):
            if j == 0:
                temp = int(temp)
                if id2label is not None:
                    label = id2label.get(temp)
                else:
                    label = temp
            else:
                if j == len(temps) - 1:
                    detail[index_values[j - 1]] = int(temp)
                else:
                    detail[index_values[j - 1]] = round(float(temp), 4)

        details[label] = detail


    return details


def re_calc_details(details, id2label, id_beta):
    for key, value in id_beta.items():
        label = id2label.get(key)

        if details.__contains__(label):
            precision = details.get(label).get('精确率')
            recall = details.get(label).get("召回率")
            f_beta = safe_divide((1 + value ** 2) * precision * recall, value ** 2 * precision + recall)
            details[label]['F1'] = f_beta

    return details


if __name__ == '__main__':
    labels_num = 5
    batch_size = 3
    classes = np.arange(0, labels_num, 1)
    print("标签id", classes)

    id2label = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4
    }

    mlb = MultiLabelBinarizer(classes=classes)
    classes = np.reshape(classes, [labels_num, 1])
    mlb.fit_transform(classes)

    y_trues = [[0], [1, 2], [1]]
    y_preds = [[0], [1], [0, 1]]
    y_trues = mlb.transform(y_trues)
    y_preds = mlb.transform(y_preds)

    print('y_trues:\n', y_trues)
    print('y_preds:\n', y_preds)
    result = calc_metrics(y_trues, y_preds, id2label, False)
    print('result:\n', result)
