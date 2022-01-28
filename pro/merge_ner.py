import json

def save_json(data, file_out):
    op = open(file_out, 'w', encoding="utf-8")

    json.dump(data, op, ensure_ascii=False, indent=1)
    op.close()

def get_dir_data(dir_path):
    with open(dir_path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
        data_list = data['result']
        print(len(data_list))
    return data_list


def get_offsets(spans):
    offsets = set()
    for span in spans:
        offsets.add((span['start_offset'], span['end_offset']))
    return list(offsets)


def merge_spans(data_list1, data_list2, file_out):
    new_data_list = []
    for data1, data2 in zip(data_list1, data_list2):
        spans1 = data1['pred_spans']
        spans2 = data2['pred_spans']
        pred_spans = []
        if len(spans1) > 0 and len(spans2) > 0:
            offsets2 = get_offsets(spans2)

            for span in spans1:
                start = span['start_offset']
                end = span['end_offset']
                if (start, end) in offsets2:
                    pred_spans.append(span)

        new_data = data1
        new_data['pred_spans'] = pred_spans
        new_data_list.append(new_data)

    result = {
        "result":new_data_list
    }

    save_json(result,file_out)


def union_spans(data_list1, data_list2, file_out):
    new_data_list = []

    for data1, data2 in zip(data_list1, data_list2):
        spans1 = data1['pred_spans']
        spans2 = data2['pred_spans']
        pred_spans = spans1
        if len(spans1) > 0 :
            offsets1 = get_offsets(spans1)
            for span in spans2:
                start = span['start_offset']
                end = span['end_offset']
                if (start, end) not in offsets1:
                    pred_spans.append(span)

        new_data = data1
        new_data['pred_spans'] = pred_spans
        new_data_list.append(new_data)

    result = {
        "result": new_data_list
    }

    save_json(result, file_out)


if __name__ == '__main__':
    data1 = get_dir_data('D:/PycharmProjects/single-cls/data/test.json')
    data2 = get_dir_data('D:/PycharmProjects/single-cls/data/test1.json')
    file_out = 'D:/PycharmProjects/single-cls/data/test_union.json'

    # merge_spans(data1, data2,file_out)

    union_spans(data1, data2, file_out)

