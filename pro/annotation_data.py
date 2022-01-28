import json


def get_offsets(spans):
    offsets = set()
    for span in spans:
        offsets.add((span['start_offset'], span['end_offset']))
    return list(offsets)


def find_all_indexs(input_str, search_str):
    l1 = []
    length = len(input_str)
    index = 0
    while index < length:
        i = input_str.find(search_str, index)
        if i == -1:
            return l1
        l1.append((i, i+len(search_str)))
        index = i+1
    return l1


def annotation_data(data_list):

    for data in data_list:
        spans = data['spans']
        content = data['content']
        if isinstance(content, list):
            content = "".join(content)
        if len(spans) > 0:
            true_offsets = get_offsets(spans)
            new_spans = []
            extra_span = {}
            for span in spans:
                span_name = span['span_name']
                label = span['label']
                if label != "NA":
                    all_pos = find_all_indexs(content, span_name)
                    for pos in all_pos:
                        if pos not in true_offsets:
                            extra_span[pos] = span_name

            for pos, name in extra_span.items():
                one_span = {
                    'span_name': name,
                    'label': 'NA',
                    'start_offset': pos[0],
                    'end_offset': pos[1]
                }
                new_spans.append(one_span)

            new_spans += spans
            data['spans'] = new_spans

    return data_list


def annotation_file(file_path, file_out):
    fp = open(file_path, 'r', encoding='utf-8')
    op = open(file_out, 'w', encoding="utf-8")
    json_data = json.load(fp)
    data_list = json_data['result']
    
    data_list = annotation_data(data_list)
    
    result = {
        'result': data_list
    }
    
    json.dump(result, op, ensure_ascii=False, indent=1)
    
    fp.close()
    op.close()



if __name__ == '__main__':
    ''
    file_path = "/home/aipf/work/single-cls/data/v0_0_8/train_spans.json"
    file_out = "/home/aipf/work/single-cls/data/v0_0_8/train_aug_spans.json"

    annotation_file(file_path, file_out)