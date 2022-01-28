from LAC import LAC
import json


def get_names(spans):
    names = set()
    for span in spans:
        names.add((span['span_name']))
    return list(names)


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
        l1.append((i, i + len(search_str)))
        index = i + 1
    return l1


def annotation_lac_data(data_list):
    # LAC实体识别
    lac = LAC(mode='lac')

    print("数据总数为%d"%len(data_list))

    for i, data in enumerate(data_list):
        spans = data['spans']
        content = data['content']
        if i%1000==0:
            print("已分词的数据量为%d"%i)

        lac_content = lac.run(content)
        # print(lac_content)
        org_words = set()
        for word, ner in zip(lac_content[0], lac_content[1]):
            if ner == 'ORG':
                org_words.add(word)

        names = get_names(spans)
        offsets = get_offsets(spans)
        new_spans = []
        extra_span = {}
        # 回标
        if len(spans) > 0:
            for org_w in org_words:
                all_pos = find_all_indexs(content, org_w)
                if org_w not in names:
                    for pos in all_pos:
                        extra_span[pos] = org_w
                else:
                    for pos in all_pos:
                        if pos not in offsets:
                            extra_span[pos] = org_w
        else:
            for org_w in org_words:
                all_pos = find_all_indexs(content, org_w)
                for pos in all_pos:
                    extra_span[pos] = org_w

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

    data_list = annotation_lac_data(data_list)

    result = {
        'result': data_list
    }

    json.dump(result, op, ensure_ascii=False, indent=1)

    fp.close()
    op.close()


if __name__ == '__main__':
    ''
    file_path = "D:/PycharmProjects/event-extract-join/data/test2.json"
    file_out = "D:/PycharmProjects/event-extract-join/data/result.json"

    annotation_file(file_path, file_out)
