# _*_ coding: utf-8 _*_
# @File : sentence_split.py
# @Time : 2021/10/29 9:27
# @Author : Yan Qiuxia
import re
import json
import math

http_pattern = re.compile(r'http[s]?://[a-zA-Z0-9.?/&=:]*', re.S)
www_pattern = re.compile(r'www.[a-zA-Z0-9.?/&=:]*', re.S)
html_pattern = re.compile(r'<[^>]+>', re.S)
html_en_pattern = re.compile(r'&lt;(.*?)&gt;', re.S)
sentence_separator = re.compile(r"(。|？|/?|！|!|/.{6}|…{2}|/.{3})", re.S)
min_sentence_length = 10


def cut_sent(text, k=1):
    new_sentences = []
    sentences = []
    if text is not None:
        sentences = re.split(sentence_separator, text)
        if len(sentences) <= k:
            return [''.join(sentences)]
        else:

            for var1, var2 in zip(sentences[0::2], sentences[1::2]):
                sentence = var1 + var2

                if var1.startswith("“") or var1.startswith("’") or len(var1) < min_sentence_length:
                    if len(new_sentences) > 1:
                        new_sentences[-1] = new_sentences[-1] + sentence
                    else:
                        new_sentences.append(sentence)
                else:
                    new_sentences.append(sentence)

            if (len(sentences) % 2) != 0:
                if len(sentences[-1]) < min_sentence_length:
                    new_sentences[-1] = new_sentences[-1] + sentences[-1]
                else:
                    new_sentences.append(sentences[-1])

            batch = int(len(new_sentences) / k)
            sentences = []
            i = 0
            if len(new_sentences) <= k:
                return [''.join(new_sentences)]
            else:
                for i in range(batch):
                    sentences.append(''.join(new_sentences[i * k:(i + 1) * k]))
                if (i + 1) * k < len(new_sentences):
                    sentences.append(''.join(new_sentences[(i + 1) * k:]))
            return sentences
    else:
        return sentences

def get_offsets(spans):
    offsets = set()
    for span in spans:
        offsets.add((span['start_offset'], span['end_offset']))
    return list(offsets)


def is_inoffsets(offset, offsets):
    flag = False
    drange = None
    for drange in offsets:
        if offset >= drange[0] and offset < drange[1]:
            flag = True
            break
    return flag, drange


def cut_sent_by_max_length(text, spans, max_length=510):
    '''
    @param text:
    @return:
    '''
    # 先用标点符号分句
    sentences = cut_sent(text)
    offsets = get_offsets(spans)
    # 使用最大长度分句
    new_sentences = []
    for sentence in sentences:
        if len(sentence)>max_length:
            chunk_contents = []

            chunk_end_offset = 0

            while chunk_end_offset <= len(sentence):
                chunk_start_offset = chunk_end_offset
                chunk_end_offset += max_length
                flag, drange = is_inoffsets(chunk_end_offset-1, offsets)
                if flag:
                    '''
                    最后截止位置刚好为drange 的位置，则重新切分
                    '''
                    chunk_end_offset = drange[0]

                chunk_content = sentence[chunk_start_offset:chunk_end_offset]
                chunk_contents.append(chunk_content)
            new_sentences.extend(chunk_contents)

        else:
            new_sentences.append(sentence)

    return new_sentences


def split_dataset_by_sent(file_path, file_out, k=1, dataset_type="train",max_length=510):
    fp = open(file_path, 'r', encoding='utf-8')
    json_data = json.load(fp)
    fp.close()
    data_list = json_data['result']
    new_data_list = []
    for data in data_list:
        doc_name = data['doc_name']
        data_set_name = data['data_set_name']
        content = data['content']
        spans = data['spans']

        sorted_spans = sorted(spans, key=lambda keys: keys.get("start_offset"), reverse=False)

        # sentences = cut_sent(content)
        sentences = cut_sent_by_max_length(content, spans, max_length)
        if dataset_type == "train":
            content_list, spans_list = combine_k_sentence(sentences, k, sorted_spans)
        elif dataset_type == 'test':
            content_list, spans_list = split_k_sentence(sentences, k, sorted_spans)

        for content_, spans_ in zip(content_list, spans_list):
            new_data = {
                "doc_name": doc_name,
                "data_set_name": data_set_name,
                "content": content_,
                "spans": spans_
            }
            new_data_list.append(new_data)
    op = open(file_out, 'w', encoding='utf-8')

    result_data = {
        "result": new_data_list
    }
    json.dump(result_data, op, ensure_ascii=False, indent=4)
    op.close()


def combine_k_sentence(sentences, k, spans=[]):
    '''
    将文章中所有句子进行组合 1,...,K为一块 2,...,k+1为一块，组成多个样本，并且更新spans的信息
    :param sentences:
    :param k:
    :param spans:
    :return:
    '''
    content_list = []
    spans_list = []  # 二维list
    cur_chunk_start_offset = 0
    cur_chunk_end_offset = 0
    for i in range(len(sentences)):
        content = sentences[i:i + k]

        cur_chunk_end_offset = cur_chunk_start_offset + len("".join(content))

        if len(sentences[i:i + k]) > 0:
            content_list.append(content)

            if len(spans) > 0:
                ''
                candicate_spans = get_chunk_spans(spans, cur_chunk_start_offset, cur_chunk_end_offset)
                spans_list.append(candicate_spans)
            else:
                spans_list.append([])
        cur_chunk_start_offset = cur_chunk_start_offset + len("".join(sentences[i]))

    return content_list, spans_list


def split_k_sentence(sentences, k, spans=[]):
    '''
    将文章中句子按k句切分，1,...,k 为一块， k+1,...,2k为一块，并且更新spans的信息
    :param sentences:
    :param k:
    :param spans:
    :return:
    '''

    content_list = []
    spans_list = []  # 二维list
    cur_chunk_start_offset = 0
    cur_chunk_end_offset = 0

    batch = math.ceil(len(sentences) / k)
    i = 0

    for i in range(batch):
        content = sentences[i * k:(i + 1) * k]
        cur_chunk_end_offset = cur_chunk_start_offset + len("".join(content))

        if len(sentences[i * k:(i + 1) * k]) > 0:
            content_list.append(content)
            if len(spans) > 0:
                ''
                candicate_spans = get_chunk_spans(spans, cur_chunk_start_offset, cur_chunk_end_offset)
                spans_list.append(candicate_spans)
            else:
                spans_list.append([])
        cur_chunk_start_offset = cur_chunk_end_offset

    return content_list, spans_list


def get_chunk_spans(spans, chunk_start_offset, chunk_end_offset):
    candicate_spans = []
    for span in spans:
        start_offset = span['start_offset']
        end_offset = span['end_offset']
        if start_offset < chunk_start_offset:
            continue
        if start_offset >= chunk_start_offset and end_offset <= chunk_end_offset:
            start_offset = start_offset - chunk_start_offset
            end_offset = end_offset - chunk_start_offset

            new_span = {
                "span_name": span['span_name'],
                "label": span['label'],
                "start_offset": start_offset,
                "end_offset": end_offset,

            }
            candicate_spans.append(new_span)

        if start_offset > chunk_end_offset:
            break

    return candicate_spans


def split_test_data_bychunk(content,spans=None):
    if isinstance(content,str):
        sentences = cut_sent(content)
    else:
        sentences = content
    chunk_start_offset = 0
    chunk_end_offset = 0
    batch_content = []
    batch_start_offset = []
    chunk_content = []
    chunk_length = 0
    i = 0
    batch_spans = []
    while i < len(sentences):

        if chunk_length + len(sentences[i]) <= 512:
            chunk_content.append(sentences[i])
            chunk_length += len(sentences[i])
        else:
            if i > 0:
                batch_content.append(chunk_content)
                batch_start_offset.append(chunk_start_offset)
                cur_chunk_spans = get_chunk_spans(spans, chunk_start_offset,
                                                  chunk_start_offset + len("".join(chunk_content)))
                batch_spans.append(cur_chunk_spans)
            chunk_content = [sentences[i]]
            chunk_length = len(sentences[i])
            chunk_start_offset = chunk_end_offset

        chunk_end_offset += len(sentences[i])

        i += 1

    if len(chunk_content)>0:
        batch_content.append(chunk_content)
        batch_start_offset.append(chunk_start_offset)
        cur_chunk_spans = get_chunk_spans(spans, chunk_start_offset,
                                          chunk_start_offset + len("".join(chunk_content)))
        batch_spans.append(cur_chunk_spans)

    return batch_content, batch_start_offset, batch_spans


def split_test_data_bysentence(content,spans):
    # sentences = cut_sent(content)
    sentences = cut_sent_by_max_length(content, spans, max_length=510)
    chunk_start_offset = 0
    chunk_end_offset = 0
    batch_content = []
    batch_start_offset = []
    batch_spans = []

    i = 0
    while i < len(sentences):
        chunk_content = [sentences[i]]
        batch_content.append(chunk_content)


        chunk_length = len(sentences[i])
        chunk_end_offset += chunk_length

        cur_chunk_spans = get_chunk_spans(spans, chunk_start_offset,
                                          chunk_end_offset)
        batch_spans.append(cur_chunk_spans)

        batch_start_offset.append(chunk_start_offset)
        chunk_start_offset = chunk_end_offset
        i += 1

    return batch_content, batch_start_offset,batch_spans

def recalc_offset(batch_start_offset, batch_entities):
    all_entities = []
    for start_offset, entities in zip(batch_start_offset, batch_entities):
        for entity in entities:
            entity['start_offset'] = entity['start_offset'] + start_offset
            entity['end_offset'] = entity['end_offset'] + start_offset
            all_entities.append(entity)

    sorted_spans = sorted(all_entities, key=lambda keys: keys.get("start_offset"), reverse=False)
    return sorted_spans


def get_stand_res(spans):
    res_spans = []
    for span in spans:
        label = span['label']
        if '-' in label:
            new_labels = label.split('-')
            for l in new_labels:
                span['label'] = l
                res_spans.append(span)
        else:
            res_spans.append(span)
    return res_spans

def find_all_indexs(input_str,search_str):
    l1 = []
    length = len(input_str)
    index = 0
    while index <length:
        i = input_str.find(search_str, index)
        if i==-1:
            return l1
        l1.append((i,i+len(search_str)))
        index = i+1
    return l1

def generare_org_label(file_path,file_out):
    '''
    将所有事件主体 替换为组织标签
    :param file_path:
    :return:
    '''
    fp = open(file_path, 'r', encoding='utf-8')
    json_data = json.load(fp)
    data_list = json_data['result']
    new_data_list = []
    for data in data_list:
        content = "".join(data['content'])
        spans = data['spans']
        new_spans = []
        new_spans_off = []
        if len(spans)>0:
            for span in spans:
                ''
                span_name = span['span_name']
                span['label'] = "ORG"
                span_all_offset = find_all_indexs(content, span_name)
                for span_offset in span_all_offset:
                    if span_offset not in new_spans_off:
                        new_spans_off.append(span_offset)
                        span['start_offset'] = span_offset[0]
                        span['end_offset'] = span_offset[1]
                        new_spans.append(span)

                # if (span["start_offset"], span['end_offset']) not in new_spans_off:
                #     new_spans.append(span)
                #     new_spans_off.append((span["start_offset"], span['end_offset']))

            data['spans'] = new_spans
            new_data_list.append(data)
        else:
            '''
            暂时不加 没有实体的数据，负样本太多了
            '''

    result = {
        "result":new_data_list
    }

    op = open(file_out, 'w', encoding='utf-8')
    json.dump(result, op, ensure_ascii=False, indent=1)
    op.close()


if __name__ == '__main__':
    ''

    file_path = "D:/PycharmProjects/single-cls/data/train.json"
    file_out = "D:/PycharmProjects/single-cls/data/dev_sentences.json"
    split_dataset_by_sent(file_path,file_out, k=1,dataset_type="test",max_length=510)

    file_path = "D:/PycharmProjects/single-cls/data/train.json"
    file_out = "D:/PycharmProjects/single-cls/data/train_sentences.json"
    split_dataset_by_sent(file_path, file_out, k=1, dataset_type="train",max_length=510)

    # file_path = "D:/PycharmProjects/event-extract-join/data/train_sentences.json"
    # check_span(file_path)
    #
    # file_path = "D:/PycharmProjects/event-extract-join/data/dev_sentences.json"
    # check_span(file_path)

    # file_path = "D:/PycharmProjects/yuqing_event_extract/data/train_sentences.json"
    # file_out = "D:/PycharmProjects/yuqing_event_extract/data/train_sentences_org.json"
    # generare_org_label(file_path, file_out)
