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

    print("召回率%f, 准确率%f, 准确的个数%d，真实实体个数%d, 预测错误的个数%d, 预测实体个数%d"%(recall,pred,TP,true_count,FN,pred_count))



if __name__ == '__main__':
    ''
    text = """三大航业绩领跑行业，产能逐渐释放，机场航运概念依然可期\t11月4日上午，机场航运概念集体走强，截至发稿，南方航空涨近5%，中国国航、东方航空、吉祥航空涨逾3%，华夏航空、白云机场等多股纷纷跟涨
。近期，多家航空企业相继公布三季报，航空板块中，8家上市航企2019年前三季度营收全部同比增长，其中，南方航空以1166.65亿元的营收领跑，同比增速也最快。春秋航空仍保持率润率高速增长，归属上市公司股东净利润达
到17.19亿，同比增长21.7%。华夏航空前三季度净利润3.66亿元，同比增长85.05%。财报数据显示，三大航（中国国航、东方航空和南方航空）领跑行业，其中南航和国航2019年前三季度营收均超千亿。从全行业来看，三大航的
营收与净利润在8家上市公司的营收和净利润总和中占比均超七成。天风证券指出，各航司三季报披露完毕，三季度中国国航、南方航空、东方航空、春秋航空、吉祥航空归母净利润分别同比升4.4%、17.2%、9.8%、26.1%、-19.5%（吉祥去年投资收益较大，扣非正增长）。三季度公商务需求相对低迷，三大航运价普跌，而成本端因飞机利用效率提升等因素，单位非油成本出现明显节约的态势；春秋吉祥运价相对坚挺，但单位非油成本同比去年变化不大>。冬春换季后民航正班时刻量同比升5.7%，增速大幅放缓，伴随着波音737MAX停飞及R5实施，行业供给从多维度受到限制，叠加国庆后公商务需求回流及去年年底的低基数，认为票价同比表现有望显著恢复。考虑到相比于去年四
季度油价有明显下降，预计淡季业绩无虞。从历史回测数据来看，11月航空股相对大盘有正超额收益的概率较大，过去19年胜率为74%，平均超额为3.7%。继续推荐三大航，同步推荐春秋航空、吉祥航空。安信证券认为，东航正>式入驻北京大兴机场和上海浦东机场S1卫星厅运营，原核心京沪航线仍保留在首都机场运营，公司开启了在京沪双枢纽、两市四场的运营新格局，公司的航线品质有望持续领先。天风证券表示，大兴机场已经投产，南航作为未来
大兴机场的最大承运人，南北齐飞战略将稳步推进。申万宏源认为，转场搬迁后国航在首都机场占比将超过50%，有利于国航形成国际航班波，提升中转效率，首都机场枢纽功能进一步加强。另外，今年前三季度，民航市场需求>量仍处于缓慢增长阶段。民航局数据显示，2019年前9个月，民航全行业共完成旅客运输量5.0亿人次，同比增长8.7%。而2018年同期增长率为11.6%。对此，光大证券表示，一线机场虽然受制于民航总局“控总量、调结构”，航空>性业务增长缓慢，但随着一线机场产能逐渐释放，航空性业务增长依然可期；免税业务依托机场免税店独特竞争优势以及国内居民消费水平的提高，处于快速成长期，给公司带来新的盈利增长点，建议积极布局航空机场板块。
"""
    seg_result = lac.run(text)
    for w,tag in zip(seg_result[0],seg_result[1]):
        print(w,tag)

