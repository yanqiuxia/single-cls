# _*_ coding: utf-8 _*_
# @Time : 2021/3/9 下午2:04 
# @Author : yanqiuxia
# @Version：V 0.1
# @File : lm_encoder.py
import torch
import torch.nn as nn
from transformers import XLNetTokenizer, XLNetModel, XLNetConfig


class XlnetEncoder(nn.Module):
    def __init__(self, **kwargs) -> None:
        super(XlnetEncoder, self).__init__()
        model_name = 'hfl/chinese-xlnet-base'

        self.model = XLNetModel.from_pretrained(model_name)

        self.config = XLNetConfig.from_pretrained(model_name)
        # 修改配置model_name
        self.config.output_hidden_states = False
        self.output_attentions = False

        dropout = kwargs.get('drop_rate')
        self.dropout = nn.Dropout(dropout)

    def forward(self, word_inputs, word_masks):
        """
        Parameters
        ----------
        word_inputs : ``torch.LongTensor``, required
        (batch_size, sent_size)

        word_masks :(batch_size, sent_size)
        Returns
        -------
        bert_seq_out : ``torch.Tensor``, required
            (batch_size, sent_size, -1)
        """
        with torch.no_grad():
            (word_seq_out, _) = self.model(word_inputs,
                                            token_type_ids=None,
                                            attention_mask=word_masks,
                                            )
            word_seq_out = self.dropout(word_seq_out)
        return word_seq_out

if __name__ == '__main__':
    ''
    model_path = ""
    tokenizer = XLNetTokenizer.from_pretrained('hfl/chinese-xlnet-base')
    text = "我爱中国！"
    tokenized_sequence = tokenizer.tokenize(text)
    input_ids_method2 = tokenizer.convert_tokens_to_ids(tokenized_sequence)


    text = "新生儿重度窒息多脏器损伤首当其冲是中枢神经系统损伤，因为大脑对缺氧最敏感，但奇葩的是：儿科未有早产儿重度窒息多脏器损伤，特别是脑损伤并发症缺血缺氧性脑病，颅内出血预防、诊疗和告知，自己不救治脑损伤也不告脑损伤，不让家属转院救治脑损伤，性质恶劣，儿科住院16天，出院记录只有早产儿重度窒息、吸入性肺炎诊断，无有早产儿重度窒息多脏器损伤，特别是脑损伤并发症缺血缺氧性脑病，颅内出血的诊断就是儿科无诊无治，严重违反早产儿重度窒息护理规范，存在重大责任过错的铁证，儿科脑损伤诊疗告知不作为严重违法过错导致新儿窒息后丧失转院三甲专业医院救治脑损伤，出院康复补救脑损伤甚至放弃救治的宝贵机会，最终剥度新生儿生而为人至今20年，且失智真相被蒙在鼓里13年，凭常识就能断定，三级医院新生儿科医生不知早产儿重度窒息本质多脏器损伤，特别是能引起脑损伤并发症缺血缺氧性脑病，颅内出血是绝对不能成立的，故儿科脑损伤诊疗、告知不作为是明知故犯的犯罪行为，儿科同样也是为了出于逃避责任，收取大笔诊疗费用目的，故意脑损伤诊疗不作为，隐瞒早产儿重度窒息近期脑损伤并发症，远期智残后遗症，儿科同样确犯《刑法》故意伤害罪，医疗事故罪铁板钉钉，但儿科如此严重过错导致如此严重损害后果，诉讼8年，民事只需承担20%责任，逻辑颠倒，申诉人已提起刑事自诉指控儿科确犯《刑法》故意伤害罪，医疗事故罪、伪证罪、非法行医罪，虐待罪，已在杭州中院一级刑事申诉，明知不可为而为之？"
    text = list(text)
    input_ids_method2 = tokenizer.convert_tokens_to_ids(text)
    print(input_ids_method2)
    print(len(input_ids_method2))

    # token2id = tokenizer.encode_plus(
    #     text,  ## 输入文本
    #     add_special_tokens=False,  # 添加 '[CLS]' 和 '[SEP]'
    #     pad_to_max_length=True,
    #     max_length=613,
    #     return_tensors='np',  ## 返回 pytorch tensors 格式的数据
    # )  # 返回是字典dict,使用字典访问的方式取出结果数据
    # print(token2id)
    # input_ids = token2id['input_ids']
    # token_type_ids = token2id['token_type_ids']
    # attention_mask = token2id['attention_mask']

    # xlnet_encoder = XlnetEncoder(drop_rate=0.1)
    # xlnet_encoder(input_ids, token_type_ids, attention_mask)

