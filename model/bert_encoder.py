# _*_ coding: utf-8 _*_
# @Time : 2021/3/4 下午3:12 
# @Author : yanqiuxia
# @Version：V 0.1
# @File : bert_encoder.py
import torch
import torch.nn as nn

import transformers
from transformers import BertTokenizer, BertModel


class BertEncoder(nn.Module):

    def __init__(self, **kwargs) -> None:
        super(BertEncoder, self).__init__()

        model_path = kwargs.get('bert_model_path')

        self.bert_config = transformers.BertConfig.from_pretrained(model_path)
        # 修改配置
        self.bert_config.output_hidden_states = False
        self.bert_config.output_attentions = False
        # 通过配置和路径导入模型
        self.bert_model = transformers.BertModel.from_pretrained(model_path, config=self.bert_config)

        dropout = kwargs.get('drop_rate', 0.1)
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
        # with torch.no_grad():
        out = self.bert_model(word_inputs,
                              token_type_ids=None,
                              attention_mask=word_masks,
                              output_hidden_states=True
                              )
        bert_seq_out = out.last_hidden_state
        bert_seq_out = self.dropout(bert_seq_out)
        return bert_seq_out


if __name__ == '__main__':
    ""
    tokenizer = BertTokenizer.from_pretrained("D:/PycharmProjects/event-extract-join/data/chinese-roberta-wwm-ext")
    sequence = ["[CLS]"]
    sequence.extend(list("我爱中国！祖国您好！，天气非常不好！"))
    # tokenized_sequence = tokenizer.tokenize(sequence)
    sequence.append("[SEP]")
    input_ids_method2 = tokenizer.convert_tokens_to_ids(sequence)
    print(input_ids_method2)
    # print(tokenized_sequence)
    bertModel = BertEncoder(bert_model_path="D:/PycharmProjects/event-extract-join/data/chinese-roberta-wwm-ext")

    token2id = tokenizer.encode_plus(
        sequence,  ## 输入文本
        add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
        max_length=20,  # 填充 & 截断长度
        pad_to_max_length=False,
        return_tensors='pt',  ## 返回 pytorch tensors 格式的数据
    )  # 返回是字典dict,使用字典访问的方式取出结果数据
    print(token2id)

    # out = bertModel(word_inputs=token2id['input_ids'], word_masks=token2id['attention_mask'])
    #
    # for name, param in bertModel.named_parameters():
    #     print(name)
    #     print(param.requires_grad)
