# _*_ coding: utf-8 _*_
# @Time : 2021/1/4 下午4:17 
# @Author : yanqiuxia
# @Version：V 0.1
# @File : bilstm_crf.py
import os
import sys
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchcrf import CRF

from model.word_encoder import WordEncoder
from model.xlnet_encoder import XlnetEncoder
from model.bert_encoder import BertEncoder
from model.word_sequence import WordSequence


class BiLSTM_CRF(nn.Module):

    def __init__(self,
                 num_entity_labels,
                 drop_rate=0.1,
                 lstm_drop_rate=0.5,
                 hidden_size=128,
                 use_bert=False,
                 word_vec=None,
                 bert_model_path=None,
                 gpu=False,
                 use_sent=False,
                 ):
        super(BiLSTM_CRF, self).__init__()

        self.hidden_size = hidden_size
        self.use_bert = use_bert
        if not use_bert:
            self.word_encoder = WordEncoder(word_vec, drop_rate=drop_rate)
            self.embedding_size = word_vec.shape[1]

            self.lstm = nn.LSTM(self.embedding_size, self.hidden_size,
                                num_layers=1, bidirectional=True)
        else:
            self.bert_encoder = BertEncoder(drop_rate=drop_rate, bert_model_path=bert_model_path)
            # self.xlnet_encoder = XlnetEncoder(drop_rate=drop_rate)

            self.lstm = nn.LSTM(self.bert_encoder.bert_config.hidden_size, self.hidden_size,
                                num_layers=1, bidirectional=True)

        # Map token-level hidden state into tag scores
        self.tag_size = num_entity_labels
        self.hidden2tag = nn.Linear(hidden_size * 2, self.tag_size)
        self.crf = CRF(self.tag_size, batch_first=True)
        self.droplstm = nn.Dropout(lstm_drop_rate)

        # 定义一个wordsequence 编码句子之间信息以及句内信息
        self.word_sequence = WordSequence(lstm_drop_rate=lstm_drop_rate,
                                          wordrep=self.bert_encoder,
                                          input_size=self.bert_encoder.bert_config.hidden_size,
                                          lstm_hidden=hidden_size,
                                          gpu=gpu,
                                          use_sent=use_sent)

    def _get_lstm_features(self, word_inputs, word_seq_lengths, seq_token_masks, sent_tokens_list):
        # batch_size = word_inputs.size()[0]
        #
        # # [batch_size, seq_len, embed_size]
        # if self.use_bert:
        #     embeds = self.bert_encoder(word_inputs, seq_token_masks)
        # else:
        #     embeds = self.word_encoder(word_inputs)
        #
        # # word_embs (batch_size, seq_len, embed_size)
        # packed_words = pack_padded_sequence(embeds, word_seq_lengths.cpu().numpy(), True, enforce_sorted=True)
        #
        # lstm_out, (h_n, c_n) = self.lstm(packed_words)
        #
        # lstm_out, _ = pad_packed_sequence(lstm_out)
        # # lstm_out (seq_len, batch_size, hidden_size*2)
        # # feature_out (batch_size, seq_len, hidden_size*2)
        # feature_out = self.droplstm(lstm_out.transpose(1, 0))
        # h_n = self.droplstm(h_n.transpose(1, 0))
        # h_n = h_n.reshape([batch_size, -1])
        feature_out = self.word_sequence(word_inputs,
                                         sent_tokens_list,
                                         word_seq_lengths,
                                         seq_token_masks)
        h_n = None
        return feature_out, h_n

    def forward(self, word_inputs,
                word_seq_lengths,
                seq_token_label,
                seq_token_masks,
                sent_tokens_list,
                train_flag=True,
                decode_flag=True,

                ):
        '''

        :param word_inputs: [batch_size, seq_len]
        :param word_seq_lengths: [batch_size]
        :param seq_token_label: [batch_size, seq_len]
        :param seq_token_masks: [batch_size, seq_len]
        :param sent_words_list [batch_size,sent_num,sent_len]
        :param train_flag:
        :param decode_flag:
        :return:
        '''

        # (batch_size, seq_len , hidden_size*2)
        lstm_feats, last_lstm_feat = self._get_lstm_features(word_inputs,
                                                             word_seq_lengths,
                                                             seq_token_masks,
                                                             sent_tokens_list)

        # 使用pytorch-crf layer
        seq_emit_score = self.hidden2tag(lstm_feats)  # [batch_size, seq_len, tag_size]
        seq_token_masks = seq_token_masks.byte()
        if train_flag:
            ner_loss = -self.crf(seq_emit_score, seq_token_label, seq_token_masks, reduction="mean")
        else:
            ner_loss = None
        batch_seq_pred = None
        if decode_flag:
            batch_seq_pred = self.crf.decode(seq_emit_score, seq_token_masks)  # [batch_size,seq_len]
        return ner_loss, batch_seq_pred, lstm_feats


if __name__ == '__main__':
    ''
    # import numpy as np
    # from train.data_helper import load_word_vectors
    #
    # char_vec, char2id = load_word_vectors("D:/PycharmProjects/ner/data/wor2vec\sgns.sogou.need.char", norm=True, biovec=False)
    #
    # hidden_size = 128
    # num_entity_labels = 5  # [OUBIE]
    # drop_rate = 0.1
    # batch_size = 4
    # seq_len = 200
    # lstm_drop_rate = 0.5
    #
    # ner_model = BiLSTM_CRF(
    #     num_entity_labels,
    #     drop_rate=0.1,
    #     lstm_drop_rate=0.5,
    #     hidden_size=128,
    #     use_bert=False,
    #     word_vec=char_vec
    # )
    #
    # word_inputs = np.random.randint(0, len(char2id), size=[batch_size, seq_len])
    # word_inputs = torch.tensor(word_inputs).long()
    #
    # word_seq_lengths = np.array([seq_len] * batch_size)
    # word_seq_lengths = torch.tensor(word_seq_lengths).long()
    #
    # seq_token_masks = np.ones([batch_size, seq_len])
    # seq_token_masks = torch.tensor(seq_token_masks).long()
    #
    # seq_token_label = np.random.randint(0, num_entity_labels, size=[batch_size, seq_len])
    # seq_token_label = torch.tensor(seq_token_label).long()
    #
    # nll_loss, batch_seq_pred = ner_model(word_inputs,
    #             word_seq_lengths,
    #             seq_token_label,
    #             seq_token_masks,
    #             train_flag=True,
    #             decode_flag=True,)
    # print(nll_loss)
