# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2017-10-17 16:47:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-02-01 15:59:26
from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from torch.autograd import Variable

seed_num = 42
torch.manual_seed(seed_num)

torch.cuda.manual_seed_all(seed_num)
torch.backends.cudnn.deterministic = True


class WordSequence(nn.Module):
    def __init__(self, lstm_drop_rate=0.5,
                 wordrep=None,
                 input_size=768,
                 lstm_hidden=128,
                 gpu=False,
                 use_sent=True,
                 ):
        super(WordSequence, self).__init__()
        self.wordrep = wordrep
        print("build word sequence feature extractor")
        self.droplstm = nn.Dropout(lstm_drop_rate)

        self.lstm_hidden = lstm_hidden
        self.gpu = gpu
        self.use_sent = use_sent

        self.lstm = nn.LSTM(input_size, lstm_hidden, num_layers=1, batch_first=True, bidirectional=True)


        if self.use_sent:
            self.droplstm_sent = nn.Dropout(lstm_drop_rate - 0.1)
            self.sent_lstm = nn.LSTM(input_size, lstm_hidden, num_layers=1, batch_first=True, bidirectional=True)
            self.lstm2 = nn.LSTM(lstm_hidden * 2, lstm_hidden, num_layers=1, batch_first=True, bidirectional=True)

            HP_hidden_dim = lstm_hidden * 2
            self.gate = nn.Linear(HP_hidden_dim * 2, HP_hidden_dim)
            self.sigmoid = nn.Sigmoid()

    def get_sent_rep(self, sent):
        sent_np = np.array(sent).reshape([1, len(sent)])
        word_masks = np.ones([1, len(sent)])
        word_inputs = Variable(torch.from_numpy(sent_np).long())
        word_masks = Variable(torch.from_numpy(word_masks).float())

        sent_length = np.array([len(sent)])
        sent_length = Variable(torch.from_numpy(sent_length).long())

        if self.gpu:
            word_inputs = word_inputs.cuda()
            word_masks = word_masks.cuda()

        word_represent = self.wordrep(word_inputs, word_masks)

        ## word_embs (1, seq_len, embed_size)
        packed_words = pack_padded_sequence(word_represent, sent_length, True)
        hidden = None
        lstm_out, hidden = self.sent_lstm(packed_words, hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out)
        ## lstm_out (seq_len, 1, hidden_size)
        feature_out_sent = self.droplstm_sent(lstm_out.transpose(1, 0))
        ## feature_out (1, seq_len, hidden_size)
        return feature_out_sent

    def forward(self, word_inputs, sent_tokens_list,
                word_seq_lengths, seq_token_masks
                ):
        '''

        :param word_inputs: (batch_size,seq_len)
        :param sent_tokens_list: (batch_size,sent_num,sent_len)
        :param word_seq_lengths: (batch_size)
        :param seq_token_masks: [batch_size, seq_len]
        :return:
        '''
        self.word_represent = self.wordrep(word_inputs, seq_token_masks)  # (batch_size,embed_size)

        ## parah level
        ## word_embs (batch_size, seq_len, embed_size)
        packed_words = pack_padded_sequence(self.word_represent, word_seq_lengths.cpu().numpy(), True, enforce_sorted=True)
        hidden = None
        lstm_out, hidden = self.lstm(packed_words, hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out)
        ## lstm_out (seq_len, batch_size, hidden_size)
        feature_out = self.droplstm(lstm_out.transpose(1, 0))
        ## feature_out (batch_size, seq_len, hidden_size)

        if self.use_sent:
            # !!! sent-level reps
            ## feature_out_sents (batch_size, seq_len, hidden_size)
            feature_out_sents = torch.zeros((feature_out.size()[0], feature_out.size()[1], feature_out.size()[2]),
                                            requires_grad=False).float()
            if self.gpu:
                feature_out_sents = feature_out_sents.cuda()

            for idx in range(len(sent_tokens_list)):
                feature_out_seq = []

                seq = sent_tokens_list[idx]
                # seq (sent_num, sent_len)
                for sent in seq:
                    feature_out_sent = self.get_sent_rep(sent)
                    # feature_out_sent (1, sent_len, hidden_size)
                    feature_out_seq.append(feature_out_sent.squeeze(0))
                # (seq_len, hidden_size)
                feature_out_seq = torch.cat(feature_out_seq, 0)

                feature_out_sents[idx][:len(feature_out_seq)][:] = feature_out_seq

            gamma = self.sigmoid(self.gate(torch.cat((feature_out, feature_out_sents), 2)))
            outputs_final = gamma * feature_out + (1 - gamma) * feature_out_sents
        else:
            outputs_final = feature_out
        # outputs_final (batch_size, seq_len, hidden_size)
        return outputs_final


if __name__ == '__main__':
    ''
    from model.bert_encoder import BertEncoder

    wordrep = BertEncoder(
        drop_rate=0.1,
        bert_model_path="D:/PycharmProjects/yuqing_event_extract/data/bert-base-chinese"
    )

    wordseq = WordSequence(
        lstm_drop_rate=0.5,
        wordrep=wordrep,
        input_size=768,
        lstm_hidden=128,
    )
