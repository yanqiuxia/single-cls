# _*_ coding: utf-8 _*_
# @Time : 2020/12/24 上午10:51 
# @Author : yanqiuxia
# @Version：V 0.1
# @File : word_encoder.py
import torch
import torch.nn as nn

class WordEncoder(nn.Module):

    def __init__(self, word_vec=None, **kwargs) -> None:
        super(WordEncoder, self).__init__()

        if word_vec is not None:
            self.vocab_size = word_vec.shape[0]
            self.embed_size = word_vec.shape[1]
        else:
            self.vocab_size = kwargs.get('vocab_size')
            self.embed_size = kwargs.get('embed_size')

        self.padding_idx = kwargs.get('padding_idx', 0)
        self.embed = nn.Embedding(self.vocab_size,
                                  self.embed_size,
                                  padding_idx=self.padding_idx)
        dropout = kwargs.get('drop_rate')
        self.dropout = nn.Dropout(dropout)

        if word_vec is not None:
            self.embed.weight = nn.Parameter(torch.from_numpy(word_vec).float())
            self.embed.weight.requires_grad = kwargs.get('update_word_embed', False)

    def forward(self, batch_sent_input):
        """
        Parameters
        ----------
        batch_sent_input : ``torch.LongTensor``, required
            (batch_size, sent_size)

        Returns
        -------
        batch_sent_vecs : ``torch.Tensor``, required
            (batch_size, sent_size, -1)
        """
        batch_sent_vecs = self.embed(batch_sent_input)  # (batch_size, sent_size,embedding_size)
        batch_sent_vecs = self.dropout(batch_sent_vecs)
        return batch_sent_vecs
