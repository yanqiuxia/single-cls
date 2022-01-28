# _*_ coding: utf-8 _*_
# @Time : 2020/12/10 上午11:25 
# @Author : yanqiuxia
# @Version：V 0.1
# @File : config.py
import os
import json
from util.log import logger


class Parameter(object):
    def __init__(self, parameter_file):

        if parameter_file is not None and os.path.isfile(parameter_file):

            with open(parameter_file, 'r', encoding='utf-8') as fp:
                try:
                    self._parameter = json.load(fp)
                except Exception as e:
                    logger.exception(e)
                    self._parameter = {}
        else:
            self._parameter = {}

    @property
    def lr(self):
        return self._parameter.get('lr', 1e-3)

    @property
    def lr_weight_decay(self):
        return self._parameter.get('lr_weight_decay', 1e-6)

    @property
    def num_epoch_lr_decay(self):
        return self._parameter.get('num_epoch_lr_decay', 5)

    @property
    def lr_decay_gamma(self):
        return self._parameter.get('lr_decay_gamma', 0.5)

    @property
    def num_epoch(self):
        return self._parameter.get('num_epoch', 50)

    @property
    def batch_size(self):
        return self._parameter.get('batch_size', 64)

    @property
    def test_batch_size(self):
        return self._parameter.get('test_batch_size', 64)

    @property
    def display_num_one_epoch(self):
        return self._parameter.get('display_num_one_epoch', 1)

    @property
    def accumulation_steps(self):
        return self._parameter.get('accumulation_steps', 1)

    @property
    def is_train(self):
        return self._parameter.get('is_train', True)

    @property
    def is_finetune(self):
        return self._parameter.get('is_finetune', False)
    @property
    def is_predict(self):
        return self._parameter.get("is_predict",False)


    @property
    def pretrained_word_embed(self):
        return self._parameter.get('pretrained_word_embed', True)

    @property
    def update_word_embed(self):
        return self._parameter.get('update_word_embed', True)

    @property
    def pred_thred(self):
        return self._parameter.get('pred_thred', 0.5)

    @property
    def gpu(self):
        return self._parameter.get('gpu', True)

    @property
    def max_tokens(self):
        return self._parameter.get('max_tokens', 512)

    @property
    def hidden_size(self):
        return self._parameter.get('hidden_size', 128)

    @property
    def cnn_hidden_size(self):
        return self._parameter.get("cnn_hidden_size", 128)


    @property
    def drop_rate(self):
        return self._parameter.get('drop_rate', 0.1)

    @property
    def lstm_drop_rate(self):
        return self._parameter.get("lstm_drop_rate", 0.5)

    @property
    def clip_c(self):
        return self._parameter.get('clip_c', 10)

    @property
    def gpu(self):
        return self._parameter.get('gpu', True)

    @property
    def use_bert(self):
        return self._parameter.get('use_bert', False)

    @property
    def use_sent(self):
        return self._parameter.get("use_sent", False)


    def __str__(self):
        info = "\n"
        info += "\t{} : {}\n".format('lr', str(self.lr))
        info += "\t{} : {}\n".format('lr_weight_decay', str(self.lr_weight_decay))
        info += "\t{} : {}\n".format('num_epoch_lr_decay', str(self.num_epoch_lr_decay))
        info += "\t{} : {}\n".format('lr_decay_gamma', str(self.lr_decay_gamma))
        info += "\t{} : {}\n".format('num_epoch', str(self.num_epoch))
        info += "\t{} : {}\n".format('batch_size', str(self.batch_size))
        info += "\t{} : {}\n".format('test_batch_size', str(self.test_batch_size))
        info += "\t{} : {}\n".format('display_num_one_epoch', str(self.display_num_one_epoch))
        info += "\t{} : {}\n".format('accumulation_steps', str(self.accumulation_steps))
        info += "\t{} : {}\n".format('is_train', str(self.is_train))
        info += "\t{} : {}\n".format('is_finetune', str(self.is_finetune))
        info += "\t{} : {}\n".format('is_predict', str(self.is_predict))
        info += "\t{} : {}\n".format('pretrained_word_embed', str(self.pretrained_word_embed))
        info += "\t{} : {}\n".format('update_word_embed', str(self.update_word_embed))
        info += "\t{} : {}\n".format('hidden_size', str(self.hidden_size))
        info += "\t{} : {}\n".format('drop_rate', str(self.drop_rate))
        info += "\t{} : {}\n".format('lstm_drop_rate', str(self.lstm_drop_rate))

        info += "\t{} : {}\n".format('clip_c', str(self.clip_c))
        info += "\t{} : {}\n".format('use_bert', str(self.use_bert))
        info += "\t{} : {}\n".format('use_sent', str(self.use_sent))

        return info
