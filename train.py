# _*_ coding: utf-8 _*_
# @Time : 2021/2/7 下午5:42 
# @Author : yanqiuxia
# @Version：V 0.1
# @File : train.py

import datetime
import json
import math
import os

# from tkinter import _flatten
from itertools import chain

import torch
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from transformers import BertTokenizer
import numpy as np

from util.log import logger
from util.create_log import print_config
# from model.ner_model import BiLSTM_CRF
from model.event_extract import EventExtract
from pro.sentence_split import split_test_data_bysentence, recalc_offset, split_test_data_bychunk, get_stand_res

from data_helper import load_word_vectors, load_corpus_file, build_BIOES2id, build_event_entitylabel2id, \
    data2bert_number
from analysis import calc_metrics, safe_divide
from data_loader import gen_minibatch
from conlleval import get_metric_by_competition
from predict import Predict
from parameter import Parameter

np.random.seed(2021)


class Train(object):

    def __init__(self, config, param: Parameter):

        ''

        self.config = config
        self.param = param
        self.wv = None

        print_config(config, logger)

        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_model_path)

        if not param.use_bert:
            self.wv, self.word2id = load_word_vectors(self.config.char_vec_file, norm=True, biovec=False)

        self._load_dataset()

        self._create_model()

        self.best_model_path = None
        if self.param.gpu:
            torch.cuda.manual_seed(2021)
        else:
            torch.manual_seed(2021)

    def _load_dataset(self):

        self.event_entity_label2id = build_event_entitylabel2id(self.config.event_entity_label2id_file)
        self.event_entity_id2label = {int(v): k for k, v in self.event_entity_label2id.items()}

        if self.param.is_train:
            train_data = load_corpus_file(self.config.train_data_file)
        dev_data = load_corpus_file(self.config.dev_data_file)

        if self.param.use_bert:
            if self.param.is_train:
                self.train_data = data2bert_number(train_data,
                                                   self.event_entity_label2id,
                                                   self.tokenizer)
            self.dev_data = data2bert_number(dev_data,
                                             self.event_entity_label2id,
                                             self.tokenizer)
        else:
            '''
            后续实现 word2vec
            '''

        print('The training set, verification set split are completed!')
        if self.param.is_train:
            print('the train data\'s num is  %d, the dev data\'s num is %d' % (
                len(train_data), len(dev_data)))
        else:
            print("the dev data\'s num is %d" % len(dev_data))

    def _create_model(self):

        self.model = EventExtract(
            num_event_entity_labels=len(self.event_entity_label2id),
            drop_rate=self.param.drop_rate,
            cnn_hidden_size=self.param.cnn_hidden_size,
            bert_model_path=self.config.bert_model_path,
            gpu=self.param.gpu,

        )

        if self.param.gpu:
            self.model.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.param.lr,
                                          weight_decay=self.param.lr_weight_decay)

        self.scheduler = StepLR(self.optimizer, step_size=self.param.num_epoch_lr_decay,
                                gamma=self.param.lr_decay_gamma)
        print(self.model)
        print('build model sucess!')

    def step(self, batch_tokens_inputs,
             batch_tokens_masks,
             batch_tokens_length,
             batch_spans_dranges,
             batch_spans_masks,
             batch_spans_labels,
             ):

        # 将这些数据转换成Variable类型
        batch_tokens_inputs = Variable(batch_tokens_inputs)
        batch_tokens_masks = Variable(batch_tokens_masks)
        batch_tokens_length = Variable(batch_tokens_length)
        batch_spans_dranges = Variable(batch_spans_dranges)
        batch_spans_masks = Variable(batch_spans_masks)
        batch_spans_labels = Variable(batch_spans_labels)

        if self.param.gpu:
            batch_tokens_inputs = batch_tokens_inputs.cuda()
            batch_tokens_masks = batch_tokens_masks.cuda()
            batch_tokens_length = batch_tokens_length.cuda()
            batch_spans_dranges = batch_spans_dranges.cuda()
            batch_spans_masks = batch_spans_masks.cuda()
            batch_spans_labels = batch_spans_labels.cuda()

        loss, spans_pred_ids,_ = self.model(
            word_inputs=batch_tokens_inputs,
            word_seq_lengths=batch_tokens_length,
            seq_token_masks=batch_tokens_masks,
            spans_dranges=batch_spans_dranges,
            spans_masks=batch_spans_masks,
            spans_labels=batch_spans_labels,
        )

        return loss, spans_pred_ids

    def train(self):
        if self.train_data is not None:
            self._train(self.train_data)
            self.save_model_result(self.best_model_path)
            self.rename_best_model_file()
        else:
            logger.info('the train data is None!')

    def _train(self, dataset):
        ''
        mertrics_file = self.config.metrics_file
        if os.path.exists(mertrics_file):
            os.remove(mertrics_file)
        op = open(mertrics_file, 'a+', encoding='utf-8')

        # 创建模型存储路径
        if not os.path.exists(self.config.save_model_dir):
            os.makedirs(self.config.save_model_dir)

        display_num_one_epoch = self.param.display_num_one_epoch  # Display 1 pre epoch
        train_batch_num = math.ceil(len(dataset['tokens']) / self.param.batch_size)
        display_batch = int(train_batch_num / display_num_one_epoch)
        if display_batch == 0:
            display_batch = 1

        best_f1 = 0.0
        show_loss = 0.0

        total_step = train_batch_num

        for i in range(self.param.num_epoch):

            logger.info("epoch %d, lr:%f" % (i + 1, self.optimizer.param_groups[0]['lr']))
            # self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], i + 1)
            num_step = 0

            for batch, (batch_tokens_inputs, batch_tokens_masks, batch_tokens_length
                        , batch_spans_dranges, batch_spans_masks, batch_spans_labels) \
                    in enumerate(gen_minibatch(self.param, dataset, len(self.event_entity_label2id))):
                self.model.train()
                num_step += 1

                loss, _ = self.step(batch_tokens_inputs,
                                    batch_tokens_masks,
                                    batch_tokens_length,
                                    batch_spans_dranges,
                                    batch_spans_masks,
                                    batch_spans_labels,
                                    )

                loss = self._update_gradient(loss, num_step
                                             )
                show_loss += loss.item()

                # self.writer.add_scalar("train/loss", loss.item(), num_step + i * total_step)

                if self.dev_data is not None:

                    if batch % 100 == 0:
                        logger.info("train step: %d, total_step:%d,train_loss %.4e"
                                    % (num_step, train_batch_num, loss.item()))

                    if (batch + 1) % display_batch == 0:
                        ''
                        logger.info("step %d, total_step:%d" % (num_step, train_batch_num))

                        eval_result = self.eval_dataset(self.dev_data, is_train=False)
                        self.save_metrics(i + 1, num_step, total_step, show_loss, eval_result, op)

                        logger.info("train_loss: %.4e, "
                                    "dev_loss: %.4e"
                                    % (show_loss, eval_result["loss"]))
                        print(eval_result)

                        # if eval_result['平均F1'] >= best_f1:
                        best_f1 = eval_result['平均F1']
                        self.save_model_state(num_step + i * total_step)

                        show_loss = 0.0
                else:
                    logger.info('the dev data is None!')

            self.scheduler.step()

        op.close()

    def _update_gradient(self, loss, num_step):
        # 1 loss regularization
        loss = loss / self.param.accumulation_steps
        # 2 back propagation
        loss.backward()
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(parameters, self.param.clip_c)

        # 3. update parameters of net
        if ((num_step + 1) % self.param.accumulation_steps) == 0:
            # optimizer the net
            self.optimizer.step()  # update parameters of net
            self.optimizer.zero_grad()  # reset gradient
        return loss

    def eval_dataset(self, dataset, is_train=False):

        self.model.eval()

        total_loss = 0
        spans_true_labels = []
        spans_pred_labels = []

        '''
        竞赛换将sklearn 只有0.20版本暂时不支持多标签f1值计算，自己实现f1值计算，1105上线后对方会更新sklearn
        '''

        batch_num = math.ceil(len(dataset['tokens']) / self.param.batch_size)

        for batch, (batch_tokens_inputs, batch_tokens_masks,
                    batch_tokens_length, batch_spans_dranges,
                    batch_spans_masks, batch_spans_labels) in enumerate(
            gen_minibatch(self.param, dataset, len(self.event_entity_label2id))):

            if batch % 100 == 0:
                logger.info("eval step: %d, total_step:%d, " % (batch, batch_num))

            loss, spans_preds = self.step(batch_tokens_inputs,
                                          batch_tokens_masks,
                                          batch_tokens_length,
                                          batch_spans_dranges,
                                          batch_spans_masks,
                                          batch_spans_labels,
                                          )
            total_loss += loss.item()

            batch_spans_labels = list(batch_spans_labels.cpu().numpy())
            batch_spans_masks = batch_spans_masks.cpu().numpy()

            # [batch*non_mask_num_spans, C]
            span_label_list = self.get_nonmask_span_label(batch_spans_labels, batch_spans_masks)
            spans_true_labels.extend(span_label_list)

            # [batch, num_spans, C]
            spans_preds = list(spans_preds.cpu().numpy())
            # [batch*non_mask_num_spans, event_num]
            spans_preds = self.get_nonmask_span_label(spans_preds, batch_spans_masks)
            spans_pred_labels.extend(spans_preds)

        avg_loss = total_loss / batch_num
        eval_result = {}
        eval_result['loss'] = avg_loss

        cls_result = calc_metrics(spans_true_labels, spans_pred_labels, is_train, self.event_entity_id2label)
        eval_result.update(cls_result)
        eval_result['平均F1'] = cls_result['宏F1']

        return eval_result

    def get_nonmask_span_label(self, spans_labels, spans_masks):
        '''

        @param spans_labels: [batch_size, num_spans,C]
        @param spans_masks: [batch_size,num_spans]
        @return:
        '''
        spans_length = np.sum(spans_masks, axis=1).tolist()  # [batch_size]

        spans_labels_list = []  # [batch_size*num_spans, C]
        for span_label, span_length in zip(spans_labels, spans_length):
            if span_length > 0:
                # [non_mask_spans, C]
                spans_label = span_label[:int(span_length), :]
                # [non_mask_spans*C]
                spans_labels_list.extend(spans_label)
        return spans_labels_list

    def save_model_state(self, num_step):
        # 最后一轮训练结束的模型保存
        model_name = 'model-' + str(num_step) + '.pth'
        save_path = os.path.join(self.config.save_model_dir, model_name)
        torch.save(self.model.state_dict(), save_path)

        self.best_model_path = save_path
        logger.info('best_model_path:%s' % self.best_model_path)

    def save_model_result(self, model_path):
        op = open(self.config.evaluate_result_file, 'w', encoding='utf-8')
        result = {}
        if model_path is not None and os.path.isfile(model_path):
            state_dict = torch.load(model_path)
            self.model.load_state_dict(state_dict)
            '''
            去掉验证集上的评估
            '''
            # if self.train_data is not None:
            #     train_result = self.eval_dataset(self.train_data, False)
            #     train_result['loss'] = format(train_result['loss'], '.2e')
            # else:
            train_result = {}

            if self.dev_data is not None:
                dev_result = self.eval_dataset(self.dev_data, False)
                dev_result['loss'] = format(dev_result['loss'], '.2e')

            else:
                dev_result = {}

            result = {
                "train": train_result,
                "valid": dev_result
            }

        json.dump(result, op, ensure_ascii=False, indent=1)
        op.close()

    def save_metrics(self, epoch, num_step, total_step, train_loss, eval_result, op):
        curr_time = datetime.datetime.now()
        time_str = datetime.datetime.strftime(curr_time, '%Y-%m-%d %H:%M:%S')

        metrics = {
            "time": time_str,
            "epoch": epoch,
            "total_epoch": self.param.num_epoch,
            "step": num_step,
            "total_step": total_step,
            "train_loss": format(train_loss, '.2e')
        }

        metrics['valid_loss'] = format(eval_result['loss'], '.2e')

        for key, value in eval_result.items():

            if not key.endswith("loss"):
                metrics[key] = value

        json.dump(metrics, op, ensure_ascii=False)
        op.write('\n')

    def rename_best_model_file(self):
        if os.path.exists(self.best_model_path):
            model_file_dir, model_file_name = os.path.split(self.best_model_path)
            new_best_model_path = os.path.join(model_file_dir, 'model-best.pth')
            if os.path.exists(new_best_model_path):
                os.remove(new_best_model_path)
            try:
                os.rename(self.best_model_path, new_best_model_path)
            except Exception as e:
                logger.exception(e)

    def save_dict(self):
        ''
        if not self.param.use_bert:
            op = open(self.config.word2input_id_file, 'w', encoding='utf-8')
            json.dump(self.word2id, op, ensure_ascii=False, indent=1)
            op.close()

        op = open(self.config.entitylabel2id_file, 'w', encoding='utf-8')
        json.dump(self.entitylabel2id, op, ensure_ascii=False, indent=1)
        op.close()
        self.entityid2label = {int(v): k for k, v in self.entitylabel2id.items()}

    def _load_dict(self):
        if not self.param.use_bert:
            fp = open(self.config.word2input_id_file, 'r', encoding='utf-8')
            self.word2id = json.load(fp)
            fp.close()

        fp = open(self.config.entitylabel2id_file, 'r', encoding='utf-8')
        self.entitylabel2id = json.load(fp)
        fp.close()
        self.entityid2label = {int(v): k for k, v in self.entitylabel2id.items()}

    def load_model(self):
        model_file_path = self.config.load_model_file
        if os.path.isfile(model_file_path):
            '''
            '''
            state_dict = torch.load(model_file_path)
            self.model.load_state_dict(state_dict)
            print("the model file is %s" % model_file_path)
        else:
            print('the model file is error! %s' % self.config.load_model_file)
            raise Exception("the model file is error!")

    def eval(self):
        ''
        self.load_model()

        eval_result = self.eval_dataset(self.dev_data, is_train=False)
        eval_result['loss'] = format(eval_result['loss'], '.2e')

        op = open(self.config.evaluate_result_file, 'w', encoding='utf-8')

        json.dump(eval_result, op, ensure_ascii=False, indent=1)
        op.close()

        return eval_result

    def predict(self):
        ''
        self.load_model()
        test_data = load_corpus_file(self.config.test_data_file)
        print('the test data\'s num is  %d' % len(test_data))
        predict = Predict(self.tokenizer, self.param, self.model, self.event_entity_id2label)
        for i, data in enumerate(test_data):
            if i % 100 == 0:
                print("现在运行的数据量%d" % i)
            content = data['content']
            spans = data['pred_spans']

            if len(spans) == 0:
                continue

            if self.param.use_sent:
                #  分块实体预测
                batch_content, batch_start_offset = split_test_data_bychunk(content)
            else:
                # 分句实体预测
                batch_content, batch_start_offset, batch_spans = split_test_data_bysentence(content, spans)
            '''
            文本分块结果进行校验
            '''
            if batch_start_offset[-1] + len("".join(batch_content[-1])) != len("".join(content)):
                print("将content切块结果不对！%s" % content)
            # 一整批输入模型
            # batch_entities = predict.batch_predict(batch_content)

            # 分批输入模型
            batch_num = math.ceil(len(batch_content) / self.param.test_batch_size)
            batch_entities = []
            for j in range(batch_num):
                cur_batch_content = batch_content[
                                    j * self.param.test_batch_size:(j + 1) * self.param.test_batch_size]

                cur_batch_spans = batch_spans[j * self.param.test_batch_size:(j + 1) * self.param.test_batch_size]

                cur_batch_entity = predict.batch_predict(cur_batch_content, cur_batch_spans)
                batch_entities.extend(cur_batch_entity)

            # 一句一句输入模型

            # #内存不够改成单样本预测
            # batch_entities = []
            # for content in batch_content:
            #     entities = predict.predict(content)
            #     batch_entities.append(entities)

            spans = recalc_offset(batch_start_offset, batch_entities)

            # 单句实体预测
            data['pred_spans'] = spans

        result = {
            "result": test_data
        }
        op = open(self.config.predict_result_file, 'w', encoding="utf-8")
        json.dump(result, op, indent=1, ensure_ascii=False)
        op.close()
        logger.info("预测结果写入文件%s" % self.config.predict_result_file)
