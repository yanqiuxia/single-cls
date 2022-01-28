# _*_ coding: utf-8 _*_
import os
import sys
import torch

root_path = os.getcwd()
print("root path is : %s " % root_path)
sys.path.append(root_path)

from util import log
from util.log import logger

from parameter import Parameter
from arg import config
from train import Train


def make_dirs(file_path, isfile=False):

    if isfile:
        file_dir = os.path.split(file_path)[0]
    else:
        file_dir = file_path
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
        print("创建路径%s"%file_dir)


if __name__ == '__main__':
    root_path = "/home/aipf/work/userdata"
    print("data path is : %s " % root_path)
    log.change_work_dir(__file__)
    log_dir = os.path.join(root_path, "./logs/")
    log.auto_init_logger(log_dir, prefix="ner_train", debug=False)

    args = config()
    args.train_data_file = os.path.join(root_path, args.train_data_file)
    args.dev_data_file = os.path.join(root_path, args.dev_data_file)
    args.test_data_file = os.path.join(root_path, args.test_data_file)
    args.predict_result_file = os.path.join(root_path, args.predict_result_file)
    args.char_vec_file = os.path.join(root_path, args.char_vec_file)

    args.save_model_dir = os.path.join(root_path, args.save_model_dir)
    make_dirs(args.save_model_dir, isfile=False)

    args.param_file = os.path.join(root_path,args.param_file)
    args.summary_dir = os.path.join(root_path, args.summary_dir)
    make_dirs(args.summary_dir, isfile=False)

    args.metrics_file = os.path.join(root_path, args.metrics_file)
    make_dirs(args.metrics_file,isfile=True)

    args.evaluate_result_file = os.path.join(root_path, args.evaluate_result_file)
    make_dirs(args.evaluate_result_file, isfile=True)

    args.event_entity_label2id_file = os.path.join(root_path, args.event_entity_label2id_file)

    args.word2input_id_file = os.path.join(root_path, args.word2input_id_file)
    make_dirs(args.word2input_id_file, isfile=True)

    args.entitylabel2id_file = os.path.join(root_path, args.entitylabel2id_file)
    make_dirs(args.entitylabel2id_file, isfile=True)

    args.load_model_file = os.path.join(root_path, args.load_model_file)
    args.bert_model_path = os.path.join(root_path, args.bert_model_path)

    param = Parameter(args.param_file)
    logger.info(param)

    train = Train(args, param)

    if param.is_finetune:
        state_dict = torch.load(args.load_model_file)
        train.model.load_state_dict(state_dict)
        print("the model file is %s" % args.load_model_file)

    if param.is_train:
        train.train()
        logger.info("Training completed!")

    elif param.is_predict:
        train.predict()
        logger.info("predict completed!")
    else:
        train.eval()
        logger.info("eval completed!")
    # train.writer.close()
