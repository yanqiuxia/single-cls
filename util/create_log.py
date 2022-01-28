# _*_ coding: utf-8 _*_
# @Time : 2019/12/4 下午3:32
# @Author : yanqiuxia
# @Version：V 0.1
# @File : utils.py

import logging
import sys
import os


def get_logger(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def print_config(config, logger=None):
    config = vars(config)
    info = "Running with the following configs:\n"
    for k, v in config.items():
        info += "\t{} : {}\n".format(k, str(v))
    if not logger:
        print("\n" + info + "\n")
    else:
        logger.info("\n" + info + "\n")

# cur_path = os.path.abspath(os.path.dirname(__file__))
# root_path = os.path.split(cur_path)[0]
#
# log_dir = os.path.join(root_path, 'logs')
#
# if not os.path.exists(log_dir):
#     os.mkdir(log_dir)
#
# log_file = os.path.join(log_dir, 'log.txt')
# logger = get_logger(log_file)
