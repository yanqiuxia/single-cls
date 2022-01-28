#!/bin/bash

# 启动训练脚本
python3 -u  ${CODEREPO_PATH}/single-cls/train_main.py  &> /home/aipf/work/userdata/cls.log
echo "train start finish"