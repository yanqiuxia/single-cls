#!/bin/bash

APP_HOME="D:/PycharmProjects/yuqing_event_extract"

# 启动训练脚本
nohup python3 -u $APP_HOME/train_main.py  > $APP_HOME/logs/log.txt 2>&1 &
echo "train start finish"