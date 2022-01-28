APP_HOME="/home/aipf/work/event-extract-join"

# 启动训练脚本
nohup python3 -u $APP_HOME/pro/online_evaluate_scorer.py  > $APP_HOME/output/online_evaluate.txt 2>&1 &
echo "evaluate start finish"