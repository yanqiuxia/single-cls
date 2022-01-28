import argparse


def config():

    ap = argparse.ArgumentParser()

    ap.add_argument('--train_data_file', default='./data/train_spans.json', help='tran data for training')
    ap.add_argument('--dev_data_file', default='./data/dev_spans.json', help='dev data for training')
    ap.add_argument('--test_data_file', default='./data/test.json', help='test data ')
    ap.add_argument('--predict_result_file', default='./data/predict.json', help='the predict result file ')

    ap.add_argument('--char_vec_file', default='./data/word2vec/sgns.sogou.need.char', help='pretrain char vector')

    ap.add_argument('--save_model_dir', default='./output/model', help='model file save path')

    ap.add_argument('--param_file', default='./conf.json', help='the param file')
    ap.add_argument('--summary_dir', default='./output/summary', help='the tensorboard summary file output')

    ap.add_argument('--metrics_file', default='./output/metrics.txt', help='the training result output file')
    ap.add_argument('--evaluate_result_file', default='./output/evaluate_result.json', help='the final evaluate result file')

    ap.add_argument("--event_entity_label2id_file",default="./data/event_entity_label2id.json",help='the event entity file')

    ap.add_argument('--word2input_id_file', default='./output/dict/word2input_id.json', help='the word to input id file')
    ap.add_argument('--entitylabel2id_file', default='./output/dict/entitylabel2id.json', help='the entity label to  outputid file')

    ap.add_argument('--load_model_file', default='./output/model/model-best.pth', help='the save model file ')
    ap.add_argument('--bert_model_path', default='./data/FinBERT', help='the bert model dir ')

    args = ap.parse_args()

    return args


