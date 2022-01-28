import argparse


def config():

    ap = argparse.ArgumentParser()

    ap.add_argument('--train_data_file', default='./single-cls/data/train_3fold_spans.json', help='tran data for training')
    ap.add_argument('--dev_data_file', default='./single-cls/data/dev_3fold_spans.json', help='dev data for training')
    ap.add_argument('--test_data_file', default='./single-cls/data/test_pred_3.json', help='test data ')
    ap.add_argument('--predict_result_file', default='./single-cls/data/test_cls_pred_3.json', help='the predict result file ')

    ap.add_argument('--char_vec_file', default='./data/word2vec/sgns.sogou.need.char', help='pretrain char vector')

    ap.add_argument('--save_model_dir', default='./single-cls/output/kfold_3/model', help='model file save path')

    ap.add_argument('--param_file', default='./single-cls/conf.json', help='the param file')
    ap.add_argument('--summary_dir', default='./single-cls/output/summary', help='the tensorboard summary file output')

    ap.add_argument('--metrics_file', default='./single-cls/output/metrics.txt', help='the training result output file')
    ap.add_argument('--evaluate_result_file', default='./single-cls/output/evaluate_result.json', help='the final evaluate result file')

    ap.add_argument("--event_entity_label2id_file",default="./single-cls/data/event_entity_label2id.json",help='the event entity file')

    ap.add_argument('--word2input_id_file', default='./single-cls/output/dict/word2input_id.json', help='the word to input id file')
    ap.add_argument('--entitylabel2id_file', default='./single-cls/output/dict/entitylabel2id.json', help='the entity label to  outputid file')

    ap.add_argument('--load_model_file', default='./single-cls/output/kfold_3/model/model-best.pth', help='the save model file ')
    ap.add_argument('--bert_model_path', default='./FinBERT', help='the bert model dir ')

    args = ap.parse_args()

    return args


