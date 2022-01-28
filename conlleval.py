# Python version of the evaluation script from CoNLL'00-
# Originates from: https://github.com/spyysalo/conlleval.py


# Intentional differences:
# - accept any space as delimiter by default
# - optional file argument (default STDIN)
# - option to set boundary (-b argument)
# - LaTeX output (-l argument) not supported
# - raw tags (-r argument) not supported

# add function :evaluate(predicted_label, ori_label): which will not read from file

import sys
import re
import codecs
from collections import defaultdict, namedtuple

ANY_SPACE = '<SPACE>'


class FormatError(Exception):
    pass

Metrics = namedtuple('Metrics', 'tp fp fn prec rec fscore')


class EvalCounts(object):
    def __init__(self):
        self.correct_chunk = 0    # number of correctly identified chunks
        self.correct_tags = 0     # number of correct chunk tags
        self.found_correct = 0    # number of chunks in corpus
        self.found_guessed = 0    # number of identified chunks
        self.token_counter = 0    # token counter (ignores sentence breaks)

        # counts by type
        self.t_correct_chunk = defaultdict(int)
        self.t_found_correct = defaultdict(int)
        self.t_found_guessed = defaultdict(int)


def parse_args(argv):
    import argparse
    parser = argparse.ArgumentParser(
        description='evaluate tagging results using CoNLL criteria',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arg = parser.add_argument
    arg('-b', '--boundary', metavar='STR', default='-X-',
        help='sentence boundary')
    arg('-d', '--delimiter', metavar='CHAR', default=ANY_SPACE,
        help='character delimiting items in input')
    arg('-o', '--otag', metavar='CHAR', default='O',
        help='alternative outside tag')
    arg('file', nargs='?', default=None)
    return parser.parse_args(argv)


def parse_tag(t):
    m = re.match(r'^([^-]*)-(.*)$', t)
    return m.groups() if m else (t, '')

def evaluate_by_real_time(words,trues,preds,options=None):
    '''
    added by yanqiuxia
    :param words:
    :param trues:
    :param preds:
    :param options:
    :return:
    '''
    if options is None:
        options = parse_args([])    # use defaults


    counts = EvalCounts()
    num_features = None       # number of features per line
    in_correct = False        # currently processed chunks is correct until now
    last_correct = 'O'        # previous chunk tag in corpus
    last_correct_type = ''    # type of previously identified chunk tag
    last_guessed = 'O'        # previously identified chunk tag
    last_guessed_type = ''    # type of previous chunk tag in corpus

    for word, true_, pred in zip(words,trues,preds):
        features = [word,true_, pred]
        guessed, guessed_type = parse_tag(features.pop())
        correct, correct_type = parse_tag(features.pop())
        first_item = features.pop(0)

        if first_item == options.boundary:
            guessed = 'O'

        end_correct = end_of_chunk(last_correct, correct,
                                   last_correct_type, correct_type)
        end_guessed = end_of_chunk(last_guessed, guessed,
                                   last_guessed_type, guessed_type)
        start_correct = start_of_chunk(last_correct, correct,
                                       last_correct_type, correct_type)
        start_guessed = start_of_chunk(last_guessed, guessed,
                                       last_guessed_type, guessed_type)

        if in_correct:
            if (end_correct and end_guessed and
                    last_guessed_type == last_correct_type):
                in_correct = False
                counts.correct_chunk += 1
                counts.t_correct_chunk[last_correct_type] += 1
            elif (end_correct != end_guessed or guessed_type != correct_type):
                in_correct = False

        if start_correct and start_guessed and guessed_type == correct_type:
            in_correct = True

        if start_correct:
            counts.found_correct += 1
            counts.t_found_correct[correct_type] += 1
        if start_guessed:
            counts.found_guessed += 1
            counts.t_found_guessed[guessed_type] += 1
        if first_item != options.boundary:
            if correct == guessed and guessed_type == correct_type:
                counts.correct_tags += 1
            counts.token_counter += 1

        last_guessed = guessed
        last_correct = correct
        last_guessed_type = guessed_type
        last_correct_type = correct_type

    if in_correct:
        counts.correct_chunk += 1
        counts.t_correct_chunk[last_correct_type] += 1

    return counts

def get_metric_by_competition(words,trues,preds):
    '''
    added by yanqiuxia
    :param words:
    :param trues:
    :param preds:
    :return:
    '''
    c = evaluate_by_real_time(words,trues,preds)
    overall, by_type = metrics(c)
    eval_metric = {
        "token_acc":round(c.correct_tags/c.token_counter,4),
        "micro_p":round(overall.prec,4),
        "micro_p":round(overall.rec,4),
        "micro_f1":round(overall.fscore,4),

    }
    details = {}

    macro_p = 0
    macro_r = 0
    macro_f1 = 0

    for i, m in sorted(by_type.items()):
        i_mertirc = {
            "p":round(m.prec,4),
            "r":round(m.rec,4),
            "f1":round(m.fscore,4)
        }
        details[i]=i_mertirc
        macro_p += m.prec
        macro_r += m.rec
        macro_f1 += m.fscore

    macro_p = macro_p/len(by_type)
    macro_r = macro_r/len(by_type)
    macro_f1 = macro_f1/len(by_type)
    eval_metric['macro_p'] = round(macro_p,4)
    eval_metric['macro_r'] = round(macro_r,4)
    eval_metric['macro_f1'] = round(macro_f1,4)
    eval_metric['details'] = details

    return eval_metric


def evaluate(iterable, options=None):
    if options is None:
        options = parse_args([])    # use defaults

    counts = EvalCounts()
    num_features = None       # number of features per line
    in_correct = False        # currently processed chunks is correct until now
    last_correct = 'O'        # previous chunk tag in corpus
    last_correct_type = ''    # type of previously identified chunk tag
    last_guessed = 'O'        # previously identified chunk tag
    last_guessed_type = ''    # type of previous chunk tag in corpus

    for line in iterable:
        line = line.rstrip('\r\n')

        if options.delimiter == ANY_SPACE:
            features = line.split()
        else:
            features = line.split(options.delimiter)

        if num_features is None:
            num_features = len(features)
        elif len(features) == 2:
                features = [" ", 'O', 'O']
        elif num_features != len(features) and len(features) != 0:
            raise FormatError('unexpected number of features: %d (%d)' %
                              (len(features), num_features))

        if len(features) == 0 or features[0] == options.boundary:
            features = [options.boundary, 'O', 'O']
        if len(features) < 3:
            raise FormatError('unexpected number of features in line %s' % line)

        guessed, guessed_type = parse_tag(features.pop())
        correct, correct_type = parse_tag(features.pop())
        first_item = features.pop(0)

        if first_item == options.boundary:
            guessed = 'O'

        end_correct = end_of_chunk(last_correct, correct,
                                   last_correct_type, correct_type)
        end_guessed = end_of_chunk(last_guessed, guessed,
                                   last_guessed_type, guessed_type)
        start_correct = start_of_chunk(last_correct, correct,
                                       last_correct_type, correct_type)
        start_guessed = start_of_chunk(last_guessed, guessed,
                                       last_guessed_type, guessed_type)

        if in_correct:
            if (end_correct and end_guessed and
                last_guessed_type == last_correct_type):
                in_correct = False
                counts.correct_chunk += 1
                counts.t_correct_chunk[last_correct_type] += 1
            elif (end_correct != end_guessed or guessed_type != correct_type):
                in_correct = False

        if start_correct and start_guessed and guessed_type == correct_type:
            in_correct = True

        if start_correct:
            counts.found_correct += 1
            counts.t_found_correct[correct_type] += 1
        if start_guessed:
            counts.found_guessed += 1
            counts.t_found_guessed[guessed_type] += 1
        if first_item != options.boundary:
            if correct == guessed and guessed_type == correct_type:
                counts.correct_tags += 1
            counts.token_counter += 1

        last_guessed = guessed
        last_correct = correct
        last_guessed_type = guessed_type
        last_correct_type = correct_type

    if in_correct:
        counts.correct_chunk += 1
        counts.t_correct_chunk[last_correct_type] += 1

    return counts



def uniq(iterable):
  seen = set()
  return [i for i in iterable if not (i in seen or seen.add(i))]


def calculate_metrics(correct, guessed, total):
    tp, fp, fn = correct, guessed-correct, total-correct
    p = 0 if tp + fp == 0 else 1.*tp / (tp + fp)
    r = 0 if tp + fn == 0 else 1.*tp / (tp + fn)
    f = 0 if p + r == 0 else 2 * p * r / (p + r)
    return Metrics(tp, fp, fn, p, r, f)


def metrics(counts):
    c = counts
    overall = calculate_metrics(
        c.correct_chunk, c.found_guessed, c.found_correct
    )
    by_type = {}
    for t in uniq(list(c.t_found_correct) + list(c.t_found_guessed)):
        by_type[t] = calculate_metrics(
            c.t_correct_chunk[t], c.t_found_guessed[t], c.t_found_correct[t]
        )
    return overall, by_type


def report(counts, out=None):
    if out is None:
        out = sys.stdout

    overall, by_type = metrics(counts)

    c = counts
    out.write('processed %d tokens with %d phrases; ' %
              (c.token_counter, c.found_correct))
    out.write('found: %d phrases; correct: %d.\n' %
              (c.found_guessed, c.correct_chunk))

    if c.token_counter > 0:
        out.write('token accuracy: %6.2f%%; \n' %
                  (100.*c.correct_tags/c.token_counter))
        out.write('实体 precision: %6.2f%%; ' % (100.*overall.prec))
        out.write('recall: %6.2f%%; ' % (100.*overall.rec))
        out.write('FB1: %6.2f\n' % (100.*overall.fscore))

    for i, m in sorted(by_type.items()):
        out.write('%17s: ' % i)
        out.write('precision: %6.2f%%; ' % (100.*m.prec))
        out.write('recall: %6.2f%%; ' % (100.*m.rec))
        out.write('FB1: %6.2f  数据总数: %d\n' % (100.*m.fscore, c.t_found_guessed[i]))


def report_notprint(counts, out=None):
    if out is None:
        out = sys.stdout

    overall, by_type = metrics(counts)

    c = counts
    final_report = []
    line = []
    line.append('processed %d tokens with %d phrases; ' %
              (c.token_counter, c.found_correct))
    line.append('found: %d phrases; correct: %d.\n' %
              (c.found_guessed, c.correct_chunk))
    final_report.append("".join(line))

    if c.token_counter > 0:
        line = []
        line.append('token accuracy: %6.2f%%; \n' %
                  (100.*c.correct_tags/c.token_counter))
        line.append('实体 precision: %6.2f%%; ' % (100.*overall.prec))
        line.append('recall: %6.2f%%; ' % (100.*overall.rec))
        line.append('FB1: %6.2f\n' % (100.*overall.fscore))
        final_report.append("".join(line))

    for i, m in sorted(by_type.items()):
        line = []
        line.append('%17s: ' % i)
        line.append('precision: %6.2f%%; ' % (100.*m.prec))
        line.append('recall: %6.2f%%; ' % (100.*m.rec))
        line.append('FB1: %6.2f  数据总数: %d\n' % (100.*m.fscore, c.t_found_correct[i]))
        final_report.append("".join(line))
    return final_report


def end_of_chunk(prev_tag, tag, prev_type, type_):
    # check if a chunk ended between the previous and current word
    # arguments: previous and current chunk tags, previous and current types
    chunk_end = False

    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'S': chunk_end = True

    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'S': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    # these chunks are assumed to have length 1
    if prev_tag == ']': chunk_end = True
    if prev_tag == '[': chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    # check if a chunk started between the previous and current word
    # arguments: previous and current chunk tags, previous and current types
    chunk_start = False

    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True

    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'S' and tag == 'E': chunk_start = True
    if prev_tag == 'S' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    # these chunks are assumed to have length 1
    if tag == '[': chunk_start = True
    if tag == ']': chunk_start = True

    return chunk_start


def return_report(input_file):
    with codecs.open(input_file, "r", "utf8") as f:
        counts = evaluate(f)
    return report_notprint(counts)


def main(argv):
    args = parse_args(argv[1:])

    if args.file is None:
        counts = evaluate(sys.stdin, args)
    else:
        with open(args.file) as f:
            counts = evaluate(f, args)
    report(counts)

if __name__ == '__main__':
    # sys.exit(main(sys.argv))

    # final_reports = return_report('./data/ner.result.txt')
    #
    # for final_report in final_reports:
    #     print(final_report)

    words = list("天目药业涉嫌信披违法遭立案调查 与控股股东违规占用资金等有关。")
    trues = ["O"]*len(words)
    preds = ["O"]*len(words)
    trues[0]="B-e"
    trues[1] = "I-e"
    trues[2] = "I-e"
    trues[3] = "E-e"

    trues[5] = "B-e1"
    trues[6] = "E-e1"

    preds[0] = "B-e"
    preds[1] = "I-e"
    preds[2] = "I-e"
    preds[3] = "E-e"

    preds[4] = "B-e1"
    preds[5] = "E-e1"

    counts = evaluate_by_real_time(words,trues,preds)

    report(counts)
