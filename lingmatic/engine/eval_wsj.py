import argparse
import os
import json

from tqdm import tqdm

from nltk.tree import Tree


punctuation_tags = ['.', ',', ':', '-LRB-', '-RRB-', "''", '``']


def tokeep_punct_using_labels(pt):
    if isinstance(pt, str):
        return pt
    if pt.label() in punctuation_tags or pt.label() in ('$', '#'):
        return None

    node = []
    for subtree in pt:
        x = tokeep_punct_using_labels(subtree)
        if isinstance(x, bool):
            node.append(x)
        elif isinstance(x, list):
            node += x
        elif isinstance(x, str):
            node.append(True)
        else:
            node.append(False)

    if len(node) == 1:
        node = node[0]

    if isinstance(node, bool):
        node = [node]

    return node


def remove_using_mask(pt, mask):
    def func(pt, mask, pos=0):
        if isinstance(pt, str):
            if mask[pos]:
                return pt, 1
            return None, 1

        node = []
        sofar = pos
        for subtree in pt:
            x, size = func(subtree, mask, pos=sofar)
            if x is not None and (isinstance(x, str) or len(x) > 0):
                node.append(x)
            sofar += size

        node = tuple(node)

        if len(node) == 1:
            node = node[0]

        size = sofar - pos

        return node, size

    node = func(pt, mask)[0]

    return node


def remove_entire_span(spans, length):
    spans.remove((0, length))
    return spans


def remove_word_spans(spans, length):
    for i in range(length):
        spans.remove((i, i+1))
    return spans


def get_len(tr):
    return len(get_tokens(tr))


def get_tokens(tr):
    if not isinstance(tr, (list, tuple)):
        return [tr]
    tokens = []
    for subtree in tr:
        tokens += get_tokens(subtree)
    return tokens


def get_spans(tree):
    def helper(tr, idx=0):
        if isinstance(tr, (str, int)):
            return 1, [(idx, idx+1)]

        spans = []
        sofar = idx

        for subtree in tr:
            size, subspans = helper(subtree, idx=sofar)
            spans += subspans
            sofar += size

        size = sofar - idx
        spans += [(idx, sofar)]

        return size, spans

    _, spans = helper(tree)

    return set(spans)


def example_f1(c1, c2):
    correct = len(c1.intersection(c2))
    if correct == 0:
        return 0.
    gt_total = len(c2)
    pred_total = len(c1)
    prec = float(correct) / gt_total
    recall = float(correct) / pred_total
    return 2 * (prec * recall) / (prec + recall)


def convert_binary_bracketing(parse):
    transitions = []
    tokens = []

    for word in parse.split(' '):
        if word[0] != "(":
            if word == ")":
                transitions.append(1)
            else:
                tokens.append(word)
                transitions.append(0)

    return tokens, transitions


def to_indexed_contituents(parse):
    sp = parse.split()
    if len(sp) == 1:
        return set([(0, 1)])

    backpointers = []
    indexed_constituents = set()
    word_index = 0
    for index, token in enumerate(sp):
        if token == '(':
            backpointers.append(word_index)
        elif token == ')':
            start = backpointers.pop()
            end = word_index
            constituent = (start, end)
            indexed_constituents.add(constituent)
        else:
            word_index += 1
    return indexed_constituents


def build_tree(tokens, transitions):
    stack = []
    buf = tokens[::-1]

    for t in transitions:
        if t == 0:
            stack.append(buf.pop())
        elif t == 1:
            right = stack.pop()
            left = stack.pop()
            stack.append((left, right))

    assert len(stack) == 1

    return stack[0]


def get_parse(x):
    return Tree.fromstring(x.strip())


def get_binary_parse(x):
    tokens, transitions = convert_binary_bracketing(x.strip())
    tree = build_tree(tokens, transitions)
    return tree


def left_branching(tokens):
    length = len(tokens)
    if length <= 2:
        return tuple(tokens)
    return (left_branching(tokens[:length-1]), tokens[-1])


def right_branching(tokens):
    length = len(tokens)
    if length <= 2:
        return tuple(tokens)
    return (tokens[0], right_branching(tokens[1:]))


punctuation_words = set(['.', ',', ':', '-LRB-', '-RRB-', '\'\'', '``', '--', ';', '-', '?', '!', '...', '-LCB-', '-RCB-'])
currency_words = set(['#', '$', 'C$', 'A$'])
ellipsis = set(['*', '*?*', '0', '*T*', '*ICH*', '*U*', '*RNR*', '*EXP*', '*PPA*', '*NOT*'])
other = set(['HK$', '&', '**'])
tokens_to_remove = set.union(punctuation_words, currency_words)
tokens_to_remove = set.union(tokens_to_remove, ellipsis)
tokens_to_remove = set.union(tokens_to_remove, other)


def ppheuristic(bpt):
    def func(bpt, last=False):
        if not isinstance(bpt, (list, tuple)) and last and bpt in tokens_to_remove:
            return None
        if not isinstance(bpt, (list, tuple)):
            return bpt
        l = func(bpt[0])
        r = func(bpt[1], last=last)

        if r is None:
            return l
        return (l, r)

    bpt2 = func(bpt, last=True)
    tokens = get_tokens(bpt)
    tokens2 = get_tokens(bpt2)

    assert len(tokens) == len(tokens2) or len(tokens) == len(tokens2) + 1, \
        '{} != {}'.format(len(tokens), len(tokens2))

    if len(tokens) == len(tokens2) + 1:
        bpt2 = (bpt2, tokens[-1])

    return bpt2


def run(options):
    def read_pred():
        with open(options.pred) as f:
            for line in tqdm(f):
                yield json.loads(line)

    def read_ptb():
        with open(options.ptb) as f:
            for line in tqdm(f):
                yield json.loads(line)

    pred = list(read_pred())
    ptb = list(read_ptb())

    # READ GROUND TRUTH
    for data in tqdm(ptb, desc='ptb'):
        data['pt'] = get_parse(data['sentence1_parse'])
        data['bpt'] = get_binary_parse(data['sentence1_binary_parse'])
        data['tokens'] = data['pt'].leaves()
        data['s'] = ' '.join(data['tokens'])
    corpus_ptb = {x['pairID']: x for x in ptb}

    # READ PREDICTIONS
    for data in tqdm(pred, desc='pred'):
        eid = data['example_id']
        if eid not in corpus_ptb:
            continue
        data_ptb = corpus_ptb[eid]
        data['tokens'] = data_ptb['tokens']
        data['s'] = data_ptb['s']
        if options.branching is None:
            data['bpt'] = get_binary_parse(data['sent1_tree'])
        elif options.branching == 'left':
            data['bpt'] = left_branching(data['tokens'])
        elif options.branching == 'right':
            data['bpt'] = right_branching(data['tokens'])
        if options.ppheuristic:
            data['bpt'] = ppheuristic(data['bpt'])

    corpus_pred = {x['example_id']: x for x in pred}

    # (OPTIONAL): Remove punctuation.
    if options.nopunct:
        keys = list(corpus_ptb.keys())
        for k in keys:
            # Ground Truth
            data_ptb = corpus_ptb[k]
            mask = tokeep_punct_using_labels(data_ptb['pt'])
            data_ptb['pt'] = remove_using_mask(data_ptb['pt'], mask)
            data_ptb['bpt'] = remove_using_mask(data_ptb['bpt'], mask)

            # Predictions
            if k not in corpus_pred:
                continue
            data_pred = corpus_pred[k]
            data_pred['bpt'] = remove_using_mask(data_pred['bpt'], mask)

    # (OPTIONAL): Remove sentences below cutoff.
    if options.cutoff is not None:
        keys = list(corpus_ptb.keys())
        for k in keys:
            # Ground Truth
            data_ptb = corpus_ptb[k]
            length = get_len(data_ptb['bpt'])
            if length > options.cutoff:
                del corpus_ptb[k]

    print('# sentences (ground truth) = {}'.format(len(corpus_ptb)))

    # Evaluation
    f1 = 0
    n = 0

    for k in tqdm(corpus_ptb.keys(), desc='f1'):
        # Ground Truth
        data_ptb = corpus_ptb[k]
        length = get_len(data_ptb['bpt'])
        if options.full:
            gt = get_spans(data_ptb['pt'])
        else:
            gt = get_spans(data_ptb['bpt'])
        
        # (OPTIONAL): Skip words.
        if options.ignore_word:
            gt = remove_word_spans(gt, length)

        # If there are no eligible spans, then skip this example.
        if len(gt) == 0:
            continue

        # (OPTIONAL): Skip spans over entire sentence.
        if options.ignore_entire:
            gt = remove_entire_span(gt, length)

        # If there are no eligible spans, then skip this example.
        if len(gt) == 0:
            continue

        # Prediction
        if len(gt) == 1 and (0, length) in gt:
            pred = set([(0, length)])
        else:
            data_pred = corpus_pred[k]
            pred = get_spans(data_pred['bpt'])
            if options.ignore_word:
                pred = remove_word_spans(pred, length)
            if options.ignore_entire:
                pred = remove_entire_span(pred, length)

        f1 += example_f1(gt, pred)
        n += 1

    f1 /= n

    print('# of eligible sentences = {}'.format(n))
    print('f1 = {:.3f}'.format(f1))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ptb', default=os.path.expanduser('~/Downloads/ptb.jsonl'), type=str)
    parser.add_argument('--pred', default=os.path.expanduser('~/Downloads/PRPN_parses/PRPNLM_ALLNLI/parsed_WSJ_PRPNLM_AllLI_ESLM.jsonl'), type=str)
    parser.add_argument('--branching', default=None, choices=('left', 'right'))
    parser.add_argument('--nopunct', action='store_true')
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--ppheuristic', action='store_true')
    parser.add_argument('--cutoff', default=None, type=int)

    parser.add_argument('--ignore_entire', action='store_true')
    parser.add_argument('--ignore_word', action='store_true')

    parser.add_argument('--eval', default=None, choices=('wsj-10', 'wsj-10-full', 'wsj-40', 'wsj-dev', 'wsj-test'))

    options = parser.parse_args()

    if options.eval == 'wsj-10':
        options.cutoff = 10
        options.nopunct = True
        options.ignore_word = True
        options.ignore_entire = True
        options.full = False
        options.ptb = os.path.expanduser('~/Downloads/ptb.jsonl')
    elif options.eval == 'wsj-10-full':
        options.cutoff = 10
        options.nopunct = True
        options.ignore_word = True
        options.ignore_entire = True
        options.full = True
        options.ptb = os.path.expanduser('~/Downloads/ptb.jsonl')
    elif options.eval == 'wsj-40':
        options.cutoff = 40
        options.nopunct = True
        options.ignore_word = True
        options.ignore_entire = True
        options.full = True
        options.ptb = os.path.expanduser('~/Downloads/ptb-test.jsonl')
    elif options.eval == 'wsj-dev':
        options.cutoff = None
        options.nopunct = False
        options.ignore_word = True
        options.ignore_entire = False
        options.full = False
        options.ptb = os.path.expanduser('~/Downloads/ptb-dev.jsonl')
    elif options.eval == 'wsj-test':
        options.cutoff = None
        options.nopunct = False
        options.ignore_word = True
        options.ignore_entire = False
        options.full = False
        options.ptb = os.path.expanduser('~/Downloads/ptb-test.jsonl')

    print(json.dumps(options.__dict__, sort_keys=True, indent=4))

    run(options)
