import os
import json

from collections import OrderedDict


punctuation_words = set(['.', ',', ':', '-LRB-', '-RRB-', '\'\'', '``', '--', ';', '-', '?', '!', '...', '-LCB-', '-RCB-'])


def tree_to_string(parse):
    if not isinstance(parse, (list, tuple)):
        return parse
    if len(parse) == 1:
        return parse[0]
    else:
        return '( ' + tree_to_string(parse[0]) + ' ' + tree_to_string(parse[1]) + ' )'


def average_depth(parse):
    depths = []
    current_depth = 0
    for token in parse.split():
        if token == '(':
            current_depth += 1
        elif token == ')':
            current_depth -= 1
        else:
            depths.append(current_depth)
    return float(sum(depths)) / len(depths)


def example_f1(c1, c2):
    prec = float(len(c1.intersection(c2))) / len(c2)  # TODO: More efficient.
    return prec  # For strictly binary trees, P = R = F1


class AverageDepth(object):
    name = 'depth'

    def __init__(self):
        self.score = 0

    def compare(self, gt, pred):
        parse = pred.raw_binary_parse
        if parse is None:
            parse = tree_to_string(pred.binary_parse_tree)  # This is necessary when post-processing.
        self.score += average_depth(parse)

    def finish(self, count):
        self.score /= count

    def print(self):
        return '{} {:.1f}'.format(self.name, self.score)


class CompareF1(object):
    name = 'f1'

    def __init__(self, verbose=False):
        self.score = 0
        self.verbose = verbose

    def compare(self, gt, pred):
        gt_spans = gt.binary_parse_spans
        pred_spans = pred.binary_parse_spans
        f1 = example_f1(pred_spans, gt_spans)
        self.score += f1

        if self.verbose:
            if f1 < 1:
                out = OrderedDict()
                out['example_id'] = gt.example_id
                out['length'] = len(gt.tokens)
                out['f1'] = f1
                print(json.dumps(out))

    def finish(self, count):
        self.score /= count

    def print(self):
        return '{} {:.3f}'.format(self.name, self.score)


def get_spans(tree):
    def helper(tr, idx=0):
        if isinstance(tr, (str, int)):
            return 1, []
        left, left_spans = helper(tr[0], idx=idx)
        right, right_spans = helper(tr[1], idx=idx+left)
        span = [(idx, idx + left + right)]
        spans = span + left_spans + right_spans
        return left + right, spans

    _, spans = helper(tree)

    return spans


def should_remove(x):
    if not isinstance(x, str):
        return False
    if x in punctuation_words:
        return True
    return False


def remove_punctuation(tr, last_only=False):
    def helper(tree, last_only=False):
        removed = None
        if isinstance(tree, str):
            return tree, removed

        if last_only:
            left = tree[0]
        else:
            left, _ = helper(tree[0], last_only=False)
        right, removed = helper(tree[1], last_only=last_only)

        if last_only:
            left_skip = False
        else:
            left_skip = left is None or should_remove(left)
        right_skip = right is None or should_remove(right)

        if last_only and right_skip:
            removed = right

        if left_skip and right_skip:
            return None, removed
        elif left_skip:
            return right, removed
        elif right_skip:
            return left, removed
        return (left, right), removed

    result, removed = helper(tr, last_only=last_only)

    return result, removed


def heuristic(pt):
    tr, removed = remove_punctuation(pt.binary_parse_tree, last_only=True)
    if removed is None:
        return pt.binary_parse_spans, pt.binary_parse_tree
    tr = (tr, removed)
    spans = set(get_spans(tr))
    return spans, tr


class ParseComparison(object):
    def __init__(self, comparisons=[CompareF1(), AverageDepth()], count_missing=False, postprocess=False,
                 guide_mode='constrain'):
        self.stats = OrderedDict()
        self.stats['count'] = 0
        self.stats['missing'] = 0
        self.stats['skipped-key'] = 0
        self.stats['skipped-len'] = 0
        self.stats['skipped-guide'] = 0
        self.comparisons = comparisons
        self.count_missing = count_missing
        self.postprocess = postprocess
        self.guide_mode = guide_mode

    def should_run(self, corpus_gt, corpus_pred, corpus_guide, key):
        gt, pred, skip = None, None, False

        if key not in corpus_gt:
            skip = True
            self.stats['skipped-key'] += 1

        if not skip and corpus_guide is not None:
            if self.guide_mode == 'constrain':
                if key not in corpus_guide:
                    skip = True
                    self.stats['skipped-guide'] += 1
            elif self.guide_mode == 'skip':
                if key in corpus_guide:
                    skip = True
                    self.stats['skipped-guide'] += 1

        if not skip:
            pred = corpus_pred[key]
            gt = corpus_gt[key]

        return gt, pred, skip

    def preprocess(self, gt, pred):
        skip = False

        if self.postprocess:
            pred.binary_parse_spans, pred.binary_parse_tree = heuristic(pred)

        return gt, pred, skip

    def run(self, corpus_gt, corpus_pred, corpus_guide=None):
        seen = set()
        for key in corpus_pred.keys():

            seen.add(key)

            gt, pred, skip = self.should_run(corpus_gt, corpus_pred, corpus_guide, key)

            if skip:
                continue

            gt, pred, skip = self.preprocess(gt, pred)

            if skip:
                continue

            for judge in self.comparisons:
                judge.compare(gt, pred)

            self.stats['count'] += 1

        if self.count_missing:
            for key in corpus_gt.keys():
                if key in seen:
                    continue
                gt = corpus_gt[key]
                pred = gt
                if len(gt.tokens) > 2:
                    self.stats['skipped-len'] += 1
                    # print(gt.example_id, len(gt.tokens))
                    continue
                for judge in self.comparisons:
                    judge.compare(gt, pred)
                self.stats['count'] += 1
                self.stats['missing'] += 1

        for judge in self.comparisons:
            judge.finish(self.stats['count'])
        for judge in self.comparisons:
            print(judge.print())

        print(' '.join(['{}={}'.format(k, v) for k, v in self.stats.items()]))


if __name__ == '__main__':
    import argparse

    from lingmatic.engine.parsetree import ParseTreeReader
    from lingmatic.engine.parsetree import ParseTreeDeserializeGroundTruth
    from lingmatic.engine.parsetree import ParseTreeDeserializeInfer


    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', default=None, type=int)
    parser.add_argument('--gt', default=os.path.expanduser('~/Downloads/ptb.jsonl'), type=str)
    parser.add_argument('--pred', default=os.path.expanduser('~/Downloads/PRPN_parses/PRPNLM_ALLNLI/parsed_WSJ_PRPNLM_AllLI_ESLM.jsonl'), type=str)
    parser.add_argument('--guide', default=None, type=str)
    parser.add_argument('--guide_type', default='pred', choices=('gt', 'pred'))
    parser.add_argument('--guide_mode', default='constrain', choices=('constrain', 'skip'))
    parser.add_argument('--postprocess', action='store_true')
    parser.add_argument('--data_type', default='ptb', choices=('ptb', 'nli'))
    parser.add_argument('--print_diff', action='store_true')
    options = parser.parse_args()

    print(json.dumps(options.__dict__, sort_keys=True, indent=4))

    class DeserializeGT(ParseTreeDeserializeGroundTruth):
        def get_parse(self):
            return None

        def get_binary_parse_tree(self):
            return None


    class DeserializePred(ParseTreeDeserializeInfer):
        def get_parse(self):
            return None

        def get_raw_binary_parse(self):
            if options.postprocess:
                return None
            else:
                return super(DeserializePred, self).get_raw_binary_parse()

        def get_binary_parse_tree(self):
            if options.postprocess:
                return super(DeserializePred, self).get_binary_parse_tree()
            else:
                return None

        def get_tokens(self):
            return None


    class DesGT_NLI_1(DeserializeGT):
        LABEL_MAP = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

        def should_skip(self):
            return self.obj['gold_label'] not in self.LABEL_MAP

        def get_id(self):
            return self.obj[self.key_id] + '_1'

    class DesGT_NLI_2(DesGT_NLI_1):
        key_parse = 'sentence2_parse'
        key_binary_parse = 'sentence2_binary_parse'

        def get_id(self):
            return self.obj[self.key_id] + '_2'

    class DesPred_NLI_1(DeserializePred):
        def get_id(self):
            return self.obj[self.key_id] + '_1'

    class DesPred_NLI_2(DesPred_NLI_1):
        key_binary_parse = 'sent2_tree'

        def get_id(self):
            return self.obj[self.key_id] + '_2'


    if options.data_type == 'ptb':
        gt_deserializer_cls_lst = DeserializeGT
        pred_deserializer_cls_lst = DeserializePred
    elif options.data_type == 'nli':
        gt_deserializer_cls_lst = [DesGT_NLI_1, DesGT_NLI_2]
        pred_deserializer_cls_lst = [DesPred_NLI_1, DesPred_NLI_2]


    limit = options.limit

    gt_path = options.gt
    reader = ParseTreeReader(limit=limit, parse_tree_config=dict(deserializer_cls_lst=gt_deserializer_cls_lst))
    results = list(reader.read(gt_path))
    corpus_gt = {x.example_id: x for x in results}

    infer_path = options.pred
    reader = ParseTreeReader(limit=limit, parse_tree_config=dict(deserializer_cls_lst=pred_deserializer_cls_lst))
    results = list(reader.read(infer_path))
    corpus_pred = {x.example_id: x for x in results}

    corpus_guide = None
    if options.guide is not None:
        guide_path = options.guide
        deserializer_cls_lst = pred_deserializer_cls_lst if options.guide_type == 'pred' else gt_deserializer_cls_lst
        reader = ParseTreeReader(limit=limit, parse_tree_config=dict(deserializer_cls_lst=deserializer_cls_lst))
        results = list(reader.read(guide_path))
        corpus_guide = {x.example_id: x for x in results}

    # Comparisons.
    judge_compare_f1 = CompareF1(verbose=options.print_diff)
    judge_average_depth = AverageDepth()
    comparisons = [judge_compare_f1, judge_average_depth]

    # Corpus Stats
    ParseComparison(
        comparisons=comparisons,
        count_missing=True,
        postprocess=options.postprocess,
        guide_mode=options.guide_mode,
        ).run(corpus_gt, corpus_pred, corpus_guide)
    print('Count (Ground Truth): {}'.format(len(corpus_gt)))
    print('Count (Predictions): {}'.format(len(corpus_pred)))
