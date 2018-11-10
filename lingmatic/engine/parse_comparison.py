"""
WSJ10 (7422 sentences)

--max_length 10
--strip_punct
--trivial
--data ptb

WSJ40

--max_length 40
--strip_punct
--trivial
--data ptb-test

"""

import os
import json
from nltk.tree import Tree

from collections import OrderedDict

import numpy as np


punctuation_words = set(['.', ',', ':', '-LRB-', '-RRB-', '\'\'', '``', '--', ';', '-', '?', '!', '...', '-LCB-', '-RCB-'])
currency_tags_words = set(['#', '$', 'C$', 'A$'])
ellipsis = set(['*', '*?*', '0', '*T*', '*ICH*', '*U*', '*RNR*', '*EXP*', '*PPA*', '*NOT*'])
other = set(['HK$', '&', '**'])
other2 = set(["'", '`'])
tokens_to_remove = set.union(punctuation_words, currency_tags_words)
tokens_to_remove = set.union(tokens_to_remove, ellipsis)
tokens_to_remove = set.union(tokens_to_remove, other)
# tokens_to_remove = set.union(tokens_to_remove, other2)
# %

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


def remove_trivial_spans(spans):
    length = max(x[1] for x in spans)
    spans.remove((0, length))
    return spans


class CompareF1(object):
    name = 'f1'

    def __init__(self, verbose=False, trivial=False):
        self.score = 0
        self.verbose = verbose
        self.results = []
        self.trivial = trivial

    def compare(self, gt, pred):
        # gt_spans = gt.binary_parse_spans
        gt_spans = gt.f1_spans
        pred_spans = pred.f1_spans
        f1 = example_f1(pred_spans, gt_spans)
        self.score += f1

        if self.verbose:
            if f1 < 1:
                out = OrderedDict()
                out['example_id'] = gt.example_id
                out['length'] = len(gt.tokens)
                out['f1'] = f1
                self.results.append(out)
                # print(json.dumps(out))

    def finish(self, count):
        self.score /= count

        if self.verbose:
            lengths = {x['length'] for x in self.results}
            for k in sorted(lengths):
                f1s = [x['f1'] for x in self.results if x['length'] == k]
                n = len(f1s)
                mean_f1 = np.array(f1s).mean()
                print('{},{:.3f},{}'.format(k, mean_f1, n))

    def print(self):
        return '{} {:.3f}'.format(self.name, self.score)


# def get_parse_spans(parse):
#     stack = [(0, parse)]
#     spans = set()
#     while len(stack) > 0:
#         i, root = stack.pop()
#         size = len(root.leaves())
#         span = (i, i + size)
#         spans.add(span)
#         sofar = i
#         for x in root:
#             if not isinstance(x, str):
#                 size = len(x.leaves())
#                 stack.append((sofar, x))
#             else:
#                 size = 1
#             sofar += size
#     return spans


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


def should_remove(x, tokens_to_remove=punctuation_words):
    if not isinstance(x, str):
        return False
    if x in tokens_to_remove:
        return True
    return False


def check_tree(tr, val):
    if not isinstance(tr, (list, tuple)):
        return tr is val
    left = check_tree(tr[0], val)
    right = check_tree(tr[0], val)
    return left or right


def parse_to_tuples(parse):
    if isinstance(parse, str):
        return parse
    result = tuple(parse_to_tuples(x) for x in parse)
    if len(result) == 1: # unary
        result = result[0]
    return result


def tuples_to_spans(tuples, i=0):
    if isinstance(tuples, str):
        return set(), 1
    spans = set()
    sofar = i
    for x in tuples:
        subspans, size = tuples_to_spans(x, sofar)
        spans = set.union(spans, subspans)
        sofar += size
    spans.add((i, sofar))
    return spans, sofar - i


def remove_all_punct_parse(parse):
    def helper(tree):
        if isinstance(tree, str):
            return tree

        parts = [helper(x) for x in tree if not should_remove(x, tokens_to_remove=tokens_to_remove)]
        parts = [x for x in parts if x is not None]

        if len(parts) == 0:
            return None
        return tuple(parts)

    result = helper(parse)

    return result


def remove_all_punct(tr):
    def helper(tree):
        if isinstance(tree, str):
            return tree

        left = helper(tree[0])
        right = helper(tree[1])

        left_skip = left is None or should_remove(left, tokens_to_remove=tokens_to_remove)
        right_skip = right is None or should_remove(right, tokens_to_remove=tokens_to_remove)

        if left_skip and right_skip:
            return None
        elif left_skip:
            return right
        elif right_skip:
            return left
        return (left, right)

    result = helper(tr)

    # print(result)

    assert check_tree(result, None) == False

    return result


def get_tokens(tr):
    if not isinstance(tr, (list, tuple)):
        return [tr]
    return get_tokens(tr[0]) + get_tokens(tr[1])


# def right_branching(tokens):
#     if len(tokens) == 2:
#         return tuple(tokens)
#     return (right_branching(tokens[:-1]), tokens[-1])


def right_branching(tokens):
    if len(tokens) == 2:
        return tuple(tokens)
    return (tokens[0], right_branching(tokens[1:]))


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


def rb_baseline(tokens):
    tr = right_branching(tokens)
    spans = set(get_spans(tr))
    return spans, tr


def classic_gt(pt):
    tr = remove_all_punct(pt.binary_parse_tree)
    spans = set(get_spans(tr))
    parse = remove_all_punct_parse(pt.parse)
    return parse, spans, tr


def classic(pt):
    tr = remove_all_punct(pt.binary_parse_tree)
    spans = set(get_spans(tr))
    return spans, tr


class ParseComparison(object):
    def __init__(self, comparisons=[CompareF1(), AverageDepth()], count_missing=False,
                 postprocess=False, rbranch=False, trivial=False, max_length=None, strip_punct=False,
                 guide_mode='constrain'):
        self.stats = OrderedDict()
        self.stats['count'] = 0
        self.stats['missing'] = 0
        self.stats['skipped-key'] = 0
        self.stats['skipped-len'] = 0
        self.stats['skipped-guide'] = 0
        self.stats['skipped-1'] = 0
        self.stats['skipped-2'] = 0
        self.stats['skip-empty-parse'] = 0
        self.comparisons = comparisons
        self.count_missing = count_missing
        self.postprocess = postprocess
        self.rbranch = rbranch
        self.guide_mode = guide_mode
        self.strip_punct = strip_punct
        self.trivial = trivial
        self.max_length = max_length

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

        if len(gt.tokens) > 2 and not skip:
            if self.strip_punct:
                gt.parse, gt.binary_parse_spans, gt.binary_parse_tree = classic_gt(gt)

        if len(get_tokens(gt.binary_parse_tree)) == 1:
            self.stats['skipped-1'] += 1
            skip = True

        if self.trivial:
            if len(get_tokens(gt.binary_parse_tree)) == 2:
                self.stats['skipped-2'] += 1
                skip = True

        if self.max_length is not None:
            if len(get_tokens(gt.binary_parse_tree)) > self.max_length:
                skip = True
            # if len(get_tokens(gt.binary_parse_tree)) == self.max_length + 1:
            #     print(get_tokens(gt.binary_parse_tree))

        if not skip:
            if self.rbranch:
                pred.binary_parse_spans, pred.binary_parse_tree = rb_baseline(get_tokens(gt.binary_parse_tree))
            elif self.strip_punct:
                pred.binary_parse_spans, pred.binary_parse_tree = classic(pred)

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

            gt.f1_spans = tuples_to_spans(gt.parse)[0]
            pred.f1_spans = pred.binary_parse_spans
            if self.trivial:
                gt.f1_spans = remove_trivial_spans(gt.f1_spans)
                pred.f1_spans = remove_trivial_spans(pred.f1_spans)

            if len(gt.f1_spans) == 0:
                self.stats['skip-empty-parse'] += 1
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
                skip = False
                if len(gt.tokens) > 2:
                    self.stats['skipped-len'] += 1
                    # print(gt.example_id, len(gt.tokens))
                    continue
                for judge in self.comparisons:
                    if len(get_tokens(gt.binary_parse_tree)) <= 2 and hasattr(judge, 'trivial') and judge.trivial:
                        skip = True
                        break
                    judge.compare(gt, pred)
                if skip:
                    continue
                self.stats['count'] += 1
                self.stats['missing'] += 1

        for judge in self.comparisons:
            judge.finish(self.stats['count'])
        for judge in self.comparisons:
            print(judge.print())

        print('seen', len(seen))
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
    parser.add_argument('--rbranch', action='store_true')
    parser.add_argument('--skip_missing', action='store_true')
    parser.add_argument('--trivial', action='store_true')
    parser.add_argument('--strip_punct', action='store_true')
    parser.add_argument('--max_length', default=None, type=int)
    parser.add_argument('--data_type', default='ptb', choices=('ptb', 'nli'))
    parser.add_argument('--verbose', action='store_true')
    options = parser.parse_args()

    print(json.dumps(options.__dict__, sort_keys=True, indent=4))

    class DeserializeGT(ParseTreeDeserializeGroundTruth):
        def get_parse(self):
            parse = super(DeserializeGT, self).get_parse()
            return parse_to_tuples(parse)

        # def get_binary_parse_tree(self):
        #     return None


    class DeserializePred(ParseTreeDeserializeInfer):
        def get_parse(self):
            return None

        def get_raw_binary_parse(self):
            if options.postprocess:
                return None
            else:
                return super(DeserializePred, self).get_raw_binary_parse()

        # def get_binary_parse_tree(self):
        #     if options.postprocess:
        #         return super(DeserializePred, self).get_binary_parse_tree()
        #     else:
        #         return None

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
    print('gt-results', len(results))
    corpus_gt = {x.example_id: x for x in results}

    infer_path = options.pred
    reader = ParseTreeReader(limit=limit, parse_tree_config=dict(deserializer_cls_lst=pred_deserializer_cls_lst))
    results = list(reader.read(infer_path))
    print('pred-results', len(results))
    corpus_pred = {x.example_id: x for x in results}

    corpus_guide = None
    if options.guide is not None:
        guide_path = options.guide
        deserializer_cls_lst = pred_deserializer_cls_lst if options.guide_type == 'pred' else gt_deserializer_cls_lst
        reader = ParseTreeReader(limit=limit, parse_tree_config=dict(deserializer_cls_lst=deserializer_cls_lst))
        results = list(reader.read(guide_path))
        corpus_guide = {x.example_id: x for x in results}

    # Comparisons.
    judge_compare_f1 = CompareF1(verbose=options.verbose, trivial=options.trivial)
    judge_average_depth = AverageDepth()
    comparisons = [judge_compare_f1, judge_average_depth]

    # Corpus Stats
    ParseComparison(
        comparisons=comparisons,
        count_missing=not options.skip_missing,
        postprocess=options.postprocess,
        rbranch=options.rbranch,
        strip_punct=options.strip_punct,
        max_length=options.max_length,
        trivial=options.trivial,
        guide_mode=options.guide_mode,
        ).run(corpus_gt, corpus_pred, corpus_guide)
    print('Count (Ground Truth): {}'.format(len(corpus_gt)))
    print('Count (Predictions): {}'.format(len(corpus_pred)))
