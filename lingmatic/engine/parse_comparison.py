import os

punctuation_words = set(['.', ',', ':', '-LRB-', '-RRB-', '\'\'', '``', '--', ';', '-', '?', '!', '...', '-LCB-', '-RCB-'])


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
        self.score += average_depth(pred.raw_binary_parse)

    def finish(self, count):
        self.score /= count

    def print(self):
        return '{} {:.3f}'.format(self.name, self.score)


class CompareF1(object):
    name = 'f1'

    def __init__(self):
        self.score = 0

    def compare(self, gt, pred):
        gt_spans = gt.binary_parse_spans
        pred_spans = pred.binary_parse_spans
        self.score += example_f1(pred_spans, gt_spans)

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
    def __init__(self, comparisons=[CompareF1(), AverageDepth()], count_missing=False, postprocess=False):
        self.stats = {}
        self.stats['count'] = 0
        self.stats['skipped'] = 0
        self.comparisons = comparisons
        self.count_missing = count_missing
        self.postprocess = postprocess

    def should_run(self, corpus_gt, corpus_pred, key):
        gt, pred, skip = None, None, False
        if key in corpus_gt:
            pred = corpus_pred[key]
            gt = corpus_gt[key]
        else:
            skip = True
            self.stats['skipped'] += 1
        return gt, pred, skip

    def preprocess(self, gt, pred):
        skip = False

        if self.postprocess:
            pred.binary_parse_spans, pred.binary_parse_tree = heuristic(pred)

        return gt, pred, skip

    def run(self, corpus_gt, corpus_pred):
        seen = set()
        for key in tqdm(corpus_pred.keys()):

            seen.add(key)

            gt, pred, skip = self.should_run(corpus_gt, corpus_pred, key)

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
                for judge in self.comparisons:
                    judge.compare(gt, gt)
                self.stats['count'] += 1

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

    from tqdm import tqdm


    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', default=None, type=int)
    parser.add_argument('--gt', default=os.path.expanduser('~/Downloads/ptb.jsonl'), type=str)
    parser.add_argument('--pred', default=os.path.expanduser('~/Downloads/PRPN_parses/PRPNLM_ALLNLI/parsed_WSJ_PRPNLM_AllLI_ESLM.jsonl'), type=str)
    parser.add_argument('--postprocess', action='store_true')
    options = parser.parse_args()


    class DeserializeGT(ParseTreeDeserializeGroundTruth):
        def get_parse(self):
            return None

        def get_binary_parse_tree(self):
            return None

        def get_tokens(self):
            return None


    class DeserializePred(ParseTreeDeserializeInfer):
        def get_parse(self):
            return None

        def get_binary_parse_tree(self):
            if options.postprocess:
                return super(DeserializePred, self).get_binary_parse_tree()
            else:
                return None

        def get_tokens(self):
            return None


    limit = options.limit

    gt_path = options.gt
    reader = ParseTreeReader(limit=limit, parse_tree_config=dict(deserializer_cls=DeserializeGT))
    results = list(tqdm(reader.read(gt_path)))
    corpus_gt = {x.example_id: x for x in results}

    infer_path = options.pred
    reader = ParseTreeReader(limit=limit, parse_tree_config=dict(deserializer_cls=DeserializePred))
    results = list(tqdm(reader.read(infer_path)))
    corpus_pred = {x.example_id: x for x in results}

    # Corpus Stats
    ParseComparison(count_missing=True, postprocess=options.postprocess).run(corpus_gt, corpus_pred)
    print('Count (Ground Truth): {}'.format(len(corpus_gt)))
    print('Count (Predictions): {}'.format(len(corpus_pred)))
