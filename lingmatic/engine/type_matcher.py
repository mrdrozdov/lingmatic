import os
import json

from collections import Counter

import numpy as np

from lingmatic.engine.parse_comparison import heuristic


class SimpleSkipCriteria(object):
    def should_skip(self, corpus_gt, corpus_pred, key):
        return key not in corpus_gt


class ParseTreeHandler(object):
    def __init__(self, postprocess=False):
        self.postprocess = postprocess

    def get(self, corpus_gt, corpus_pred, key):
        gt = corpus_gt[key]
        pred = corpus_pred[key]

        if self.postprocess:
            pred.binary_parse_spans, pred.binary_parse_tree = heuristic(pred)

        return gt, pred


class TopDownS2LConverter(object):
    """ Top Down Spans-to-Labels Converter """
    def __init__(self, skip_one=False):
        self.skip_one = skip_one

    def convert(self, gt, pred):
        span2label = {}

        stack = [(0, gt.parse)]

        while len(stack) > 0:
            idx, tr = stack.pop()

            if isinstance(tr, str):
                continue

            length = len(tr.leaves())
            span = (idx, idx+length)
            label = tr.label()

            if self.skip_one and length == 1:
                continue

            span2label[span] = label

            seen_length = idx
            for subtree in tr:
                if isinstance(subtree, str):
                    length = 1
                else:
                    length = len(subtree.leaves())
                    stack.append((seen_length, subtree))
                seen_length += length

        return span2label


class PhraseCounter(object):
    def reset(self):
        self.correct = Counter()
        self.total = Counter()

    def count(self, gt, pred, span2label):
        for span in gt.binary_parse_spans:
            length = span[1] - span[0]
            if length == 1:
                continue
            if span not in span2label:
                continue
            label = span2label[span]
            if span in pred.binary_parse_spans:
                self.correct[label] += 1
            self.total[label] += 1

    def print(self):
        labels = sorted(self.total.keys())

        print('Phrase Counter')
        for label in labels:
            ncorrect = self.correct[label]
            ntotal = self.total[label]
            print('{:<6} {:<8} {:<8} {:<8.3f}'.format(label, ncorrect, ntotal, float(ncorrect)/float(ntotal)))
        print('Total', sum(self.total.values()))
        print


class PhraseLengthCounter(object):
    def reset(self):
        self.correct = {}
        self.total = {}

    def count(self, gt, pred, span2label):
        for span in gt.binary_parse_spans:
            length = span[1] - span[0]
            if length == 1:
                continue
            if span not in span2label:
                continue
            label = span2label[span]
            if span in pred.binary_parse_spans:
                self.correct.setdefault(label, Counter())[length] += 1
            self.total.setdefault(label, Counter())[length] += 1

    def print(self):
        labels = sorted(self.total.keys())
        lengths = set()

        for v in self.total.values():
            for k in v.keys():
                lengths.add(k)

        ncorrect = np.zeros((len(labels), len(lengths)))
        ntotal = np.zeros((len(labels), len(lengths)))

        for i, label in enumerate(sorted(labels)):
            for j, length in enumerate(sorted(lengths)):
                self.correct.setdefault(label, Counter())[length] += 0
                ncorrect[i, j] = self.correct[label][length]
                ntotal[i, j] = self.total[label][length]

        # Print
        line = ' ' * 12
        line += ' ' * 5
        for i, length in enumerate(sorted(lengths)):
            line += ' {:7}'.format(length)
        print(line)

        for i, label in enumerate(sorted(labels)):
            line = '{:12}'.format(label)
            line += '{:.3f}'.format(ncorrect[i].sum() / ntotal[i].sum())
            for j, length in enumerate(sorted(lengths)):
                ncorrect[i, j] = self.correct[label][length]
                ntotal[i, j] = self.total[label][length]
                line += ' {:7}'.format(int(ncorrect[i, j]))
            print(line)
            line = ' ' * 12
            line += ' ' * 5
            for j, length in enumerate(sorted(lengths)):
                ncorrect[i, j] = self.correct[label][length]
                ntotal[i, j] = self.total[label][length]
                line += ' {:7}'.format(int(ntotal[i, j]))
            print(line)
        print

        for i, label in enumerate(sorted(labels)):
            ncsum, ntsum = int(ncorrect[i].sum()), int(ntotal[i].sum())
            recall = ncsum / float(ntsum)
            line = '{:12} {:7} {:7} {:.3f}'.format(label, ncsum, ntsum, recall)
            print(line)
        print


class AttachmentCounter(object):
    def reset(self):
        self.relevant = Counter()
        self.theoretical = Counter()
        self.both = Counter()
        self.total = Counter()

    def build_recurse_tree(self, span2label, counter):
        def recurse_tree(x, sofar, guide=None):
            if isinstance(x, str):
                return 1
            l_len = recurse_tree(x[0], sofar, guide=guide)
            r_len = recurse_tree(x[1], sofar + l_len, guide=guide)

            l_span = (sofar, sofar+l_len)
            r_span = (l_span[1], l_span[1] + r_len)

            if l_span in span2label and r_span in span2label:
                if guide is None or ((l_span, r_span) in guide or (l_span in guide and r_span in guide)):
                    key = (span2label[l_span], span2label[r_span])
                    counter[key] += 1

            return l_len + r_len
        return recurse_tree

    def count(self, gt, pred, span2label):
        def build_guide(x, sofar, guide={}):
            if isinstance(x, str):
                return 1, guide

            l_len, _ = build_guide(x[0], sofar, guide=guide)
            r_len, _ = build_guide(x[1], sofar + l_len, guide=guide)

            l_span = (sofar, sofar+l_len)
            r_span = (l_span[1], l_span[1] + r_len)

            guide[(l_span, r_span)] = True

            return l_len + r_len, guide

        _, guide = build_guide(pred.binary_parse_tree, 0)

        self.build_recurse_tree(span2label, self.relevant)(pred.binary_parse_tree, 0)
        self.build_recurse_tree(span2label, self.total)(gt.binary_parse_tree, 0)
        self.build_recurse_tree(span2label, self.both)(gt.binary_parse_tree, 0, guide)
        self.build_recurse_tree(span2label, self.theoretical)(gt.binary_parse_tree, 0, pred.binary_parse_spans)

    def print(self):
        keys = sorted(self.total.keys())

        print('Attachment Counter')
        for key in keys:
            l, r = key
            npred = self.relevant[key]
            nboth = self.both[key]
            ngt = self.total[key]
            ntheoretical = self.theoretical[key]
            print('{:<6} {:<6} {:<8} {:<8} {:<8} {:<8}'.format(l, r, npred, nboth, ngt, ntheoretical))
        print('{:<6} {:<6} {:<8} {:<8} {:<8} {:<8}'.format('Total', '',
            sum(self.relevant.values()),
            sum(self.both.values()),
            sum(self.total.values()),
            sum(self.theoretical.values()),
            ))
        print


class MatchEntities(object):
    def __init__(self, skip_criteria=None, parsetree_handler=None, s2l_converter=None, phrase_counter_lst=None):
        if skip_criteria is None:
            skip_criteria = SimpleSkipCriteria()
        if parsetree_handler is None:
            parsetree_handler = ParseTreeHandler()
        if s2l_converter is None:
            s2l_converter = TopDownS2LConverter()
        if phrase_counter_lst is None:
            phrase_counter_lst = PhraseCounter()
        if not isinstance(phrase_counter_lst, (list, tuple)):
            phrase_counter_lst = [phrase_counter_lst]

        self.skip_criteria = skip_criteria
        self.parsetree_handler = parsetree_handler
        self.s2l_converter = s2l_converter
        self.phrase_counter_lst = phrase_counter_lst

        self.reset()

    def reset(self):
        stats = {}
        stats['count'] = 0
        stats['skipped'] = 0

        self.stats = stats
        for counter in self.phrase_counter_lst:
            counter.reset()

    def run(self, corpus_gt, corpus_pred):
        for key in tqdm(corpus_pred.keys()):
            if self.skip_criteria.should_skip(corpus_gt, corpus_pred, key):
                self.stats['skipped'] += 1
                continue

            pred = corpus_pred[key]
            gt = corpus_gt[key]

            gt, pred = self.parsetree_handler.get(corpus_gt=corpus_gt, corpus_pred=corpus_pred, key=key)
            span2label = self.s2l_converter.convert(gt, pred)

            for counter in self.phrase_counter_lst:
                counter.count(gt, pred, span2label)

        print('Count={} Skipped={}'.format(self.stats['count'], self.stats['skipped']))


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
    parser.add_argument('--gt_type', default='gt', choices=('gt'))
    parser.add_argument('--pred_type', default='infer', choices=('gt', 'infer'))
    parser.add_argument('--track_length', action='store_true')
    parser.add_argument('--track_attachment', action='store_true')
    parser.add_argument('--skip_one', action='store_true')
    parser.add_argument('--postprocess', action='store_true')
    options = parser.parse_args()

    print(json.dumps(options.__dict__, indent=4, sort_keys=True))

    if options.gt_type == 'gt':
        gt_deserializer_cls_lst = ParseTreeDeserializeGroundTruth

    if options.pred_type == 'gt':
        pred_deserializer_cls_lst = ParseTreeDeserializeGroundTruth
    elif options.pred_type == 'infer':
        pred_deserializer_cls_lst = ParseTreeDeserializeInfer


    gt_path = options.gt
    reader = ParseTreeReader(limit=options.limit, parse_tree_config=dict(deserializer_cls_lst=gt_deserializer_cls_lst))
    results = list(tqdm(reader.read(gt_path)))
    corpus_gt = {x.example_id: x for x in results}

    infer_path = options.pred
    reader = ParseTreeReader(parse_tree_config=dict(deserializer_cls_lst=pred_deserializer_cls_lst))
    results = list(tqdm(reader.read(infer_path)))
    corpus_pred = {x.example_id: x for x in results}

    # Match Entities
    phrase_counter_lst = [PhraseCounter()]
    if options.track_length:
        phrase_counter_lst.append(PhraseLengthCounter())
    if options.track_attachment:
        phrase_counter_lst.append(AttachmentCounter())

    matcher = MatchEntities(
        s2l_converter=TopDownS2LConverter(skip_one=options.skip_one),
        parsetree_handler=ParseTreeHandler(postprocess=options.postprocess),
        phrase_counter_lst=phrase_counter_lst)
    matcher.run(corpus_gt, corpus_pred)
    for counter in matcher.phrase_counter_lst:
        counter.print()
