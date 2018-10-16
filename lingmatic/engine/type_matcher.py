import os

from collections import Counter


def match_entities(corpus_gt, corpus_pred):
    stats = {}
    stats['count'] = 0
    stats['skipped'] = 0

    phrase_counter_correct = Counter()
    phrase_counter_total = Counter()

    for key in tqdm(corpus_pred.keys()):
        if key in corpus_gt:
            pred = corpus_pred[key]
            gt = corpus_gt[key]
        else:
            skipped += 1
            continue

        span2label = {}
        span2tr = {}

        # Both styles work and give the same result, but top-down is much faster.
        style = 'top-down'
        # style = 'span-set'

        if style == 'top-down':

            stack = [(0, gt.parse)]

            while len(stack) > 0:
                idx, tr = stack.pop()

                if isinstance(tr, str):
                    continue

                length = len(tr.leaves())
                span = (idx, idx+length)
                label = tr.label()

                if length == 1:
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

        elif style == 'span-set':
            for span in gt.binary_parse_spans:
                length = span[1] - span[0]
                treepos = gt.parse.treeposition_spanning_leaves(span[0], span[1])
                subtree = gt.parse[treepos]

                if not isinstance(subtree, str):
                    label = subtree.label()
                    if len(subtree.leaves()) == length:
                        span2label[span] = label

        for span in gt.binary_parse_spans:
            if span not in span2label:
                continue
            label = span2label[span]
            if span in pred.binary_parse_spans:
                phrase_counter_correct[label] += 1
            phrase_counter_total[label] += 1

        stats['count'] += 1

    labels = sorted(phrase_counter_total.keys())

    for label in labels:
        ncorrect = phrase_counter_correct[label]
        ntotal = phrase_counter_total[label]
        print('{:<6} {:<8} {:<8} {:<8.3f}'.format(label, ncorrect, ntotal, float(ncorrect)/float(ntotal)))
    print(sum(phrase_counter_total.values()))
    print

    print('Count={} Skipped={}'.format(
        stats['count'], stats['skipped']))


if __name__ == '__main__':
    from lingmatic.engine.parsetree import ParseTreeReader
    from lingmatic.engine.parsetree import ParseTreeDeserializeGroundTruth
    from lingmatic.engine.parsetree import ParseTreeDeserializeInfer

    from tqdm import tqdm

    limit = 10000

    gt_path = os.path.expanduser('~/Downloads/ptb.jsonl')
    reader = ParseTreeReader(limit=limit, parse_tree_config=dict(deserializer_cls=ParseTreeDeserializeGroundTruth))
    results = list(tqdm(reader.read(gt_path)))
    corpus_gt = {x.example_id: x for x in results}

    print(results[0].parse)

    infer_path = os.path.expanduser('~/Downloads/PRPN_parses/PRPNLM_ALLNLI/parsed_WSJ_PRPNLM_AllLI_ESLM.jsonl')
    reader = ParseTreeReader(limit=limit, parse_tree_config=dict(deserializer_cls=ParseTreeDeserializeInfer))
    results = list(tqdm(reader.read(infer_path)))
    corpus_pred = {x.example_id: x for x in results}

    print(results[0].binary_parse_tree)

    # Match Entities
    match_entities(corpus_gt, corpus_pred)
