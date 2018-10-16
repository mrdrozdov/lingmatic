import os
import json

from nltk.tree import Tree


def to_tokens(parse):
    return [x for x in parse.split() if x != '(' and x != ')']


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


def convert_binary_bracketing(parse):
    transitions = []
    tokens = []

    for word in parse.split():
        if word[0] != "(":
            if word == ")":
                transitions.append(1)
            else:
                tokens.append(word)
                transitions.append(0)

    return tokens, transitions


class ParseTreeDeserializeBase(object):
    key_id = None
    key_parse = None
    key_binary_parse = None

    def __init__(self, obj):
        if isinstance(obj, str):
            obj = json.loads(obj)
        self.obj = obj

    def get_id(self):
        r"""

        Returns:
            output (str): The example id.

        """
        return self.obj[self.key_id]

    def get_parse(self):
        r"""

        Returns:
            output (nltk.Tree): An nltk.Tree representing the constituency parse.

        """
        if self.key_parse is None:
            return None
        return Tree.fromstring(self.obj[self.key_parse])

    def get_binary_parse_spans(self):
        r"""

        Returns:
            output (set): A set of spans representing the binary tree.

        """
        return to_indexed_contituents(self.obj[self.key_binary_parse])

    def get_binary_parse_tree(self):
        r"""

        Returns:
            output (list): A nested list representation of the binary tree.

        """
        tokens, transitions = convert_binary_bracketing(self.obj[self.key_binary_parse])
        tree = build_tree(tokens, transitions)
        return tree

    def get_raw_binary_parse(self):
        r"""

        Returns:
            output (str): The string for the binary parse.

        """
        return self.obj[self.key_binary_parse]

    def get_tokens(self):
        r"""

        Returns:
            output (list): The list of tokens in the sentence.

        """
        return to_tokens(self.obj[self.key_binary_parse])


class ParseTreeDeserializeGroundTruth(ParseTreeDeserializeBase):
    key_id = 'pairID'
    key_parse = 'sentence1_parse'
    key_binary_parse = 'sentence1_binary_parse'


class ParseTreeDeserializeInfer(ParseTreeDeserializeBase):
    key_id = 'example_id'
    key_binary_parse = 'sent1_tree'
        


class ParseTree(object):
    def __init__(self):
        pass

    @staticmethod
    def from_parse(obj, deserializer_cls_lst=[ParseTreeDeserializeBase]):
        if not isinstance(deserializer_cls_lst, (list, tuple)):
            deserializer_cls_lst = [deserializer_cls_lst]

        pts = []

        for deserializer_cls in deserializer_cls_lst:
            pt = ParseTree()

            obj_reader = deserializer_cls(obj)
            pt.example_id = obj_reader.get_id()
            pt.parse = obj_reader.get_parse()
            pt.binary_parse_spans = obj_reader.get_binary_parse_spans()
            pt.binary_parse_tree = obj_reader.get_binary_parse_tree()
            pt.raw_binary_parse = obj_reader.get_raw_binary_parse()
            pt.tokens = obj_reader.get_tokens()

            pts.append(pt)

        return pts


class ParseTreeReader(object):
    def __init__(self, limit=None, parse_tree_config={}):
        self.limit = limit
        self.parse_tree_config = parse_tree_config

    def read(self, filename):
        with open(filename) as f:
            for i, line in enumerate(f):
                pts = ParseTree.from_parse(line, **self.parse_tree_config)
                if not isinstance(pts, (list, tuple)):
                    pts = [pts]
                for pt in pts:
                    yield pt

                if self.limit is not None and i+1 >= self.limit:
                    break


if __name__ == '__main__':
    from tqdm import tqdm

    gt_path = os.path.expanduser('~/Downloads/ptb.jsonl')
    reader = ParseTreeReader(limit=10, parse_tree_config=dict(deserializer_cls_lst=ParseTreeDeserializeGroundTruth))
    results = list(tqdm(reader.read(gt_path)))

    print(results[0].parse)

    infer_path = os.path.expanduser('~/Downloads/PRPN_parses/PRPNLM_ALLNLI/parsed_WSJ_PRPNLM_AllLI_ESLM.jsonl')
    reader = ParseTreeReader(limit=10, parse_tree_config=dict(deserializer_cls_lst=ParseTreeDeserializeInfer))
    results = list(tqdm(reader.read(infer_path)))

    print(results[0].binary_parse_tree)
