import os
import json
import sys
import transaction
from datetime import datetime

from pyramid.paster import get_appsettings
from pyramid.paster import setup_logging

from pyramid.scripts.common import parse_vars

from lingmatic.models.meta import Base
from lingmatic.models import get_engine
from lingmatic.models import get_session_factory
from lingmatic.models import get_tm_session
from lingmatic.models import ParseTree

from lingmatic.engine.parsetree import to_indexed_contituents, convert_binary_bracketing, build_tree

from nltk.tree import Tree
from nltk.treeprettyprinter import TreePrettyPrinter


defaults = dict(
    home=os.path.expanduser('~'),
)


def usage(argv):
    cmd = os.path.basename(argv[0])
    print('usage: %s <config_uri> [var=value]\n'
          '(example: "%s development.ini")' % (cmd, cmd))
    sys.exit(1)


example_data = [
    '{ "rawid": 1, "data": "( A ( B C ) )" }'
]


def tree_to_string(s, symbol='|'):
    if not isinstance(s, (list, tuple)):
        s = s.replace(')', 'RP').replace('(', 'LP')
        return '({} {})'.format(symbol, s)

    return '({} {} {})'.format(
        symbol,
        tree_to_string(s[0], symbol),
        tree_to_string(s[1], symbol),
        )


def print_tree(s):
    if isinstance(s, (list, tuple)):
        tree_string = tree_to_string(s)
    tree = Tree.fromstring(tree_string)
    out = TreePrettyPrinter(tree).text()
    return out


class ParseTreeImporter(object):
    @staticmethod
    def read(raw, source):
        raise NotImplementedError


class SimpleParseTreeImporter(ParseTreeImporter):
    @staticmethod
    def read(raw, source):
        o = json.loads(raw)
        rawid = str(o['rawid']).strip()
        data = o['data'].strip()
        tokens, transitions = convert_binary_bracketing(data)
        binary_parse = build_tree(tokens, transitions)
        binary_parse_pretty = print_tree(binary_parse)
        return ParseTree(
            rawid=rawid,
            data=data,
            source=source,
            binary_parse=json.dumps(binary_parse),
            binary_parse_pretty=binary_parse_pretty,
            )


def main(argv=sys.argv):
    if len(argv) < 2:
        usage(argv)
    config_uri = argv[1]
    options = dict()
    options.update(defaults)
    options.update(parse_vars(argv[2:]))
    setup_logging(config_uri)
    settings = get_appsettings(config_uri, options=options)

    engine = get_engine(settings)
    Base.metadata.create_all(engine)

    session_factory = get_session_factory(engine)

    #
    source = options['source']

    with transaction.manager:
        dbsession = get_tm_session(session_factory, transaction.manager)

        added = 0
        count = 0

        for raw in example_data:
            pt = SimpleParseTreeImporter.read(raw, source)
            if dbsession.query(ParseTree).filter_by(rawid=pt.rawid, source=pt.source).count() == 0:
                dbsession.add(pt)
                added += 1
            count += 1

        print("Added {} trees. Skipped {}.".format(added, count - added))


if __name__ == '__main__':
    main()
