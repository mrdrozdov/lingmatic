import json
from pyramid.compat import escape
import re
from docutils.core import publish_parts

from pyramid.httpexceptions import HTTPFound
from pyramid.httpexceptions import HTTPNotFound

from pyramid.view import view_config

from lingmatic.models import ParseTree


class Tree(object):
    def __init__(self, source, rawid):
        self.source = source
        self.rawid = rawid


@view_config(route_name='view_tree_sources', renderer='../templates/view_tree_sources.jinja2')
def view_tree_sources(request):
    sources = request.dbsession.query(ParseTree.source).distinct().all()
    sources = [x[0] for x in sources]
    return dict(sources=sources)


@view_config(route_name='view_trees', renderer='../templates/view_trees.jinja2')
def view_trees(request):
    source = str(request.matchdict['source'])
    query = request.dbsession.query(ParseTree)
    query = query.filter_by(source=source)
    trees = query.all()

    return dict(trees=trees)


@view_config(route_name='view_tree', renderer='../templates/view_tree.jinja2')
def view_tree(request):
    source = str(request.matchdict['source'])
    rawid = str(request.matchdict['rawid'])
    query = request.dbsession.query(ParseTree)
    query = query.filter_by(source=source, rawid=rawid)
    tree = query.first()

    return dict(tree=tree)
