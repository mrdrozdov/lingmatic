from pyramid.httpexceptions import HTTPNotFound
from pyramid.httpexceptions import HTTPFound
from pyramid.security import Allow
from pyramid.security import Everyone

from lingmatic.models import ParseTree


def includeme(config):
    config.add_static_view('static', 'static', cache_max_age=3600)
    config.add_route('home', '/')
    config.add_route('view_tree_sources', '/trees')
    config.add_route('view_trees', '/trees/{source}')
    config.add_route('view_tree', '/trees/{source}/{rawid}')
