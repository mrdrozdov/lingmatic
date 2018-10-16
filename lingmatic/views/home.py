from pyramid.compat import escape
import re
from docutils.core import publish_parts

from pyramid.httpexceptions import HTTPFound
from pyramid.httpexceptions import HTTPNotFound

from pyramid.view import view_config


@view_config(route_name='home', renderer='../templates/home.jinja2')
def home(request):
    return {}
