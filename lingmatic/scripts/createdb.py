import os
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
from lingmatic.models import User


defaults = dict(
    home=os.path.expanduser('~'),
)


default_users = [
    dict(name='user', password='user')
]


def usage(argv):
    cmd = os.path.basename(argv[0])
    print('usage: %s <config_uri> [var=value]\n'
          '(example: "%s development.ini")' % (cmd, cmd))
    sys.exit(1)


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

    with transaction.manager:
        dbsession = get_tm_session(session_factory, transaction.manager)

        added = 0

        for u in default_users:
            name = u['name']
            if dbsession.query(User).filter_by(name=name).count() == 0:
                new_user = User(name=name, role='editor')
                new_user.set_password(u['password'])
                dbsession.add(new_user)
                added += 1

        print("Added {} users. Skipped {}.".format(added, len(default_users) - added))


if __name__ == '__main__':
    main()
