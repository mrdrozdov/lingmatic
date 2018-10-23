from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy.orm import relationship

from .meta import Base


class ParseTree(Base):
    __tablename__ = 'parsetrees'

    id = Column(Integer(), primary_key=True)
    rawid = Column(String(256), nullable=False)
    source = Column(String(256), nullable=False)

    data = Column(Text(64000), nullable=False)

    binary_parse = Column(Text(64000))
    binary_parse_pretty = Column(Text(64000))
