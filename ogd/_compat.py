# -*- coding: utf-8 -*-

import sys

PY2 = sys.version_info[0] == 2


if PY2:
    import cPickle as pickle
    from itertools import izip
else:
    import pickle
    izip = zip
