# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
import pytest

from FTRLprox.utils import sigmoid
from FTRLprox.utils import GetList


@pytest.mark.parametrize('x, expected', [
    (0, 0.5),
    (-50, 0.),
    (50, 1.)
])
def test_sigmoid(x, expected):
    result = sigmoid(x)
    assert np.allclose(result, expected)


@pytest.mark.parametrize('key, default, expected', [
    (0, 0, 0),
    (0, 123, 0),
    (123, 0, 123),
    (123, 456, 123),
])
def test_getlist(key, default, expected):
    lst = GetList(range(1000))
    result = lst.get(key, default)
    assert result == expected
