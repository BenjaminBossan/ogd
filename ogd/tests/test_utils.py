# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
import pytest

from ogd.utils import add_interactions
from ogd.utils import sigmoid
from ogd.utils import GetList


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


@pytest.mark.parametrize('lst, depth, expected', [
    (['a', 'bb', 'c', 'd'], 0, ['a', 'bb', 'c', 'd']),
    (['a', 'bb', 'c', 'd'], 1, ['a', 'bb', 'c', 'd']),
    (['a', 'bb', 'c', 'd'], 2, ['a', 'bb', 'c', 'd', 'a bb', 'a c',
                                'a d', 'bb c', 'bb d', 'c d']),
    (['a', 'bb', 'c', 'd'], 3, ['a', 'bb', 'c', 'd', 'a bb', 'a c',
                                'a d', 'bb c', 'bb d', 'c d', 'a bb c',
                                'a bb d', 'a c d', 'bb c d']),
    (['a', 'bb', 'c', 'd'], 4, ['a', 'bb', 'c', 'd', 'a bb', 'a c',
                                'a d', 'bb c', 'bb d', 'c d', 'a bb c',
                                'a bb d', 'a c d', 'bb c d', 'a bb c d']),
])
def test_add_interactions(lst, depth, expected):
    add_interactions(lst, depth=depth)
    assert lst == expected
