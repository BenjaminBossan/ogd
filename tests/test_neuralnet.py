# -*- coding: utf-8 -*-

from __future__ import division

import pytest

from FTRLprox.neuralnet import Neuralnet
from tutils import X
from tutils import y


def test_init_args():
    clf = Neuralnet(num_units=12)
    clf.lambda1 == 0.
    clf.alr_schedule == 'gradient'
    clf.num_units == 12

    clf = Neuralnet(3, num_units=13)
    clf.lambda1 == 3
    clf.alr_schedule == 'gradient'
    clf.num_units == 13

    clf = Neuralnet(num_units=1, alr_schedule='count')
    clf.lambda1 == 0.
    clf.alr_schedule == 'count'
    clf.num_units == 1

    clf = Neuralnet(alr_schedule='count', num_units=12)
    clf.lambda1 == 0.
    clf.alr_schedule == 'count'
    clf.num_units == 12

    clf = Neuralnet(0.5, alr_schedule='count', num_units=12)
    clf.lambda1 == 0.5
    clf.alr_schedule == 'count'
    clf.num_units == 12


def test_init_exceptions():
    with pytest.raises(TypeError):
        Neuralnet()
    with pytest.raises(TypeError):
        Neuralnet(0, 0.3, 0.1, 1., alr_schedule='constant')
    with pytest.raises(NotImplementedError):
        Neuralnet(num_units=10, ndims=2 ** 20)


def test_fit():
    clf = Neuralnet(num_units=12)
    pass
