#!/usr/bin/env python

import unittest

import numpy as np

from escpy.elasticity import *


class TestElasticities(unittest.TestCase):
    def setUp(self):
        self.hexagonal = HexagonalSystemStiffnessMatrix(np.array([[297.85, 126.9, 104.5, 0., 0., 0.],
                                                                  [126.9, 297.85, 104.5, 0., 0., 0.],
                                                                  [104.5, 104.5, 286.9, 0., 0., 0.],
                                                                  [0., 0., 0., 59.225, 0., 0.],
                                                                  [0., 0., 0., 0., 59.225, 0.],
                                                                  [0., 0., 0., 0., 0., 85.475]]))

    def test_validate(self):
        self.assertTrue(self.hexagonal.validate())
