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

    def test_compliance_matrix(self):
        np.testing.assert_array_almost_equal(self.hexagonal.compliance_matrix,
                                             np.array([[0.00435904, -0.00149062, -0.00104479, 0., 0., 0.],
                                                       [-0.00149062, 0.00435904, -0.00104479, 0., 0., 0.],
                                                       [-0.00104479, -0.00104479, 0.00424664, 0., 0., 0.],
                                                       [0., 0., 0., 0.01688476, 0., 0.],
                                                       [0., 0., 0., 0., 0.01688476, 0.],
                                                       [0., 0., 0., 0., 0., 0.01169933]]))

    def test_validate(self):
        self.assertTrue(self.hexagonal.validate())


if __name__ == '__main__':
    unittest.main()
