#!/usr/bin/env python3
"""
:mod:`` -- 
========================================

.. module 
   :platform: Unix, Windows, Mac, Linux
   :synopsis: 
.. moduleauthor:: Qi Zhang <qz2280@columbia.edu>
"""

import abc
from typing import List, Tuple

import numpy as np


class CrystalSystem:
    def __init__(self, elastic_matrix):
        elastic_matrix = np.array(elastic_matrix, dtype=np.float64)

        if not elastic_matrix.shape == (6, 6):
            raise ValueError("Your *elastic_matrix* must have a shape of (6, 6)!")

        self.elastic_matrix = elastic_matrix

    @property
    @abc.abstractmethod
    def ns_conditions(self) -> List[Tuple(bool, str)]:
        ...

    def check_ns_conditions(self, outfile=None) -> bool:
        flag = True

        for criterion, criterion_latex in self.ns_conditions:
            if not criterion:  # If `criterion` evaluates to `False`.
                print("Condition {0} is not satisfied!".format(criterion_latex), file=outfile)
                flag = False

        return flag


class CubicSystem(CrystalSystem):
    @property
    def ns_conditions(self):
        _ = self.elastic_matrix
        c11, c12, c44 = _[0, 0], _[0, 1], _[3, 3]

        return [
            (c11 > np.abs(c12), "$C_{11} > | C_{12} |$"),
            (c11 + 2 * c12 > 0, "$C_{11} + 2 C_{12} > 0$"),
            (c44 > 0, "$C_{44} > 0$")
        ]

    def check_symmetry(self):
        pass


class HexagonalSystem(CrystalSystem):
    @property
    def ns_conditions(self):
        _ = self.elastic_matrix
        c11, c12, c13, c33, c44 = _[0, 0], _[0, 1], _[0, 2], _[2, 2], _[3, 3]
        c66 = (c11 - c12) / 2

        return [
            (c11 > np.abs(c12), "$C_{11} > | C_{12} |$"),
            (2 * c13 ** 2 < c33 * (c11 + c12), "$2 C_{13}^2 < C_{33} (C_{11} + C_{12})$"),
            (c44 > 0, "$C_{44} > 0$"),
            (c66 > 0, "$C_{66} > 0$")

        ]


class TetragonalSystem(CrystalSystem):
    @property
    def ns_conditions(self):
        _ = self.elastic_matrix
        c11, c12, c13, c16, c33, c44, c66 = _[0, 0], _[0, 1], _[0, 2], _[0, 5], _[2, 2], _[3, 3], _[5, 5]

        if c16 == 0:  # Tetragonal (I) class
            return [
                (c11 > np.abs(c12), "$C_{11} > | C_{12} |$"),
                (2 * c13 ** 2 < c33 * (c11 + c12), "$2 C_{13}^2 < C_{33} (C_{11} + C_{12})$"),
                (c44 > 0, "$C_{44} > 0$"),
                (c66 > 0, "$C_{66} > 0$")
            ]

        return [  # Tetragonal (II) class
            (c11 > np.abs(c12), "$C_{11} > | C_{12} |$"),
            (2 * c13 ** 2 < c33 * (c11 + c12), "$2 C_{13}^2 < C_{33} (C_{11} + C_{12})$"),
            (c44 > 0, "$C_{44} > 0$"),
            (2 * c16 ** 2 < c66 * (c11 - c12), "$2 C_{16}^2 < C_{66} (C_{11} - C_{12})$")
        ]


class RhombohedralSystem(CrystalSystem):
    @property
    def ns_conditions(self):
        _ = self.elastic_matrix
        c11, c12, c13, c14, c15, c33, c44, c66 = _[0, 0], _[0, 1], _[0, 2], _[0, 3], _[0, 4], _[2, 2], _[3, 3], _[5, 5]

        if c15 == 0:  # Rhombohedral (I) class
            return [
                (c11 > np.abs(c12), "$C_{11} > | C_{12} |$"),
                (c44 > 0, "$C_{44} > 0$"),
                (c13 ** 2 < 0.5 * c33 * (c11 + c12), "$C_{13}^2 < 1/2 C_{33} (C_{11} + C_{12})$"),
                (c14 ** 2 < 0.5 * c44 * (c11 - c12), "$C_{14}^2 < 1/2 C_{44} (C_{11} - C_{12})$"),
                (0.5 * c44 * (c11 - c12) == c44 * c66, "$C_{14}^2 < 1/2 C_{44} (C_{11} - C_{12}) = C_{44} * C_{66}$")
            ]

        return [  # Rhombohedral (II) class
            (c11 > np.abs(c12), "$C_{11} > | C_{12} |$"),
            (c44 > 0, "$C_{44} > 0$"),
            (c13 ** 2 < 0.5 * c33 * (c11 + c12), "$C_{13}^2 < 1/2 C_{33} (C_{11} + C_{12})$"),
            (c14 ** 2 + c15 ** 2 < 0.5 * c44 * (c11 - c12), "$C_{14}^2 + C_{15}^2 < 1/2 C_{44} (C_{11} - C_{12})$"),
            (0.5 * c44 * (c11 - c12) == c44 * c66, "$C_{14}^2 < 1/2 C_{44} (C_{11} - C_{12}) = C_{44} * C_{66}$")
        ]


class OrthorhombicSystem(CrystalSystem):
    @property
    def ns_conditions(self):
        _ = self.elastic_matrix
        c11, c12, c13, c22, c23, c33, c44, c55, c66 = _[0, 0], _[0, 1], _[0, 2], _[1, 1], _[1, 2], _[2, 2], _[3, 3], _[
            4, 4], _[5, 5]

        return [
            (c11 > 0, "$C_{11} > 0$"),
            (c11 * c22 > c12 ** 2, "$C_{11} C_{22} > C_{12}^2$"),
            (c11 * c22 * c33 + 2 * c12 * c13 * c23 > c11 * c23 ** 2 + c22 * c13 ** 2 + c33 * c12 ** 2,
             "$C_{11} C_{22} C_{33} + 2 C_{12} C_{13} C_{23} > C_{11} C_{23}^2 + C_{22} C_{13}^2 + C_{33} C_{12}^2$"),
            (c44 > 0, "$C_{44} > 0$"),
            (c55 > 0, "$C_{55} > 0$"),
            (c66 > 0, "$C_{66} > 0$")
        ]


class MonoclinicSystem(CrystalSystem):
    @property
    def ns_conditions(self):
        return []


class TriclinicSystem(CrystalSystem):
    @property
    def ns_conditions(self):
        return []
