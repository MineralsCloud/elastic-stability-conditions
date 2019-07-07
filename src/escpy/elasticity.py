#!/usr/bin/env python
"""
.. module:: elasticity
   :platform: Unix, Windows, Mac, Linux
   :synopsis: This module can be used to check the validity of input elasticity for 11 Laue classes.
.. moduleauthor:: Qi Zhang <qz2280@columbia.edu>
"""

import abc
import warnings

import numpy as np

__all__ = [
    'StiffnessMatrix',
    'CubicSystemStiffnessMatrix',
    'HexagonalSystemStiffnessMatrix',
    'TetragonalSystemStiffnessMatrix',
    'RhombohedralSystemStiffnessMatrix',
    'OrthorhombicSystemStiffnessMatrix',
    'MonoclinicSystemStiffnessMatrix',
    'TriclinicSystemStiffnessMatrix'
]


class StiffnessMatrix:
    def __init__(self, stiffness_matrix):
        stiffness_matrix = np.array(stiffness_matrix, dtype=np.float64)

        if stiffness_matrix.shape != (6, 6):
            raise ValueError("Your *elastic_matrix* must have a shape of (6, 6)!")

        self._stiffness_matrix = stiffness_matrix

        self._eps = 1e-8

    @property
    def eps(self):
        return self._eps

    @eps.setter
    def eps(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError("The `eps` attribute has to be a number!")
        if abs(value) > 1:
            warnings.warn("The value of `eps` attribute may be too large!", stacklevel=2)
        self._eps = value

    @property
    @abc.abstractmethod
    def symmetry_conditions_text(self):
        ...

    @property
    @abc.abstractmethod
    def symmetry_conditions(self):
        ...

    def validate(self, outfile=None):
        flag = True

        for criterion, criterion_text in zip(self.symmetry_conditions, self.symmetry_conditions_text):
            if not criterion:  # If `criterion` evaluates to `False`.
                print("Criterion ${0}$ is not satisfied!".format(criterion_text), file=outfile)
                flag = False

        return flag

    @property
    def stiffness_matrix(self):
        return self._stiffness_matrix

    @property
    def compliance_matrix(self):
        return np.linalg.inv(self._stiffness_matrix)


class CubicSystemStiffnessMatrix(StiffnessMatrix):
    @property
    def symmetry_conditions_text(self):
        return [
            "C_{11} = C_{22} = C_{33}",
            "C_{44} = C_{55} = C_{66}",
            "C_{12} = C_{13} = C_{23} = C_{21} = C_{31} = C_{32}",
        ]

    @property
    def symmetry_conditions(self):
        c = self._stiffness_matrix
        eps = self.eps
        d = np.diag(c)
        arr = np.array(c[0, 1], c[0, 2], c[1, 2], c[1, 0], c[2, 0], c[2, 1])
        return [
            np.isclose(d[0:3].min(), d[0:3].max(), atol=eps),
            np.isclose(d[3:6].min(), d[3:6].max(), atol=eps),
            np.isclose(arr.min(), arr.max(), atol=eps)
        ]


class HexagonalSystemStiffnessMatrix(StiffnessMatrix):
    @property
    def symmetry_conditions_text(self):
        return [
            "C_{11} = C_{22}",
            "C_{44} = C_{55}",
            "C_{13} = C_{23}",
            "C_{66} = 1 / 2 (C_{11} - C_{12})"
        ]

    @property
    def symmetry_conditions(self):
        c = self._stiffness_matrix
        eps = self.eps
        return [
            np.isclose(c[0, 0], c[1, 1], atol=eps),
            np.isclose(c[3, 3], c[4, 4], atol=eps),
            np.isclose(c[0, 2], c[1, 2], atol=eps),
            np.isclose(c[5, 5], 0.5 * (c[0, 0] - c[0, 1]), atol=eps)
        ]


class TetragonalSystemStiffnessMatrix(StiffnessMatrix):
    @property
    def symmetry_conditions_text(self):
        if self._stiffness_matrix[0, 5] == 0:  # Tetragonal (I) class
            return [
                "C_{11} = C_{22}",
                "C_{44} = C_{55}",
                "C_{13} = C_{23}",
            ]

        return [  # Tetragonal (II) class
            "C_{11} = C_{22}",
            "C_{44} = C_{55}",
            "C_{13} = C_{23}",
            "C_{16} = -C_{26}"
        ]

    @property
    def symmetry_conditions(self):
        c = self._stiffness_matrix
        eps = self.eps
        if self._stiffness_matrix[0, 5] == 0:  # Tetragonal (I) class
            return [
                np.isclose(c[0, 0], c[1, 1], atol=eps),
                np.isclose(c[3, 3], c[4, 4], atol=eps),
                np.isclose(c[0, 2], c[1, 2], atol=eps),
            ]

        return [
            np.isclose(c[0, 0], c[1, 1], atol=eps),
            np.isclose(c[3, 3], c[4, 4], atol=eps),
            np.isclose(c[0, 2], c[1, 2], atol=eps),
            np.isclose(c[0, 5], -c[1, 5], atol=eps)
        ]


class RhombohedralSystemStiffnessMatrix(StiffnessMatrix):
    @property
    def symmetry_conditions_text(self):
        if self._stiffness_matrix[0, 4] == 0:  # Rhombohedral (I) class
            return [
                "C_{11} == C_{22}",
                "C_{44} = C_{55}",
                "C_{13} = C_{23}",
                "C_{66} = 1 / 2 (C_{11} - C_{12})",
                "C_{14} = -C_{24} = -C_{56}"
            ]

        return [  # Rhombohedral (II) class
            "C_{11} == C_{22}",
            "C_{44} = C_{55}",
            "C_{13} = C_{23}",
            "C_{66} = 1 / 2 (C_{11} - C_{12})",
            "C_{14} = -C_{24} = -C_{56}",
            "-C_{15} = C_{25} = C_{46}"
        ]

    @property
    def symmetry_conditions(self):
        c = self._stiffness_matrix
        eps = self.eps
        if self._stiffness_matrix[0, 4] == 0:  # Rhombohedral (I) class
            return [
                np.isclose(c[0, 0], c[1, 1], atol=eps),
                np.isclose(c[3, 3], c[4, 4], atol=eps),
                np.isclose(c[0, 2], c[1, 2], atol=eps),
                np.isclose(c[5, 5], 0.5 * (c[0, 0] - c[0, 1]), atol=eps),
                np.isclose(c[0, 3], -c[1, 3], atol=eps) and np.isclose(c[0, 3], -c[4, 5], atol=eps)
            ]

        return [  # Rhombohedral (II) class
            np.isclose(c[0, 0], c[1, 1], atol=eps),
            np.isclose(c[3, 3], c[4, 4], atol=eps),
            np.isclose(c[0, 2], c[1, 2], atol=eps),
            np.isclose(c[5, 5], 0.5 * (c[0, 0] - c[0, 1]), atol=eps),
            np.isclose(c[0, 3], -c[1, 3], atol=eps) and np.isclose(c[0, 3], -c[4, 5], atol=eps),
            np.isclose(c[1, 4], -c[0, 4], atol=eps) and np.isclose(c[3, 5], -c[0, 4], atol=eps)
        ]


class OrthorhombicSystemStiffnessMatrix(StiffnessMatrix):
    @property
    def symmetry_conditions_text(self):
        return [
            "C_{14} = C_{24} = C_{34} = 0",
            "C_{15} = C_{25} = C_{35} = C_{45} = 0",
            "C_{16} = C_{26} = C_{36} = C_{46} = C_{56} = 0"
            "There are 9 independent values."
        ]

    @property
    def symmetry_conditions(self):
        c = self._stiffness_matrix
        eps = self.eps
        return [
            np.allclose(c[0:3, 3], np.full(3, 0), atol=eps),
            np.allclose(c[0:4, 4], np.full(4, 0), atol=eps),
            np.allclose(c[0:5, 5], np.full(5, 0), atol=eps),
            len(np.unique(c)) == 9
        ]


class MonoclinicSystemStiffnessMatrix(StiffnessMatrix):
    @property
    def symmetry_conditions_text(self):
        return [
            "C_{14} = C_{24} = C_{34} = 0",
            "C_{45} = 0",
            "C_{16} = C_{26} = C_{36} = C_{56} = 0",
            "There are 13 independent values."
        ]

    @property
    def symmetry_conditions(self):
        c = self._stiffness_matrix
        eps = self.eps
        return [
            np.allclose(c[0:3, 3], np.full(3, 0), atol=eps),
            abs(c[3, 4]) < eps,
            np.allclose(c[0:3, 5], np.full(3, 0), atol=eps) and abs(c[4, 5]) < eps,
            len(np.unique(c)) == 13
        ]


class TriclinicSystemStiffnessMatrix(StiffnessMatrix):
    @property
    def symmetry_conditions_text(self):
        return [
            "There are 21 independent values."
        ]

    @property
    def symmetry_conditions(self):
        c = self._stiffness_matrix
        return [
            len(np.unique(c)) == 21
        ]
