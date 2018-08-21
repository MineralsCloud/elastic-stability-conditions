#!/usr/bin/env python

import abc

import numpy as np


class StiffnessMatrix:
    def __init__(self, stiffness_matrix):
        stiffness_matrix = np.array(stiffness_matrix, dtype=np.float64)

        if stiffness_matrix.shape != (6, 6):
            raise ValueError("Your *elastic_matrix* must have a shape of (6, 6)!")

        self._stiffness_matrix = stiffness_matrix

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


class CrystalSystemStiffnessMatrix(StiffnessMatrix):
    @property
    def symmetry_conditions_text(self):
        return [
            "C_{11} = C_{22} = C_{33}",
            "C_{44} = C_{55} = C_{66}",
            "C_{12} = C_{13} = C_{23} = C_{21} = C_{31} = C_{32}",
        ]

    @property
    def symmetry_conditions(self):
        _ = self._stiffness_matrix
        diag = np.diag(_)
        return [
            diag[0:3].min() == diag[0:3].max(),
            diag[3:6].min() == diag[3:6].max(),
            _[0, 1] == _[0, 2] == _[1, 2] == _[1, 0] == _[2, 0] == _[2, 1]
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
        _ = self._stiffness_matrix
        return [
            _[0, 0] == _[1, 1],
            _[3, 3] == _[4, 4],
            _[0, 2] == _[1, 2],
            _[5, 5] == 0.5 * (_[0, 0] - _[0, 1])
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
        _ = self._stiffness_matrix
        if self._stiffness_matrix[0, 5] == 0:  # Tetragonal (I) class
            return [
                _[0, 0] == _[1, 1],
                _[3, 3] == _[4, 4],
                _[0, 2] == _[1, 2],
            ]

        return [
            _[0, 0] == _[1, 1],
            _[3, 3] == _[4, 4],
            _[0, 2] == _[1, 2],
            _[0, 5] == -_[1, 5]
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
        _ = self._stiffness_matrix
        if self._stiffness_matrix[0, 4] == 0:  # Rhombohedral (I) class
            return [
                _[0, 0] == _[1, 1],
                _[3, 3] == _[4, 4],
                _[0, 2] == _[1, 2],
                _[5, 5] == 0.5 * (_[0, 0] - _[0, 1]),
                _[0, 3] == -_[1, 3] == -_[4, 5]
            ]

        return [  # Rhombohedral (II) class
            _[0, 0] == _[1, 1],
            _[3, 3] == _[4, 4],
            _[0, 2] == _[1, 2],
            _[5, 5] == 0.5 * (_[0, 0] - _[0, 1]),
            _[0, 3] == -_[1, 3] == -_[4, 5],
            -_[0, 4] == _[1, 4] == _[3, 5]
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
        _ = self._stiffness_matrix
        return [
            np.array_equal(_[0:3, 3], np.full(3, 0)),
            np.array_equal(_[0:4, 4], np.full(4, 0)),
            np.array_equal(_[0:5, 5], np.full(5, 0)),
            len(np.unique(_)) == 9
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
        _ = self._stiffness_matrix
        return [
            np.array_equal(_[0:3, 3], np.full(3, 0)),
            _[3, 4] == 0,
            np.array_equal(_[0:3, 5], np.full(3, 0)) and _[4, 5] == 0,
            len(np.unique(_)) == 13
        ]


class TriclinicSystemStiffnessMatrix(StiffnessMatrix):
    @property
    def symmetry_conditions_text(self):
        return [
            "There are 21 independent values."
        ]

    @property
    def symmetry_conditions(self):
        _ = self._stiffness_matrix
        return [
            len(np.unique(_)) == 21
        ]
