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
