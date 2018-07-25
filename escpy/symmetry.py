#!/usr/bin/env python3
"""
:mod:`` -- 
========================================

.. module 
   :platform: Unix, Windows, Mac, Linux
   :synopsis: 
.. moduleauthor:: Qi Zhang <qz2280@columbia.edu>
"""

import numpy as np

from .crystal import *


def find_crystal_system(elastic_matrix):
    elastic_matrix = np.array(elastic_matrix, dtype=np.float64)

    if not elastic_matrix.shape == (6, 6):
        raise ValueError("Your *elastic_matrix* must have a shape of (6, 6)!")

    return CubicSystem
