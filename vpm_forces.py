#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy
from scipy.linalg import solve

import vpm_aerodynamics
import vpm_airfoil

DBL_PI = 3.14159265358979323846264338327950288
DBL_D2R = DBL_PI / 180.0
DBL_R2D = 180.0 / DBL_PI


def cos(arg):
    return numpy.cos(arg)


def sin(arg):
    return numpy.sin(arg)


def calc_forces(nx, ny, partial_matrix, gamma_matrix, boundary_pts, alpha, v):
    sina = sin(alpha)  # radian
    cosa = cos(alpha)

    rho = 1.2

    panels = (nx - 1) * (ny - 1)
    vinf = numpy.zeros((panels, 3), dtype=float)

    for i in range(3):
        vinf[:, i] = partial_matrix[:, :, i].dot(gamma_matrix)

    vinf[:, 0] += cosa * v
    vinf[:, 2] += sina * v

    boundary_vector = -boundary_pts[:, 1:, :] + boundary_pts[:, :-1, :]
    cross_vector = numpy.cross(vinf, boundary_vector.reshape(-1, boundary_vector.shape[-1], order="F"))

    forces = numpy.zeros((panels, 3), dtype=float)

    for i in range(3):
        forces[:, i] = rho * gamma_matrix * cross_vector[:, i]

    return forces.reshape((nx - 1, ny - 1, 3), order="F")
