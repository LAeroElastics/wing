#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Any, Union

import numpy
import itertools

DBL_PI = 3.14159265358979323846264338327950288
DBL_D2R = DBL_PI / 180.0
DBL_R2D = 180.0 / DBL_PI


def cos(arg: object) -> object:
    return numpy.cos(arg)


def sin(arg: object) -> object:
    return numpy.sin(arg)


def norm2(vector):
    return numpy.sum(vector ** 2)


def norm(vector):
    return numpy.sqrt(norm2(vector))


class Aerodynamics(object):
    def __init__(self, airfoil, camber):
        self.airfoil = airfoil
        self.camber = camber

        self.num_chord_max_iter = int(numpy.shape(self.airfoil)[0] / 2.0)
        self.num_span_max_iter = numpy.shape(self.airfoil)[1]
        self.num_wing_panels = self.num_chord_max_iter * self.num_span_max_iter
        self.rhs_vector = numpy.zeros((self.num_wing_panels, self.num_wing_panels))
        self.aic_matrix = numpy.zeros((self.num_wing_panels, self.num_wing_panels, 3))

    def gen_collocation_pts(self):
        boundary_pts = self.camber[:-1, :, :] * 0.75 + self.camber[1:, :, :] * 0.25
        collocation_pts = 0.5 * (self.camber[1:, :-1, :] * 0.75
                                 + self.camber[:-1, :-1, :] * 0.25
                                 + self.camber[:-1, 1:, :] * 0.25
                                 + self.camber[1:, 1:, :] * 0.75)
        return boundary_pts, collocation_pts

    def gen_normal_vector(self):
        return numpy.cross(self.camber[:-1, 1:, :] - self.camber[1:, :-1, :],
                           self.camber[:-1, :-1, :] - self.camber[1:, 1:, :], axis=2)

    def gen_normal_vector_unitized(self):
        result = self.gen_normal_vector()
        temp_norm = numpy.sqrt(numpy.sum(result ** 2.0, axis=2))

        for i_iter in range(3):
            result[:, :, i_iter] /= temp_norm

        return result

    def gen_ref_area(self):
        return 0.5 * numpy.sum(self.gen_normal_vector())

    @staticmethod
    def calc_local_vorticity(col_pts, src_pt1, src_pt2):
        temp1 = col_pts - src_pt1
        temp2 = col_pts - src_pt2
        n1 = norm(temp1)
        n2 = norm(temp2)

        return (n1 + n2) * numpy.cross(temp1, temp2) / (n1 * n2 * (n1 * n2 + temp1.dot(temp2)))

    def gen_rhs_vector(self, alpha, vinf):
        sina = sin(alpha)  # radian
        cosa = cos(alpha)
        vinf = numpy.array([cosa, 0.0, sina]) * vinf

        copied_vector = self.gen_normal_vector_unitized().copy()
        reshaped_vector = copied_vector.reshape(-1, 3, order="F")

        return -reshaped_vector.reshape(-1, reshaped_vector.shape[-1], order="F").dot(vinf)

    def gen_aic_matrix(self, alpha, num_chord_max_iter, num_span_max_iter):
        sina = sin(alpha)  # radian
        cosa = cos(alpha)
        vinf = numpy.array([cosa, 0.0, sina])
        boundary_pts, collocation_pts = self.gen_collocation_pts()
        normal_vector = self.gen_normal_vector_unitized()
        num_panels = (self.num_span_max_iter - 1) * (self.num_chord_max_iter - 1)

        partial_matrix = numpy.zeros((num_panels, num_panels, 3))
        aic_matrix = numpy.zeros((num_panels, num_panels))

        i_matrix_x_iter = 0
        for i_element_y_iter, i_element_x_iter in itertools.product(range(num_span_max_iter - 1),
                                                                    range(num_chord_max_iter - 1)):
            i_matrix_y_iter = 0
            current_col_pt = collocation_pts[i_element_x_iter, i_element_y_iter, :]

            for i_colpts_y_iter in range(num_span_max_iter - 1):
                temp1 = current_col_pt - boundary_pts[-1, i_colpts_y_iter + 0, :]
                temp2 = current_col_pt - boundary_pts[-1, i_colpts_y_iter + 1, :]
                n1 = norm(temp1)
                n2 = norm(temp2)

                trailing_edge_vortex1 = numpy.cross(vinf, temp2) / (n2 * (n2 - vinf.dot(temp2)))
                trailing_edge_vortex3 = numpy.cross(vinf, temp1) / (n1 * (n1 - vinf.dot(temp1)))

                trailing_edge_vortex = trailing_edge_vortex3 - trailing_edge_vortex1
                sum_vortex = 0.0
                for i_colpts_x_iter in range(num_chord_max_iter - 1):
                    boundary_pts1 = boundary_pts[i_colpts_x_iter, i_colpts_y_iter + 0, :]
                    boundary_pts2 = boundary_pts[i_colpts_x_iter, i_colpts_y_iter + 1, :]

                    if i_colpts_x_iter != num_chord_max_iter - 2:
                        boundary_pts3 = boundary_pts[i_colpts_x_iter + 1, i_colpts_y_iter + 1, :]
                        boundary_pts4 = boundary_pts[i_colpts_x_iter + 1, i_colpts_y_iter + 0, :]
                    else:
                        boundary_pts3 = boundary_pts[-1, i_colpts_y_iter + 1, :]
                        boundary_pts4 = boundary_pts[-1, i_colpts_y_iter + 0, :]

                    sum_vortex += self.calc_local_vorticity(current_col_pt, boundary_pts2, boundary_pts3)
                    sum_vortex += self.calc_local_vorticity(current_col_pt, boundary_pts4, boundary_pts1)
                    sum_boundary_vortex = self.calc_local_vorticity(current_col_pt, boundary_pts1, boundary_pts2)

                    partial_matrix[i_matrix_x_iter, i_matrix_y_iter, :] \
                        = trailing_edge_vortex + sum_vortex + sum_boundary_vortex
                    aic_matrix[i_matrix_x_iter, i_matrix_y_iter] \
                        = partial_matrix[i_matrix_x_iter, i_matrix_y_iter, 0] * normal_vector[
                        i_element_x_iter, i_element_y_iter, 0] \
                          + partial_matrix[i_matrix_x_iter, i_matrix_y_iter, 1] * normal_vector[
                              i_element_x_iter, i_element_y_iter, 1] \
                          + partial_matrix[i_matrix_x_iter, i_matrix_y_iter, 2] * normal_vector[
                              i_element_x_iter, i_element_y_iter, 2]

                    aic_matrix[i_matrix_x_iter, i_matrix_y_iter] /= 4.0 * DBL_PI
                    i_matrix_y_iter += 1
            i_matrix_x_iter += 1

        return partial_matrix, aic_matrix, boundary_pts, collocation_pts
