#!/usr/bin/env python
# -*- coding: utf-8 -*-
from scipy.linalg import solve

import time
import cmn_plot
import vpm_aerodynamics
import vpm_airfoil
import vpm_forces

DBL_PI = 3.14159265358979323846264338327950288
DBL_D2R = DBL_PI / 180.0
DBL_R2D = 180.0 / DBL_PI


def problem():
    num_x = 10
    num_y = 20
    span = 10.0
    chord = 1.0
    max_camber = 0.02
    location_max_camber = 0.4
    thickness = 0.12
    mach = 0.15
    alpha = 3.0

    sweep_angle = 8.0
    dihedral_angle = -5.0
    taper_ratio = 0.5

    v0 = 340.294
    vinf = v0 * mach
    wingsection = vpm_airfoil.WingSection(num_x, num_y, span, chord, max_camber, location_max_camber, thickness)
    wingsection.assemble()
    wingsection.set_sweep_angle(sweep_angle)
    wingsection.set_dihedral_angle(dihedral_angle)
    wingsection.set_taper_ratio(taper_ratio)

    mesh_airfoil = wingsection.airfoil
    mesh_camber = wingsection.camber
    aerodynamics = vpm_aerodynamics.Aerodynamics(mesh_airfoil, mesh_camber)
    matrix_part, matrix_aic, boundary_pts, collocation_pts = aerodynamics.gen_aic_matrix(alpha, num_x, num_y)
    matrix_rhs = aerodynamics.gen_rhs_vector(alpha, vinf)
    matrix_gamma = solve(matrix_aic, matrix_rhs)
    matrix_forces = vpm_forces.calc_forces(num_x, num_y, matrix_part, matrix_gamma, boundary_pts, alpha, vinf)

    return matrix_forces, mesh_airfoil, boundary_pts, collocation_pts


def main():
    start = time.time()
    force, airfoil, boundary_pts, collocation_pts = problem()
    end = time.time() - start
    cmn_plot.show(airfoil, boundary_pts, collocation_pts, force)
    #cmn_plot.show2d(airfoil, collocation_pts)
    cmn_plot.show_contour(airfoil, force)
    print(end)

if __name__ == "__main__":
    main()
