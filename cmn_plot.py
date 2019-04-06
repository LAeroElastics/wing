#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import numpy
from numpy import vstack, hstack

def show2d(bodies, c_pts):
    num_y = 20
    pyplot.plot(bodies[:, 0, 0], bodies[:, 0, 2], ".", color="red", ms=5, mew=5)
    pyplot.plot(c_pts[:, 0, 0], c_pts[:, 0, 2], ".", color="green", ms=5, mew=0.5)
    pyplot.plot(bodies[:, 1, 0], bodies[:, 1, 2], ".", color="blue", ms=5, mew=5)
    pyplot.axes().set_aspect("equal", "datalim")
    pyplot.grid()
    pyplot.show()

def show(body, boundary, collocation, force, el=25.0, az=-160.0):
    ax3d = Axes3D(pyplot.figure())

    num_x = 10
    num_y = 20

    for i in range(num_x):
        ax3d.plot(body[i, :, 0], body[i, :, 1], body[i, :, 2], '-', color='lime', ms=1.5, mew=0.5)

    for i in range(num_y):
        ax3d.plot(body[i, :, 0], body[i, :, 1], body[i, :, 2], '-', color='black', ms=0.5, mew=0.5)
        ax3d.plot(body[i, :, 0], body[i, :, 1], body[i, :, 2], '.', color='lime', ms=0.5, mew=0.5)

    for i in range(num_y - 1):
        ax3d.plot(collocation[:, i, 0], collocation[:, i, 1], collocation[:, i, 2], '.', color='blue', ms=3, mew=0.1)
        ax3d.plot(boundary[:, i, 0], boundary[:, i, 1], boundary[:, i, 2], '.', color='red', ms=3, mew=0.1)
    for i_x in range(len(collocation[:, 0, 0])):
        for i_y in range(len(collocation[0, :, 0])):
            ax3d.quiver(collocation[i_x, i_y, 0], collocation[i_x, i_y, 1], collocation[i_x, i_y, 2],
                        0., 0., force[i_x, i_y, :]*hstack((0, 0, 0.002)), color='k', linewidth=1.0)

    ax3d.set_zlim3d(-2.5, 2.5)
    ax3d.set_xlim3d(-2.5, 2.5)
    ax3d.set_ylim3d(0., 10.)
    ax3d.set_xlabel("x[m]")
    ax3d.set_ylabel("y[m]")
    ax3d.set_zlabel("z[m]")
    ax3d.view_init(elev=el, azim=az)

    pyplot.show()