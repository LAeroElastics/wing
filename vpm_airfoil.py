#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy

DBL_PI = 3.14159265358979323846264338327950288
DBL_D2R = DBL_PI / 180.0
DBL_R2D = 180.0 / DBL_PI

def cos(arg):
    return numpy.cos(arg)

def sin(arg):
    return numpy.sin(arg)

class WingSection(object):
    def __init__(self,num_chord_max_iter,num_span_max_iter,span,chord,max_camber,loc_max_camber,thickness):
        self.num_chord_max_iter=num_chord_max_iter;
        self.num_span_max_iter=num_span_max_iter;
        self.max_camber=max_camber;
        self.loc_max_camber=loc_max_camber;
        self.thickness=thickness;
        self.span_wing_section=span;
        self.chord_wing_section=chord;
        self.airfoil=numpy.array((self.num_chord_max_iter,self.num_span_max_iter,3));
        self.camber=numpy.array((self.num_chord_max_iter,3));
        self.uniform_chord_point=numpy.zeros(self.num_chord_max_iter)
        self.uniform_chord_point.fill(1.0)

    def generate(self):
        num_chord_max_iter = self.num_chord_max_iter;
        num_span_max_iter = self.num_span_max_iter;

        #コードの分割点
        chord_divs = numpy.linspace(0.0, DBL_PI, num_chord_max_iter)
        #キャンバーの分割点
        camber_divs = numpy.zeros((num_chord_max_iter))

        for i_chord_iter, loc_tan in enumerate(chord_divs):
            camber_divs[i_chord_iter] = 0.5*(1.0-cos(loc_tan))\
                                        *self.uniform_chord_point[i_chord_iter]

        #キャンバーラインを求めるサブルーチン
        def camber_line(pts, max_camber, loc_max_camber):
            return numpy.where((pts>=0)&(pts<=(1.0*loc_max_camber)),
                               max_camber*(pts/numpy.power(loc_max_camber,2.0))*(2.0*loc_max_camber-pts),
                               max_camber*((1.0-pts)/numpy.power(1.0-loc_max_camber,2.0))*(1.0+pts-2.0*loc_max_camber))

        camber_points = camber_line(camber_divs, self.max_camber, self.loc_max_camber)
        camber_dy = numpy.where((camber_divs>=0)&(camber_divs<=self.loc_max_camber),
                               ((2.0*self.max_camber)/numpy.power(self.loc_max_camber,2.0))*(self.loc_max_camber-camber_divs),
                               ((2.0*self.max_camber)/numpy.power(1.0-self.loc_max_camber,2.0))*(self.loc_max_camber-camber_divs))
        atan_dy = numpy.arctan(camber_dy)
        #キャンバーラインからの上下面高さ
        thickness = 5.0*self.thickness*(0.2969*(numpy.sqrt(camber_divs))-0.1260*camber_divs-0.3516*numpy.power(camber_divs,2.0)+\
                                0.2843*numpy.power(camber_divs,3.0)-0.1015*numpy.power(camber_divs,4.0))
        #メッシュ生成
        mesh_upper_half = numpy.zeros((num_chord_max_iter, 2))
        mesh_lower_half = numpy.zeros((num_chord_max_iter, 2))

        #上面
        mesh_upper_half[:,0] = camber_divs-thickness*sin(atan_dy)
        mesh_upper_half[:,1] = camber_points+thickness*cos(atan_dy)
        #下面
        mesh_lower_half[:,0] = camber_divs+thickness*sin(atan_dy)
        mesh_lower_half[:,1] = camber_points-thickness*cos(atan_dy)

        #結合
        mesh_result = numpy.vstack((mesh_upper_half[:,:],mesh_lower_half[:,:]))
        camber_result = numpy.zeros((num_chord_max_iter, 2))
        for i_chord_iter in range(num_chord_max_iter):
            camber_result[i_chord_iter,:]=[camber_divs[i_chord_iter],camber_points[i_chord_iter]]

        return mesh_result, camber_result

    def assemble(self):
        num_chord_max_iter = self.num_chord_max_iter
        num_span_max_iter = self.num_span_max_iter

        #矩形格子の生成
        source_span_points = numpy.linspace(0.0, 1.0, num_span_max_iter)*self.span_wing_section
        source_chord_points, source_camber_points = self.generate()

        mesh_airfoil = numpy.zeros((len(source_chord_points), num_span_max_iter,3))
        mesh_camber = numpy.zeros((num_chord_max_iter, num_span_max_iter,3))

        for i_chord_iter,_ in enumerate(source_chord_points):
            for i_span_iter in range(num_span_max_iter):
                mesh_airfoil[i_chord_iter,i_span_iter,:] \
                    = [source_chord_points[i_chord_iter,0], source_span_points[i_span_iter], source_chord_points[i_chord_iter,1]]

        for i_chord_iter,_ in enumerate(source_camber_points):
            for i_span_iter in range(num_span_max_iter):
                mesh_camber[i_chord_iter,i_span_iter,:] \
                    = [source_camber_points[i_chord_iter,0], source_span_points[i_span_iter], source_camber_points[i_chord_iter,1]]

        self.airfoil = mesh_airfoil
        self.camber = mesh_camber

    def set_position_offset(self, offset, dst):
        self.airfoil[:,:,dst] += offset

    def set_sweep_angle(self, sweep):
        num_chord_max_iter = self.num_chord_max_iter
        num_span_max_iter = self.num_span_max_iter

        leading_edge = self.airfoil[0]
        tan_theta = numpy.tan(DBL_D2R*sweep)

        offset = -(leading_edge[:,1]-leading_edge[-1,1])*tan_theta

        for i_chord_iter in range(num_chord_max_iter):
            self.airfoil[i_chord_iter,:,0]+=offset

    def set_dihedral_angle(self, dihedral):
        num_chord_max_iter = numpy.shape(self.airfoil)[0]
        num_span_max_iter = numpy.shape(self.airfoil)[1]

        leading_edge = self.airfoil[0]
        tan_theta = numpy.tan(DBL_D2R*dihedral)

        offset = -(leading_edge[:,1]-leading_edge[-1,1])*tan_theta

        for i_chord_iter in range(num_chord_max_iter):
            self.airfoil[i_chord_iter,:,0]+=offset

    def set_taper_ratio(self):
        num_chord_max_iter = numpy.shape(self.airfoil)[0]
        num_span_max_iter = numpy.shape(self.airfoil)[1]

        trailing_edge = self.airfoil[-1]
        leading_edge = self.airfoil[0]
        quater_chord = 0.25*trailing_edge+0.75*leading_edge



    def set_chord_scale(self, dst):
        trailing_edge = self.airfoil[-1]
        leading_edge = self.airfoil[0]
        quater_chord = 0.25*trailing_edge+0.75*leading_edge

        num_span_max_iter = numpy.shape(self.airfoil)[1]

        for i_span_iter in range(num_span_max_iter):
            self.airfoil[:,i_span_iter,0] \
                = (self.airfoil[:,i_span_iter,0]-quater_chord[i_span_iter,0]
                   *dst[i_span_iter]+quater_chord[i_span_iter,0])

    def set_span_scale(self, dst):
        trailing_edge = self.airfoil[-1]
        leading_edge = self.airfoil[0]
        quater_chord = 0.25 * trailing_edge + 0.75 * leading_edge

        temp = quater_chord[-1,1]-quater_chord[0,1]
        fixed_span = quater_chord[:,1]/temp

        self.airfoil[:,:,1] = fixed_span*dst

