import numpy as np
import random
import math

calcs = []

# for action in list(results_master.keys()):

results_master = {
    'Bowing': {
        'l_arm_m2': np.array([[ 5.,  5.],
                              [ 1.,  5.],
                              [ 0.,  0.],
                              [ 0.,  0.],
                              [ 0.,  0.],
                              [ 0., 12.],
                              [ 5.,  5.],
                              [ 0.,  6.],
                              [12., 12.],
                              [ 3.,  5.]]), 
        'l_arm_m3': np.array([[ 8.,  8.],
                            [ 8.,  8.],
                            [ 0.,  0.],
                            [ 0.,  0.],
                            [ 0.,  0.],
                            [ 0.,  5.],
                            [ 0.,  7.],
                            [ 7.,  7.],
                            [ 3.,  4.],
                            [11., 11.]]), 
        'r_arm_m4': np.array([[ 0.,  5.],
                            [ 0.,  6.],
                            [ 0.,  0.],
                            [ 0.,  0.],
                            [ 0.,  0.],
                            [ 8.,  9.],
                            [10., 10.],
                            [ 1.,  8.],
                            [ 3.,  3.],
                            [ 9.,  9.]]), 
        'r_arm_m5': np.array([[ 0.,  8.],
                            [ 0.,  7.],
                            [ 0.,  0.],
                            [ 0.,  0.],
                            [ 0.,  0.],
                            [ 0.,  5.],
                            [ 8.,  8.],
                            [ 2., 13.],
                            [ 0.,  9.],
                            [ 0.,  0.]]), 
        'head': np.array([[5., 5.],
                            [7., 7.],
                            [5., 5.],
                            [1., 6.],
                            [0., 6.],
                            [1., 1.],
                            [2., 2.],
                            [3., 9.],
                            [2., 2.],
                            [7., 7.]])}, 
    'Clapping': {
        'l_arm_m2': np.array([[11., 11.],
       [ 9.,  9.],
       [ 0.,  0.],
       [ 0.,  0.],
       [10., 10.],
       [ 0.,  0.],
       [ 6.,  8.],
       [ 0.,  0.],
       [12., 12.],
       [ 0.,  0.]]), 
        'l_arm_m3': np.array([[ 5.,  5.],
       [ 5.,  5.],
       [ 5.,  5.],
       [ 6., 10.],
       [ 3.,  3.],
       [ 0.,  0.],
       [ 9.,  9.],
       [ 0.,  0.],
       [ 5.,  5.],
       [ 8.,  8.]]), 
        'r_arm_m4': np.array([[10., 10.],
       [ 9.,  9.],
       [ 0.,  0.],
       [ 0.,  0.],
       [ 0.,  0.],
       [ 0.,  0.],
       [13., 13.],
       [ 0.,  0.],
       [ 8.,  8.],
       [10., 10.]]), 'r_arm_m5': np.array([[ 4.,  4.],
       [ 5.,  5.],
       [ 7.,  7.],
       [ 0.,  0.],
       [ 0.,  0.],
       [ 8., 12.],
       [ 3.,  3.],
       [ 3.,  6.],
       [ 3.,  3.],
       [10., 10.]])}, 'Handshaking': {'l_arm_m2': np.array([[0., 6.],
       [0., 6.],
       [0., 6.],
       [8., 8.],
       [0., 8.],
       [5., 5.],
       [0., 3.],
       [0., 0.],
       [0., 5.],
       [0., 3.]]), 'l_arm_m3': np.array([[2., 6.],
       [0., 4.],
       [0., 6.],
       [0., 7.],
       [0., 6.],
       [6., 6.],
       [5., 5.],
       [0., 0.],
       [4., 5.],
       [1., 5.]]), 'r_arm_m4': np.array([[ 7.,  7.],
       [ 0.,  7.],
       [ 0., 12.],
       [ 0.,  3.],
       [ 0.,  0.],
       [ 0.,  0.],
       [ 1.,  5.],
       [ 0.,  0.],
       [ 3., 10.],
       [ 0.,  6.]]), 'r_arm_m5': np.array([[ 5.,  6.],
       [ 0.,  5.],
       [ 0.,  6.],
       [ 0.,  7.],
       [ 0.,  0.],
       [ 0.,  0.],
       [ 0.,  8.],
       [ 0.,  0.],
       [ 2.,  7.],
       [ 0., 11.]])}, 'Hugging': {'l_arm_m2': np.array([[12., 12.],
       [ 0.,  0.],
       [ 0.,  0.],
       [ 0.,  0.],
       [ 0.,  7.],
       [ 9., 10.],
       [ 0.,  0.],
       [ 5.,  7.],
       [ 5., 14.],
       [ 0.,  0.]]), 'l_arm_m3': np.array([[ 0.,  7.],
       [ 0.,  0.],
       [ 0.,  0.],
       [ 0.,  0.],
       [ 0.,  3.],
       [15., 17.],
       [ 0.,  0.],
       [ 0.,  9.],
       [14., 14.],
       [ 0.,  0.]]), 'r_arm_m4': np.array([[ 0.,  0.],
       [ 5.,  5.],
       [15., 15.],
       [12., 12.],
       [ 0.,  0.],
       [ 0.,  0.],
       [ 0.,  0.],
       [12., 15.],
       [ 2.,  3.],
       [ 0.,  0.]]), 'r_arm_m5': np.array([[ 0.,  0.],
       [ 0.,  3.],
       [16., 16.],
       [ 4.,  6.],
       [ 0.,  0.],
       [ 0.,  0.],
       [ 0.,  0.],
       [ 6., 14.],
       [ 0., 11.],
       [ 0.,  0.]])}, 'Jumping': {'l_leg_m6': np.array([[ 0.,  6.],
       [ 2.,  4.],
       [ 2.,  6.],
       [ 4.,  5.],
       [ 7.,  7.],
       [ 0.,  0.],
       [ 0.,  0.],
       [ 0.,  3.],
       [ 0., 12.],
       [ 7.,  7.]]), 'l_leg_m7': np.array([[ 8.,  8.],
       [ 0.,  0.],
       [ 7.,  7.],
       [ 6.,  6.],
       [ 6.,  6.],
       [ 0.,  0.],
       [ 0.,  0.],
       [ 0.,  7.],
       [ 6.,  6.],
       [10., 10.]]), 'r_leg_m8': np.array([[ 8.,  8.],
       [ 0.,  0.],
       [ 3.,  4.],
       [11., 11.],
       [ 8.,  8.],
       [ 0.,  0.],
       [ 0.,  0.],
       [ 1.,  6.],
       [ 4.,  4.],
       [ 9.,  9.]]), 'r_leg_m9': np.array([[ 0.,  0.],
       [ 0.,  0.],
       [ 6.,  8.],
       [15., 15.],
       [ 0.,  0.],
       [ 0.,  0.],
       [ 0.,  0.],
       [ 0., 12.],
       [ 9.,  9.],
       [ 6.,  6.]])}, 'Running': {'l_arm_m2': np.array([[6., 8.],
       [1., 4.],
       [6., 6.],
       [0., 5.],
       [6., 6.],
       [1., 2.],
       [6., 6.],
       [0., 4.],
       [6., 6.],
       [2., 3.]]), 'l_arm_m3': np.array([[2., 7.],
       [4., 4.],
       [6., 6.],
       [0., 4.],
       [1., 6.],
       [1., 2.],
       [6., 6.],
       [2., 6.],
       [3., 3.],
       [6., 6.]]), 'r_arm_m4': np.array([[3., 4.],
       [3., 5.],
       [4., 4.],
       [0., 7.],
       [5., 5.],
       [9., 9.],
       [3., 3.],
       [0., 2.],
       [3., 3.],
       [5., 8.]]), 'r_arm_m5': np.array([[3., 3.],
       [4., 4.],
       [7., 7.],
       [0., 8.],
       [2., 5.],
       [3., 3.],
       [5., 6.],
       [0., 4.],
       [6., 6.],
       [2., 4.]]), 'l_leg_m6': np.array([[8., 8.],
       [7., 7.],
       [6., 6.],
       [1., 3.],
       [4., 4.],
       [0., 0.],
       [8., 8.],
       [0., 5.],
       [2., 2.],
       [2., 7.]]), 'l_leg_m7': np.array([[4., 4.],
       [5., 5.],
       [5., 5.],
       [6., 9.],
       [4., 4.],
       [0., 0.],
       [4., 5.],
       [0., 7.],
       [5., 5.],
       [0., 6.]]), 'r_leg_m8': np.array([[ 7.,  7.],
       [ 4.,  6.],
       [ 4.,  4.],
       [ 0.,  3.],
       [ 6.,  6.],
       [ 0.,  0.],
       [ 2.,  2.],
       [ 1.,  1.],
       [10., 10.],
       [ 0., 11.]]), 'r_leg_m9': np.array([[8., 8.],
       [6., 6.],
       [6., 6.],
       [0., 7.],
       [6., 6.],
       [0., 0.],
       [5., 5.],
       [2., 7.],
       [1., 1.],
       [0., 4.]]), 'head': np.array([[7., 7.],
       [0., 8.],
       [6., 6.],
       [0., 3.],
       [0., 6.],
       [0., 0.],
       [5., 5.],
       [0., 3.],
       [5., 5.],
       [1., 7.]])}, 'Seating': {'l_leg_m6': np.array([[ 0.,  4.],
       [ 0.,  0.],
       [ 7.,  7.],
       [ 5.,  5.],
       [11., 11.],
       [ 3.,  3.],
       [ 2.,  6.],
       [ 0.,  2.],
       [12., 12.],
       [ 0.,  0.]]), 'l_leg_m7': np.array([[ 0., 10.],
       [ 0.,  0.],
       [ 7.,  7.],
       [ 0.,  0.],
       [ 6.,  6.],
       [ 0.,  9.],
       [ 8.,  8.],
       [ 0.,  0.],
       [ 8.,  8.],
       [ 2.,  2.]]), 'r_leg_m8': np.array([[ 0.,  0.],
       [ 0.,  0.],
       [13., 13.],
       [ 8.,  8.],
       [ 7.,  7.],
       [ 7.,  7.],
       [ 0.,  0.],
       [ 8.,  8.],
       [ 7.,  7.],
       [ 0.,  0.]]), 'r_leg_m9': np.array([[ 0.,  0.],
       [ 0.,  0.],
       [ 2.,  2.],
       [ 9.,  9.],
       [10., 10.],
       [ 6.,  7.],
       [ 8.,  8.],
       [ 0.,  0.],
       [ 6.,  6.],
       [ 8.,  8.]]), 'head': np.array([[ 0.,  0.],
       [ 7.,  7.],
       [12., 12.],
       [ 5., 10.],
       [12., 12.],
       [ 0.,  0.],
       [ 0.,  0.],
       [ 0.,  0.],
       [ 0.,  0.],
       [ 6.,  9.]])}, 'Standing': {'l_leg_m6': np.array([[ 6.,  6.],
       [10., 10.],
       [ 5.,  5.],
       [ 0.,  0.],
       [ 0.,  0.],
       [ 5.,  9.],
       [ 8.,  9.],
       [ 6.,  6.],
       [ 5.,  5.],
       [ 0.,  0.]]), 'l_leg_m7': np.array([[ 7.,  7.],
       [12., 12.],
       [ 7.,  7.],
       [ 0.,  0.],
       [ 0.,  0.],
       [ 0.,  0.],
       [ 0.,  0.],
       [ 0.,  8.],
       [16., 16.],
       [ 0.,  0.]]), 'r_leg_m8': np.array([[ 0.,  0.],
       [ 6.,  6.],
       [ 6.,  6.],
       [ 5., 10.],
       [ 7., 10.],
       [ 3.,  6.],
       [ 0.,  0.],
       [ 0.,  0.],
       [12., 12.],
       [ 0.,  0.]]), 'r_leg_m9': np.array([[ 0.,  0.],
       [10., 10.],
       [ 0.,  0.],
       [11., 13.],
       [13., 13.],
       [ 0.,  0.],
       [ 0.,  0.],
       [ 0.,  0.],
       [ 9., 14.],
       [ 0.,  0.]]), 'head': np.array([[ 0.,  0.],
       [11., 11.],
       [ 5.,  5.],
       [ 8., 17.],
       [ 0.,  0.],
       [ 7.,  7.],
       [ 0.,  0.],
       [10., 10.],
       [ 0.,  0.],
       [ 0.,  0.]])}, 'Walking': {'l_arm_m2': np.array([[ 3.,  6.],
       [ 0.,  0.],
       [ 6.,  9.],
       [ 0., 12.],
       [ 9.,  9.],
       [ 6.,  6.],
       [ 8.,  8.],
       [ 0.,  0.],
       [ 0.,  0.],
       [ 0.,  0.]]), 'l_arm_m3': np.array([[ 8.,  9.],
       [ 0.,  0.],
       [ 0., 10.],
       [ 0.,  6.],
       [ 5.,  5.],
       [10., 10.],
       [10., 10.],
       [ 0.,  0.],
       [ 0.,  0.],
       [ 0.,  0.]]), 'r_arm_m4': np.array([[ 8.,  8.],
       [ 0.,  0.],
       [ 0.,  3.],
       [ 0.,  7.],
       [ 0.,  0.],
       [12., 13.],
       [ 9.,  9.],
       [ 0.,  0.],
       [10., 10.],
       [ 0.,  0.]]), 'r_arm_m5': np.array([[ 9.,  9.],
       [ 0.,  0.],
       [ 1., 10.],
       [ 0.,  4.],
       [ 0.,  0.],
       [ 6.,  6.],
       [12., 12.],
       [ 0.,  0.],
       [ 9.,  9.],
       [ 0.,  0.]]), 'l_leg_m6': np.array([[ 7.,  7.],
       [ 0.,  0.],
       [10., 10.],
       [ 0.,  7.],
       [ 7.,  7.],
       [ 0.,  8.],
       [11., 11.],
       [ 0.,  0.],
       [ 0.,  0.],
       [ 0.,  0.]]), 'l_leg_m7': np.array([[14., 14.],
       [ 0.,  0.],
       [ 8.,  8.],
       [ 0.,  9.],
       [ 7.,  7.],
       [ 0.,  5.],
       [ 7.,  7.],
       [ 0.,  0.],
       [ 0.,  0.],
       [ 0.,  0.]]), 'r_leg_m8': np.array([[5., 5.],
       [0., 0.],
       [6., 8.],
       [0., 9.],
       [9., 9.],
       [0., 4.],
       [6., 6.],
       [0., 0.],
       [9., 9.],
       [0., 0.]]), 'r_leg_m9': np.array([[ 5., 10.],
       [ 0.,  0.],
       [ 0.,  8.],
       [ 0.,  9.],
       [ 6.,  6.],
       [ 2.,  5.],
       [ 4.,  4.],
       [ 0.,  0.],
       [ 8.,  8.],
       [ 0.,  0.]]), 'head': np.array([[ 4.,  4.],
       [ 0.,  0.],
       [ 7.,  9.],
       [ 0., 10.],
       [ 8.,  8.],
       [10., 10.],
       [ 0.,  0.],
       [ 0.,  0.],
       [ 9.,  9.],
       [ 0.,  0.]])}, 'Waving': {'l_arm_m2': np.array([[ 0., 15.],
       [ 0., 11.],
       [ 0.,  0.],
       [ 0.,  0.],
       [ 0.,  9.],
       [ 0.,  0.],
       [ 0.,  8.],
       [ 0.,  0.],
       [ 0.,  7.],
       [ 0.,  0.]]), 'l_arm_m3': np.array([[ 0., 10.],
       [ 4.,  6.],
       [ 0.,  0.],
       [ 0.,  0.],
       [ 0.,  6.],
       [ 0.,  0.],
       [ 0., 11.],
       [ 0.,  9.],
       [ 0.,  8.],
       [ 0.,  0.]]), 'r_arm_m4': np.array([[ 5.,  5.],
       [ 0.,  9.],
       [ 0.,  0.],
       [ 0.,  0.],
       [ 0.,  7.],
       [ 0.,  0.],
       [ 0., 12.],
       [ 0.,  0.],
       [ 0., 17.],
       [ 0.,  0.]]), 'r_arm_m5': np.array([[10., 10.],
       [ 3., 12.],
       [ 0.,  6.],
       [ 0.,  0.],
       [ 0.,  0.],
       [ 0.,  0.],
       [ 0.,  9.],
       [ 0.,  0.],
       [ 0., 13.],
       [ 0.,  0.]])}, 'Elbowing': {'l_arm_m2': np.array([[5., 5.],
       [7., 7.],
       [4., 4.],
       [5., 5.],
       [5., 5.],
       [5., 5.],
       [6., 6.],
       [3., 3.],
       [3., 3.],
       [7., 7.]]), 'l_arm_m3': np.array([[4., 4.],
       [7., 7.],
       [4., 4.],
       [6., 6.],
       [3., 3.],
       [5., 5.],
       [3., 3.],
       [3., 3.],
       [8., 8.],
       [7., 7.]]), 'r_arm_m4': np.array([[4., 4.],
       [5., 5.],
       [5., 5.],
       [6., 6.],
       [4., 5.],
       [4., 4.],
       [3., 3.],
       [5., 5.],
       [8., 8.],
       [5., 5.]]), 'r_arm_m5': np.array([[5., 5.],
       [4., 4.],
       [5., 5.],
       [8., 8.],
       [6., 6.],
       [3., 3.],
       [0., 3.],
       [6., 6.],
       [7., 7.],
       [1., 3.]])}, 'Frontkicking': {'l_leg_m6': np.array([[ 9.,  9.],
       [ 0.,  0.],
       [ 0.,  6.],
       [ 5.,  7.],
       [ 6.,  6.],
       [ 0.,  0.],
       [ 3., 15.],
       [ 2.,  2.],
       [ 5.,  5.],
       [ 0.,  0.]]), 'l_leg_m7': np.array([[ 8.,  8.],
       [ 0.,  0.],
       [ 0.,  4.],
       [ 1.,  5.],
       [12., 12.],
       [ 0.,  0.],
       [ 0.,  5.],
       [12., 12.],
       [ 4.,  4.],
       [ 0.,  0.]]), 'r_leg_m8': np.array([[ 3.,  3.],
       [ 6.,  6.],
       [ 6.,  6.],
       [ 0.,  0.],
       [10., 10.],
       [ 4.,  4.],
       [ 0.,  0.],
       [ 5.,  7.],
       [ 7.,  7.],
       [ 0.,  7.]]), 'r_leg_m9': np.array([[4., 4.],
       [7., 7.],
       [6., 6.],
       [0., 0.],
       [3., 3.],
       [8., 8.],
       [0., 0.],
       [4., 5.],
       [9., 9.],
       [2., 8.]]), 'head': np.array([[ 0.,  0.],
       [ 7.,  7.],
       [ 8.,  8.],
       [ 5.,  5.],
       [ 0.,  0.],
       [17., 17.],
       [ 0.,  0.],
       [13., 13.],
       [ 0.,  0.],
       [ 0.,  0.]])}, 'Hamering': {'l_arm_m2': np.array([[ 6.,  6.],
       [ 8.,  8.],
       [ 0.,  0.],
       [ 6.,  7.],
       [ 0.,  7.],
       [ 0.,  0.],
       [ 4.,  4.],
       [ 0., 13.],
       [ 0.,  0.],
       [ 3.,  5.]]), 'l_arm_m3': np.array([[6., 6.],
       [9., 9.],
       [0., 0.],
       [4., 5.],
       [6., 9.],
       [0., 0.],
       [5., 6.],
       [5., 7.],
       [0., 0.],
       [8., 8.]]), 'r_arm_m4': np.array([[ 1.,  7.],
       [ 9.,  9.],
       [ 0.,  0.],
       [ 0.,  2.],
       [ 3.,  8.],
       [ 0.,  0.],
       [ 7., 10.],
       [ 0.,  7.],
       [ 0.,  0.],
       [ 7.,  7.]]), 'r_arm_m5': np.array([[ 2.,  8.],
       [ 7.,  7.],
       [ 0.,  0.],
       [ 1.,  7.],
       [11., 11.],
       [ 0.,  0.],
       [ 0.,  5.],
       [ 0.,  7.],
       [ 0.,  0.],
       [ 5.,  5.]])}, 'Headering': {'head': np.array([[6., 6.],
       [6., 6.],
       [4., 6.],
       [0., 5.],
       [0., 5.],
       [0., 0.],
       [4., 4.],
       [0., 8.],
       [4., 4.],
       [6., 6.]])}, 'Kneeing': {'l_leg_m6': np.array([[ 9.,  9.],
       [ 0.,  0.],
       [ 7.,  7.],
       [ 0.,  0.],
       [ 2.,  8.],
       [ 0.,  5.],
       [ 6.,  6.],
       [ 9., 10.],
       [ 5.,  5.],
       [ 0.,  0.]]), 'l_leg_m7': np.array([[ 7.,  7.],
       [ 0.,  0.],
       [ 9.,  9.],
       [ 0.,  0.],
       [ 4.,  9.],
       [ 0.,  4.],
       [ 6., 10.],
       [ 7.,  7.],
       [ 4.,  4.],
       [ 0.,  0.]]), 'r_leg_m8': np.array([[ 6.,  6.],
       [ 3.,  3.],
       [ 6.,  6.],
       [12., 12.],
       [ 0.,  7.],
       [ 0.,  0.],
       [ 8.,  8.],
       [ 2.,  2.],
       [ 6.,  6.],
       [ 0.,  0.]]), 'r_leg_m9': np.array([[8., 8.],
       [4., 7.],
       [0., 0.],
       [8., 8.],
       [8., 8.],
       [0., 2.],
       [4., 4.],
       [4., 4.],
       [5., 6.],
       [3., 3.]])}, 'Pulling': {'l_arm_m2': np.array([[6., 6.],
       [2., 7.],
       [0., 3.],
       [7., 7.],
       [1., 3.],
       [4., 4.],
       [0., 3.],
       [2., 7.],
       [0., 3.],
       [3., 7.]]), 'l_arm_m3': np.array([[4., 4.],
       [0., 3.],
       [8., 8.],
       [1., 2.],
       [4., 4.],
       [4., 8.],
       [6., 6.],
       [3., 3.],
       [3., 4.],
       [0., 8.]]), 'r_arm_m4': np.array([[ 4.,  4.],
       [ 7.,  7.],
       [ 0.,  0.],
       [ 3.,  3.],
       [ 0.,  0.],
       [ 1.,  6.],
       [ 0.,  0.],
       [11., 13.],
       [13., 13.],
       [ 4.,  4.]]), 'r_arm_m5': np.array([[ 5.,  5.],
       [ 0.,  3.],
       [11., 11.],
       [ 0.,  0.],
       [ 1.,  4.],
       [ 1.,  7.],
       [ 0.,  0.],
       [ 7.,  7.],
       [ 6.,  6.],
       [ 0.,  7.]])}, 'Punching': {'l_arm_m2': np.array([[ 8.,  8.],
       [ 9.,  9.],
       [ 7.,  7.],
       [ 0.,  0.],
       [10., 10.],
       [ 0.,  0.],
       [ 6.,  6.],
       [ 2.,  3.],
       [ 7.,  7.],
       [ 0.,  0.]]), 'l_arm_m3': np.array([[4., 4.],
       [3., 4.],
       [3., 3.],
       [0., 0.],
       [5., 5.],
       [6., 6.],
       [9., 9.],
       [5., 5.],
       [9., 9.],
       [4., 5.]]), 'r_arm_m4': np.array([[ 9.,  9.],
       [10., 10.],
       [ 0.,  0.],
       [ 4.,  4.],
       [ 0.,  7.],
       [ 3.,  4.],
       [ 2.,  2.],
       [ 8.,  8.],
       [ 6.,  6.],
       [ 0.,  0.]]), 'r_arm_m5': np.array([[ 7.,  7.],
       [ 5.,  5.],
       [ 0.,  0.],
       [ 5.,  8.],
       [ 4.,  4.],
       [ 0.,  2.],
       [ 7.,  7.],
       [ 9., 10.],
       [ 7.,  7.],
       [ 0.,  0.]])}, 'Pushing': {'l_arm_m2': np.array([[10., 10.],
       [ 7.,  8.],
       [ 6.,  6.],
       [ 7.,  7.],
       [ 7.,  7.],
       [ 6.,  6.],
       [ 0.,  0.],
       [ 0.,  0.],
       [ 6.,  6.],
       [ 0.,  0.]]), 'l_arm_m3': np.array([[ 6.,  6.],
       [ 6.,  9.],
       [ 8.,  8.],
       [ 4.,  4.],
       [ 5.,  5.],
       [ 7.,  8.],
       [ 0.,  0.],
       [ 0.,  0.],
       [10., 10.],
       [ 0.,  0.]]), 'r_arm_m4': np.array([[6., 8.],
       [7., 7.],
       [6., 6.],
       [0., 0.],
       [9., 9.],
       [3., 3.],
       [0., 0.],
       [5., 5.],
       [4., 4.],
       [7., 8.]]), 'r_arm_m5': np.array([[0., 5.],
       [8., 8.],
       [6., 6.],
       [0., 0.],
       [8., 8.],
       [6., 6.],
       [0., 0.],
       [5., 5.],
       [7., 7.],
       [5., 5.]])}, 'Sidekicking': {'l_leg_m6': np.array([[ 0.,  0.],
       [ 0., 10.],
       [ 1.,  2.],
       [ 0.,  9.],
       [ 7.,  7.],
       [ 0.,  6.],
       [ 0.,  4.],
       [ 0.,  7.],
       [ 5.,  5.],
       [ 0.,  0.]]), 'l_leg_m7': np.array([[0., 0.],
       [0., 8.],
       [0., 5.],
       [6., 6.],
       [0., 7.],
       [0., 7.],
       [0., 8.],
       [0., 5.],
       [4., 4.],
       [0., 0.]]), 'r_leg_m8': np.array([[14., 14.],
       [ 5.,  5.],
       [10., 10.],
       [ 0.,  0.],
       [ 8.,  9.],
       [ 0.,  0.],
       [ 7.,  7.],
       [ 5.,  5.],
       [ 0.,  0.],
       [ 0.,  0.]]), 'r_leg_m9': np.array([[10., 10.],
       [ 5.,  8.],
       [ 7.,  7.],
       [ 0.,  0.],
       [ 7.,  7.],
       [ 0.,  0.],
       [ 9.,  9.],
       [ 9.,  9.],
       [ 0.,  0.],
       [ 0.,  0.]])}, 'Slapping': {'l_arm_m2': np.array([[9., 9.],
       [7., 7.],
       [4., 4.],
       [5., 6.],
       [0., 5.],
       [5., 5.],
       [6., 6.],
       [2., 3.],
       [5., 5.],
       [0., 0.]]), 'l_arm_m3': np.array([[4., 4.],
       [4., 4.],
       [4., 4.],
       [7., 9.],
       [1., 4.],
       [0., 6.],
       [4., 4.],
       [8., 8.],
       [4., 4.],
       [3., 3.]]), 'r_arm_m4': np.array([[ 5.,  5.],
       [ 1.,  3.],
       [11., 11.],
       [ 6.,  7.],
       [ 6.,  7.],
       [ 0.,  5.],
       [ 0.,  0.],
       [ 7.,  8.],
       [ 4.,  4.],
       [ 0.,  0.]]), 'r_arm_m5': np.array([[11., 11.],
       [ 8.,  8.],
       [ 2.,  2.],
       [ 4.,  4.],
       [ 3.,  9.],
       [ 0.,  0.],
       [ 0.,  0.],
       [ 6.,  6.],
       [ 5.,  5.],
       [ 5.,  5.]])}}



calcs = []

for action in list(results_master.keys()):
    print(f"Action: {action}")
    total_action = np.zeros((1, 2))
    for limb in list(results_master[action].keys()):
        results_master[action][limb]
        total_action += np.sum(results_master[action][limb], axis=0)
    print(f"Acc: {total_action[0, 0]/total_action[0, 1]}")



# count = 0
# correct_master = 0
# total_master = 0
# for action, dic in results_master.items():
#     print("")
#     for limb, results in results_master[action].items():
#         correct = np.sum(results, axis=0)[0]
#         total = np.sum(results, axis=0)[1]
#         print(f"{action} + {limb} Total Accuracy: \t\t{round(correct/total*100, 2)}%")
#         correct_master += correct
#         total_master += total

# print(f"\nTOTAL ACCURACY: {round(correct_master/total_master*100, 2)}%\n")

# a = np.array([[8, 1], [3, 4], [4, 5]])
# loc, = np.where(a[:, 0] == 8)
# print(loc[0])

# print(np.sum(a, axis=0))