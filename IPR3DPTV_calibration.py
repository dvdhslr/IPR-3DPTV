import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob

from scipy.linalg import norm

import IPR3DPTV_module as ipr3dptv


#------------------------------------------------------------
# SCALES & CALIBRATION NAME
#------------------------------------------------------------

# calibration name
cal = 'lastlast'

# scales
scx, scy = 0.0355148, -0.0355148

# image size
h, w = 2488, 1656
shp = (h, w)

np.save('calibration/scales',  np.array([scx, scy, h, w]))
np.save('calibration/name', cal)

#------------------------------------------------------------
# READ IMAGES
#------------------------------------------------------------
flist1 = glob.glob('calibration/images/*.raww')
flist2 = glob.glob('calibration/images/*.raw')

if len(flist1) > 1:

    # read dewarped images of calibration target
    Icalcorr = ipr3dptv.read_single_image(flist1[0], (2517, 2083), '16')

    np.save('calibration/dewarped_calibration_image', Icalcorr)

else:
    print('Did not find image of type .raww in subfolder <calibration> that could be used as dewarped image of calibration target')

if len(flist2) == 4:

    # read image of calibration target
    Icalcam1 = ipr3dptv.read_single_image(flist2[0], (h, w), '8')
    Icalcam2 = ipr3dptv.read_single_image(flist2[2], (h, w), '8')
    Icalcam3 = ipr3dptv.read_single_image(flist2[1], (h, w), '8')
    Icalcam4 = ipr3dptv.read_single_image(flist2[3], (h, w), '8')

    Icalcamall = [Icalcam1, Icalcam2, Icalcam3, Icalcam4]
    np.save('calibration/calibration_images', Icalcamall)

else:
    print('Did not find 4 images of type .raw in subfolder <calibration> that could be used as raw images of calibration target')

#------------------------------------------------------------
# SAVE COEFFICIENTS FROM POLYNOMIAL CALIBRATION (AS TAKEN FROM DAVIS, MANUALY)
#------------------------------------------------------------

# Lastlast calibration (from volume self-calibration in DAVIS)
#------------------------------------------------------------
d_x = np.array([507., 1015.])
d_y = np.array([803., 1606.])

# camera 1, plane 1
c_x = np.array([-21.1453,
                134.475, 4.8113, -2.95383,
                -6.19382, -1.09263, 1.15985,
                0.564189, -0.237491, 0.847562])

c_y = np.array([-204.482,
                -2.74979, -0.042329, 0.565728,
                4.63241, -5.06856, 5.24771,
                9.99215, -0.003559, 0.0598747])

cc11 = [d_x, d_y, c_x, c_y]

# camera 1, plane 2
c_x = np.array([-370.058,
                133.691, 6.21981, 1.49051,
                -7.02896, 0.557357, -0.196226,
                -0.511337, 0.104403, -0.615648])

c_y = np.array([-206.804,
                -0.65754, 0.0070577, -0.405748,
                -0.837185, 2.78789, -3.43702,
                9.54329, -0.410446, 0.280706])

cc12 = [d_x, d_y, c_x, c_y]

# camera 2, plane 1
c_x = np.array([-372.74,
                32.9066, -1.39457, 0.506197,
                -17.2106, -1.48957, 3.28767,
                -0.25464, 0.778092, 0.460909])

c_y = np.array([-201.91,
                9.36593, -0.025216, 0.908298,
                12.567, -4.71678, 5.87607,
                -3.17965, 0.138703, -0.300586])

cc21 = [d_x, d_y, c_x, c_y]

# camera 2, plane 2
c_x = np.array([-227.875,
                26.9828, -5.45544, 1.51374,
                -15.0493, 1.38896, -1.74265,
                -0.713999, -0.586955, -0.718693])

c_y = np.array([-205.644,
                11.7428, -0.092354, -0.641578,
                3.71433, 2.73398, -2.70289,
                -3.95469, -0.318164, 0.0948225])

cc22 = [d_x, d_y, c_x, c_y]

# camera 3, plane 1
c_x = np.array([-125.033,
                22.9881, 2.25693, -0.182298,
                -3.92376, -1.76052, 2.10013,
                0.0561675, -0.393107, 1.03019])

c_y = np.array([-199.973,
                -0.5797, -0.006130, 0.459492,
                5.28392, -5.57354, 5.20158,
                2.85581, 0.543471, -0.126591])

cc31 = [d_x, d_y, c_x, c_y]

# camera 3, plane 2
c_x = np.array([-262.777,
                13.5625, 0.839143, 1.3934,
                -3.62916, 0.798617, -1.18376,
                -0.691538, 0.0523463, -0.377444])

c_y = np.array([-201.194,
                1.32232, 0.0285261, 0.192932,
                -3.78169, 2.02324, -3.48839,
                2.31585, -0.093783, 0.225158])

cc32 = [d_x, d_y, c_x, c_y]

# camera 4, plane 1
c_x = np.array([-632.943,
                154.829, -5.12234, 1.98226,
                -14.045, -1.49712, 3.06976,
                -0.284294, -0.550609, -0.272509])

c_y = np.array([-198.98,
                11.1556, 0.263177, 0.413035,
                10.4015, -4.72354, 5.41349,
                -10.3602, 0.171884, -0.545251])

cc41 = [d_x, d_y, c_x, c_y]

# camera 4, plane 2
c_x = np.array([-279.348,
                159.484, -3.66924, -2.90329,
                -12.1647, 0.765153, -1.66158,
                0.481198, 0.0193581, -0.183151])

c_y = np.array([-207.43,
                12.6972, 0.158176, -0.139817,
                6.11216, 2.77188, -3.15512,
                -11.397, -0.501836, 0.21936])

cc42 = [d_x, d_y, c_x, c_y]

ccall = [cc11, cc12, cc21, cc22, cc31, cc32, cc41, cc42]

np.save('calibration/calibration_coefficients_'+cal, ccall)

#------------------------------------------------------------
# POINT CORRESPONDANCE (SYNTETIC)
#------------------------------------------------------------

# world points - box seeding
dims = np.array([[4.,8.,4.],[10.,10.,5.8],[13., 26., 13.],[20., 40., 20.]])
B = ipr3dptv.gen_world_points_box(dims).T

# world points - box seeding
C = ipr3dptv.gen_world_points_calibration_plate().T

# world points - cylinder seeding
Z = ipr3dptv.gen_world_points_cylinder([3.,7.,12.], [50.,50.,50.], [0.,-5.,0.]).T

# mapping world --> image
# !!! mind camera order form DaVis 1 (outer left), 2 (inner right), 3 (inner left) 4 (outer right)!!!
xcrr1 = ipr3dptv.world2image_poly3(Z, 1, cal)
xcrr2 = ipr3dptv.world2image_poly3(Z, 3, cal)
xcrr3 = ipr3dptv.world2image_poly3(Z, 2, cal)
xcrr4 = ipr3dptv.world2image_poly3(Z, 4, cal)

#------------------------------------------------------------
# PRE-CONDITIONNING
#------------------------------------------------------------

# transform to prospective space (homogeneous coordinates)
xcrr1h = ipr3dptv.homogeneous_coordinates(xcrr1)
xcrr2h = ipr3dptv.homogeneous_coordinates(xcrr2)
xcrr3h = ipr3dptv.homogeneous_coordinates(xcrr3)
xcrr4h = ipr3dptv.homogeneous_coordinates(xcrr4)

Xcrrh = ipr3dptv.homogeneous_coordinates(Z)

xcrrall = [xcrr1h, xcrr2h, xcrr3h, xcrr4h]

np.save('calibration/correspondance_image_points', xcrrall)
np.save('calibration/correspondance_world_points', Xcrrh)

# similarity transforms
T1 = ipr3dptv.similarity_transform(shp)
T2 = ipr3dptv.similarity_transform(shp)
T3 = ipr3dptv.similarity_transform(shp)
T4 = ipr3dptv.similarity_transform(shp)

U  = ipr3dptv.similarity_transform_world([10, 20, 10])

Tall = [T1, T2, T3, T4]

np.save('calibration/similarity_transform_image', Tall)
np.save('calibration/similarity_transform_world', U)

#------------------------------------------------------------
# CAMERA MATRIX
#------------------------------------------------------------
P1, err1 = ipr3dptv.camera_matrix(xcrr1h, Xcrrh, T1, U)
P2, err2 = ipr3dptv.camera_matrix(xcrr2h, Xcrrh, T2, U)
P3, err3 = ipr3dptv.camera_matrix(xcrr3h, Xcrrh, T3, U)
P4, err4 = ipr3dptv.camera_matrix(xcrr4h, Xcrrh, T4, U)

Pall = np.array([P1, P2, P3, P4])
np.save('calibration/camera_matrices', Pall)

#------------------------------------------------------------
# FUNDAMENTAL MATRIX
#------------------------------------------------------------
F12 = ipr3dptv.fundamental_matrix_normalized8point(xcrr1h, xcrr2h, T1, T2)
F13 = ipr3dptv.fundamental_matrix_normalized8point(xcrr1h, xcrr3h, T1, T3)
F14 = ipr3dptv.fundamental_matrix_normalized8point(xcrr1h, xcrr4h, T1, T4)

F21 = F12.T
F23 = ipr3dptv.fundamental_matrix_normalized8point(xcrr2h, xcrr3h, T2, T3)
F24 = ipr3dptv.fundamental_matrix_normalized8point(xcrr2h, xcrr4h, T2, T4)

F31 = F13.T
F32 = F23.T
F34 = ipr3dptv.fundamental_matrix_normalized8point(xcrr3h, xcrr4h, T3, T4)

F41 = F14.T
F42 = F24.T
F43 = F34.T

NM = np.zeros([3,3])
Fall = np.array([[NM,  F12, F13, F14],
                 [F21,  NM, F23, F24],
                 [F31, F32,  NM, F34],
                 [F41, F42, F43, NM]])

F12mae = ipr3dptv.fundamental_matrix_minAlgebraicError(xcrr1h, xcrr2h, T1, T2)
F13mae = ipr3dptv.fundamental_matrix_minAlgebraicError(xcrr1h, xcrr3h, T1, T3)
F14mae = ipr3dptv.fundamental_matrix_minAlgebraicError(xcrr1h, xcrr4h, T1, T4)

F21mae = F12mae.T
F23mae = ipr3dptv.fundamental_matrix_minAlgebraicError(xcrr2h, xcrr3h, T2, T3)
F24mae = ipr3dptv.fundamental_matrix_minAlgebraicError(xcrr2h, xcrr4h, T2, T4)

F31mae = F13mae.T
F32mae = F23mae.T
F34mae = ipr3dptv.fundamental_matrix_minAlgebraicError(xcrr3h, xcrr4h, T3, T4)

F41mae = F14mae.T
F42mae = F24mae.T
F43mae = F34mae.T

NM = np.zeros([3,3])
Fallmae = np.array([[NM,     F12mae, F13mae, F14mae],
                    [F21mae,     NM, F23mae, F24mae],
                    [F31mae, F32mae,     NM, F34mae],
                    [F41mae, F42mae, F43mae, NM]])

np.save('calibration/fundamental_matrices', Fallmae)
