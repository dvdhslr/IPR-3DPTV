import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import sys
import time

from scipy.linalg import norm

import IPR3DPTV_module as ipr3dptv

#------------------------------------------------------------
# INITIALIZE
#------------------------------------------------------------
plistall = np.load('initial/original_particle_lists.npy')

plistallF1 = plistall[:4]
plistallF2 = plistall[4:]

Iall = np.load('initial/original_images.npy')
IallF1 = [I1, I2, I3, I4] = Iall[:4]
IallF2 = [I5, I6, I7, I8] = Iall[4:]

#------------------------------------------------------------
# ITERATIVE PARTICLE RECONSTRUCTION
#------------------------------------------------------------

# number of iterations for different loops: outer loop 4 cams, inner loop 4 cams, outer loop 3 cams, inner loop 3 cams
Niters = [5, 5, 5, 5]

# triangulation and candidate limit errors
rlim_pxl, dlim_pxl = 1.0, 1.0

# maximal first shake shift
shift_max = 1.0

# method for shaking:
# GE = True # geometric error based position correction
GE = False # projection residual based position correction

# set casename
if GE:
    casename = 'IPR_GE'+str(Niters[0])+str(Niters[1])+str(Niters[2])+str(Niters[3])
else:
    casename =   'IPR_'+str(Niters[0])+str(Niters[1])+str(Niters[2])+str(Niters[3])


"""
t_start = time.time()

# 1st frame
print('\n 1st FRAME: ITERATIVE PARTICLE RECONSTRUCTION')
print('------------------------------------------------------')
XPF1, IresallF1 = ipr3dptv.IPR_loop(plistallF1, IallF1, Niters, rlim_pxl, dlim_pxl, shift_max, GE)

# 2nd frame
print('\n 2nd FRAME: ITERATIVE PARTICLE RECONSTRUCTION')
print('------------------------------------------------------')
XPF2, IresallF2 = ipr3dptv.IPR_loop(plistallF2, IallF2, Niters, rlim_pxl, dlim_pxl, shift_max, GE)

t_end = time.time() - t_start
print('\n ELAPSED TIME: '+str(t_end)+' seconds')

# save
np.save('lists/particle_3D_positions_F1_'+casename, XPF1)
np.save('lists/particle_3D_positions_F2_'+casename, XPF2)
np.save('temp/images_original_F1_'+casename, IallF1)
np.save('temp/images_residual_F1_'+casename, IresallF1)
"""
XPF1 = np.load('lists/particle_3D_positions_F1_'+casename+'.npy')
XPF2 = np.load('lists/particle_3D_positions_F2_'+casename+'.npy')
IallF1    = np.load('temp/images_original_F1_'+casename+'.npy')
IresallF1 = np.load('temp/images_residual_F1_'+casename+'.npy')

#------------------------------------------------------------
# PARTICLE TRACKING
#------------------------------------------------------------
"""
Tracks, Tracks_prob, part_ind_vec, Prob, Prob_star = ipr3dptv.particle_tracking_relaxation(XPF1, XPF2)
np.save('lists/particle_tracks_'+casename, Tracks)
np.save('lists/particle_tracks_probabilities'+casename, Tracks_prob)
np.save('lists/particle_tracks_particle_index_'+casename, part_ind_vec)
"""
Tracks = np.load('lists/particle_tracks_'+casename+'.npy')
Tracks_prob = np.load('lists/particle_tracks_probabilities'+casename+'.npy')
part_ind_vec = np.load('lists/particle_tracks_particle_index_'+casename+'.npy')

dt = 0.000200
vel_thresh = 4.0
Velocities, Points = ipr3dptv.tracks_to_velocity_vectors(dt, XPF1, Tracks, part_ind_vec, vel_thresh)

# outlier detection
N_neigh, sigma_ol = 7, 1.5
Points, Velocities, outlier_vec = ipr3dptv.outlier_detection(Points, Velocities, N_neigh, sigma_ol)


filename = 'results/sparse_velocity_data_'+casename+'.vtk'
ipr3dptv.write_VTK_files(filename, Points, Velocities)

#------------------------------------------------------------
# INTERPOLATION
#------------------------------------------------------------


#------------------------------------------------------------
# PLOT RESULTS
#------------------------------------------------------------
"""
fig = plt.figure(1)

cmap = 'gray'
xlim, ylim = (500, 800), (1300, 600)

ax = fig.add_subplot(111, xlim=xlim, ylim=ylim)
ax.imshow(IresallF1[0], cmap, vmin=0, vmax=1000, interpolation='none')

plt.show()

plt.savefig('pix/proj_residual_'+casename+'.png')


fig = plt.figure(2)

cmap = 'gray'
xlim, ylim = (329, 1048), (1814, 272)

ax = fig.add_subplot(111, xlim=xlim, ylim=ylim)
ax.imshow(IresallF1[0], cmap, vmin=0, vmax=1000, interpolation='none')

plt.show()

plt.savefig('pix/proj_residual'+casename+'_full.png')
"""
