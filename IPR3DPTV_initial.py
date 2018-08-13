import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import sys

from scipy.linalg import norm

import IPR3DPTV_module as ipr3dptv

#------------------------------------------------------------
# LOAD FROM CALIBRATION
#------------------------------------------------------------
[scx, scy, h, w] = np.load('calibration/scales.npy')
h, w = int(h), int(w)

#------------------------------------------------------------
# READ IMAGES
#------------------------------------------------------------
flist = glob.glob('images/*.raww')

if len(flist)==1:

    print('\n'+flist[0]+' is used as input for IPR')

    # read double frame (particle image)
    I1, I2, I3, I4, I5, I6, I7, I8 = ipr3dptv.read_full_frame(flist[0], (h, w))

elif len(flist) > 1:
    sys.exit('\n More than one file of type .raww found in subfolder images.')

else:
    sys.exit('\n No file of type .raww found in subfolder images.')

#------------------------------------------------------------
# IMAGE PRE-PROCESSING
#------------------------------------------------------------

# threshold for back ground removal
plt.plot(I1[int(h/2), :], 'k')
plt.show()
trsh = input('Set the threshold intensity value for background removal: ')
plt.close()

# check if masks exist already and/or should be newly computed
flist = glob.glob('initial/masks.npy')

if len(flist) > 0:
    flag_new_masks = bool(input('\n There exist already masks for the current frame. Should masks be re-defined? yes [1] / no [0]: '))
else:
    flag_new_masks = True

# get graphical input to define a rectangular mask for each image
if flag_new_masks:

    cmap = 'gray'; plt.clf

    plt.imshow(I1, cmap, vmin=0, vmax=1000)
    m1 = plt.ginput(n=2, show_clicks=True)
    plt.show()
    plt.close()

    plt.imshow(I2, cmap, vmin=0, vmax=1000)
    m2 = plt.ginput(n=2, show_clicks=True)
    plt.show()
    plt.close()

    plt.imshow(I3, cmap, vmin=0, vmax=1000)
    m3 = plt.ginput(n=2, show_clicks=True)
    plt.show()
    plt.close()

    plt.imshow(I4, cmap, vmin=0, vmax=1000)
    m4 = plt.ginput(n=2, show_clicks=True)
    plt.show()
    plt.close()

    mask1 = np.round([m1[0][0], m1[1][0], m1[0][1], m1[1][1]])
    mask2 = np.round([m2[0][0], m2[1][0], m2[0][1], m2[1][1]])
    mask3 = np.round([m3[0][0], m3[1][0], m3[0][1], m3[1][1]])
    mask4 = np.round([m4[0][0], m4[1][0], m4[0][1], m4[1][1]])

    maskall = [mask1, mask2, mask3, mask4]
    np.save('initial/masks', maskall)

else:
    [mask1, mask2, mask3, mask4] = np.load('initial/masks.npy')

print('\n ...pre-processing (original) images')
# image background removal
I1 = ipr3dptv.image_preprocessing(I1, mask1, trsh)
I2 = ipr3dptv.image_preprocessing(I2, mask2, trsh)
I3 = ipr3dptv.image_preprocessing(I3, mask3, trsh)
I4 = ipr3dptv.image_preprocessing(I4, mask4, trsh)

I5 = ipr3dptv.image_preprocessing(I5, mask1, trsh)
I6 = ipr3dptv.image_preprocessing(I6, mask2, trsh)
I7 = ipr3dptv.image_preprocessing(I7, mask3, trsh)
I8 = ipr3dptv.image_preprocessing(I8, mask4, trsh)


# store
Iall = [I1, I2, I3, I4, I5, I6, I7, I8]
np.save('initial/original_images', Iall)
print('\n => pre-processed (original) images stored in initial subfolder')

#------------------------------------------------------------
# 2D PARTICLE IMAGE COORDINATES
#------------------------------------------------------------

print('\n ...finding (original) 2D particle peaks and OTF parameters')
plist1 = ipr3dptv.find_particle_2D_position(I1)
plist2 = ipr3dptv.find_particle_2D_position(I2)
plist3 = ipr3dptv.find_particle_2D_position(I3)
plist4 = ipr3dptv.find_particle_2D_position(I4)

plist5 = ipr3dptv.find_particle_2D_position(I5)
plist6 = ipr3dptv.find_particle_2D_position(I6)
plist7 = ipr3dptv.find_particle_2D_position(I7)
plist8 = ipr3dptv.find_particle_2D_position(I8)

# list of all particle lists. (lists have different size => lists of object)
plistall = [plist1, plist2, plist3, plist4, plist5, plist6, plist7, plist8]

# store
np.save('initial/original_particle_lists', plistall)
print('\n => (original) particle list stored in initial subfolder')
