import numpy as np

from numpy import pi, cos, sin, sqrt
from scipy.linalg import svd, block_diag, inv, norm, pinv, solve
from scipy.optimize import least_squares, leastsq, minimize
from scipy.ndimage import center_of_mass
from scipy.interpolate import LinearNDInterpolator

from math import log

import visit_writer as vw

########
# READ FULL FRAME RAW IMAGE
########

def read_full_frame(name, dim):

    # number of frames
    Nf = 8;

    # load image
    I = np.empty((dim[0]*dim[1]*Nf), np.uint16)
    I.data[:] = open(name).read()

    I = I.reshape([dim[0]*Nf, dim[1]])

    Iall = []

    for j in range(Nf):

        Iall.append(I[j*dim[0]:(j+1)*dim[0],:])

    I1 = Iall[0]
    I2 = Iall[4]
    I3 = Iall[2]
    I4 = Iall[6]
    I5 = Iall[1]
    I6 = Iall[5]
    I7 = Iall[3]
    I8 = Iall[7]

    return I1, I2, I3, I4, I5, I6, I7, I8

#---------------------------------------------------------------------------------

def read_single_image(name, dim, bit):

    # load image
    if bit == '16':
        type = np.uint16
    elif bit == '8':
        type = np.uint8
    else:
        print("choose '16' or '8' bit")
        return 0

    I = np.empty((dim[0]*dim[1]), type)
    I.data[:] = open(name).read()

    I = I.reshape([dim[0], dim[1]])

    return I


########
# GENERATE WORLD POINTS
########

def gen_world_points_random(N, limits):
    #------------------------------------
    # Description:
    #
    # Input:
    #
    # Output:
    #
    # Reference:
    #
    #------------------------------------

    # limits for random number
    w, h, d = limits[0], limits[1], limits[2]

    # random numbers
    R = 2*(np.random.rand(N, 3) - 0.5)

    R[:,0] = R[:,0]*w
    R[:,1] = R[:,1]*h
    R[:,2] = R[:,2]*d

    return R

#---------------------------------------------------------------------------------

def gen_world_points_box(dim):
    #------------------------------------
    # Description:
    # Computes the world coordinates of a cube with side length 'a' and with the
    # center in the middle.
    #
    # Input:
    #
    # N: number of boxes
    # dim: vector (or matrix) width side lengths of box (length in x, width in y, depth in z). If dim is a matrix (numpy array) then each row defines the dimensions of a box
    #
    # Output:
    # B: Matrix with (8x3) 8 edge coordinate points
    #
    # Reference:
    #
    #------------------------------------

    m, n = dim.shape

    # limits for random number
    B = np.zeros([m*8,3])
    for i in range(m):
        w = dim[i,0]
        h = dim[i,1]
        d = dim[i,2]

        k = i*8

        B[k+0,0] = w/2.; B[k+0,1] = h/2.; B[k+0,2] = d/2.
        B[k+1,0] =-w/2.; B[k+1,1] = h/2.; B[k+1,2] = d/2.
        B[k+2,0] =-w/2.; B[k+2,1] =-h/2.; B[k+2,2] = d/2.
        B[k+3,0] = w/2.; B[k+3,1] =-h/2.; B[k+3,2] = d/2.
        B[k+4,0] = w/2.; B[k+4,1] = h/2.; B[k+4,2] =-d/2.
        B[k+5,0] =-w/2.; B[k+5,1] = h/2.; B[k+5,2] =-d/2.
        B[k+6,0] =-w/2.; B[k+6,1] =-h/2.; B[k+6,2] =-d/2.
        B[k+7,0] = w/2.; B[k+7,1] =-h/2.; B[k+7,2] =-d/2.

    return B

def gen_world_points_calibration_plate():
    #------------------------------------
    # Description:
    #
    # Input:
    #
    # Output:
    #
    # Reference:
    #
    #------------------------------------

    C1 = np.zeros([11**2,3]);
    for k in range(-5,6):
        for l in range(-5,6):
            C1[(k+5)*11 + (l+5), 0] = k*5
            C1[(k+5)*11 + (l+5), 1] = l*5
            C1[(k+5)*11 + (l+5), 2] = 2.9


    C2 = np.zeros([10**2,3]);
    for k in range(-5,5):
        for l in range(-5,5):
            C2[(k+5)*10 + (l+5), 0] = (k+0.5)*5
            C2[(k+5)*10 + (l+5), 1] = (l+0.5)*5
            C2[(k+5)*10 + (l+5), 2] = 1.9

    C = np.concatenate([C1, C2], axis=0);

    return C


def gen_world_points_cylinder(radius, height, height_offset):
    #------------------------------------
    # Description:
    #
    # Input:
    #
    # Output:
    #
    # Reference:
    #
    #------------------------------------

    Nr = 8
    Nh = 7
    NZ = len(radius)

    Z = []
    for i in range(NZ):

        Zi = np.zeros([Nr*Nh, 3])
        for l in range(Nh):
            for k in range(Nr):

                theta = k*2*pi/Nr

                x = radius[i] * cos(theta)
                y = height[i] * ((l+1)/float(Nh) - 0.5) + height_offset[i]
                z = radius[i] * sin(theta)

                Zi[l*Nr + k, 0] = x
                Zi[l*Nr + k, 1] = y
                Zi[l*Nr + k, 2] = z

        Z.append(Zi)


    return np.concatenate(Z, axis=0);

########
# MAPPING
########

# world2image polynomial
def world2image_poly3(P, cam, cal):
    #------------------------------------
    # Description:
    # Maps the the coordinates of a 3D space point to the image plane based on 3rd order polynomial functions for different Z-planes. In Z-direction the coordinates are interpolated.
    #
    # Input:
    # X: 3D space points in mm with origin in center (as given by calibration plate)
    # cam: Camera index 1,2,3,4
    # cal: Calibration (e.g. 'orig', 'last', 'lastlast')
    #------------------------------------

    dims = P.shape

    # for general treatment, adjust if dimensions is 1
    if len(dims)==1:
        P = np.array([P]).T
        Npts = 1

    else:
        Npts = dims[1]

    X = np.zeros_like(P)

    # transform coordinates (only X and Y) to pixel coordinates and add offset
    [scx, scy, h, w] = np.load('calibration/scales.npy')
    h, w = int(h), int(w)
    cal = str(np.load('calibration/name.npy'))

    if cal == 'lastlast':
        OXdw = 506.83
        OYdw = 985.50
    else:
        print('Calibration name not known.')
        return 0

    X[0,:], X[1,:], X[2,:] = P[0,:]/scx + OXdw, P[1,:]/scy + OYdw, P[2,:];

    # Z-coordinates of support points. (depth of plane for given calibration)
    if cal == 'lastlast':
        ZA = -9.0;
        ZB =  9.0;
    else:
        print('Calibration name not known.')
        return 0

    # allocate
    xA = np.zeros([2, Npts]);
    xB = np.zeros([2, Npts]);
    x  = np.zeros([2, Npts]);

    # support point 1
    cc = np.load('calibration/calibration_coefficients_'+cal+'.npy')
    ccii = cc[(cam-1)*2]
    dx = ccii[0]; dy = ccii[1]
    cx = ccii[2]; cy = ccii[3]

    for i in range(Npts):

        s = 2*(X[0,i] - dx[0])/dx[1]
        t = 2*(X[1,i] - dy[0])/dy[1]

        Dx = cx[0] \
           + cx[1]*s   + cx[2]*s**2   + cx[3]*s**3 \
           + cx[4]*t   + cx[5]*t**2   + cx[6]*t**3 \
           + cx[7]*s*t + cx[8]*s**2*t + cx[9]*s*t**2

        Dy = cy[0] \
           + cy[1]*s   + cy[2]*s**2   + cy[3]*s**3 \
           + cy[4]*t   + cy[5]*t**2   + cy[6]*t**3 \
           + cy[7]*s*t + cy[8]*s**2*t + cy[9]*s*t**2

        xA[0,i] = X[0,i] - Dx
        xA[1,i] = X[1,i] - Dy

    # support point 2
    cc = np.load('calibration/calibration_coefficients_'+cal+'.npy')
    ccii = cc[(cam-1)*2 + 1]

    dx = ccii[0]; dy = ccii[1];
    cx = ccii[2]; cy = ccii[3];

    for i in range(Npts):

        s = 2*(X[0,i] - dx[0])/dx[1];
        t = 2*(X[1,i] - dy[0])/dy[1];


        Dx = cx[0] \
           + cx[1]*s   + cx[2]*s**2   + cx[3]*s**3 \
           + cx[4]*t   + cx[5]*t**2   + cx[6]*t**3 \
           + cx[7]*s*t + cx[8]*s**2*t + cx[9]*s*t**2

        Dy = cy[0] \
           + cy[1]*s   + cy[2]*s**2   + cy[3]*s**3 \
           + cy[4]*t   + cy[5]*t**2   + cy[6]*t**3 \
           + cy[7]*s*t + cy[8]*s**2*t + cy[9]*s*t**2

        xB[0,i] = X[0,i] - Dx
        xB[1,i] = X[1,i] - Dy


    # linear interpolation
    for i in range(Npts):
        x[0,i] = xA[0,i] + (X[2,i] - ZA)/(ZB - ZA)*(xB[0,i] - xA[0,i]);
        x[1,i] = xA[1,i] + (X[2,i] - ZA)/(ZB - ZA)*(xB[1,i] - xA[1,i]);

    return x

# world2image pinhole
# --> see camera_matrix

# corrected image reference frame

# homogeneous coordinates
def homogeneous_coordinates(x):
    #------------------------------------
    # Description:
    # Simply transforms a 2-space or 3-space vector into its corresponding normalized homogeneous coordinate.
    #
    # Input:
    # x: array of 2-space (2xn) or 3-space (3xn) vector, where n is the number of vectors to be transformed
    #
    # Output:
    # X: array of 2-space (3xn) or 3-space (4xn) homogeneous vector.
    #------------------------------------

    # shape
    n, m = x.shape

    # append ones
    x = np.append(x, np.ones([1,m]), axis=0)

    return x



########
# SIMILARITY TRANSFORMATION
########
def similarity_transform(shp, alpha=0):
    #------------------------------------
    # Description:
    # Transformation of image coordinates to normalized image coordinates (homogeneous coordinates)
    #
    # Input:
    # shp: shape of the image: width (w) and height (h)
    #
    # Outut:
    # T: (3x3) transformation matrix.
    #
    # Reference:
    # p. 104, Hartley & Zisserman, 2003, 'Multiple View Geometry in Computer Vision'
    #------------------------------------

    # dimensions
    (h, w) = shp

    # translation
    t = np.array([np.floor(w/2), np.floor(h/2)])

    # rotation
    R = np.array([[cos(alpha), sin(alpha)],[-sin(alpha), cos(alpha)]])

    # scaling
    s = 4./(w + h)

    # transformation
    T = np.zeros([3,3])
    T[0:2,0:2] = s*R
    T[0:2,2]   = -s*t
    T[2,2]     = 1

    return T

def similarity_transform_world(cub_dim):
    #------------------------------------
    # Description:
    # Transformation of world coordinates to normalized image coordinates (homogeneous coordinates)
    #
    # Input:
    # cub_dim: (3) typical dimensions of measurement volume
    #
    # Outut:
    # T: (4x4) transformation matrix.
    #
    # Reference:
    # p. 104, Hartley & Zisserman, 2003, 'Multiple View Geometry in Computer Vision'
    #------------------------------------

    # dimensions
    h = cub_dim[0]
    w = cub_dim[1]
    d = cub_dim[2]

    # translation: t = 0 if world reference frame is in the center of the mesurement volume
    t = np.zeros([1,3])

    # rotation
    R = np.eye(3)

    # scaling
    s = 6./(w + h + d)

    # transformation
    U = np.zeros([4,4])
    U[0:3,0:3] = s*R
    U[0:3,3]   = -s*t
    U[3,3]     = 1

    return U


########
# CAMERA MATRIX
########
def camera_matrix(x, X, T, U):
    #------------------------------------
    # Description:
    # Computes the camera matrix P (x = PX) from n >= 6 world to image
    # correspondances {x_i <--> X_i} for i = 1,2,...,n.
    #
    # Input:
    # X: (4 x n) world points (3-space) in homogeneous coordinates.
    # x: (3 x n) corresponding image points in homogeneous coordinates.
    # T: (3 x 3) similarty transformation matrix for image points
    # U: (4 x 4) similarty transformation matrix for world points
    #
    # Output:
    # P: camera matrix (3x4) such that x = PX
    #
    # Reference:
    # - p. 181, Hartley & Zisserman, 2003, 'Multiple View Geometry in computer
    #   vision'
    #
    #------------------------------------

    m, n = x.shape
    # store
    xx = np.copy(x); XX = np.copy(X);

    # normalization
    x = np.dot(T,x)
    X = np.dot(U,X)

    # direct linear transformation
    A = []
    z14 = np.zeros(4)
    for i in range(n):

        # for sub-matrix A (Hartley & Zisserman, equ. 7.2)
        Xi = X[:,i]
        xi, yi, wi = x[0,i], x[1,i], x[2,i]
        A.append(list(np.concatenate([z14, -wi*Xi, yi*Xi])))
        A.append(list(np.concatenate([wi*Xi, z14, -xi*Xi])))


    A = np.array(A)

    Usvd, s, VH = svd(A)
    p0 = VH.T[:,-1];

    # minimize geometric error
    [p, resnorm] = leastsq(geometric_error, p0, args = (x, X));

    # camera matrix
    P = p.reshape([3,4]);

    # denormalize
    P = np.dot(inv(T), np.dot(P, U));
    p_vec = P.reshape([1, 12]);

    # error
    #err = norm(geometric_error(p, x, X));
    ge = geometric_error(p_vec, xx, XX);
    err = norm(ge);

    return P, err


#--------------------------------------------------------------------------

def geometric_error(p, x, X):

    # camera matrix
    P = p.reshape([3, 4])

    # projected image coordinates
    x_bp = np.dot(P,X)

    # geometric error
    xi, yi, wi = x[0,:], x[1,:], x[2,:]
    xi_bp, yi_bp, wi_bp = x_bp[0,:], x_bp[1,:], x_bp[2,:]

    return np.sqrt((xi/wi - xi_bp/wi_bp)**2 + (yi/wi - yi_bp/wi_bp)**2)


# Compare calibration functions
def compare_polynomial_and_pinhole(x, P, cam, cal):
    #------------------------------------
    # Description:
    # Backprojection of x onro ray X(lambda). Points on ray are mapped back on
    # image with polynomial calibration (world2image_poly3). The difference between the
    # point x and the mapped world point on the ray is a measure for the
    # difference between the polynomial and the pinhole calibration as a
    # functin of the ray parameter 'lambda'
    #
    # Input:
    # x: (3 x n) image points in homogeneous coordinates.
    # P: (3 x 4) camera matrix
    # cam: Camera index
    # cal: Calibration (e.g. 'orig')
    #
    # Output:
    # d: geometric difference d = d(lambda)
    # Dist: Norm of 3-space point on ray
    #
    # Reference:
    #
    #------------------------------------


    # pseudo-inverse of camera matrix
    Ppsi = pinv(P)

    # camera center
    C = nullspace(P)
    C = C/C[3]

    # camera center finite camera
    M = P[:,0:3]
    p4 = P[:,3]

    C_fc = solve(-M, p4)

    # parameter domain
    N = 100
    lAmbda = np.linspace(0,0.015,N);
    mu     = np.linspace(-0.958,-0.946,N);

    # ray
    diff, diff_fc, dist, dist_fc = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    for i in range(N):

        # 3-space point on ray
        X_bar = np.dot(Ppsi,x[:,0])
        X_bar = X_bar/X_bar[3]

        X = X_bar + lAmbda[i]*C
        X = X/X[3]
        X = np.array([X]).T

        dist[i] = norm(X[0:3])

        # 3-space point on ray finite camera
        X_fc = solve(M, -mu[i]*x[:,0]-p4)
        X_fc = np.append(X_fc, 1.)
        X_fc = np.array([X_fc]).T

        dist_fc[i] = norm(X_fc[0:3])

        # mapping
        x_m3 = world2image_poly3(X, cam, cal)

        # mapping finite camera
        x_m3_fc = world2image_poly3(X_fc, cam, cal)

        # difference
        diff[i] = sqrt((x[0,0] - x_m3[0,0])**2 + (x[1,0] - x_m3[1,0])**2)

        # difference finite camera
        diff_fc[i] = sqrt((x[0,0] - x_m3_fc[0,0])**2 + (x[1,0] - x_m3_fc[1,0])**2)

    return diff, dist, diff_fc, dist_fc





########
# FUNDAMENTAL MATRIX
########
def fundamental_matrix_normalized8point(xA, xB, TA, TB):
    #------------------------------------
    # Description:
    # Calculates the fundamental matrix for a set of n >= 8 point
    # correspondances using the normalized 8-point algorithm.
    #
    # Input:
    # xA, xB: Matrix 3 x n with each n corresponding point in homogeneous
    #           coordinates.
    # TA, TB: similarty transformation matrix for image A and B
    #
    # Output:
    # F: Fundamental matrix (3x3) for which det(F) = 0 (singular).
    #
    # Reference:
    # - p. 282, Hartley & Zisserman, 2003, 'Multiple View Geometry in computer
    #   vision'
    #
    #------------------------------------

    # allocate
    m, n = xA.shape

    # Normalize coordinates
    xA_prime = np.dot(TA, xA)
    xB_prime = np.dot(TB, xB)

    # build homogenous linear equation matrix A (n x 9)
    A = np.zeros([n,9]);
    for i in range(n):
        x1 = xA_prime[0,i]
        y1 = xA_prime[1,i]
        w1 = xA_prime[2,i]

        x2 = xB_prime[0,i]
        y2 = xB_prime[1,i]
        w2 = xB_prime[2,i]

        A[i,:] = np.array([x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1.])

    # least square solution
    U, s, VH = svd(A)
    f = VH.T[:,-1]

    F = np.reshape(f,[3,3])

    # singularity constraint enforcement
    U, s, VH = svd(F);
    s[2] = 0;
    S = np.diag(s)
    F = np.dot(U, np.dot(S,VH))

    # denormalization
    F = np.dot(TB.T, np.dot(F,TA))

    return F



def compute_flattend_fundamental_matrix_for_given_nullspace(e, A):

    # build epipole block diagonal matrix
    ex = np.array([[ 0,    -e[2],  e[1]],
                   [ e[2],  0,    -e[0]],
                   [-e[1],  e[0],  0]])

    E = block_diag(ex, ex, ex)
    r = np.linalg.matrix_rank(E)

    # find f = E*m that minimizes ||A*f|| subject to ||E*m||=1
    U, s, VH = svd(E)
    U_prime = U[:,:r]

    U, s, VH = svd(np.dot(A,U_prime))
    f_prime = (VH.T)[:,-1]

    # flattened fundamental matrix in row major order
    f = np.dot(U_prime, f_prime)

    return  f

def compute_algebraic_error(e, *args):

    A = np.array([args[0]])
    for arg in args[1:]:
        A = np.append(A, np.array([arg]), axis=0)

    # flattend fundamental matrix
    f = compute_flattend_fundamental_matrix_for_given_nullspace(e, A)

    # algebraic error
    epsilon_vec = np.dot(A,f)

    return epsilon_vec

def compute_algebraic_error_II(e, A):


    # flattend fundamental matrix
    f = compute_flattend_fundamental_matrix_for_given_nullspace(e, A)

    # algebraic error
    epsilon_vec = np.dot(A,f)

    return epsilon_vec

def fundamental_matrix_minAlgebraicError(xA, xB, TA, TB):
    # Description:
    # Calculates the fundamental matrix (for a set of n >= 8 point correspondances) that minimizes the algebraic error ||A*f|| subject to
    # ||f|| = 1 and det(F) = 0,
    #
    # Input:
    # xA, xB: Matrix 3 x n with each n corresponding point in homogeneous
    #           coordinates.
    # TA, TB: similarty transformation matrix for image A and B
    #
    # Output:
    # F: Fundamental matrix (3x3) (singular)
    #
    #
    # Reference:
    # - p. 284, Hartley & Zisserman, 2003, 'Multiple View Geometry in computer
    #   vision'
    #


    # find a first good guess of the fundamental matrix using the normalized
    # 8-point algorithm
    F = fundamental_matrix_normalized8point(xA, xB, TA, TB)

    # build homogenous linear equation matrix A (n x 9)
    m, n = xA.shape
    A = np.zeros([n,9])
    for i in range(n):
        x1 = xA[0,i]
        y1 = xA[1,i]
        w1 = xA[2,i]

        x2 = xB[0,i]
        y2 = xB[1,i]
        w2 = xB[2,i]

        A[i,:] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]

    # initial epipole
    e0 = nullspace(F);

    # minimize algebraic error
    e = least_squares(compute_algebraic_error, e0, method='lm', args=(A))

    # reshape fundamental matrix
    f = compute_flattend_fundamental_matrix_for_given_nullspace(e.x, A);

    # compute algebraic error
    eps = compute_algebraic_error_II(e.x, A);
    eps0 = compute_algebraic_error_II(e0, A);

    rn = np.linalg.norm(eps)/np.linalg.norm(eps0)

    F = np.reshape(f,[3,3])

    return F



########
# FIND CANDIDATES
########
def candidates(F, pA, plistB, dlim):
    # Description:
    # Searches the particles that lie within a orthogonal distance of d_Lim to
    # the epipolar line corresponding to particle p = (px, py) and defined by
    # the fundamental matrix F.
    #
    # Input:
    # F:     fundamental matrix between image A and B. Image A contains the particle
    #        p and image B on which candidates for the corresponding particle
    #        are located.
    # pA:   2D position vector of a particle located on image A (search
                                                                  #        point).
    # plistB: list of all particles in image B
    # dlim: maximal normal distance of a candidate particle in image B to the
    #        epipolar line corresponding to particle p (2x1)
    #
    #
    # Output:
    # clist: list of candidate particles
    #
    # Reference:
    #

    # epipolar line parameters
    epil = np.dot(F,pA)
    A, B, C = epil[0], epil[1], epil[2];

    # inclination of epipolar line
    alpha = np.arctan(-A/B)

    # non-orthogonal distance
    d_no = dlim/np.cos(alpha)

    # particle coordinates
    px, py = plistB[:,0], plistB[:,1]

    # upper and lower bound
    p_uy = -C/B -A/B*px - d_no
    p_ly = -C/B -A/B*px + d_no

    # candidate list
    mask = (py > p_uy) & (py < p_ly)

    if np.any(mask):
        clist = plistB[mask,:]
        found = 1
    else:
        found = 0
        clist = np.array([[],[]])

    return clist, found



########
# EPIPOLAR LINE
########
def epipolar_line(xA, FAB, Idim):

    # Description:
    # Computes two points at the left and right border of an image with dimensions 'dim'. The line connecting the two points is the epipolar line on image B for point xA in image A.
    #
    # Input:
    # xA: (3) image point in image A in homogeneous coordinates
    # FAB: fundamental matrix xB^T*FAB*xA = 0
    # Idim: dimension of image B
    #
    # Output:
    # elx, ely: x and y coordinates of the points defining the epipolar line


    # line coefficients A*x + B*y + C = 0
    el = np.dot(FAB, xA)
    A, B, C = el[0], el[1], el[2]

    # point coordiantes
    elx, ely = [0, Idim[1]], [-C/B, -C/B-A/B*Idim[1]]

    return elx, ely




########
# GAUSS FIT 2D
########

# find particle 2D position coordinate in image
def find_particle_2D_position(I):
    #------------------------------------
    # Description:
    #
    # Input:
    # I: Pre-processed* image (intesity array).
    #    * background subtraction
    #
    # Output:
    # plist: list with sub-pixel particle position
    #
    # Reference:
    #
    #------------------------------------

    # size of image
    h, w = I.shape

    # initialize particle list
    plist = []

    # loop over each particle
    for j in range(2,h-2):
        for i in range(2,w-2):

            # check for intensity
            if I[j,i] < 1:
                continue

            # 5x5 neighborhood
            nbh55 = np.zeros([5,5])
            for jj in range(-2,3):
                for ii in range(-2,3):

                    # make pixels with equal intensity distinguishable
                    #if I[i,j] == I[j+jj, i+ii]:
                    #      I[j+jj, i+ii] = I[j+jj, i+ii]-1

                    nbh55[jj+2, ii+2] = I[j+jj, i+ii]

            # find particle based on maximum-in-3x3-neighborhood criterium
            nbh33 = nbh55[1:4,1:4]

            # only consider peaks if pixel is maximum in neighborhood and at least least 3 other
            # neighboring pixel have intensity above 'thrsh_nbh'
            thrsh_nbh = 100.
            if (I[j,i] == nbh33.max()) & (nbh33[nbh33>thrsh_nbh].size >= 4):

                # define neighborhood for 2D Gauss-fit
                inner_nbh = nbh33.copy(); inner_nbh[1,1] = 0
                outer_nbh = nbh55.copy(); outer_nbh[1:4,1:4] = 0
                if inner_nbh.max() > outer_nbh.max():

                    r = 2
                    nbh = nbh55

                    # 2D Gauss-fit
                    x = np.arange(i-r,i+r+1)
                    y = np.arange(j-r,j+r+1)

                    # particle properties from fit: peak position (x, y), peak height, peak width (wx, wy)
                    part_prop = fit_2D_Gaussian(x, y, nbh, I[j,i])

                else:
                    # particle properties from fit: peak position (x, y), peak height, peak width (wx, wy)
                    part_prop = [i, j, I[j,i], 1.0, 1.0]


                # extend list if gaussian fit was found
                if len(part_prop)==5:
                    plist.append(part_prop)

            else:
                continue

    return np.array(plist)


# 2D Gaussian
def func_2D_Gaussian(p, X, Y):
    #------------------------------------
    # Description:
    # Returns a two-dimensional Gaussian function for given parameters
    #
    # Reference:
    #
    #------------------------------------
    return p[0]*np.exp( -(X - p[1])**2/(2*p[2]**2) -(Y - p[3])**2/(2*p[4]**2))


# error function
def error_func_2D_Gaussian(p, X, Y, F):

    return np.ravel(func_2D_Gaussian(p, X, Y) - F)

# 2D Gaussian fit
def fit_2D_Gaussian(x, y, F, I0):
    #------------------------------------
    # Description:
    # Returns the parameters of the 2D Gaussian function that best fit the sampled data.
    #
    # Reference:
    #
    #------------------------------------

    # sample points
    X, Y = np.meshgrid(x, y)

    # initial parameter vector
    x0, y0 = x.mean(), y.mean()
    wx0, wy0 = 1, 1
    p0 = np.array([I0, x0, wx0, y0, wy0])

    # non-linear least-square fit
    p = least_squares(error_func_2D_Gaussian, p0, args=(X, Y, F), method='lm')

    if p.success:
        return [p.x[1], p.x[3], p.x[0], p.x[2], p.x[4]]
    else:
        return []


def find_particle_2D_position_COM(I):
    #------------------------------------
    # Description:
    # Computes the centroid of a particle image based on its center of mass (COM)
    #
    # Input:
    # I: Pre-processed* image (intesity array).
    #    * background subtraction
    #
    # Output:
    # plist: list with sub-pixel particle position
    #
    # Reference:
    #
    #------------------------------------

    # size of image
    h, w = I.shape

    # initialize particle list
    plist = []

    # loop over each particle
    for j in range(2,h-2):
        for i in range(2,w-2):

            # check for intensity
            if I[j,i] < 1:
                continue

            # 5x5 neighborhood
            nbh55 = np.zeros([5,5])
            for jj in range(-2,3):
                for ii in range(-2,3):

                    # make pixels with equal intensity distinguishable
                    #if I[i,j] == I[j+jj, i+ii]:
                    #      I[j+jj, i+ii] = I[j+jj, i+ii]-1

                    nbh55[jj+2, ii+2] = I[j+jj, i+ii]

            # find particle based on maximum-in-3x3-neighborhood criterium
            nbh33 = nbh55[1:4,1:4]

            # only consider peaks if pixel is maximum in neighborhood and at least least 3 other
            # neighboring pixel have intensity above 'thrsh_nbh'
            thrsh_nbh = 100.
            if (I[j,i] == nbh33.max()) & (nbh33[nbh33>thrsh_nbh].size >= 4):

                # define neighborhood for 2D Gauss-fit
                inner_nbh = nbh33.copy(); inner_nbh[1,1] = 0
                outer_nbh = nbh55.copy(); outer_nbh[1:4,1:4] = 0
                if inner_nbh.max() > outer_nbh.max():
                    r = 2
                    nbh = nbh55
                else:
                    r = 1
                    nbh = nbh33

                # 2D Gauss-fit
                x = np.arange(i-r,i+r+1)
                y = np.arange(j-r,j+r+1)

                # center of mass
                xsItot, ysItot, Itot = 0, 0, 0
                jj = 0
                for yi in y:
                    ii = 0
                    for xi in x:
                        xsItot += xi*nbh[jj,ii]
                        ysItot += yi*nbh[jj,ii]
                        Itot   += nbh[jj,ii]
                        ii += 1
                    jj += 1

                xs = xsItot/Itot
                ys = ysItot/Itot

                part_prop = [xs, ys, I[j,i], 1.0, 1.0]

                # extend list if gaussian fit was found
                if len(part_prop)==5:
                    plist.append(part_prop)

            else:
                continue

    return np.array(plist)


########
# IMAGE PRE-PROCESS
########
def image_preprocessing(I, mask, thrsh):
    #------------------------------------
    # Description:
    #
    # Input:
    # I: image (intesity array);
    # para.mask: border for black-maksing (mask(1) = x_min, mask(2) = x_max, mask(3) = y_max, mask(4) = y_max)
    # para.thrsh: threshold for background subtraction
    #
    # Output:
    # I_proc: processed image
    #
    # Reference:
    #
    #------------------------------------

    I_proc = np.copy(I)

    # size of image
    h, w = I_proc.shape;

    # index array
    ii, jj = np.arange(w), np.arange(h)
    II, JJ = np.meshgrid(ii, jj)

    # black-mask image
    x_min, x_max = mask[0], mask[1]
    y_min, y_max = mask[2], mask[3]

    m = ((II < x_min) | (II > x_max)) | ((JJ < y_min) | (JJ > y_max));

    I_proc[m] = 0.;

    # background removal
    m = I_proc < thrsh
    I_proc[m] = 0.

    return I_proc


def residual_image_to_image(I, thrsh):
    #------------------------------------
    # Description:
    #
    # Input:
    # I: residual image (intesity array);
    # para.mask: border for black-maksing (mask(1) = x_min, mask(2) = x_max, mask(3) = y_max, mask(4) = y_max)
    # para.thrsh: threshold for background subtraction
    #
    # Output:
    # I_proc: processed image
    #
    # Reference:
    #
    #------------------------------------

    I_proc = np.copy(I)

    # background removal
    m = I_proc < thrsh
    I_proc[m] = 0.

    return I_proc


########
# TRIANGULATION
########

# triangulation - homogeneous method

def triangulate_homogeneous_method(xA, xB, PA, PB):
    #------------------------------------
    # Description:
    # Simple linear triangulation of space point X based on its images xA,
    # xB.
    #
    # Input:
    # xA, xB: (3x1) image points in homogeneous coordinates on camera A & B
    # PA, PB: (3x4) camera matrices for camera A and &
    #
    # Output:
    # X: corresponding 3D-space point in homogeneous coordinates (normalized,
                                                                  #    X(4) = 1)
    #
    # Reference:
    # - p. 312, Hartley & Zisserman, 2003, 'Multiple View Geometry in computer
    #   vision'
    #
    #------------------------------------

    # write system matrix
    A = np.array([xA[0]*PA[2,:] - PA[0,:], \
                  xA[1]*PA[2,:] - PA[1,:], \
                  xB[0]*PB[2,:] - PB[0,:], \
                  xB[1]*PB[2,:] - PB[1,:]])



    # DLT
    U, s, VH = svd(A)
    X = VH.T[:,-1]

    # normalize
    X = X/X[3];

    return X



# triangulate optima method
def triangulate_optimal_method(xA, xB, FAB, PA, PB):
    #------------------------------------
    # Description:
    # Optimal triangulation of space point X based on its images xA,
    # xB.
    #
    # Input:
    # xA, xB: (3x1) image points in homogeneous coordinates on camera A & B
    # F: (3x3) fundamental matrix A & B
    # PA, PB: (3x4) camera matrices for camera A & B
    #
    # Output:
    # X: corresponding 3D-space point in homogeneous coordinates (normalized,
                                                                   #    X(4) = 1)
    #
    # Reference:
    # - p. 318, Hartley & Zisserman, 2003, 'Multiple View Geometry in computer
    #   vision'
    #
    #------------------------------------

    # transformation
    TA = np.array([[1, 0, -xA[0]], \
                   [0, 1, -xA[1]], \
                   [0, 0,  1]])

    TB = np.array([[1, 0, -xB[0]], \
                   [0, 1, -xB[1]], \
                   [0, 0,  1]])

    invTB = inv(TB)
    invTA = inv(TA)
    F = np.dot(invTB.T, np.dot(FAB, invTA))

    # right and left epipoles (normalized)
    eA = nullspace(F)
    eA = eA/(eA[0]**2 + eA[1]**2)

    eB = nullspace(F.T)
    eB = eB/(eB[0]**2 + eB[1]**2)

    # rotation
    RA = np.array([[ eA[0], eA[1], 0], \
                   [-eA[1], eA[0], 0], \
                   [ 0,      0,      1]])

    RB = np.array([[ eB[0], eB[1], 0], \
                   [-eB[1], eB[0], 0], \
                   [ 0,      0,      1]])


    F =  np.dot(RB, np.dot(F, RA.T))

    # find roots polynomial g(t)
    # coefficients
    a, b, c, d = F[1,1], F[1,2], F[2,1], F[2,2]
    fA, fB = eA[2], eB[2]

    p = np.zeros(7)
    p[0] = a * b * c**2 * fA**4 - a**2 * c * d * fA**4
    p[1] = a**4 + b**2 * c**2 * fA**4 - a**2 * d**2 * fA**4 + 2 * a**2 * c**2 * fB**2 + c**4 * fB**4
    p[2] = 4 * a**3 * b + 2 * a * b * c**2 * fA**2 - 2 * a**2 * c * d * fA**2 + b**2 * c * d * fA**4 \
         - a * b * d**2 * fA**4 + 4 * a * b * c**2 * fB**2 + 4 * a**2 * c * d * fB**2 + 4*c**3 * d * fB**4
    p[3] = 6 * a**2 * b**2 + 2 * b**2 * c**2 * fA**2 - 2 * a**2 * d**2 * fA**2 + 2 * b**2 * c**2 * fB**2 \
         + 8 * a * b * c * d * fB**2 + 2 * a**2 * d**2 * fB**2 + 6 * c**2 * d**2 * fB**4
    p[4] = 4 * a * b**3 + a * b * c**2 - a**2 * c * d + 2 * b**2 * c * d * fA**2 - 2 * a * b * d**2 * fA**2 \
         + 4 * b**2 * c * d * fB**2 + 4 * a * b * d**2 * fB**2 + 4 * c * d**3 * fB*4
    p[5] = b**4 + b**2 * c**2 - a**2 * d**2 + 2 * b**2 * d*2 * fB**2 + d**4 * fB**4
    p[6] = b**2 * c * d - a * b * d**2

    r = np.roots(p)

    # evaluate cost function (at real parts of roots)
    t = r.real
    s = t**2 / (1. + fA**2 * t**2) + (c*t + d)**2 / ((a*t + b)**2 + fB**2 * (c*t + d)**2);

    # asymptotic value
    s_asym = (a**2 + c**2 * (fA**2 + fB**2))/(fA**2 * (a**2 + c**2 * fB**2))
    s = np.append(s, s_asym)

    # choose minimum
    smin_ind = s.argmin() # !!! check min and index of min in numpy !!!
    tmin = t[smin_ind]

    # evaluate lines
    lambdaA = tmin*fA
    muA = 1
    nuA = -tmin

    lambdaB = -fB*(c*tmin + d)
    muB = a*tmin + b
    nuB = c*tmin + d

    # closest point on line
    xoptA = np.zeros(3)
    xoptA[0] = -lambdaA*nuA
    xoptA[1] = -muA*nuA
    xoptA[2] = lambdaA**2 + muA**2

    xoptB = np.zeros(3)
    xoptB[0] = -lambdaB*nuB
    xoptB[1] = -muB*nuB
    xoptB[2] = lambdaB**2 + muB**2

    # back-transformation
    xoptA = np.dot(inv(TA), np.dot(RA.T, xoptA))
    xoptB = np.dot(inv(TB), np.dot(RB.T, xoptB))

    # normalize
    xoptA = xoptA/xoptA[2]
    xoptB = xoptB/xoptB[2]

    norm(xA - xoptA)
    norm(xB - xoptB)

    # homogeneous method (DLT)
    X = triangulate_homogeneous_method(xoptA, xoptB, PA, PB)

    return X


def triangulate_particles_3cameras(rlim_pxl, ind_fst, plistallF, dlim_pxl=1):
    #------------------------------------
    # Description:
    # Determine particle correspndance based on maximal error between triangulated 3-space coordinates
    # from different candidate pairs on different cameras. A total of 4 cameras and hence 4 lists of particle
    # lists is at hand.
    #
    # Input:
    # rlim_pxl: Allowed triangulation error in pixel units
    # ind_fst: Index of the bases first camera (base camera) 1,2,3 or 4 (non-zero indexing)
    # plistallF: List of particle lists of one frame - total of 4 particle lists.
    #
    # Output:
    #
    # Reference:
    # - p. 318, Hartley & Zisserman, 2003, 'Multiple View Geometry in computer
    #   vision'
    #
    #------------------------------------

    ind_not = 2 # this is fix

    #--------------------------
    # PARTICLE LIST
    #--------------------------

    # relation: camera order to camera indices.
    # order of cameras to calcualte candidates: A is the base camera (ind_fst), B, C, D correspond to the remaining
    ol = range(4)
    ol.remove(ind_fst-1)
    ol.remove(ind_not-1)
    ol.insert(0, ind_fst-1)

    # relation: camera indices to camera order (used later to assign OTF parameters to specific camera)
    al = range(1,3)
    al.insert(ind_fst-1,0)

    #--------------------------
    # PARTICLE LIST
    #--------------------------
    [plistA, plistB, plistC] = plistallF[ol]

    #--------------------------
    # CALIBRATION
    #--------------------------
    # calibration parameters
    [scx, scy, h, w] = np.load('calibration/scales.npy')
    h, w = int(h), int(w)
    cal = str(np.load('calibration/name.npy'))

    # camera matrices: PA, PB, PC, PD
    Pall = np.load('calibration/camera_matrices.npy')
    [PA, PB, PC] = Pall[ol,:,:]

    # fundamental matrix: FAB, FAC, FAD
    Fall = np.load('calibration/fundamental_matrices.npy')
    FAB = Fall[ol[0], ol[1], :, :]
    FAC = Fall[ol[0], ol[2], :, :]

    #--------------------------
    # FUTHER PARAMETERS
    #--------------------------

    # candicate search distance to epipolar line
    rlim = rlim_pxl*scx

    #--------------------------
    # TRIANGULATION
    #--------------------------
    npA = plistA[:,0].size

    # allocate
    XP = []
    xp1, xp2, xp3, xp4 = np.array([[],[]]).T, np.array([[],[]]).T, np.array([[],[]]).T, np.array([[],[]]).T
    xb1, xb2, xb3, xb4 = np.array([[],[]]).T, np.array([[],[]]).T, np.array([[],[]]).T, np.array([[],[]]).T
    Matches = np.zeros(npA)


    # loop over particles in first image
    for k in range(npA):

        if k%1000 == 0:
            print('TRI: particle k = '+str(k)+' of '+str(npA))

        # current paricle
        pkh = np.array([np.append(plistA[k,:2], 1)]).T

        # compute candidate lists for current particle
        clistAB, foundAB = candidates(FAB, pkh, plistB, dlim_pxl)
        clistAC, foundAC = candidates(FAC, pkh, plistC, dlim_pxl)

        ncanAB, nouse = clistAB.shape
        ncanAC, nouse = clistAC.shape

        if (not(foundAB)) | (not(foundAC)):
            continue

        # triangulation (for ech candiate in each image)
        XtAB = np.zeros([3, ncanAB])
        for i in range(ncanAB):

            # candidate image point
            cAB = np.array([np.append(clistAB[i,:2], 1)]).T

            # triangulate
            XthAB = triangulate_optimal_method(pkh, cAB, FAB, PA, PB);

            # store
            XtAB[:,i] = XthAB[:3]

        XtAC = np.zeros([3, ncanAC])
        for i in range(ncanAC):

            # candidate image point
            cAC = np.array([np.append(clistAC[i,:2], 1)]).T

            # triangulate
            XthAC = triangulate_optimal_method(pkh, cAC, FAC, PA, PC);

            # store
            XtAC[:,i] = XthAC[:3]



        # Find canidate matches in 3-space
        matches = 0
        for i in range(ncanAB):

            # current point in 3-space is from the set of triangulated points
            # with images form cam 1 and cam 2
            point = XtAB[:,i]

            # check whether one of the points triangulated with images from cam 1
            # and cam 3 are in the defined neighborhood of the current point
            miBC = find_neighbors_in_radius(point, XtAC, rlim)

            if (miBC >= 0) & (matches == 0):

                matches += 1

                # consider matches only: for information particle has five elements, peak coordinates (x,y), peak amplitude, peak widths (x,y)
                partA, partB, partC = plistA[k,:], clistAB[i,:], clistAC[miBC,:]
                partall = [partA, partB, partC]

                # assign to specific camera. camera 2 is excluded
                part1, part3, part4 = partall[al[0]], partall[al[1]], partall[al[2]]

                # Gaussian OTF parameter guess:
                # particle intensty (mean from all fit amplitudes)
                I = np.mean([part1[2], part3[2], part4[2]])

                # amplitudes of 2D Gaussian fit
                a1, a2, a3, a4 = part1[2], I, part3[2], part4[2]

                # widths of 2D Gaussian fit
                w1mean = np.mean([part1[3], part3[3], part4[3]])
                w2mean = np.mean([part1[4], part3[4], part4[4]])
                w11, w12, w13, w14 = part1[3], w1mean, part3[3], part4[3]
                w21, w22, w23, w24 = part1[4], w2mean, part3[4], part4[4]

                # base particle image coordinates
                b1x, b3x, b4x = part1[0], part3[0], part4[0] # not given for part2
                b1y, b3y, b4y = part1[1], part3[1], part4[1] # not given for part2

                # 3-space point of matched triangulation
                TMB = point
                TMC = XtAC[:, miBC]

                # final triangulation
                Pk = np.mean(np.array([TMB, TMC]), axis=0)

                # projection
                p1 = world2image_poly3(np.array([Pk]).T, 1, cal)
                p2 = world2image_poly3(np.array([Pk]).T, 3, cal)
                p3 = world2image_poly3(np.array([Pk]).T, 2, cal)
                p4 = world2image_poly3(np.array([Pk]).T, 4, cal)

                p1x, p1y = p1[0,0], p1[1,0]
                p2x, p2y = p2[0,0], p2[1,0]
                p3x, p3y = p3[0,0], p3[1,0]
                p4x, p4y = p4[0,0], p4[1,0]

                # compute errors between projection and base positions
                err1 = np.sqrt((p1x - b1x)**2 + (p1y - b1y)**2)
                err3 = np.sqrt((p3x - b3x)**2 + (p3y - b3y)**2)
                err4 = np.sqrt((p4x - b4x)**2 + (p4y - b4y)**2)

                # only add to list if error stays within limit for all cameras
                if (err1 < rlim_pxl) & (err3 < rlim_pxl) & (err4 < rlim_pxl):

                    # mapping of triangulation matches
                    xp1 = np.append(xp1, np.array([[p1x], [p1y]]).T, axis=0)
                    xp2 = np.append(xp2, np.array([[p2x], [p2y]]).T, axis=0)
                    xp3 = np.append(xp3, np.array([[p3x], [p3y]]).T, axis=0)
                    xp4 = np.append(xp4, np.array([[p4x], [p4y]]).T, axis=0)

                    # base particle image coordinates
                    xb1  = np.append(xb1, np.array([[b1x], [b1y]]).T, axis=0)
                    xb2  = np.append(xb2, np.array([[p2x], [p2y]]).T, axis=0)
                    xb3  = np.append(xb3, np.array([[b3x], [b3y]]).T, axis=0)
                    xb4  = np.append(xb4, np.array([[b4x], [b4y]]).T, axis=0)

                    # store triangulated 3D position (final triangulation) and
                    # OTF parameter guess
                    XP.append([Pk[0], Pk[1], Pk[2], I, a1, a2, a3, a4, w11, w12, w13, w14, w21, w22, w23, w24])


    # convert list to numpy array
    XP = np.array(XP)

    # find narrow particles in 3-space and delete one
    print('Number of triangulated particles before search for narrow neighbor: '+str(XP[:,0].size))
    ind_vec_nrw3D = find_narrow_particles_in_3space(XP)

    # delete dublicate particles
    XP  =  XP[ind_vec_nrw3D,:]

    xp1 = xp1[ind_vec_nrw3D,:]
    xp2 = xp2[ind_vec_nrw3D,:]
    xp3 = xp3[ind_vec_nrw3D,:]
    xp4 = xp4[ind_vec_nrw3D,:]

    xb1 = xb1[ind_vec_nrw3D,:]
    xb2 = xb2[ind_vec_nrw3D,:]
    xb3 = xb3[ind_vec_nrw3D,:]
    xb4 = xb4[ind_vec_nrw3D,:]


    print('Number of triangulated particles after search for narrow neighbor: '+str(XP[:,0].size))

    xpall = [xp1, xp2, xp3, xp4]
    xball = [xb1, xb2, xb3, xb4]
    return XP, xpall, xball

def triangulate_1particle_3cameras(rlim_pxl, ind_fst, ind_part, plistallF, dlim_pxl=1):
    #------------------------------------
    # Description:
    # Determine particle correspndance based on maximal error between triangulated 3-space coordinates
    # from different candidate pairs on different cameras. A total of 3 cameras and hence 3 lists of particle
    # lists is at hand. The excluded camera is cam 2.
    #
    # Input:
    # rlim_pxl: Allowed triangulation error in pixel units
    # ind_fst: Index of the bases first camera (base camera) 1,2,3 or 4 (non-zero indexing)
    # plistallF: List of particle lists of one frame - total of 4 particle lists.
    #
    # Output:
    #
    # Reference:
    # - p. 318, Hartley & Zisserman, 2003, 'Multiple View Geometry in computer
    #   vision'
    #
    #------------------------------------

    ind_not = 2

    #--------------------------
    # PARTICLE LIST
    #--------------------------

    # relation: camera order to camera indices.
    # order of cameras to calcualte candidates: A is the base camera (ind_fst), B, C, D correspond to the remaining
    ol = range(4)
    ol.remove(ind_fst-1)
    ol.remove(ind_not-1)
    ol.insert(0, ind_fst-1)

    # relation: camera indices to camera order (used later to assign OTF parameters to specific camera)
    al = range(1,3)
    al.insert(ind_fst-1,0)

    print(ol)
    print(al)

    #--------------------------
    # PARTICLE LIST
    #--------------------------
    [plistA, plistB, plistC] = plistallF[ol]

    #--------------------------
    # CALIBRATION
    #--------------------------
    # calibration parameters
    [scx, scy, h, w] = np.load('calibration/scales.npy')
    h, w = int(h), int(w)
    cal = str(np.load('calibration/name.npy'))

    # camera matrices: PA, PB, PC, PD
    Pall = np.load('calibration/camera_matrices.npy')
    [PA, PB, PC] = Pall[ol,:,:]

    # fundamental matrix: FAB, FAC, FAD
    Fall = np.load('calibration/fundamental_matrices.npy')
    FAB = Fall[ol[0], ol[1], :, :]
    FAC = Fall[ol[0], ol[2], :, :]

    #--------------------------
    # FUTHER PARAMETERS
    #--------------------------

    # candicate search distance to epipolar line
    rlim = rlim_pxl*scx

    #--------------------------
    # TRIANGULATION
    #--------------------------

    # allocate
    XP = []
    xp1, xp2, xp3, xp4 = np.array([[],[]]).T, np.array([[],[]]).T, np.array([[],[]]).T, np.array([[],[]]).T

    # loop over particles in first image
    ghost_particle_count = 0

    # current paricle
    pkh = np.array([np.append(plistA[ind_part,:2], 1)]).T

    # compute candidate lists for current particle
    clistAB, foundAB = candidates(FAB, pkh, plistB, dlim_pxl)
    clistAC, foundAC = candidates(FAC, pkh, plistC, dlim_pxl)

    ncanAB, nouse = clistAB.shape
    ncanAC, nouse = clistAC.shape

    if (not(foundAB)) | (not(foundAC)):
        print('There were no candidates found in at least one image.')

    # triangulation (for ech candiate in each image)
    XtAB = np.zeros([3, ncanAB])
    for i in range(ncanAB):

        # candidate image point
        cAB = np.array([np.append(clistAB[i,:2], 1)]).T

        # triangulate
        XthAB = triangulate_optimal_method(pkh, cAB, FAB, PA, PB);

        # store
        XtAB[:,i] = XthAB[:3]

    XtAC = np.zeros([3, ncanAC])
    for i in range(ncanAC):

        # candidate image point
        cAC = np.array([np.append(clistAC[i,:2], 1)]).T

        # triangulate
        XthAC = triangulate_optimal_method(pkh, cAC, FAC, PA, PC);

        # store
        XtAC[:,i] = XthAC[:3]



    # Find canidate matches in 3-space
    matches = 0
    for i in range(ncanAB):

        # current point in 3-space is from the set of triangulated points
        # with images form cam 1 and cam 2
        point = XtAB[:,i]

        # check whether one of the points triangulated with images from cam 1
        # and cam 3 are in the defined neighborhood of the current point
        miBC = find_neighbors_in_radius(point, XtAC, rlim)

        if (miBC >= 0) & matches == 0:

            matches += 1

            # consider matches only: for information particle has five elements, peak coordinates (x,y), peak amplitude, peak widths (x,y)
            partA, partB, partC = plistA[ind_part,:], clistAB[i,:], clistAC[miBC,:]
            partall = [partA, partB, partC]

            # assign to specific camera. camera 2 is excluded
            part1, part3, part4 = partall[al[0]], partall[al[1]], partall[al[2]]

            # Gaussian OTF parameter guess:
            # particle intensty (mean from all fit amplitudes)
            I = np.mean([part1[2], part3[2], part4[2]])

            # amplitudes of 2D Gaussian fit
            a1, a2, a3, a4 = part1[2], I, part3[2], part4[2]

            # widths of 2D Gaussian fit
            w1mean = np.mean([part1[3], part3[3], part4[3]])
            w2mean = np.mean([part1[4], part3[4], part4[4]])
            w11, w12, w13, w14 = part1[3], w1mean, part3[3], part4[3]
            w21, w22, w23, w24 = part1[4], w2mean, part3[4], part4[4]

            # image points of matched triangulations
            tm1 = part1[:2]
            tm3 = part3[:2]
            tm4 = part4[:2]

            # 3-space point of matched triangulation
            TMB = point
            TMC = XtAC[:, miBC]

            # final triangulation
            Pk = np.mean(np.array([TMB, TMC]), axis=0)

            # store triangulated 3D position (final triangulation) and
            # OTF parameter guess
            XP.append([Pk[0], Pk[1], Pk[2], I, a1, a2, a3, a4, w11, w12, w13, w14, w21, w22, w23, w24])

            # mapping of triangulation matches
            xp1 = np.append(xp1, world2image_poly3(np.array([Pk]).T, 1, cal).T, axis=0)
            xp2 = np.append(xp2, world2image_poly3(np.array([Pk]).T, 3, cal).T, axis=0)
            xp3 = np.append(xp3, world2image_poly3(np.array([Pk]).T, 2, cal).T, axis=0)
            xp4 = np.append(xp4, world2image_poly3(np.array([Pk]).T, 4, cal).T, axis=0)

            tm2 = xp2[0,:]


    if matches == 0:
        print('... no matching triangulation found!')

    # convert list to numpy array
    XP = np.array(XP)

    return XP, xp1, xp2, xp3, xp4, XtAB, XtAC, tm1, tm2, tm3, tm4


def triangulate_particles_4cameras(rlim_pxl, ind_fst, plistallF, dlim_pxl=1):
    #------------------------------------
    # Description:
    # Determine particle correspndance based on maximal error between triangulated 3-space coordinates
    # from different candidate pairs on different cameras. A total of 4 cameras and hence 4 lists of particle
    # lists is at hand.
    #
    # Input:
    # rlim_pxl: Allowed triangulation error in pixel units
    # ind_fst: Index of the bases first camera (base camera) 1,2,3 or 4 (non-zero indexing)
    # plistallF: List of particle lists of one frame - total of 4 particle lists.
    #
    # Output:
    #
    # Reference:
    # - p. 318, Hartley & Zisserman, 2003, 'Multiple View Geometry in computer
    #   vision'
    #
    #------------------------------------

    # relation: camera order to camera indices.
    # order of cameras to calcualte candidates: A is the base camera (ind_fst), B, C, D correspond to the remaining
    ol = range(4)
    ol.remove(ind_fst-1)
    ol.insert(0, ind_fst-1)

    # relation: camera indices to camera order (used later to assign OTF parameters to specific camera)
    al = range(1,4)
    al.insert(ind_fst-1,0)

    #--------------------------
    # PARTICLE LIST
    #--------------------------
    [plistA, plistB, plistC, plistD] = plistallF[ol]


    #--------------------------
    # CALIBRATION
    #--------------------------
    # calibration parameters
    [scx, scy, h, w] = np.load('calibration/scales.npy')
    h, w = int(h), int(w)
    cal = str(np.load('calibration/name.npy'))

    # camera matrices: PA, PB, PC, PD
    Pall = np.load('calibration/camera_matrices.npy')
    [PA, PB, PC, PD] = Pall[ol,:,:]

    # fundamental matrix: FAB, FAC, FAD
    Fall = np.load('calibration/fundamental_matrices.npy')
    FAB = Fall[ol[0], ol[1], :, :]
    FAC = Fall[ol[0], ol[2], :, :]
    FAD = Fall[ol[0], ol[3], :, :]

    #--------------------------
    # FUTHER PARAMETERS
    #--------------------------

    # candicate search distance to epipolar line
    rlim = rlim_pxl*scx

    #--------------------------
    # TRIANGULATION
    #--------------------------
    npA = plistA[:,0].size

    # allocate
    XP = []
    xp1, xp2, xp3, xp4 = np.array([[],[]]).T, np.array([[],[]]).T, np.array([[],[]]).T, np.array([[],[]]).T
    xb1, xb2, xb3, xb4 = np.array([[],[]]).T, np.array([[],[]]).T, np.array([[],[]]).T, np.array([[],[]]).T
    Matches = np.zeros(npA)


    # loop over particles in first image
    for k in range(npA):

        if k%1000 == 0:
            print('TRI: particle k = '+str(k)+' of '+str(npA))

        # current paricle
        pkh = np.array([np.append(plistA[k,:2], 1)]).T

        # compute candidate lists for current particle
        clistAB, foundAB = candidates(FAB, pkh, plistB, dlim_pxl)
        clistAC, foundAC = candidates(FAC, pkh, plistC, dlim_pxl)
        clistAD, foundAD = candidates(FAD, pkh, plistD, dlim_pxl)

        ncanAB, nouse = clistAB.shape
        ncanAC, nouse = clistAC.shape
        ncanAD, nouse = clistAD.shape

        if (not(foundAB)) | (not(foundAC)) | (not(foundAD)):
            continue

        # triangulation (for ech candiate in each image)
        XtAB = np.zeros([3, ncanAB])
        for i in range(ncanAB):

            # candidate image point
            cAB = np.array([np.append(clistAB[i,:2], 1)]).T

            # triangulate
            XthAB = triangulate_optimal_method(pkh, cAB, FAB, PA, PB);

            # store
            XtAB[:,i] = XthAB[:3]

        XtAC = np.zeros([3, ncanAC])
        for i in range(ncanAC):

            # candidate image point
            cAC = np.array([np.append(clistAC[i,:2], 1)]).T

            # triangulate
            XthAC = triangulate_optimal_method(pkh, cAC, FAC, PA, PC);

            # store
            XtAC[:,i] = XthAC[:3]

        XtAD = np.zeros([3, ncanAD])
        for i in range(ncanAD):

            # candidate image point
            cAD = np.array([np.append(clistAD[i,:2], 1)]).T

            # triangulate
            XthAD = triangulate_optimal_method(pkh, cAD, FAD, PA, PD);

            # store
            XtAD[:,i] = XthAD[:3]


        # Find canidate matches in 3-space
        matches = 0
        CI = []
        for i in range(ncanAB):

            # current point in 3-space is from the set of triangulated points
            # with images form cam 1 and cam 2
            point = XtAB[:,i]

            # check whether one of the points triangulated with images from cam 1
            # and cam 3 are in the defined neighborhood of the current point
            miBC = find_neighbors_in_radius(point, XtAC, rlim)

            if miBC >= 0:

                # check whether one of the points triangulated with images from cam 1
                # and cam 4 are in the defined neighborhood of the current point
                miBD = find_neighbors_in_radius(point, XtAD, rlim)

                if (miBD >= 0) & (matches == 0):

                    matches += 1

                    # consider matches only: for information particle has five elements, peak coordinates (x,y), peak amplitude, peak widths (x,y)
                    partA, partB, partC, partD = plistA[k,:], clistAB[i,:], clistAC[miBC,:], clistAD[miBD,:]
                    partall = [partA, partB, partC, partD]

                    # assign to specific camera
                    part1, part2, part3, part4 = partall[al[0]], partall[al[1]], partall[al[2]], partall[al[3]]

                    # Gaussian OTF parameter guess:
                    # particle intensty (mean from all fit amplitudes)
                    I = np.mean([part1[2], part2[2], part3[2], part4[2]])

                    # amplitudes of 2D Gaussian fit
                    a1, a2, a3, a4 = part1[2], part2[2], part3[2], part4[2]

                    # widths of 2D Gaussian fit
                    w11, w12, w13, w14 = part1[3], part2[3], part3[3], part4[3]
                    w21, w22, w23, w24 = part1[4], part2[4], part3[4], part4[4]

                    # base particle image coordinates
                    b1x, b2x, b3x, b4x = part1[0], part2[0], part3[0], part4[0]
                    b1y, b2y, b3y, b4y = part1[1], part2[1], part3[1], part4[1]

                    # 3-space point of matched triangulation
                    TMB = point
                    TMC = XtAC[:, miBC]
                    TMD = XtAD[:, miBD]

                    # final triangulation
                    Pk = np.mean(np.array([TMB, TMC, TMD]), axis=0)

                    # projection
                    p1 = world2image_poly3(np.array([Pk]).T, 1, cal)
                    p2 = world2image_poly3(np.array([Pk]).T, 3, cal)
                    p3 = world2image_poly3(np.array([Pk]).T, 2, cal)
                    p4 = world2image_poly3(np.array([Pk]).T, 4, cal)

                    p1x, p1y = p1[0,0], p1[1,0]
                    p2x, p2y = p2[0,0], p2[1,0]
                    p3x, p3y = p3[0,0], p3[1,0]
                    p4x, p4y = p4[0,0], p4[1,0]

                    # compute errors between projection and base positions
                    err1 = np.sqrt((p1x - b1x)**2 + (p1y - b1y)**2)
                    err2 = np.sqrt((p2x - b2x)**2 + (p2y - b2y)**2)
                    err3 = np.sqrt((p3x - b3x)**2 + (p3y - b3y)**2)
                    err4 = np.sqrt((p4x - b4x)**2 + (p4y - b4y)**2)

                    # only add to list if error stays within limit for all cameras
                    if (err1 < rlim_pxl) & (err2 < rlim_pxl) & (err3 < rlim_pxl) & (err4 < rlim_pxl):

                        # mapping of triangulation matches
                        xp1 = np.append(xp1, np.array([[p1x], [p1y]]).T, axis=0)
                        xp2 = np.append(xp2, np.array([[p2x], [p2y]]).T, axis=0)
                        xp3 = np.append(xp3, np.array([[p3x], [p3y]]).T, axis=0)
                        xp4 = np.append(xp4, np.array([[p4x], [p4y]]).T, axis=0)

                        # base particle image coordinates
                        xb1  = np.append(xb1, np.array([[b1x], [b1y]]).T, axis=0)
                        xb2  = np.append(xb2, np.array([[b2x], [b2y]]).T, axis=0)
                        xb3  = np.append(xb3, np.array([[b3x], [b3y]]).T, axis=0)
                        xb4  = np.append(xb4, np.array([[b4x], [b4y]]).T, axis=0)

                        # store candidate indices
                        CI.append([i, miBC, miBD])

                        # store triangulated 3D position (final triangulation) and
                        # OTF parameter guess
                        XP.append([Pk[0], Pk[1], Pk[2], I, a1, a2, a3, a4, w11, w12, w13, w14, w21, w22, w23, w24])


    # convert list to numpy array
    XP = np.array(XP)

    # find narrow particles in 3-space and delete one
    print('Number of triangulated particles before search for narrow neighbor: '+str(XP[:,0].size))
    ind_vec_nrw3D = find_narrow_particles_in_3space(XP)

    # delete dublicate particles
    XP  =  XP[ind_vec_nrw3D,:]

    xp1 = xp1[ind_vec_nrw3D,:]
    xp2 = xp2[ind_vec_nrw3D,:]
    xp3 = xp3[ind_vec_nrw3D,:]
    xp4 = xp4[ind_vec_nrw3D,:]

    xb1 = xb1[ind_vec_nrw3D,:]
    xb2 = xb2[ind_vec_nrw3D,:]
    xb3 = xb3[ind_vec_nrw3D,:]
    xb4 = xb4[ind_vec_nrw3D,:]


    print('Number of triangulated particles after search for narrow neighbor: '+str(XP[:,0].size))

    xpall = [xp1, xp2, xp3, xp4]
    xball = [xb1, xb2, xb3, xb4]
    return XP, xpall, xball


def find_narrow_particles_in_3space(XP):

    # calibration parameters
    [scx, scy, h, w] = np.load('calibration/scales.npy')
    h, w = int(h), int(w)

    # shift for strictly positive values
    XPs = XP[:,:3].copy()
    for i in range(3): XPs[:,i] -= XPs[:,i].min()

    # round to accuracy of 1 pixel. Particle 3-space positions within one pixel will have same coordinates.
    XPrnd = np.round(XPs/scx)

    # transform rounded particle coordinates to corresponding nine-digit 1D array to compute unique rows
    XPrnd3digit = XPrnd[:,0]*1e06 + XPrnd[:,1]*1e3 + XPrnd[:,2]
    XPrnd3digit, ind_vec_nrw3D = np.unique(XPrnd3digit, return_index=True)

    return ind_vec_nrw3D


def triangulate_1particle_4cameras(rlim_pxl, ind_fst, ind_prt, plistallF):
    #------------------------------------
    # Description:
    # Compute all possible triangulation for each camera pair combinatino with a base camera.
    # Find cooresponding particle.
    #
    # Input:
    # rlim_pxl: Allowed triangulation error in pixel units
    # ind_fst: Index of the bases first camera (base camera) 1,2,3 or 4 (non-zero indexing)
    # ind_prt: Index of treated particle
    # plistallF: List of particle lists of one frame - total of 4 particle lists.
    #
    # Output:
    #
    # Reference:
    # - p. 318, Hartley & Zisserman, 2003, 'Multiple View Geometry in computer
    #   vision'
    #
    #------------------------------------

    # relation: camera order to camera indices.
    # order of cameras to calcualte candidates: A is the base camera (ind_fst), B, C, D correspond to the remaining
    ol = range(4)
    ol.remove(ind_fst-1)
    ol.insert(0, ind_fst-1)

    # relation: camera indices to camera order (used later to assign OTF parameters to specific camera)
    al = range(1,4)
    al.insert(ind_fst-1,0)

    print(ol)
    print(al)
    #--------------------------
    # PARTICLE LIST
    #--------------------------
    [plistA, plistB, plistC, plistD] = plistallF[ol]


    #--------------------------
    # CALIBRATION
    #--------------------------
    # calibration parameters
    [scx, scy, h, w] = np.load('calibration/scales.npy')
    h, w = int(h), int(w)
    cal = str(np.load('calibration/name.npy'))

    # camera matrices: PA, PB, PC, PD
    Pall = np.load('calibration/camera_matrices.npy')
    [PA, PB, PC, PD] = Pall[ol,:,:]

    # fundamental matrix: FAB, FAC, FAD
    Fall = np.load('calibration/fundamental_matrices.npy')
    FAB = Fall[ol[0], ol[1], :, :]
    FAC = Fall[ol[0], ol[2], :, :]
    FAD = Fall[ol[0], ol[3], :, :]

    #--------------------------
    # FUTHER PARAMETERS
    #--------------------------

    # candicate search distance to epipolar line
    dlim_pxl = 1
    rlim = rlim_pxl*scx

    #--------------------------
    # TRIANGULATION
    #--------------------------

    # allocate
    XP = []
    xp1, xp2, xp3, xp4 = np.array([[],[]]).T, np.array([[],[]]).T, np.array([[],[]]).T, np.array([[],[]]).T

    # particle to be treated
    pkh = np.array([np.append(plistA[ind_prt,:2], 1)]).T

    # compute candidate lists for current particle
    clistAB, foundAB = candidates(FAB, pkh, plistB, dlim_pxl)
    clistAC, foundAC = candidates(FAC, pkh, plistC, dlim_pxl)
    clistAD, foundAD = candidates(FAD, pkh, plistD, dlim_pxl)

    ncanAB, nouse = clistAB.shape
    ncanAC, nouse = clistAC.shape
    ncanAD, nouse = clistAD.shape

    if (not(foundAB)) | (not(foundAC)) | (not(foundAD)):
        print('There were no candidates found in at least one image.')

    # triangulation (for ech candidate in each image)
    XtAB = np.zeros([3, ncanAB])
    for i in range(ncanAB):

        # candidate image point
        cAB = np.array([np.append(clistAB[i,:2], 1)]).T

        # triangulate
        XthAB = triangulate_optimal_method(pkh, cAB, FAB, PA, PB)
        #XthAB = triangulate_homogeneous_method(pkh, cAB, PA, PB)

        # store
        XtAB[:,i] = XthAB[:3]

    XtAC = np.zeros([3, ncanAC])
    for i in range(ncanAC):

        # candidate image point
        cAC = np.array([np.append(clistAC[i,:2], 1)]).T

        # triangulate
        XthAC = triangulate_optimal_method(pkh, cAC, FAC, PA, PC)
        #XthAC = triangulate_homogeneous_method(pkh, cAC, PA, PC)


        # store
        XtAC[:,i] = XthAC[:3]

    XtAD = np.zeros([3, ncanAD])
    for i in range(ncanAD):

        # candidate image point
        cAD = np.array([np.append(clistAD[i,:2], 1)]).T

        # triangulate
        XthAD = triangulate_optimal_method(pkh, cAD, FAD, PA, PD)
        #XthAD = triangulate_homogeneous_method(pkh, cAD, PA, PD)


        # store
        XtAD[:,i] = XthAD[:3]


    # Find canidate matches in 3-space
    matches = 0
    CI = []
    for i in range(ncanAB):

        # current point in 3-space is from the set of triangulated points
        # with images form cam 1 and cam 2
        point = XtAB[:,i]

        # check whether one of the points triangulated with images from cam 1
        # and cam 3 are in the defined neighborhood of the current point
        miBC = find_neighbors_in_radius(point, XtAC, rlim)

        if miBC >= 0:

            # check whether one of the points triangulated with images from cam 1
            # and cam 4 are in the defined neighborhood of the current point
            miBD = find_neighbors_in_radius(point, XtAD, rlim)

            if (miBD >= 0) & (matches == 0):

                matches += 1

                # consider matches only: for information particle has five elements, peak coordinates (x,y), peak amplitude, peak widths (x,y)
                partA, partB, partC, partD = plistA[ind_prt,:], clistAB[i,:], clistAC[miBC,:], clistAD[miBD,:]
                partall = [partA, partB, partC, partD]

                # assign to specific camera
                part1, part2, part3, part4 = partall[al[0]], partall[al[1]], partall[al[2]], partall[al[3]]

                # Gaussian OTF parameter guess:
                # particle intensty (mean from all fit amplitudes)
                I = np.mean([part1[2], part2[2], part3[2], part4[2]])

                # amplitudes of 2D Gaussian fit
                a1, a2, a3, a4 = part1[2], part2[2], part3[2], part4[2]

                # widths of 2D Gaussian fit
                w11, w12, w13, w14 = part1[3], part2[3], part3[3], part4[3]
                w21, w22, w23, w24 = part1[4], part2[4], part3[4], part4[4]

                # image points of matched triangulations
                tm1 = part1[:2]
                tm2 = part2[:2]
                tm3 = part3[:2]
                tm4 = part4[:2]

                # 3-space point of matched triangulation
                TMB = point
                TMC = XtAC[:, miBC]
                TMD = XtAD[:, miBD]

                # final triangulation
                Pk = np.mean(np.array([TMB, TMC, TMD]), axis=0)

                # store candidate indices
                CI.append([i, miBC, miBD])

                # store triangulated 3D position (final triangulation) and
                # OTF parameter guess
                XP.append([Pk[0], Pk[1], Pk[2], I, a1, a2, a3, a4, w11, w12, w13, w14, w21, w22, w23, w24])

                # mapping of triangulation matches
                xp1 = np.append(xp1, world2image_poly3(np.array([Pk]).T, 1, cal).T, axis=0)
                xp2 = np.append(xp2, world2image_poly3(np.array([Pk]).T, 3, cal).T, axis=0)
                xp3 = np.append(xp3, world2image_poly3(np.array([Pk]).T, 2, cal).T, axis=0)
                xp4 = np.append(xp4, world2image_poly3(np.array([Pk]).T, 4, cal).T, axis=0)

    if matches == 0:
        print('... no matching triangulation found!')

    XP = np.array(XP)
    return XP, xp1, xp2, xp3, xp4, XtAB, XtAC, XtAD, tm1, tm2, tm3, tm4


# find particles in neighborhood
def find_neighbors_in_radius(X, Xns, r, opt=0):
    # Description:
    # Find all (3-space-)particles from a set of neighborparticles that are
    # within a distance 'r'
    #
    # Input:
    # X: (3) 3D position of particle of consideration. ((4x1) if given in
    # homogeneous coordinates.)
    # Xns: (3xn) 3D positions of particles in neighborhood-set with n
    # neighbors. ((4xn) if given in homogeneous coordinates.)
    #
    # Output:
    # mi: (1xk) vector of indiecs of all particle matching the distance
    # criteria
    #
    # Reference:
    #


    # length of neighborhood-set
    m, n = Xns.shape

    # calculate euclidean distances
    Xd = Xns - np.repeat(np.array([X]).T, n, axis=1)
    d = np.sqrt(np.sum(Xd**2, axis=0))

    # sort distances. smallest first
    sort_ind = d.argsort()
    d_sorted = d[sort_ind]

    # distance criteria
    if d_sorted[0] < r:
        mi = sort_ind[0]
        sort_ind = sort_ind[d_sorted < r]
    else:
        mi = -1
        sort_ind = np.array([])

    if opt == 0:
        return mi

    else:
        return mi, sort_ind


def optical_transfer_function(Xp, Nnbh, cam, cal, csp=np.array(0)):
    # Description:
    # Computes the pixel intensity values on the image (of camera 'cam') for a
    # 3-space particle position mapped to a image position 'xp'. Intensity
    # values are computed for a 'Nnbh'-pixel neighborhood of the pixel
    # floor(xp).
    #
    # Input:
    # Xp(1:3): 3-space particle position
    # Xp(4): particle intesity. This is calculated as the mean of all
    # corresponding particle (image) intesity.
    # Xp(5:8): particle image intensity (OTF peak height) normalized with
    # particle intensity. for each camera.
    # Xp(8:16): particle image width. for each camera
    # Nnbh: size of neighborhood in pixel
    # cam: camera index (i.e. 1,2,3,4)
    # cal: calibration (i.e. 'orig' or 'last' or 'lastlast')
    # csp: central sampling points. csp_x = csp(1), csp_y = csp(2)
    #
    # Output:
    # Fi: (Nnbh, Nnbh) OTF values sampled at pixels in neighborhood of xp
    # Xi, Yi: pixel neighborhood
    #
    # Reference:
    # - Wieneke, 2013, 'Iterative reconstruction of volumetric particle distribution'

    # particle properties
    X = np.array([Xp[0], Xp[1], Xp[2]])
    X = np.array([X]).T

    # particle intensity
    I = Xp[3]

    # OTF parameters
    if cam == 1:
        a, w1, w2 = Xp[4], Xp[8], Xp[12]

    elif cam == 2:
        a, w1, w2 = Xp[5], Xp[9], Xp[13]

    elif cam == 3:
        a, w1, w2 = Xp[6], Xp[10], Xp[14]

    elif cam == 4:
        a, w1, w2 = Xp[7], Xp[11], Xp[15]


    # use polynomial calibratoin to mapp 3-space particle position on image
    if cam == 2:
        cam_poly3 = 3
    elif cam == 3:
        cam_poly3 = 2
    else:
        cam_poly3 = cam

    xp = world2image_poly3(X, cam_poly3, cal)

    # define neighborhood (sampling points)
    if csp.any():
        xip, yip = np.round(csp[0,0]).astype(np.int), np.round(csp[1,0]).astype(np.int)

    else:
        xip, yip = np.round(xp[0,0]).astype(np.int), np.round(xp[1,0]).astype(np.int)

    xi, yi = [xip], [yip]
    for j in range(int(Nnbh/2.)):
        k = j+1
        xi = [xip-k]+xi+[xip+k]
        yi = [yip-k]+yi+[yip+k]

    xi, yi = np.array(xi), np.array(yi)
    Xi, Yi = np.meshgrid(xi, yi)

    # compute OTF values
    Fi = gaussian_OTF_model(xp[0,0], xp[1,0], Xi, Yi, a, w1, w2)

    return Fi, xi, yi, xp


def gaussian_OTF_model(x_p, y_p, x_ij, y_ij, a, w1, w2):

    return a*np.exp(-(x_ij - x_p)**2/(2*w1**2) - (y_ij - y_p)**2/(2*w2**2))



# project 3-space particle positions to images for each camera
def projected_images(XP, iter=0, Niter=0):
    # Description:
    # Project (new, shaked) particle 3-space position on all camera images.
    #
    # Input:
    # X: 3-space particle position and OTF parameters
    #
    # Output:
    # Iproji: projected images for camera i = 1, 2, 3, 4
    #
    # Reference:
    # - Wieneke, 2013, 'Iterative reconstruction of volumetrix particle distribution'
    # - Schanz et al, 2016, 'Shake-the-box: Lagrangian particle tracking at high particle densities'

    # calibration parameters
    [scx, scy, h, w] = np.load('calibration/scales.npy')
    h, w = int(h), int(w)
    cal = str(np.load('calibration/name.npy'))

    # number if cameras
    Ni = 4;

    # allocate
    Iproj1, Iproj2, Iproj3, Iproj4 = np.zeros([h,w]), np.zeros([h,w]), np.zeros([h,w]), np.zeros([h,w])

    # back-projection of triangulated particles
    for k in range(XP[:,0].size):

        if k%1000 == 0:
            print('PRJ: particle k = '+str(k)+' of '+str(XP[:,0].size)+' in iteration '+str(iter)+' of '+str(Niter))

        # OTF
        F1, x1, y1, xp1 = optical_transfer_function(XP[k,:], 5, 1, cal)
        F2, x2, y2, xp2 = optical_transfer_function(XP[k,:], 5, 2, cal)
        F3, x3, y3, xp3 = optical_transfer_function(XP[k,:], 5, 3, cal)
        F4, x4, y4, xp4 = optical_transfer_function(XP[k,:], 5, 4, cal)

        # increment intesity counts
        Iproj1[y1[0]:y1[-1]+1, x1[0]:x1[-1]+1] = Iproj1[y1[0]:y1[-1]+1, x1[0]:x1[-1]+1] + F1
        Iproj2[y2[0]:y2[-1]+1, x2[0]:x2[-1]+1] = Iproj2[y2[0]:y2[-1]+1, x2[0]:x2[-1]+1] + F2
        Iproj3[y3[0]:y3[-1]+1, x3[0]:x3[-1]+1] = Iproj3[y3[0]:y3[-1]+1, x3[0]:x3[-1]+1] + F3
        Iproj4[y4[0]:y4[-1]+1, x4[0]:x4[-1]+1] = Iproj4[y4[0]:y4[-1]+1, x4[0]:x4[-1]+1] + F4

    return Iproj1, Iproj2, Iproj3, Iproj4


########
# FIND NARROW PARTICLES
########
def find_narrow_particles(dlim, xpall):
    # Description:
    # Get index of particles that have projected image coordinates very close to
    # each other.
    #
    # Input:
    # dlim: minimum distance. If distance smaller than dlim particles are considered narrow
    # xpall: list of projected 2D image coordiates for all cameras
    #
    # Output:
    # nrw_ind_veci: index vector for all narrow particles in xpi, with i=1,2,3,4
    #
    # Reference:
    #

    [xp1, xp2, xp3, xp4] = xpall

    # sizes
    Nk = xp1[:,0].size
    Ni = 4

    # index array
    [I, J] = np.meshgrid(np.arange(Nk), np.arange(Nk))
    nask = (np.ones([Nk, Nk]) - np.eye(Nk)).astype(bool)

    nrw_ind_vec_all = []
    for i in range(Ni):

        # compute difference matrices
        x1_1 = xpall[i][:,0]
        x1_2 = x1_1

        y1_1 = xpall[i][:,1]
        y1_2 = y1_1

        X1_1, X1_2 = np.meshgrid(x1_1, x1_2)
        Y1_1, Y1_2 = np.meshgrid(y1_1, y1_2)

        DX = abs(X1_1 - X1_2)
        DY = abs(Y1_1 - Y1_2)

        D = np.sqrt(DX**2 + DY**2)

        # calculate indices of particles with minimum distance
        mask = (D < dlim)*nask

        a, b = I[mask], J[mask]

        # add pairs in order to list
        nrw_ind_vec = [];
        for aa, bb in zip(a, b):

            ab = np.array([aa, bb])

            # add pairs (smaller index first)
            nrw_ind_vec.append([min(ab), max(ab)])

        if len(nrw_ind_vec) > 0:
            nrw_ind_vec_all.append(unique_rows(np.array(nrw_ind_vec)))
        else:
            nrw_ind_vec_all.append(np.array([[],[]]).T)

    nrw_ind_vec1 = nrw_ind_vec_all[0]
    nrw_ind_vec2 = nrw_ind_vec_all[1]
    nrw_ind_vec3 = nrw_ind_vec_all[2]
    nrw_ind_vec4 = nrw_ind_vec_all[3]

    return nrw_ind_vec_all


########
# SHAKE (3-SPACE POSITION CORRECTION)
########
def shake(XP, Iresall, iter=0, Niter=0, shift_max=1.0):
    # Description:
    # Move particles by dX, dY, dX away form the triangulated 3-space position.
    # Calculated resiudual for each new projection. Choose new particle 3-space
    # position for minimum residual
    #
    # Input:
    # XP: Triangulated particles: 3-space positions, OTF parameters
    # Iresall: list containing residual images for all cameras
    # iter: current iteration number
    # Niter: number of (total) iterations
    #
    # Output:
    # (as save) XP: new, shaked particle 3-space positions
    #
    # Reference:
    # - Wieneke, 2013, 'Iterative reconstruction of volumetrix particle distribution'
    # - Schanz et al, 2016, 'Shake-the-box: Lagrangian particle tracking at high particle densities'


    # camera / calibration parameters (scales)
    [scx, scy, h, w] = np.load('calibration/scales.npy')
    h, w = int(h), int(w)
    cal = str(np.load('calibration/name.npy'))

    #--------------------------
    # LOAD DATA
    #--------------------------

    # number if cameras
    Ni = 4;

    #--------------------------
    # SHAKE LOOP
    #--------------------------
    # move step-size [pixel*scale]
    if iter == 0:
        dX, dY, dZ = 0.5*scx, 0.5*scx, 0.5*scx
    elif iter == 1:
        dX, dY, dZ = 1.0*shift_max*scx, 1.0*shift_max*scx, 1.0*shift_max*scx
    elif iter == Niter:
        dX, dY, dZ = 0.1*scx, 0.1*scx, 0.1*scx
    else:
        dX, dY, dZ = 0.2*shift_max*scx, 0.2*shift_max*scx, 0.2*shift_max*scx

    # window size
    ws = 5

    # loop over triangulated particles
    for k in range(XP[:,0].size):

        if k%1000 == 0:
            print('POS: particle k = '+str(k)+' of '+str(XP[:,0].size)+' in iteration '+str(iter)+' of '+str(Niter))

        #----------------------------- Y-shake ----------------------------
        # initialize for residual
        Res1, Res2, Res3 = 0, 0, 0

        for i in range(Ni):

            cam = i+1

            # moved particle world positions
            Xp1 = XP[k,:].copy()
            Xp1[1] -= dY

            Xp2 = XP[k,:].copy()

            Xp3 = XP[k,:].copy()
            Xp3[1] += dY;

            # back-projection of all particles particle positions
            # OTF
            F2, x2, y2, xp2 = optical_transfer_function(Xp2, ws, cam, cal);

            csp = xp2 # central OTF sampling point for shifted particle positions
            F1, x1, y1, xp1 = optical_transfer_function(Xp1, ws, cam, cal, csp)
            F3, x3, y3, xp2 = optical_transfer_function(Xp3, ws, cam, cal, csp)

            # window
            wx, wy = x2, y2

            # particle augmented images
            Ipartaug = Iresall[i][wy[0]:wy[-1]+1, wx[0]:wx[-1]+1] + F2

            # compute image residual
            Ires1 = (Ipartaug - F1)**2
            Ires2 = (Ipartaug - F2)**2
            Ires3 = (Ipartaug - F3)**2

            Res1 += Ires1.sum()
            Res2 += Ires2.sum()
            Res3 += Ires3.sum()

        # define new X-coordinate
        if (Res2 < Res1) & (Res2 < Res3):

            # fit polynomial through all-image residual
            xx = np.array([Xp1[1], Xp2[1], Xp3[1]])
            ff = np.array([Res1, Res2, Res3])
            pp = np.polyfit(xx, ff, 2)
            Y_new = -pp[1]/(2*pp[0])

        elif (Res1 < Res2) & (Res2 < Res3):
            Y_new = Xp1[1]

        elif (Res1 < Res2) & (Res1 < Res3):
            Y_new = Xp1[1]

        elif (Res3 < Res2) & (Res2 < Res1):
            Y_new = Xp3[1]

        elif (Res3 < Res2) & (Res3 < Res1):
            Y_new = Xp3[1]

        else:
            Y_new = Xp2[1]

        # correct position
        XP[k,1] = Y_new
        #------------------------------------------------------------------

        #----------------------------- Z-shake ----------------------------
        # initialize for residual
        Res1, Res2, Res3 = 0, 0, 0

        for i in range(Ni):

            cam = i+1

            # moved particle world positions
            Xp1 = XP[k,:].copy()
            Xp1[2] -=dZ

            Xp2 = XP[k,:].copy()

            Xp3 = XP[k,:].copy()
            Xp3[2] += dZ;

            # back-projection of all particles particle positions
            # OTF
            F2, x2, y2, xp2 = optical_transfer_function(Xp2, ws, cam, cal);

            csp = xp2 # central OTF sampling point for shifted particle positions
            F1, x1, y1, xp1 = optical_transfer_function(Xp1, ws, cam, cal, csp)
            F3, x3, y3, xp2 = optical_transfer_function(Xp3, ws, cam, cal, csp)

            # window
            wx, wy = x2, y2

            # particle augmented images
            Ipartaug = Iresall[i][wy[0]:wy[-1]+1, wx[0]:wx[-1]+1] + F2

            # compute image residual
            Ires1 = (Ipartaug - F1)**2
            Ires2 = (Ipartaug - F2)**2
            Ires3 = (Ipartaug - F3)**2

            Res1 += Ires1.sum()
            Res2 += Ires2.sum()
            Res3 += Ires3.sum()

        # define new X-coordinate
        if (Res2 < Res1) & (Res2 < Res3):

            # fit polynomial through all-image residual
            xx = np.array([Xp1[2], Xp2[2], Xp3[2]])
            ff = np.array([Res1, Res2, Res3])
            pp = np.polyfit(xx, ff, 2)
            Z_new = -pp[1]/(2*pp[0])

        elif (Res1 < Res2) & (Res2 < Res3):
            Z_new = Xp1[2]

        elif (Res1 < Res2) & (Res1 < Res3):
            Z_new = Xp1[2]

        elif (Res3 < Res2) & (Res2 < Res1):
            Z_new = Xp3[2]

        elif (Res3 < Res2) & (Res3 < Res1):
            Z_new = Xp3[2]

        else:
            Z_new = Xp2[2]

        # correct position
        XP[k,2] = Z_new
        #------------------------------------------------------------------


        #----------------------------- X-shake ----------------------------
        # initialize for residual
        Res1, Res2, Res3 = 0, 0, 0

        for i in range(Ni):

            cam = i+1

            # moved particle world positions
            Xp1 = XP[k,:].copy()
            Xp1[0] -= dX

            Xp2 = XP[k,:].copy()

            Xp3 = XP[k,:].copy()
            Xp3[0] += dX

            # back-projection of all particles particle positions
            # OTF
            F2, x2, y2, xp2 = optical_transfer_function(Xp2, ws, cam, cal);

            csp = xp2 # central OTF sampling point for shifted particle positions
            F1, x1, y1, xp1 = optical_transfer_function(Xp1, ws, cam, cal, csp)
            F3, x3, y3, xp2 = optical_transfer_function(Xp3, ws, cam, cal, csp)

            # window
            wx, wy = x2, y2

            # particle augmented images
            Ipartaug = Iresall[i][wy[0]:wy[-1]+1, wx[0]:wx[-1]+1] + F2

            # compute image residual
            Ires1 = (Ipartaug - F1)**2
            Ires2 = (Ipartaug - F2)**2
            Ires3 = (Ipartaug - F3)**2

            Res1 += Ires1.sum()
            Res2 += Ires2.sum()
            Res3 += Ires3.sum()

        # define new X-coordinate
        if (Res2 < Res1) & (Res2 < Res3):

            # fit polynomial through all-image residual
            xx = np.array([Xp1[0], Xp2[0], Xp3[0]])
            ff = np.array([Res1, Res2, Res3])
            pp = np.polyfit(xx, ff, 2)
            X_new = -pp[1]/(2*pp[0])

        elif (Res1 < Res2) & (Res2 < Res3):
            X_new = Xp1[0]

        elif (Res1 < Res2) & (Res1 < Res3):
            X_new = Xp1[0]

        elif (Res3 < Res2) & (Res2 < Res1):
            X_new = Xp3[0]

        elif (Res3 < Res2) & (Res3 < Res1):
            X_new = Xp3[0]

        else:
            X_new = Xp2[0]

        # correct position
        XP[k,0] = X_new
        #------------------------------------------------------------------

    return XP


def shake_GE(XP, xball, iter=0, Niter=0, shift_max=1):
    # Description:
    # Move particles by dX, dY, dX away form the triangulated 3-space position.
    # Calculated geometric error (GE) for each new projection. Choose new particle 3-space
    # position for minimum GE.
    #
    # Input:
    # XP: Triangulated particles: 3-space positions, OTF parameters
    # Iresall: list containing residual images for all cameras
    # iter: current iteration number
    # Niter: number of (total) iterations
    #
    # Output:
    # (as save) XP: new, shaked particle 3-space positions
    #
    # Reference:


    # camera / calibration parameters (scales)
    [scx, scy, h, w] = np.load('calibration/scales.npy')
    h, w = int(h), int(w)
    cal = str(np.load('calibration/name.npy'))

    #--------------------------
    # LOAD DATA
    #--------------------------

    # number if cameras
    Ni = 4;

    #--------------------------
    # SHAKE LOOP
    #--------------------------
    # move step-size [pixel*scale]
    if iter == 0:
        dX, dY, dZ = 0.5*scx, 0.5*scx, 0.5*scx
    elif iter == 1:
        dX, dY, dZ = 1.0*shift_max*scx, 1.0*shift_max*scx, 1.0*shift_max*scx
    elif iter == Niter:
        dX, dY, dZ = 0.1*scx, 0.1*scx, 0.1*scx
    else:
        dX, dY, dZ = 0.2*shift_max*scx, 0.2*shift_max*scx, 0.2*shift_max*scx

    # window size
    ws = 5

    # loop over triangulated particles
    for k in range(XP[:,0].size):

        if k%1000 == 0:
            print('POS: particle k = '+str(k)+' of '+str(XP[:,0].size)+' in iteration '+str(iter)+' of '+str(Niter))

        #----------------------------- Y-shake ----------------------------
        # initialize for residual
        GE1, GE2, GE3 = 0, 0, 0

        for i in range(Ni):

            cam = i+1

            # base particle image coordinates
            xb = xball[i][k,:]

            # moved particle world positions
            Xp1 = XP[k,:].copy()
            Xp1[1] -= dY

            Xp2 = XP[k,:].copy()

            Xp3 = XP[k,:].copy()
            Xp3[1] += dY;

            # change array shape for mapping
            Xp1 = np.array([Xp1]).T
            Xp2 = np.array([Xp2]).T
            Xp3 = np.array([Xp3]).T

            # account for camera index change for calibration coeffs of cam 1 and 2
            if cam == 2:
                cam_poly3 = 3
            elif cam == 3:
                cam_poly3 = 2
            else:
                cam_poly3 = cam

            # projection of all sheked particle positions
            xp1 = world2image_poly3(Xp1, cam_poly3, cal)[:,0];
            xp2 = world2image_poly3(Xp2, cam_poly3, cal)[:,0];
            xp3 = world2image_poly3(Xp3, cam_poly3, cal)[:,0];

            # compute geometric error
            GE1 += (xp1[0] - xb[0])**2 + (xp1[1] - xb[1])**2
            GE2 += (xp2[0] - xb[0])**2 + (xp2[1] - xb[1])**2
            GE3 += (xp3[0] - xb[0])**2 + (xp3[1] - xb[1])**2

        # define new X-coordinate
        if (GE2 < GE1) & (GE2 < GE3):

            # fit polynomial through all-image residual
            xx = np.array([Xp1[1,0], Xp2[1,0], Xp3[1,0]])
            ff = np.array([GE1, GE2, GE3])
            pp = np.polyfit(xx, ff, 2)
            Y_new = -pp[1]/(2*pp[0])

        elif (GE1 < GE2) & (GE2 < GE3):
            Y_new = Xp1[1]

        elif (GE1 < GE2) & (GE1 < GE3):
            Y_new = Xp1[1]

        elif (GE3 < GE2) & (GE2 < GE1):
            Y_new = Xp3[1]

        elif (GE3 < GE2) & (GE3 < GE1):
            Y_new = Xp3[1]

        else:
            Y_new = Xp2[1]

        # correct position
        XP[k,1] = Y_new
        #------------------------------------------------------------------

        #----------------------------- Z-shake ----------------------------
        # initialize for residual
        GE1, GE2, GE3 = 0, 0, 0

        for i in range(Ni):

            cam = i+1

            # base particle image coordinates
            xb = xball[i][k,:]

            # moved particle world positions
            Xp1 = XP[k,:].copy()
            Xp1[2] -=dZ

            Xp2 = XP[k,:].copy()

            Xp3 = XP[k,:].copy()
            Xp3[2] += dZ;

            # change array shape for mapping
            Xp1 = np.array([Xp1]).T
            Xp2 = np.array([Xp2]).T
            Xp3 = np.array([Xp3]).T

            # account for camera index change for calibration coeffs of cam 1 and 2
            if cam == 2:
                cam_poly3 = 3
            elif cam == 3:
                cam_poly3 = 2
            else:
                cam_poly3 = cam

            # projection of all sheked particle positions
            xp1 = world2image_poly3(Xp1, cam_poly3, cal)[:,0];
            xp2 = world2image_poly3(Xp2, cam_poly3, cal)[:,0];
            xp3 = world2image_poly3(Xp3, cam_poly3, cal)[:,0];

            # compute geometric error
            GE1 += (xp1[0] - xb[0])**2 + (xp1[1] - xb[1])**2
            GE2 += (xp2[0] - xb[0])**2 + (xp2[1] - xb[1])**2
            GE3 += (xp3[0] - xb[0])**2 + (xp3[1] - xb[1])**2

        # define new X-coordinate
        if (GE2 < GE1) & (GE2 < GE3):

            # fit polynomial through all-image residual
            xx = np.array([Xp1[2,0], Xp2[2,0], Xp3[2,0]])
            ff = np.array([GE1, GE2, GE3])
            pp = np.polyfit(xx, ff, 2)
            Z_new = -pp[1]/(2*pp[0])

        elif (GE1 < GE2) & (GE2 < GE3):
            Z_new = Xp1[2]

        elif (GE1 < GE2) & (GE1 < GE3):
            Z_new = Xp1[2]

        elif (GE3 < GE2) & (GE2 < GE1):
            Z_new = Xp3[2]

        elif (GE3 < GE2) & (GE3 < GE1):
            Z_new = Xp3[2]

        else:
            Z_new = Xp2[2]

        # correct position
        XP[k,2] = Z_new
        #------------------------------------------------------------------


        #----------------------------- X-shake ----------------------------
        # initialize for residual
        GE1, GE2, GE3 = 0, 0, 0

        for i in range(Ni):

            cam = i+1

            # base particle image coordinates
            xb = xball[i][k,:]

            # moved particle world positions
            Xp1 = XP[k,:].copy()
            Xp1[0] -= dX

            Xp2 = XP[k,:].copy()

            Xp3 = XP[k,:].copy()
            Xp3[0] += dX

            # change array shape for mapping
            Xp1 = np.array([Xp1]).T
            Xp2 = np.array([Xp2]).T
            Xp3 = np.array([Xp3]).T

            # account for camera index change for calibration coeffs of cam 1 and 2
            if cam == 2:
                cam_poly3 = 3
            elif cam == 3:
                cam_poly3 = 2
            else:
                cam_poly3 = cam

            # projection of all sheked particle positions
            xp1 = world2image_poly3(Xp1, cam_poly3, cal)[:,0];
            xp2 = world2image_poly3(Xp2, cam_poly3, cal)[:,0];
            xp3 = world2image_poly3(Xp3, cam_poly3, cal)[:,0];

            # compute geometric error
            GE1 += (xp1[0] - xb[0])**2 + (xp1[1] - xb[1])**2
            GE2 += (xp2[0] - xb[0])**2 + (xp2[1] - xb[1])**2
            GE3 += (xp3[0] - xb[0])**2 + (xp3[1] - xb[1])**2

        # define new X-coordinate
        if (GE2 < GE1) & (GE2 < GE3):

            # fit polynomial through all-image residual
            xx = np.array([Xp1[0,0], Xp2[0,0], Xp3[0,0]])
            ff = np.array([GE1, GE2, GE3])
            pp = np.polyfit(xx, ff, 2)
            X_new = -pp[1]/(2*pp[0])

        elif (GE1 < GE2) & (GE2 < GE3):
            X_new = Xp1[0]

        elif (GE1 < GE2) & (GE1 < GE3):
            X_new = Xp1[0]

        elif (GE3 < GE2) & (GE2 < GE1):
            X_new = Xp3[0]

        elif (GE3 < GE2) & (GE3 < GE1):
            X_new = Xp3[0]

        else:
            X_new = Xp2[0]

        # correct position
        XP[k,0] = X_new
        #------------------------------------------------------------------

    return XP


def intensity_update_new(XP, Iall, Iresall, iter=0, Niter=0):
    # Description:
    #
    # Input:
    #
    # Output:
    #
    # Reference:
    # - Wieneke, 2013, 'Iterative reconstruction of volumetrix particle distribution'
    # - Schanz et al, 2016, 'Shake-the-box: Lagrangian particle tracking at high particle densities'

    # camera / calibration parameters (scales)
    [scx, scy, h, w] = np.load('calibration/scales.npy')
    h, w = int(h), int(w)
    cal = str(np.load('calibration/name.npy'))

    # number of cameras
    Ni = 4;

    # particle intensity average and standard deviation
    Iavg = XP[:,3].mean()
    Istd = XP[:,3].std()

    print('\n average particle intensiy: '+str(Iavg)+' ('+str(Istd)+')')

    dlt_ind_vec = []
    for k in range(XP[:,0].size):

        if k%1000 == 0:
            print('INT: particle k = '+str(k)+' of '+str(XP[:,0].size)+' in iteration '+str(iter)+' of '+str(Niter))

        # particle position in 3-space
        Xp = XP[k,:].copy()

        # template index list for exclusion of brigthest
        ind_exclude = range(3,8)

        # 1st step intensity correction: pre-adjust intensity for OTF
        if iter <= 1:
            for i in range(Ni):

                cam = i+1

                # projected particle intesity (OTF)
                F, wx, wy, xp = optical_transfer_function(Xp, 5, cam, cal)

                # window intensity in original
                Iorig = Iall[i][wy[0]:wy[-1]+1, wx[0]:wx[-1]+1]

                # Compute peak intensity ratio
                int_max_ratio = Iorig.max()/F.max()

                if int_max_ratio > 1:
                    Xp[i+4] *= int_max_ratio


        # 2nd step intensity correction:
        delete_flag = False
        for i in range(Ni):

            cam = i+1

            # projected particle intesity (OTF)
            F, wx, wy, xp = optical_transfer_function(Xp, 3, cam, cal)

            # window intensity in original
            Iorig = Iall[i][wy[0]:wy[-1]+1, wx[0]:wx[-1]+1]

            # center of mass (COM) in original
            COM_orig = center_of_mass(Iorig)
            COM_orig_x, COM_orig_y = COM_orig[1]+wx[0], COM_orig[0]+wy[0]

            # distance of projection center to COM in original
            dist = np.sqrt((COM_orig_x - xp[0,0])**2 + (COM_orig_y - xp[1,0])**2)

            if dist > 1.0:
                delete_flag = True

        if delete_flag:
            dlt_ind_vec.append(k)

        # insert particle back to list
        XP[k,:] = Xp


    return XP, np.array(dlt_ind_vec)



def intensity_update(XP, Iall, Iresall, iter=0, Niter=0):
    # Description:
    #
    # Input:
    #
    # Output:
    #
    # Reference:
    # - Wieneke, 2013, 'Iterative reconstruction of volumetrix particle distribution'
    # - Schanz et al, 2016, 'Shake-the-box: Lagrangian particle tracking at high particle densities'

    # camera / calibration parameters (scales)
    [scx, scy, h, w] = np.load('calibration/scales.npy')
    h, w = int(h), int(w)
    cal = str(np.load('calibration/name.npy'))

    # number of cameras
    Ni = 4;

    # index list for intensity entries in particle list
    ind_intensity = range(3,8)

    # particle intensity average and standard deviation
    Iavg = XP[:,3].mean()
    Istd = XP[:,3].std()

    print('\n average particle intensiy: '+str(Iavg)+' ('+str(Istd)+')')

    dlt_ind_vec = []
    for k in range(XP[:,0].size):

        if k%1000 == 0:
            print('INT: particle k = '+str(k)+' of '+str(XP[:,0].size)+' in iteration '+str(iter)+' of '+str(Niter))

        # particle position in 3-space
        Xp = XP[k,:].copy()

        # 1st step intensity correction: pre-adjust intensity for OTF
        if iter <= 1:
            for i in range(Ni):

                cam = i+1

                # projected particle intesity (OTF)
                F, wx, wy, xp = optical_transfer_function(Xp, 5, cam, cal)

                # ration of summed intensity
                Iorig = Iall[i][wy[0]:wy[-1]+1, wx[0]:wx[-1]+1]

                # Compute peak intensity ratiocompute ratio of maximum
                int_max_ratio = Iorig.max()/F.max()

                if int_max_ratio > 1:
                    Xp[i+4] *= int_max_ratio


        # 2nd step intensity correction:
        Ipartaugsum, Ipartsum = 0., 0.
        Ipartsummax = 0.
        for i in range(Ni):

            cam = i+1

            # projected particle intesity (OTF)
            F, wx, wy, xp = optical_transfer_function(Xp, 3, cam, cal)

            # ration of summed intensity
            Ipartaug = Iresall[i][wy[0]:wy[-1]+1, wx[0]:wx[-1]+1] + F

            # increase sums
            Ipartaugsum += Ipartaug.sum()
            Ipartsum += F.sum()


        # check ratio of sums
        ratio_of_sums = Ipartaugsum/Ipartsum

        if ratio_of_sums < 0:
            dlt_ind_vec.append(k)

        else:
            # apply 2nd intensity correction (insert corrected OTF intensity value for this camera)
            Xp[ind_intensity] *= ratio_of_sums

            # check wheter particle intensity is to low
            if Xp[3] < 0.1*Iavg:
                dlt_ind_vec.append(k)

            # insert updated particle properties
            XP[k,:] = Xp


    return XP, np.array(dlt_ind_vec)



def intensity_update_single_cam_check(XP, Iresall, iter=0, Niter=0):
    # Description:
    #
    # Input:
    #
    # Output:
    #
    # Reference:
    # - Wieneke, 2013, 'Iterative reconstruction of volumetrix particle distribution'
    # - Schanz et al, 2016, 'Shake-the-box: Lagrangian particle tracking at high particle densities'

    # camera / calibration parameters (scales)
    [scx, scy, h, w] = np.load('calibration/scales.npy')
    h, w = int(h), int(w)
    cal = str(np.load('calibration/name.npy'))

    # number of cameras
    Ni = 4;

    dlt_ind_vec = []
    for k in range(XP[:,0].size):

        if k%1000 == 0:
            print('INT: particle k = '+str(k)+' of '+str(XP[:,0].size)+' in iteration '+str(iter)+' of '+str(Niter))

        # particle position in 3-space
        Xp = XP[k,:].copy()

        for i in range(Ni):

            cam = i+1

            # indices of other cameras
            ind_othcam = [1,2,3,4]
            ind_othcam.remove(cam)

            # projected particle intesity (OTF)
            F, wx, wy, xp = optical_transfer_function(Xp, 5, cam, cal)

            # ration of summed intensity
            Ipartaug = Iresall[i][wy[0]:wy[-1]+1, wx[0]:wx[-1]+1] + F

            ratio_of_sums = Ipartaug.sum()/F.sum()

            if ratio_of_sums < 0.1:

                # check if particle has narrow neighbor in image
                xpa = xpall[i][k,:]
                xpi_search = xpall[i].copy()
                xpi_search[k,:] *= 0
                nrw_match_mask = np.isclose(xpi_search, xpa, atol=1.)
                nrw_match = nrw_match_mask[:,0] & nrw_match_mask[:,1]

                # if particle has narrow neighbor, check which higher total residual
                if np.any(nrw_match):
                    nrw_match_ind = np.where(nrw_match)[0][0]

                    #--------------- narrow particle a --------------------
                    Xpa = XP[k,:].copy()

                    # OTF for remaining cameras
                    FB, wxB, wyB, xpB = optical_transfer_function(Xpa, 5, ind_othcam[0], cal)
                    FC, wxC, wyC, xpC = optical_transfer_function(Xpa, 5, ind_othcam[1], cal)
                    FD, wxD, wyD, xpD = optical_transfer_function(Xpa, 5, ind_othcam[2], cal)

                    # compute total squared residual intensity
                    IpartaugA = Ipartaug
                    IpartaugB = (Iresall[ind_othcam[0]-1][wyB[0]:wyB[-1]+1, wxB[0]:wxB[-1]+1] + FB)**2
                    IpartaugC = (Iresall[ind_othcam[1]-1][wyC[0]:wyC[-1]+1, wxC[0]:wxC[-1]+1] + FC)**2
                    IpartaugD = (Iresall[ind_othcam[2]-1][wyD[0]:wyD[-1]+1, wxD[0]:wxD[-1]+1] + FD)**2

                    Ipartaugtota = IpartaugA.sum() + IpartaugB.sum() + IpartaugC.sum() + IpartaugD.sum()
                    #------------------------------------------------------

                    #--------------- narrow particle b --------------------
                    Xpb = XP[nrw_match_ind,:]

                    # OTF for remaining cameras
                    FA, wxA, wyA, xpA = optical_transfer_function(Xpb, 5, cam, cal)
                    FB, wxB, wyB, xpB = optical_transfer_function(Xpb, 5, ind_othcam[0], cal)
                    FC, wxC, wyC, xpC = optical_transfer_function(Xpb, 5, ind_othcam[1], cal)
                    FD, wxD, wyD, xpD = optical_transfer_function(Xpb, 5, ind_othcam[2], cal)

                    # compute total squared residual intensity
                    IpartaugA = (Iresall[ind_othcam[0]-1][wyA[0]:wyA[-1]+1, wxA[0]:wxA[-1]+1] + FA)**2
                    IpartaugB = (Iresall[ind_othcam[0]-1][wyB[0]:wyB[-1]+1, wxB[0]:wxB[-1]+1] + FB)**2
                    IpartaugC = (Iresall[ind_othcam[1]-1][wyC[0]:wyC[-1]+1, wxC[0]:wxC[-1]+1] + FC)**2
                    IpartaugD = (Iresall[ind_othcam[2]-1][wyD[0]:wyD[-1]+1, wxD[0]:wxD[-1]+1] + FD)**2

                    Ipartaugtotb = IpartaugA.sum() + IpartaugB.sum() + IpartaugC.sum() + IpartaugD.sum()
                    #------------------------------------------------------

                    # delete particle with higher total augmented intensity
                    if Ipartaugtota > Ipartaugtotb:
                        dlt_ind_vec.append(k)
                    else:
                        dlt_ind_vec.append(nrw_match_ind)

                else:

                    # particle has no narrow neighbor
                    dlt_ind_vec.append(k)

            else:

                # particle intensity update (insert corrected OTF intensity value for this camera)
                Xp[i+4] *= sqrt(ratio_of_sums)

        # insert updated particle properites
        XP[k,:] = Xp

    return XP, np.array(dlt_ind_vec)



########
# ITERATIVE PARTICLE RECONSTRUCTION
########

def IPR_loop(plistallF, Iall, Niters, rlim_pxl, dlim_pxl, shift_max, GE=False):
    # Description:
    #
    # Input:
    #
    # Output:
    #
    # Reference:
    # - Wieneke, 2013, 'Iterative reconstruction of volumetrix particle distribution'
    # - Schanz et al, 2016, 'Shake-the-box: Lagrangian particle tracking at high particle densities'

    # number of iterations for different loops
    [Nouteriter4cam, Ninneriter4cam, Nouteriter3cam, Ninneriter3cam] = Niters

    # unpack original images
    [I1, I2, I3, I4] = Iall

    # allocate for 3-space particle list
    XP = np.array([[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]).T

    # loop: 4 camera triangulation
    #------------------------------------------------------------
    print('\n 4 CAMERA TRIANGULATION:')
    for j in range(1, Nouteriter4cam+1):

        print('\n OUTTER ITERATION: j= '+str(j)+' of '+str(Nouteriter4cam)+'\n')

        # triangulation parameters adapted for each iteration
        if j == 1:
            rlim_pxl_j = 1.0*rlim_pxl
            dlim_pxl_j = 1.0*dlim_pxl
            shift_max_j = 1.0*shift_max

        if j == 2:
            rlim_pxl_j = 1.5*rlim_pxl
            dlim_pxl_j = 1.5*dlim_pxl
            shift_max_j = 2.0*shift_max

        if j > 2:
            rlim_pxl_j = 2.0*rlim_pxl
            dlim_pxl_j = 2.0*dlim_pxl
            shift_max_j = 3.0*shift_max


        # triangulation based on 4 cameras
        ind_fst_cam = 1
        XPj, xpjall, xbjall = triangulate_particles_4cameras(rlim_pxl_j, ind_fst_cam, plistallF, dlim_pxl_j)

        # projection
        Iproj1, Iproj2, Iproj3, Iproj4 = projected_images(XPj)

        # re-compute residual images
        Ires1 = I1 - Iproj1
        Ires2 = I2 - Iproj2
        Ires3 = I3 - Iproj3
        Ires4 = I4 - Iproj4

        Iresall = [Ires1, Ires2, Ires3, Ires4]

        # 3-space correction loop (shake, project (compute resiudal), intensity update, delete)
        for iter in range(1, Ninneriter4cam+1):

            # shake - 3-space position correction
            if GE:
                XPj = shake_GE(XPj, xbjall, iter, Ninneriter4cam, shift_max)
            else:
                XPj = shake(XPj, Iresall, iter, Ninneriter4cam, shift_max)

            # projection & new residual
            Iproj1, Iproj2, Iproj3, Iproj4 = projected_images(XPj, iter, Ninneriter4cam)

            Ires1 = I1 - Iproj1
            Ires2 = I2 - Iproj2
            Ires3 = I3 - Iproj3
            Ires4 = I4 - Iproj4

            Iresall = [Ires1, Ires2, Ires3, Ires4]

            # intensity correction
            XPj, dlt_ind_vec = intensity_update_new(XPj, Iall, Iresall, iter, Ninneriter4cam)
            dlt_ind_vec = np.unique(dlt_ind_vec)

            # delete particles in lists
            XPj = np.delete(XPj, dlt_ind_vec, axis=0)

            if GE:
                xbjall[0] = np.delete(xbjall[0], dlt_ind_vec, axis=0)
                xbjall[1] = np.delete(xbjall[1], dlt_ind_vec, axis=0)
                xbjall[2] = np.delete(xbjall[2], dlt_ind_vec, axis=0)
                xbjall[3] = np.delete(xbjall[3], dlt_ind_vec, axis=0)


        # append 3-space and image positions
        XP = np.append(XP, XPj, axis=0)

        # check wheter new particle set has narrow neighbor in 3-space
        ind_vec_nrw3D = find_narrow_particles_in_3space(XP)
        XP = XP[ind_vec_nrw3D,:]

        # projection & new residual
        Iproj1, Iproj2, Iproj3, Iproj4 = projected_images(XPj)

        Ires1 = I1 - Iproj1
        Ires2 = I2 - Iproj2
        Ires3 = I3 - Iproj3
        Ires4 = I4 - Iproj4

        Iprojall = [Iproj1, Iproj2, Iproj3, Iproj4]

        # residual image is new image
        thrsh = 80
        I1 = residual_image_to_image(Ires1, thrsh)
        I2 = residual_image_to_image(Ires2, thrsh)
        I3 = residual_image_to_image(Ires3, thrsh)
        I4 = residual_image_to_image(Ires4, thrsh)

        # Find 2D particle position
        plist1 = find_particle_2D_position(I1)
        plist2 = find_particle_2D_position(I2)
        plist3 = find_particle_2D_position(I3)
        plist4 = find_particle_2D_position(I4)

        plistallF = np.array([plist1, plist2, plist3, plist4])
    #------------------------------------------------------------

    # loop: 3 camera triangulation
    #------------------------------------------------------------
    print('\n 3 CAMERA TRIANGULATION:')
    for j in range(1, Nouteriter3cam+1):

        print('\n OUTTER ITERATION: j= '+str(j)+' of '+str(Nouteriter3cam)+'\n')

        # triangulation parameters adapted for each iteration
        if j == 1:
            rlim_pxl_j = 1.0*rlim_pxl
            dlim_pxl_j = 1.0*dlim_pxl
            shift_max_j = 1.0*shift_max

        if j == 2:
            rlim_pxl_j = 1.5*rlim_pxl
            dlim_pxl_j = 1.5*dlim_pxl
            shift_max_j = 2.0*shift_max

        if j > 2:
            rlim_pxl_j = 2.0*rlim_pxl
            dlim_pxl_j = 2.0*dlim_pxl
            shift_max_j = 3.0*shift_max

        # triangulation based on 3 cameras
        ind_fst_cam = 1
        XPj, xpjall, xbjall = triangulate_particles_3cameras(rlim_pxl_j, ind_fst_cam, plistallF, dlim_pxl_j)

        # projection & new residual
        Iproj1, Iproj2, Iproj3, Iproj4 = projected_images(XPj)

        Ires1 = I1 - Iproj1
        Ires2 = I2 - Iproj2
        Ires3 = I3 - Iproj3
        Ires4 = I4 - Iproj4

        Iresall = [Ires1, Ires2, Ires3, Ires4]

        # 3-space correction loop (shake, project (compute resiudal residual), intensity update, delete)
        for iter in range(1, Ninneriter3cam+1):

            # shake - 3-space position correction
            if GE:
                XPj = shake_GE(XPj, xbjall, iter, Ninneriter3cam, shift_max)
            else:
                XPj = shake(XPj, Iresall, iter, Ninneriter3cam, shift_max)

            # projection & new residual
            Iproj1, Iproj2, Iproj3, Iproj4 = projected_images(XPj, iter, Ninneriter3cam)

            Ires1 = I1 - Iproj1
            Ires2 = I2 - Iproj2
            Ires3 = I3 - Iproj3
            Ires4 = I4 - Iproj4

            Iresall = [Ires1, Ires2, Ires3, Ires4]

            # intensity correction
            XPj, dlt_ind_vec = intensity_update_new(XPj, Iall, Iresall, iter, Ninneriter3cam)
            dlt_ind_vec = np.unique(dlt_ind_vec)

            # delete particles in lists
            XPj = np.delete(XPj, dlt_ind_vec, axis=0)

            if GE:
                xbjall[0] = np.delete(xbjall[0], dlt_ind_vec, axis=0)
                xbjall[1] = np.delete(xbjall[1], dlt_ind_vec, axis=0)
                xbjall[2] = np.delete(xbjall[2], dlt_ind_vec, axis=0)
                xbjall[3] = np.delete(xbjall[3], dlt_ind_vec, axis=0)


        # append 3-space and image positions
        XP = np.append(XP, XPj, axis=0)

        # check wheter new particle set has narrow neighbor in 3-space
        ind_vec_nrw3D = find_narrow_particles_in_3space(XP)
        XP = XP[ind_vec_nrw3D,:]

        # recalculate residual images
        Iproj1, Iproj2, Iproj3, Iproj4 = projected_images(XPj)
        Ires1 = I1 - Iproj1
        Ires2 = I2 - Iproj2
        Ires3 = I3 - Iproj3
        Ires4 = I4 - Iproj4

        # residual image is new image
        I1 = residual_image_to_image(Ires1, 80)
        I2 = residual_image_to_image(Ires2, 80)
        I3 = residual_image_to_image(Ires3, 80)
        I4 = residual_image_to_image(Ires4, 80)

        # Find 2D particle position
        plist1 = find_particle_2D_position(I1)
        plist2 = find_particle_2D_position(I2)
        plist3 = find_particle_2D_position(I3)
        plist4 = find_particle_2D_position(I4)

        plistallF = np.array([plist1, plist2, plist3, plist4])
    #------------------------------------------------------------

    # final residual
    Iresall = [Ires1, Ires2, Ires3, Ires4]

    return XP, Iresall



########
# PARTICLE TRACKING
########


# particle tracking: relaxation
def particle_tracking_relaxation(XPF1, XPF2, dt=0, mean_field=0, rms_field=0, Taylor_ms=2, constant_estimate_flag=True):
    # Description:
    # Chooses the most plausible particle track based on a quasi-rigid environment condition. Match probabilities are obtained via a relaxation method.
    #
    # Input:
    # XPF1: whole list of particles in first frame
    # XPF2: whole list of particles in second frame
    #
    # Output:
    # Tracks: 2D array of all matched 2D position
    # part_ind_vec: index of particles, for which a track is established, stored in a vector
    #
    # Reference:
    # [Pereira 2006] Two-frame 3D particle tracking


    # number of iterations in relaxation
    n_iter_rel = 7

    # weight constants for probability update
    A, B, C = 0.3, 3.0, 2.0

    # quasi-rigid radius fraction (of search volume)
    qrr = 0.2

    # track probability threshold
    tp_thresh   = 0.90
    nmtp_thresh = 0.10

    # either use constant values for neiborhood and search volum radius ...
    if constant_estimate_flag:

        # constant estimate for neighborhood radius (in mm)
        nbh_rad = 1.5

        # constant estimate for search volume radius (in mm)
        svm_rad = 1.0

    # ... or use mean vel. field and rms vel. fluc. field to estimate the neighborhood or search volume radius
    else:

        # choose nbd_rad as a function of Taylor microscale estimation
        nbh_rad = 1*Taylor_ms

        # extract
        X, Y, Z = mean_field[:,0], mean_field[:,1], mean_field[:,2]
        U, V, W = mean_field[:,3], mean_field[:,4], mean_field[:,5]
        urms = rms_field[:,3]

        # domain
        Xd, Yd, Zd = np.unique(X),  np.unique(Y),  np.unique(Z)


    # allocate
    TrackID, Neigh, Dist, Prob, Prob_star, Pred = [], [], [], [], [], []
    no_track_matches, no_track_match_counter, lonely_particle_counter = [], 0, 0


    # --------------- initialize particle tracking --------------
    print('\n INITIALIZE PARTICLE TRACKING: (finding neighbors and possible tracks)')
    for k in range(np.size(XPF1, axis=0)):

        if k%1000 == 0:
            print('TRK: particle k = '+str(k)+' of '+str(np.size(XPF1, axis=0)) )

        # current particle
        XpF1 = XPF1[k,:3]

        # ------------------ search volume ----------------------
        if constant_estimate_flag:

            # for unknown mean displacement: center of search radius is paricle position in first frame
            XpF2 = XpF1.copy()

        else:
            # interpolate mean velocity field to current particle position (local mean velocity). use local mean velocity to predict center of search volume.

            # neighbors above and below
            Xl, Xu = Xd[Xd < XpF1[0]].max(), Xd[Xd > XpF1[0]].min()
            Yl, Yu = Yd[Yd < XpF1[1]].max(), Yd[Yd > XpF1[1]].min()
            Zl, Zu = Zd[Zd < XpF1[2]].max(), Zd[Zd > XpF1[2]].min()

            maskX, maskY, maskZ = (X >= Xl) & (X <= Xu), (Y >= Yl) & (Y <= Yu), (Z >= Zl) & (Z <= Zu)
            mask = maskX & maskY & maskZ

            XYZ_pts = np.array([X[mask], Y[mask], Z[mask]]).T
            U_vls, V_vls, W_vls = U[mask], V[mask], W[mask]

            # interpolate mean velocity to search particle position
            U_interpolator = LinearNDInterpolator(XYZ_pts, U_vls)
            V_interpolator = LinearNDInterpolator(XYZ_pts, V_vls)
            W_interpolator = LinearNDInterpolator(XYZ_pts, W_vls)

            UpF1, VpF1, WpF1 = U_interpolator(XpF1)[0], V_interpolator(XpF1)[0], W_interpolator(XpF1)[0]

            # search volume center is given by mean velocity advection.
            XpF2x, XpF2y, XpF2z = XpF1[0] + dt*UpF1, XpF1[1] + dt*VpF1, XpF1[2] + dt*WpF1
            XpF2 = np.array([XpF2x, XpF2y, XpF2z])

            # interpolate rms velocity fluctuation field to predicted search volume center (local rms vel. fluc.). use local rms vel. fluc. to predict radius of search volume.

            # neighbors above and below
            Xl, Xu = Xd[Xd < XpF2[0]].max(), Xd[Xd > XpF2[0]].min()
            Yl, Yu = Yd[Yd < XpF2[1]].max(), Yd[Yd > XpF2[1]].min()
            Zl, Zu = Zd[Zd < XpF2[2]].max(), Zd[Zd > XpF2[2]].min()

            maskX, maskY, maskZ = (X >= Xl) & (X <= Xu), (Y >= Yl) & (Y <= Yu), (Z >= Zl) & (Z <= Zu)
            mask = maskX & maskY & maskZ

            XYZ_pts = np.array([X[mask], Y[mask], Z[mask]]).T
            urms_vls = urms[mask]

            # interpolate rms vel. fluc
            urms_interpolator = LinearNDInterpolator(XYZ_pts, urms_vls)

            urmspF1 = urms_interpolator(XpF2)[0]

            # search volume estimate is porportinal (e.g. scaling factor 2) to local rms velocity fluctuation.
            svm_rad = 2.0*dt*urmpF1

        # list of track candidates - search volume members
        ind_svm, ind_vec_svm  = find_neighbors_in_radius(XpF2, XPF2[:,:3].T, svm_rad, 1)

        # do not make a track search entry if there are no possible matches
        if ind_svm < 0:
            no_track_matches.append(k)
            no_track_match_counter += 1
            continue

        XP_svm = XPF2[ind_vec_svm,:]
        # -------------------------------------------------------

        # number of possible links
        Np = np.size(XP_svm[:,0])
        if k%1000 == 0:
            print('Np = '+str(Np))

        # use current particle index as track ID
        TrackID.append(k)

        # shift vector of possible link:
        # this has to be a 2d array, even if only one sv
        Dk = XP_svm[:,:3] - XpF1

        Dist.append(Dk)

        # initial match probability values
        Pk = np.ones(Np)/(Np + 1.)
        Pk_star = 1./(Np + 1.)

        Prob.append(Pk)
        Prob_star.append(Pk_star)

        # -------------------- neighborhood ---------------------
        ind_nbh, ind_vec_nbh  = find_neighbors_in_radius(XpF1, XPF1[:,:3].T, nbh_rad, 1)
        Nn = ind_vec_nbh.size
        if Nn < 2:
            lonely_particle_counter += 1

        Neigh.append(ind_vec_nbh)
        # -------------------------------------------------------
        if k%1000 == 0:
            print('Nn = '+str(Nn))

        # mean flow prediction based weight
        Fk = np.zeros(Np)

        # evaluate difference vector
        Gk = Dk - (XpF2 - XpF1)

        # enter mean flow prediction based weights
        mask = norm(Gk, axis=1) < qrr*svm_rad
        Fk[mask] = 1

        Pred.append(Fk)

    # convert to numpy array
    TrackID, Neigh, Dist, Prob, Prob_star, Pred = np.array(TrackID), np.array(Neigh), np.array(Dist), np.array(Prob), np.array(Prob_star), np.array(Pred)

    no_track_matches = np.array(no_track_matches)

    print('\n ...possible tracks found for '+str(np.size(TrackID))+' particles' )
    print('...'+str(lonely_particle_counter)+' particles with one or no neighbors' )


    # ---------------- update match probabilities ---------------
    print('\n MATCH PROBABILITY RELAXATION:')
    for iter in range(n_iter_rel):
        Prob_new, Prob_star_new = [], []
        track_counter = 0
        for trackID, neigh, dist, prob, prob_star, pred in zip(TrackID, Neigh, Dist, Prob, Prob_star, Pred):

            if track_counter%1000 == 0:
                print('RLX: track = '+str(trackID)+' in relax. iteration '+str(iter+1)+' of '+str(n_iter_rel) )

            # number of possible matches of current particle
            Np = np.size(dist, axis=0)

            # allocate for new match probability
            prob_new = np.zeros_like(prob)

            for j in range(Np):

                # -------- compute weight based on quasi-rigid conditions -------
                Qk_j = 0

                # do not compute this weight if particle has no neighbors
                if neigh.size == 0:
                    continue

                # shift vector
                dk_j = dist[j,:]

                for nb in neigh:

                    # continue if neighbor does not belong to possible tracks (no match found for this guy and therefore not in the lists)
                    if np.any(nb == no_track_matches):
                        continue

                    # mask to find properties of neighbor
                    # note: if mask_nb is used for 'indexing' the result is returned within a numpy.array. with X[mask_nb][0] the actual element is obtained
                    mask_nb = TrackID == nb

                    # number of possible matches of current neighbor
                    Np_nb = np.size(Dist[mask_nb][0], axis=0)

                    for l in range(Np_nb):

                        # shift vector of neighbor particle
                        d_nb_l = Dist[mask_nb][0][l,:]

                        # quasi rigid condition
                        if norm(dk_j - d_nb_l) < qrr*svm_rad:
                            Qk_j += Prob[mask_nb][0][l]
                # -------------------------------------------------------------------

                # update probability
                prob_new[j] = prob[j] * (A + B*Qk_j + C*pred[j])

            # normalize updated probability
            denom = prob_new.sum() + prob_star
            prob_new /= denom

            Prob_new.append(prob_new)

            # update no-match probability
            prob_star_new = prob_star / denom

            Prob_star_new.append(prob_star_new)

            track_counter += 1

        # replace all probabilities with updated values
        Prob, Prob_star = np.array(Prob_new), np.array(Prob_star_new)


    # ---------------------- determine tracks -------------------
    Tracks, Tracks_prob, part_ind_vec = [], [], []
    for prob, prob_star, dist, trackID in zip(Prob, Prob_star, Dist, TrackID):

        # check whether no match probability is bellow threshold
        if prob_star > nmtp_thresh:
            continue

        # check whether max. match probability is above threshold
        if prob.max() < tp_thresh:
            continue

        # get track with highest probability
        track = dist[prob.argmax(),:].copy()
        Tracks.append(track)

        # corresponding probabilities (last value is no-match probability)
        Tracks_prob.append(np.append(prob, prob_star))

        # corresponding particle index
        part_ind_vec.append(trackID)

    Tracks, Tracks_prob, part_ind_vec = np.array(Tracks), np.array(Tracks_prob), np.array(part_ind_vec)

    print('\n TRACKS DETERMINED:')


    return Tracks, Tracks_prob, part_ind_vec, Prob, Prob_star


# calculate sparsely located velocity vectors from tracks
def tracks_to_velocity_vectors(dt, XPF1, Tracks, part_ind_vec, vel_thresh):
    # Description:
    # For each track, the corresponding velocity vector is computed allocated at the mid-point of the track.
    #
    # Input:
    # dt: interframe time
    # XPF1: whole list of particles in first frame
    # Tracks: (N_tr x 3) list with N_tr determined three-dimensional particle tracks
    # part_ind_vec: (N_tr) list of indices for particles for which a particle track is establishes (corrsponds to Tracks)
    #
    # Output:
    # Velocities: (N_tr x 3) list with N_tr three-dimensional velocity vectors (in m/s)
    # Points: (N_tr x 3) list with N_tr node coordinates in 3D-space (in mm)
    #
    # Reference:
    #

    # allocate
    Velocities, Points = [], []

    for track, part_ind in zip(Tracks, part_ind_vec):

        # track base point
        Xp = XPF1[part_ind,:3]

        # determine mid-point
        mid_point = Xp + 0.5*track

        # velocity
        vel_vec = track*1e-03/dt

        # filter field according to velocity threshold
        if norm(vel_vec) > vel_thresh:
            continue

        # append
        Velocities.append(vel_vec)
        Points.append(mid_point)

    Velocities, Points = np.array(Velocities), np.array(Points)

    return Velocities, Points

def outlier_detection(Points, Velocities, N_neigh, sigma_ol):
    # Description:
    # A particle track (velocity vector) with N_neigh neighbors is considered to be an outlier
    # if the 3rd biggest magnitude of the difference vectors to its neighbors is
    # above some threshold value.
    #
    # Input:
    # Points: (N_tr x 3) list with N_tr node coordinates in 3D-space (in mm)
    # Velocities: (N_tr x 3) list with N_tr three-dimensional velocity vectors (in m/s)
    # N_neigh: number of neighbors that are considered
    # sigma_ol: velocity vector outlier threshold
    #
    # Output:
    # Points_new: 'Points' where outliers have been removed
    # Velocities_new: 'Velocities' where outliers have beed reduced
    # N_ol: number of outliers
    #
    #
    # Reference:
    #

    # number of velocity vectors before outlier detection
    N_tr = np.size(Points, axis=0)

    print('\n VELOCITY VECTOR OUTLIER DETECTION:')
    outlier_vec, k = [], 0
    for point, velocity in zip(Points, Velocities):

        if k%1000 == 0:
            print('OLD: track = '+str(k))

        # find neighbors
        Xk = np.repeat(np.array([point]), N_tr, axis=0)
        dXkm = norm(Points - Xk, axis=1)
        neigh = np.argsort(dXkm)[1:N_neigh+1]

        # compute difference vectors
        Vk = np.repeat(np.array([velocity]), N_neigh, axis=0)
        dVkm = norm(Velocities[neigh,:] - Vk, axis=1)
        dVkm = np.sort(dVkm)

        # evaluate outliers
        if dVkm[-3] > sigma_ol:
            outlier_vec.append(k)

        k += 1

    # delete outlier tracks
    outlier_vec = np.array(outlier_vec)

    Points     = np.delete(Points, outlier_vec, axis=0)
    Velocities = np.delete(Velocities, outlier_vec, axis=0)

    return Points, Velocities, outlier_vec






# write VTK files for visualization (e.g. using VisIt form LLNL)
def write_VTK_files(filename, Points, Velocities):
    # Description:
    # Creates a points mesh VTK file with scalar data (e.g. velocity magnitude) and vector data (velocity vector).
    #
    # Input:
    # filename: string containing the name of the vtk-file, and, if needed, the directory where to write it.
    # Points: list with point mesh coordinates
    # Velocities: list with velocity vectors for each point in 'Points'
    #
    # Output:
    #
    # Reference:
    # "Getting Data into VisIt" (LLNL VisIt manuals)
    # https://wci.llnl.gov/simulation/computer-codes/visit/manuals
    #

    U, V, W = Velocities[:,0], Velocities[:,1], Velocities[:,2]
    X, Y, Z = Points[:,0], Points[:,1], Points[:,2]

    # velocity X-component
    vel_U = U.copy().tolist()

    # velocity Y-component
    vel_V = V.copy().tolist()

    # velocity Z-component
    vel_W = W.copy().tolist()

    # velocity magnitude
    vel_mag = np.sqrt(U**2 + V**2 + W**2).tolist()

    # velocity vector field
    vel_vec_field = Velocities.copy()
    vel_vec_field = vel_vec_field.reshape(1, vel_vec_field.size).tolist()[0]

    # pint mesh node coordinates
    pmnc_xyz = Points.copy()
    pmnc_xyz = pmnc_xyz.reshape(1, pmnc_xyz.size).tolist()[0]

    # list of variables lists
    vars = [['vel_mag', 1, 1, vel_mag], ['vel_U', 1, 1, vel_U], ['vel_V', 1, 1, vel_V], ['vel_W', 1, 1, vel_W], ['vel_vec_field', 3, 1, vel_vec_field]]

    # save the vtk-file using the visit_writer module
    vw.WritePointMesh(filename, 1, pmnc_xyz, vars)

    return 0



########
# INTERPOLATION
########

def interpolate_on_regular_grid(Points, Velocities, RegGrid):
    # Description:
    #
    # Input:
    # Points: list with point mesh coordinates
    # Velocities: list with velocity vectors for each point in 'Points'
    #
    # Output:
    #
    # Reference:
    #

    return 0



########
# FUTHER FUNCTIONS
########

# null space
#(copied from: http://scipy-cookbook.readthedocs.io/items/RankNullspace.html)
def nullspace(A, atol=1e-16, rtol=0):
    """
    Compute an approximate basis for the nullspace of A.

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    Parameters
    ----------
    A : ndarray
    A should be at most 2-D.  A 1-D array with length k will be treated
    as a 2-D with shape (1, k)
    atol : float
    The absolute tolerance for a zero singular value.  Singular values
    smaller than `atol` are considered to be zero.
    rtol : float
    The relative tolerance.  Singular values less than rtol*smax are
    considered to be zero, where smax is the largest singular value.

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
    tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    Return value
    ------------
    ns : ndarray
    If `A` is an array with shape (m, k), then `ns` will be an array
    with shape (k, n), where n is the estimated dimension of the
    nullspace of `A`.  The columns of `ns` are a basis for the
    nullspace; each element in numpy.dot(A, ns) will be approximately
    zero.
    """

    A = np.atleast_2d(A)
    u, s, vh = svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T

    # return 1D array
    return ns[:,0]


# unique rows
def unique_rows(x):


    nr, nc = x.shape

    row_arr = []
    for i in range(nr):

        row_str = ''
        for j in range(nc):

            if log(x[i,j], 10) < 0:
                a = int(x[i,j]*10000*(-log(x[i,j], 10)))
            else:
                a = int(x[i,j]*1000)

            row_str = row_str+str(a)

        row_arr.append(int(row_str))

    nouse, uix = np.unique(row_arr, return_index=True)

    return x[uix,:]
