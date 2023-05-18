import numpy as np
import math
import cv2
import itertools as it
import matplotlib.pyplot as plt

###########################
######### WEEK 1 ##########
###########################
def box3d(n=16):
    points = []
    N = tuple(np.linspace(-1, 1, n))
    for i, j in [(-1, -1), (-1, 1), (1, 1), (0, 0)]:
        points.extend(set(it.permutations([(i, )*n, (j, )*n, N])))
    return np.hstack(points)/2

def Pi(p):
    '''Homogenous to inhomogenous'''
    return p[:-1]/p[-1]

def PiInv_h(p, scale=1):
    '''Hstack version: Imhomogenous to homogenous w/ scale 1'''
    p = np.hstack((p,scale*np.ones(p.shape[0]).reshape(-1,1)))
    return scale*p

def PiInv_v(p, scale=1):
    '''Vstack version of PiInv_h()'''
    p = np.vstack((p,scale*np.ones(p.shape[1]).reshape(1,-1)))
    return scale*p


def projectpoints(K, R, t, Q):
    ''' Projects 3D points into the camera plane.
    INPUTS:
        -   K: Camera matrix
        -   (R, t): Rotation/translational pose of the camera
        -   Q: Represents n points in 3D to be projected into camera
    OUTPUTS:
        - Projected 2D matrix as a 2 x n matrix.
    '''
    P_cam = R@Q+t.reshape(-1,1)
    return Pi(K@P_cam)

def ProjectPoints_( 
        K: np.array, 
        R: np.array, 
        t: np.array,
        Q: np.array,
    )-> np.array:
    P = K @ np.hstack((R, t[:, np.newaxis]))
    Ph = PiInv(Q)
    return Pi(P @ Ph)



###########################
######### WEEK 2 ##########
###########################
def deltaR(x, y, distCoeffs):
    output = 0
    power = 2
    for coef in distCoeffs:
        output += coef * np.sqrt(x**2 + y**2)**power
        power += 2
    return output

def distortion(P_cam2d, distCoeffs):
    output = P_cam2d * (1 + deltaR(P_cam2d[0,:],P_cam2d[1,:],distCoeffs))
    return output

def projectpoints_dist(K, R, t, Q, distCoeffs):
    '''See projectpoints. distCoeffs rely on distortion()'''
    P_cam = R@Q+t.reshape(-1,1)
    P_cam2d = Pi(P_cam)
    P_cam3d = PiInv_v(distortion(P_cam2d, distCoeffs))
    return Pi(K@P_cam3d)

def undistortImage(
        im: np.array, 
        K: np.array, 
        distCoeffs: np.array
    ) -> np.array:
    '''Undistorts image. Takes image, Camera matrix K and distCoeffs'''
    x, y = np.meshgrid(np.arange(im.shape[1]), np.arange(im.shape[0]))
    p = np.stack((x, y, np.ones(x.shape))).reshape(3, -1)

    Kinv = np.linalg.inv(K)
    q = Kinv @ p
    q_d = PiInv_v(distortion(Pi(q), distCoeffs))
    p_d = K @ q_d
    
    x_d = p_d[0].reshape(x.shape).astype(np.float32)
    y_d = p_d[1].reshape(y.shape).astype(np.float32)
    assert (p_d[2]==1).all(), 'You messed up.'
    im_undistorted = cv2.remap(im, x_d, y_d, cv2.INTER_LINEAR)
    return im_undistorted

def cross_op(x:np.ndarray):
    x1i = x[0].item()
    y1i = x[1].item()
    return np.array([[0, -1, y1i], [1, 0, -x1i], [-y1i, x1i, 0]])

PiInv = lambda p: np.vstack((p, np.ones(p.shape[1]))) if p.ndim > 1 \
    else np.append(p, 1) # Imhomogenous to homogenous
    
def hest(qs,ps):
    """
    INPUT:
        - List of arrays:
        [array([1, 1]), array([0, 3]), array([2, 3]), array([2, 4])]
    
    Estimate homography from two sets of points"""
    N = qs[0].shape[0]+1
    K = len(qs)
    B = np.zeros((N*K, N**2))

    for i, (q,p) in enumerate(zip(qs,ps)):
        q = PiInv(q)
        p = PiInv(p)
        # print(i*N,(i+1)*N)
        B[i*N:(i+1)*N] = np.kron(p.T, cross_op(q))

    U,S, Vh = np.linalg.svd(B)
    V = Vh.T
    H = V[:,-1].reshape(3,3).T
    
    return H

def hestRune(
        q1: np.array,
        q2: np.array,
        normalize: bool=True
    ):
    """
    Estimates the Homography from q2 to q1.
    q1 and q2 are homogenous points."""
    if normalize:
        T1, T1inv = normalize2d(Pi(q1))
        T2, T2inv = normalize2d(Pi(q2))
        q1, q2 = T1 @ q1, T2 @ q2
    
    CrossOp = lambda q: np.cross(q, np.identity(q.size) * -1)
    B = np.vstack([
        np.kron(q2i, CrossOp(q1i)) 
        for (q1i, q2i) in zip(q1.T, q2.T)
    ])

    u, s, vh = np.linalg.svd(B)
    hflat = vh[-1]
    
    h = hflat.reshape(3, 3).T
    return T1inv @ h @ T2 if normalize else h

def normalize2d(
        p: np.array
    ):
    """p are inhomogenous points."""
    mu = p.mean(1)
    std = p.std(1)
    Tinv = np.array([
        [std[0], 0, mu[0]],
        [0, std[1], mu[1]],
        [0, 0, 1]
    ])
    T = np.linalg.inv(Tinv)
    return T, Tinv

def normalize3d(
        p: np.array
    ):
    """p are inhomogenous points."""
    mu = p.mean(1)
    std = p.std(1)
    Tinv = np.array([
        [std[0], 0, 0, mu[0]],
        [0, std[1], 0, mu[1]],
        [0, 0, std[2], mu[2]],
        [0, 0, 0, 1]
    ])
    T = np.linalg.inv(Tinv)
    return T, Tinv

    
    
def warpImage(im, H):
    imWarp = cv2.warpPerspective(im, H, (im.shape[1], im.shape[0]))
    return imWarp

###########################
######### WEEK 3 ##########
###########################

def triangulate(
        q: np.array, 
        P: np.array
    ) -> np.array:
    B = np.vstack([
        np.array([
            Pj[2] * xj - Pj[0], Pj[2] * yj - Pj[1]
            ])
        for Pj, (xj, yj) in zip(P, q.T)])
    u, s, v = np.linalg.svd(B)
    Q = v[-1]
    return Pi(Q)

###########################
######### WEEK 4 ##########
###########################

def estimateHomographies(
        Q_omega: np.array, 
        qs):
    # Construct Q tilde.
    Q_tilde = Q_omega.copy()
    Q_tilde[-1] = 1
    # Estimate homographies.
    Hs = [hestRune(qi, Q_tilde) for qi in qs]
    return Hs


###########################
######### WEEK 13 #########
###########################
# Courtesey of Yucheng
import re
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)


