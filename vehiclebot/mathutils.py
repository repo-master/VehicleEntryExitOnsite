
import numpy as np

def angle_between(ang1, ang2):
    delta = np.abs(ang1 - ang2)
    return np.minimum(delta, 2*np.pi-delta)

def angle_between_vectors(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return angle_between(ang1, ang2)

def point_angle(p1, p2):
    delta = np.array(p1[::-1]) - np.array(p2[::-1])
    return np.arctan2(*delta)
