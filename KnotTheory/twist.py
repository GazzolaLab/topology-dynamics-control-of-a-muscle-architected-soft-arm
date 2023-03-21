__docs__ = """
This script is for computing the twist of a rod using the method 2a from Klenin & Langowski 2000 paper.
"""

import numpy as np
from numba import njit

from elastica._elastica_numba._linalg import _batch_norm, _batch_dot, _batch_cross

"""
Following codes are adapted from section S2 of Charles et. al. PRL 2019 paper.
"""


def compute_twist(center_line, normal_collection):
    """
    Compute the twist of a rod, using center_line and normal collection. Methods used in this function is
    adapted from method 2a Klenin & Langowski 2000 paper.

    Parameters
    ----------
    center_line : numpy.ndarray
        3D (time, 3, n_nodes) array containing data with 'float' type.
        Time history of rod node positions.
    normal_collection : numpy.ndarray
        3D (time, 3, n_elems) array containing data with 'float' type.
        Time history of rod elements normal direction.

    Warnings
    -------
    If center line is straight, although the normals of each element is pointing different direction computed twist
    will be zero.

    Returns
    -------

    """

    total_twist, local_twist = _compute_twist(center_line, normal_collection)

    return total_twist, local_twist


@njit(cache=True)
def _compute_twist(center_line, normal_collection):
    """
    Compute the twist of a rod, using center_line and normal collection. Methods used in this function is
    adapted from method 2a Klenin & Langowski 2000 paper.

    Parameters
    ----------
    center_line : numpy.ndarray
        3D (time, 3, n_nodes) array containing data with 'float' type.
        Time history of rod node positions.
    normal_collection : numpy.ndarray
        3D (time, 3, n_elems) array containing data with 'float' type.
        Time history of rod elements normal direction.

    Returns
    -------

    """

    timesize, _, blocksize = center_line.shape

    total_twist = np.zeros((timesize))
    local_twist = np.zeros((timesize, blocksize - 2))  # Only consider interior nodes

    # compute s vector
    for k in range(timesize):
        # s is a secondary vector field.
        s = center_line[k, :, 1:] - center_line[k, :, :-1]
        # Compute tangents
        tangent = s / _batch_norm(s)

        # Compute the projection of normal collection (d1) on normal-binormal plane.
        projection_of_normal_collection = (
            normal_collection[k, :, :]
            - _batch_dot(tangent, normal_collection[k, :, :]) * tangent
        )
        projection_of_normal_collection /= _batch_norm(projection_of_normal_collection)

        # Eq 27 in Klenin & Langowski 2000
        # p is defined on interior nodes
        p = _batch_cross(s[:, :-1], s[:, 1:])
        p /= _batch_norm(p)

        # Compute the angle we need to turn d1 around s to get p
        # sign part tells whether d1 must be rotated ccw(+) or cw(-) around s
        alpha = np.sign(
            _batch_dot(
                _batch_cross(projection_of_normal_collection[:, :-1], p), s[:, :-1]
            )
        ) * np.arccos(_batch_dot(projection_of_normal_collection[:, :-1], p))

        gamma = np.sign(
            _batch_dot(
                _batch_cross(p, projection_of_normal_collection[:, 1:]), s[:, 1:]
            )
        ) * np.arccos(_batch_dot(projection_of_normal_collection[:, 1:], p))

        # An angle 1 is a full rotation, 0.5 is rotation by pi, 0.25 is pi/2 etc.
        alpha /= 2 * np.pi
        gamma /= 2 * np.pi
        twist_temp = alpha + gamma
        # Make sure twist is between (-1/2 to 1/2) as defined in pg 313 Klenin & Langowski 2000
        idx = np.where(twist_temp > 0.5)[0]
        twist_temp[idx] -= 1
        idx = np.where(twist_temp < -0.5)[0]
        twist_temp[idx] += 1

        # Check if there is any Nan. Nan's appear when rod tangents are parallel to each other.
        idx = np.where(np.isnan(twist_temp))[0]
        twist_temp[idx] = 0.0

        local_twist[k, :] = twist_temp
        total_twist[k] = np.sum(twist_temp)

    return total_twist, local_twist
