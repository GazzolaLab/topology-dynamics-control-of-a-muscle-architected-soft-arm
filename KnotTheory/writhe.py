__docs__ = """
This script is for computing the writhe of a rod using the method 1a from Klenin & Langowski 2000 paper.
"""
import numpy as np
from numba import njit
from KnotTheory.pre_processing import *


"""
Following codes are adapted from section S2 of Charles et. al. PRL 2019 paper.
"""


@njit(cache=True)
def _compute_writhe(center_line):
    """
    This function computes the total writhe history of a rod.
    Equations used are from method 1a from Klenin & Langowski 2000 paper.

    Parameters
    ----------
    center_line : numpy.ndarray
        3D (time, 3, n_nodes) array containing data with 'float' type.
        Time history of rod node positions.

    Returns
    -------

    """

    time, _, blocksize = center_line.shape

    omega_star = np.zeros((blocksize - 2, blocksize - 1))
    segment_writhe = np.zeros((blocksize - 2, blocksize - 1))
    total_writhe = np.zeros((time))

    # Compute the writhe between each pair first.
    for k in range(time):
        for i in range(blocksize - 2):
            for j in range(i + 1, blocksize - 1):

                point_one = center_line[k, :, i]
                point_two = center_line[k, :, i + 1]
                point_three = center_line[k, :, j]
                point_four = center_line[k, :, j + 1]

                # Eq 15 in Klenin & Langowski 2000
                r12 = point_two - point_one
                r34 = point_four - point_three
                r14 = point_four - point_one
                r13 = point_three - point_one
                r23 = point_three - point_two
                r24 = point_four - point_two

                n1 = np.cross(r13, r14)
                n1 /= np.linalg.norm(n1)
                n2 = np.cross(r14, r24)
                n2 /= np.linalg.norm(n2)
                n3 = np.cross(r24, r23)
                n3 /= np.linalg.norm(n3)
                n4 = np.cross(r23, r13)
                n4 /= np.linalg.norm(n4)

                # Eq 16a in Klenin & Langowski 2000
                omega_star[i, j] = (
                    np.arcsin(np.dot(n1, n2))
                    + np.arcsin(np.dot(n2, n3))
                    + np.arcsin(np.dot(n3, n4))
                    + np.arcsin(np.dot(n4, n1))
                )

                if np.isnan(omega_star[i, j]):
                    omega_star[i, j] = 0

                # Eq 16b in Klenin & Langowski 2000
                segment_writhe[i, j] = (
                    omega_star[i, j]
                    * np.sign(np.dot(np.cross(r34, r12), r13))
                    / (4 * np.pi)
                )

        # Compute the total writhe
        # Eq 13 in Klenin & Langowski 2000
        total_writhe[k] = 2 * np.sum(segment_writhe)

    return total_writhe, segment_writhe


def compute_writhe(center_line, segment_length, type_of_additional_segment):
    """
    This function computes the total writhe history of a rod.

    Parameters
    ----------
    center_line : numpy.ndarray
        3D (time, 3, n_nodes) array containing data with 'float' type.
        Time history of rod node positions.
    segment_length : float
        Length of added segments.
    type_of_additional_segment : str
        Determines the method to compute new segments (elements) added to the rod.
        Valid inputs are "next_tangent", "end_to_end", "net_tangent", otherwise program uses the center line.

    Returns
    -------

    """

    center_line_with_added_segments, _, _ = compute_additional_segment(
        center_line, segment_length, type_of_additional_segment
    )

    """
    Total writhe of a rod is computed using the method 1a from Klenin & Langowski 2000 
    """
    total_writhe, segment_writhe = _compute_writhe(center_line_with_added_segments)

    return total_writhe, segment_writhe
