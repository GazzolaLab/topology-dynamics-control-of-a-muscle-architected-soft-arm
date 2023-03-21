__docs__ = """
This script is for computing the link of a rod using the method 1a from Klenin & Langowski 2000 paper.
"""

import numpy as np
from numba import njit
from KnotTheory.pre_processing import *
from elastica._elastica_numba._linalg import _batch_norm, _batch_dot

"""
Following codes are adapted from section S2 of Charles et. al. PRL 2019 paper.
"""


@njit(cache=True)
def _compute_auxiliary_line(center_line, normal_collection, radius):
    """
    This function computes the auxiliary line using rod center line and normal collection.

    Parameters
    ----------
    center_line : numpy.ndarray
        3D (time, 3, n_nodes) array containing data with 'float' type.
        Time history of rod node positions.
    normal_collection : numpy.ndarray
        3D (time, 3, n_elems) array containing data with 'float' type.
        Time history of rod elements normal direction.
    radius : numpy.ndarray
        2D (time, n_elems) array containing data with 'float' type.
        Time history of rod element radius.

    Returns
    -------

    """
    time, _, blocksize = center_line.shape
    auxiliary_line = np.zeros(center_line.shape)
    projection_of_normal_collection = np.zeros((3, blocksize))
    radius_on_nodes = np.zeros((blocksize))

    for i in range(time):
        tangent = center_line[i, :, 1:] - center_line[i, :, :-1]
        tangent /= _batch_norm(tangent)
        # Compute the projection of normal collection (d1) on normal-binormal plane.
        projection_of_normal_collection_temp = (
            normal_collection[i, :, :]
            - _batch_dot(tangent, normal_collection[i, :, :]) * tangent
        )
        projection_of_normal_collection_temp /= _batch_norm(
            projection_of_normal_collection_temp
        )

        # First node have the same direction with second node. They share the same element.
        # TODO: Instead of this maybe we should use the trapezoidal rule or averaging operator for normal and radius.
        projection_of_normal_collection[:, 0] = projection_of_normal_collection_temp[
            :, 0
        ]
        projection_of_normal_collection[:, 1:] = projection_of_normal_collection_temp[:]
        radius_on_nodes[0] = radius[i, 0]
        radius_on_nodes[1:] = radius[i, :]

        auxiliary_line[i, :, :] = (
            radius_on_nodes * projection_of_normal_collection + center_line[i, :, :]
        )

    return auxiliary_line


@njit(cache=True)
def _compute_link(center_line, auxiliary_line):
    """
    This function computes the total link history of a rod.
    Equations used are from method 1a from Klenin & Langowski 2000 paper.

    Parameters
    ----------
    center_line : numpy.ndarray
        3D (time, 3, n_nodes) array containing data with 'float' type.
        Time history of rod node positions.
    auxiliary_line : numpy.ndarray
        3D (time, 3, n_nodes) array containing data with 'float' type.
        Time history of auxiliary line.

    Returns
    -------

    """
    timesize, _, blocksize_center_line = center_line.shape
    blocksize_auxiliary_line = auxiliary_line.shape[-1]

    omega_star = np.zeros((blocksize_center_line - 1, blocksize_auxiliary_line - 1))
    segment_link = np.zeros((blocksize_center_line - 1, blocksize_auxiliary_line - 1))
    total_link = np.zeros((timesize))

    # Compute the writhe between each pair first.
    for k in range(timesize):
        for i in range(blocksize_center_line - 1):
            for j in range(blocksize_auxiliary_line - 1):

                point_one = center_line[k, :, i]
                point_two = center_line[k, :, i + 1]
                point_three = auxiliary_line[k, :, j]
                point_four = auxiliary_line[k, :, j + 1]

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
                segment_link[i, j] = (
                    omega_star[i, j]
                    * np.sign(np.dot(np.cross(r34, r12), r13))
                    / (4 * np.pi)
                )

        # Compute the total writhe
        # Eq 6 in Klenin & Langowski 2000
        # Unlike the writhe, link computed using two curves so we do not multiply by 2
        total_link[k] = np.sum(segment_link)

    return total_link


@njit(cache=True)
def _compute_auxiliary_line_added_segments(
    beginning_direction, end_direction, auxiliary_line, segment_length
):
    """
    This code is for computing position of added segments to the auxiliary line.

    Parameters
    ----------
    beginning_direction : numpy.ndarray
        2D (time, 3) array containing data with 'float' type.
        Time history of center line tangent at the beginning.
    end_direction : numpy.ndarray
        2D (time, 3) array containing data with 'float' type.
        Time history of center line tangent at the end.
    auxiliary_line : numpy.ndarray
        3D (time, 3, n_nodes) array containing data with 'float' type.
        Time history of auxiliary line.
    segment_length : float
        Length of added segments.

    Returns
    -------

    """
    timesize, _, blocksize = auxiliary_line.shape

    new_auxiliary_line = np.zeros((timesize, 3, blocksize + 2))

    new_auxiliary_line[:, :, 1:-1] = auxiliary_line

    new_auxiliary_line[:, :, 0] = (
        auxiliary_line[:, :, 0] + beginning_direction * segment_length
    )

    new_auxiliary_line[:, :, -1] = (
        auxiliary_line[:, :, -1] + end_direction * segment_length
    )

    return new_auxiliary_line


def compute_link(
    center_line, normal_collection, radius, segment_length, type_of_additional_segment
):
    """
    Compute the link of a rod.

    Parameters
    ----------
    center_line : numpy.ndarray
        3D (time, 3, n_nodes) array containing data with 'float' type.
        Time history of rod node positions.
    normal_collection : numpy.ndarray
        3D (time, 3, n_elems) array containing data with 'float' type.
        Time history of rod elements normal direction.
    radius : numpy.ndarray
        2D (time, n_elems) array containing data with 'float' type.
        Time history of rod element radius.
    segment_length : float
        Length of added segments.
    type_of_additional_segment : str
        Determines the method to compute new segments (elements) added to the rod.
        Valid inputs are "next_tangent", "end_to_end", "net_tangent", otherwise program uses the center line.

    Returns
    -------

    """

    # Compute auxiliary line
    auxiliary_line = _compute_auxiliary_line(center_line, normal_collection, radius)

    # Add segments at the beginning and end of the rod center line and auxiliary line.
    (
        center_line_with_added_segments,
        beginning_direction,
        end_direction,
    ) = compute_additional_segment(
        center_line, segment_length, type_of_additional_segment
    )
    auxiliary_line_with_added_segments = _compute_auxiliary_line_added_segments(
        beginning_direction, end_direction, auxiliary_line, segment_length
    )

    """
    Total link of a rod is computed using the method 1a from Klenin & Langowski 2000 
    """
    total_link = _compute_link(
        center_line_with_added_segments, auxiliary_line_with_added_segments
    )

    return total_link
