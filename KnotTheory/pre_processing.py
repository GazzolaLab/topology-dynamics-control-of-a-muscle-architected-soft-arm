__docs__ = """This script is for computing additional segments (elements) added to beginning and end of center line."""
__all__ = ["compute_additional_segment"]
import numpy as np
from numba import njit
from numpy import sqrt

"""
Following codes are adapted from section S2 of Charles et. al. PRL 2019 paper.
"""


@njit(cache=True)
def _compute_additional_segment_using_next_tangent(center_line, segment_length):
    """
    This function adds a two new point at the begining and end of the center line. Distance of these points are
    given in segment_length. Direction of these points are computed using the rod tangents at the begining and end.

    Parameters
    ----------
    center_line : numpy.ndarray
        3D (time, 3, n_nodes) array containing data with 'float' type.
        Time history of rod node positions.
    segment_length : float
        Length of added segments.

    Returns
    -------

    """

    timesize, _, blocksize = center_line.shape
    new_center_line = np.zeros(
        (timesize, 3, blocksize + 2)
    )  # +2 is for added two new points
    beginning_direction = np.zeros((timesize, 3))
    end_direction = np.zeros((timesize, 3))

    for i in range(timesize):
        # Direction of the additional point at the beginning of the rod
        direction_of_rod_begin = center_line[i, :, 0] - center_line[i, :, 1]
        direction_of_rod_begin /= np.linalg.norm(direction_of_rod_begin)

        # Direction of the additional point at the end of the rod
        direction_of_rod_end = center_line[i, :, -1] - center_line[i, :, -2]
        direction_of_rod_end /= np.linalg.norm(direction_of_rod_end)

        first_point = center_line[i, :, 0] + segment_length * direction_of_rod_begin
        last_point = center_line[i, :, -1] + segment_length * direction_of_rod_end

        new_center_line[i, :, 1:-1] = center_line[i, :, :]
        new_center_line[i, :, 0] = first_point
        new_center_line[i, :, -1] = last_point
        beginning_direction[i, :] = direction_of_rod_begin
        end_direction[i, :] = direction_of_rod_end

    return new_center_line, beginning_direction, end_direction


@njit(cache=True)
def _compute_additional_segment_using_end_to_end(center_line, segment_length):
    """
    This function adds a two new point at the begining and end of the center line. Distance of these points are
    given in segment_length. Direction of these points are computed using the rod node end positions.

    Parameters
    ----------
    center_line : numpy.ndarray
        3D (time, 3, n_nodes) array containing data with 'float' type.
        Time history of rod node positions.
    segment_length : float
        Length of added segments.

    Returns
    -------

    """
    timesize, _, blocksize = center_line.shape
    new_center_line = np.zeros(
        (timesize, 3, blocksize + 2)
    )  # +2 is for added two new points
    beginning_direction = np.zeros((timesize, 3))
    end_direction = np.zeros((timesize, 3))

    for i in range(timesize):
        # Direction of the additional point at the beginning of the rod
        direction_of_rod_begin = center_line[i, :, 0] - center_line[i, :, -1]
        direction_of_rod_begin /= np.linalg.norm(direction_of_rod_begin)

        # Direction of the additional point at the end of the rod
        direction_of_rod_end = -direction_of_rod_begin

        first_point = center_line[i, :, 0] + segment_length * direction_of_rod_begin
        last_point = center_line[i, :, -1] + segment_length * direction_of_rod_end

        new_center_line[i, :, 1:-1] = center_line[i, :, :]
        new_center_line[i, :, 0] = first_point
        new_center_line[i, :, -1] = last_point

        beginning_direction[i, :] = direction_of_rod_begin
        end_direction[i, :] = direction_of_rod_end

    return new_center_line, beginning_direction, end_direction


@njit(cache=True)
def _compute_additional_segment_using_net_tangent(center_line, segment_length):
    """
    This function adds a two new point at the begining and end of the center line. Distance of these points are
    given in segment_length. Direction of these points are point wise avarege of nodes at the first and second half
    of the rod.

    Parameters
    ----------
    center_line : numpy.ndarray
        3D (time, 3, n_nodes) array containing data with 'float' type.
        Time history of rod node positions.
    segment_length : float
        Length of added segments.

    Returns
    -------

    """
    timesize, _, blocksize = center_line.shape
    new_center_line = np.zeros(
        (timesize, 3, blocksize + 2)
    )  # +2 is for added two new points
    beginning_direction = np.zeros((timesize, 3))
    end_direction = np.zeros((timesize, 3))

    for i in range(timesize):
        # Direction of the additional point at the beginning of the rod
        n_nodes_begin = int(np.floor(blocksize / 2))
        average_begin = (
            np.sum(center_line[i, :, :n_nodes_begin], axis=1) / n_nodes_begin
        )
        n_nodes_end = int(np.ceil(blocksize / 2))
        average_end = np.sum(center_line[i, :, n_nodes_end:], axis=1) / (
            blocksize - n_nodes_end + 1
        )

        direction_of_rod_begin = average_begin - average_end
        direction_of_rod_begin /= np.linalg.norm(direction_of_rod_begin)

        direction_of_rod_end = -direction_of_rod_begin

        first_point = center_line[i, :, 0] + segment_length * direction_of_rod_begin
        last_point = center_line[i, :, -1] + segment_length * direction_of_rod_end

        new_center_line[i, :, 1:-1] = center_line[i, :, :]
        new_center_line[i, :, 0] = first_point
        new_center_line[i, :, -1] = last_point

        beginning_direction[i, :] = direction_of_rod_begin
        end_direction[i, :] = direction_of_rod_end

    return new_center_line, beginning_direction, end_direction


# @njit(cache=True)
# def _compute_additional_segment_using_average_tangent(center_line, segment_length, n_avg_points=1):
#     """
#     This function adds a two new point at the begining and end of the center line. Distance of these points are
#     given in segment_length. Direction of these points are point wise avarege of nodes at the first and second half
#     of the rod.
#
#     Parameters
#     ----------
#     center_line : numpy.ndarray
#         3D (time, 3, n_nodes) array containing data with 'float' type.
#         Time history of rod node positions.
#     segment_length : float
#         Length of added segments.
#
#     Returns
#     -------
#
#     """
#     timesize, _, blocksize = center_line.shape
#     new_center_line = np.zeros(
#         (timesize, 3, blocksize + 2)
#     )  # +2 is for added two new points
#     beginning_direction = np.zeros((timesize, 3))
#     end_direction = np.zeros((timesize, 3))
#
#     for i in range(timesize):
#         # Direction of the additional point at the beginning of the rod
#
#         tangent = center_line[i, :, 1:] - center_line[i, :, :-1]
#         tangent /= _batch_norm(tangent)
#
#         direction_of_rod_begin = np.zeros((3))
#         direction_of_rod_end = np.zeros((3))
#
#         direction_of_rod_begin[0] = np.mean(tangent[0, :n_avg_points])
#         direction_of_rod_begin[1] = np.mean(tangent[1, :n_avg_points])
#         direction_of_rod_begin[2] = np.mean(tangent[2, :n_avg_points])
#
#         direction_of_rod_end[0] = np.mean(tangent[0, -n_avg_points:])
#         direction_of_rod_end[1] = np.mean(tangent[1, -n_avg_points:])
#         direction_of_rod_end[2] = np.mean(tangent[2, -n_avg_points:])
#
#
#         first_point = center_line[i, :, 0] + segment_length * direction_of_rod_begin
#         last_point = center_line[i, :, -1] + segment_length * direction_of_rod_end
#
#         new_center_line[i, :, 1:-1] = center_line[i, :, :]
#         new_center_line[i, :, 0] = first_point
#         new_center_line[i, :, -1] = last_point
#
#         beginning_direction[i, :] = direction_of_rod_begin
#         end_direction[i, :] = direction_of_rod_end
#
#     return new_center_line, beginning_direction, end_direction

def compute_additional_segment(center_line, segment_length, type_of_additional_segment):
    """
    This function adds two points at the end of center line. Distance from the center line is given by segment_length.
    Direction from center line to the new point locations can be computed using 3 methods, which can be selected by
    type. For more details section S2 of Charles et. al. PRL 2019 paper.

    Parameters
    ----------
    center_line : numpy.ndarray
        3D (time, 3, n_nodes) array containing data with 'float' type.
        Time history of rod node positions.
    segment_length : float
        Length of added segments.
    type_of_additional_segment : str
        Determines the method to compute new segments (elements) added to the rod.
        Valid inputs are "next_tangent", "end_to_end", "net_tangent", otherwise it returns the center line.

    Returns
    -------

    """

    if type_of_additional_segment == "next_tangent":
        (
            center_line,
            beginning_direction,
            end_direction,
        ) = _compute_additional_segment_using_next_tangent(center_line, segment_length)
    elif type_of_additional_segment == "end_to_end":
        (
            center_line,
            beginning_direction,
            end_direction,
        ) = _compute_additional_segment_using_end_to_end(center_line, segment_length)
    elif type_of_additional_segment == "net_tangent":
        (
            center_line,
            beginning_direction,
            end_direction,
        ) = _compute_additional_segment_using_net_tangent(center_line, segment_length)
    # elif type_of_additional_segment == "average_tangent":
    #     (
    #         center_line,
    #         beginning_direction,
    #         end_direction,
    #     ) = _compute_additional_segment_using_average_tangent(center_line, segment_length,n_avg_points)
    else:
        print("Additional segments are not used")
        beginning_direction = np.zeros((center_line.shape[0], 3))
        end_direction = np.zeros((center_line.shape[0], 3))

    return center_line, beginning_direction, end_direction
