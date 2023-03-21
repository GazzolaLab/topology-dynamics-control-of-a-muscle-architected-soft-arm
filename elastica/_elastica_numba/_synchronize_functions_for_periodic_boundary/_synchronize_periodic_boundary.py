__doc__ = """These functions are used to synchronize periodic boundaries for circular rods. Numba implementations are
located in this file. """

import numpy as np

import numba
from numba import njit


@njit(cache=True)
def _synchronize_periodic_boundary_of_vector_collection(input, periodic_idx):
    """
    This function synchronizes the periodic boundaries of a vector collection.
    Parameters
    ----------
    input : numpy.ndarray
        2D (dim, blocksize) array containing data with 'float' type. Vector that is going to be synched.
    periodic_idx : numpy.ndarray
        2D (2, n_periodic_boundary) array containing data with 'float' type. Vector containing periodic boundary
        index. First dimension is the periodic boundary index, second dimension is the referenced cell index.

    Returns
    -------

    """
    for i in range(3):
        for k in range(periodic_idx.shape[1]):
            input[i, periodic_idx[0, k]] = input[i, periodic_idx[1, k]]


@njit(cache=True)
def _synchronize_periodic_boundary_of_matrix_collection(input, periodic_idx):
    """
    This function synchronizes the periodic boundaries of a matrix collection.
    Parameters
    ----------
    input : numpy.ndarray
        2D (dim, dim, blocksize) array containing data with 'float' type. Matrix collection that is going to be synched.
    periodic_idx : numpy.ndarray
        2D (2, n_periodic_boundary) array containing data with 'float' type. Vector containing periodic boundary
        index. First dimension is the periodic boundary index, second dimension is the referenced cell index.

    Returns
    -------

    """
    for i in range(3):
        for j in range(3):
            for k in range(periodic_idx.shape[1]):
                input[i, j, periodic_idx[0, k]] = input[i, j, periodic_idx[1, k]]


@njit(cache=True)
def _synchronize_periodic_boundary_of_scalar_collection(input, periodic_idx):
    """
    This function synchronizes the periodic boundaries of a scalar collection.

    Parameters
    ----------
    input : numpy.ndarray
        2D (dim, dim, blocksize) array containing data with 'float' type. Scalar collection that is going to be synched.
    periodic_idx : numpy.ndarray
        2D (2, n_periodic_boundary) array containing data with 'float' type. Vector containing periodic boundary
        index. First dimension is the periodic boundary index, second dimension is the referenced cell index.

    Returns
    -------

    """
    for k in range(periodic_idx.shape[1]):
        input[periodic_idx[0, k]] = input[periodic_idx[1, k]]


@njit(cache=True)
def _synchronize_periodic_boundary_of_nine_dim_vector_collection(input, periodic_idx):
    """
    This function synchronizes the periodic boundaries of a nine dimensional vector collection.
    Parameters
    ----------
    input : numpy.ndarray
        2D (dim, blocksize) array containing data with 'float' type. Vector that is going to be synched.
    periodic_idx : numpy.ndarray
        2D (2, n_periodic_boundary) array containing data with 'float' type. Vector containing periodic boundary
        index. First dimension is the periodic boundary index, second dimension is the referenced cell index.

    Returns
    -------

    """
    for i in range(9):
        for k in range(periodic_idx.shape[1]):
            input[i, periodic_idx[0, k]] = input[i, periodic_idx[1, k]]
