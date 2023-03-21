__doc__ = """Numba implementation of connecting inner and outer ring rods"""
__all__ = [
    "get_ring_ring_connection_reference_index",
    "OuterRingRingRodConnectionDifferentLevel",
]
import numpy as np
import numba
from numba import njit
import copy
from elastica.joint import FreeJoint
from elastica._elastica_numba._linalg import _batch_norm, _batch_dot, _batch_cross
from numpy.testing import assert_allclose


def get_ring_ring_connection_reference_index(rod_one, rod_two, n_straight_rods):
    """
    This function is computing reference indices for connecting two neighbor ring rods. In order to connect
    two ring rods, a vector pointing from ring rod one to ring rod two have to be computed. First we find a unit
    vector pointing from ring rod one element 0 to element/2 and called this as unit_circle_vector_one. Second we find
    another vector perpendicular to unit_circle_vector_one which also lives on the same plane as rod_one. This vector
    is called unit_circle_vector_two. Cross product of these two vectors should be parallel to a vector which is
    from center of ring rod one to ring rod two. If this is the case then this function returns the reference index
    element of rod one to compute a vector from rod_one to rod_two.


    Parameters
    ----------
    rod_one
    rod_two

    Returns
    -------

    """

    center_of_ring_rod_one = np.mean(rod_one.position_collection, axis=1)
    center_of_ring_rod_two = np.mean(rod_two.position_collection, axis=1)

    # This is the vector pointing from rod one center to rod two center.
    vector_from_rod_one_to_rod_two = center_of_ring_rod_two - center_of_ring_rod_one
    vector_from_rod_one_to_rod_two /= np.linalg.norm(vector_from_rod_one_to_rod_two)

    assert (
        rod_one.n_elems == rod_two.n_elems
    ), " Ring rods should have same number of elements "
    assert rod_one.n_elems % 4 == 0, "Ring rods elements should be multiple of 4"

    index_connection = 0
    index_connection_opposite = int(
        (index_connection + rod_one.n_elems / 2) % rod_one.n_elems
    )
    index_reference = int((index_connection + rod_one.n_elems / 4) % rod_one.n_elems)
    index_reference_opposite = int(
        (index_connection + 3 * rod_one.n_elems / 4) % rod_one.n_elems
    )

    # Unit circle vector one is a vector starting from 0th element and pointing toward the n_elem/2. We call it unit
    # circle because we will unitize it.
    unit_circle_vector_one = (
        rod_one.position_collection[..., index_connection_opposite]
        - rod_one.position_collection[..., index_connection]
    )
    unit_circle_vector_one /= np.linalg.norm(unit_circle_vector_one)

    # unit circle vector two is a vector living on the plane of unit circle where ring rod one lives. Also it is
    # perpendicular to the unit circle vector.
    unit_circle_vector_two = (
        rod_one.position_collection[..., index_reference_opposite]
        - rod_one.position_collection[..., index_reference]
    )
    unit_circle_vector_two /= np.linalg.norm(unit_circle_vector_two)

    # Cross product of unit_circle_vector_one and unit_circle_vector_two gives a unit vector perpendicular to the unit
    # circle. This should be the direction a vector pointing from center of rod one to rod two.
    perpendicular_vector = np.cross(unit_circle_vector_one, unit_circle_vector_two)

    # If perpendicular_vector is parallel to vector_from_rod_one_to_rod_two then indexes calculated are correct and
    # can be used for connections. Otherwise we need to change the index order so perpendicular_vector is parallel
    # to vector_from_rod_one_to_rod_two.

    if not np.dot(perpendicular_vector, vector_from_rod_one_to_rod_two) == 1:
        index_two_temp = index_reference
        index_two_temp_opposite = index_reference_opposite

        # change the index_two and index_reference_opposite
        index_reference = index_two_temp_opposite
        index_reference_opposite = index_two_temp

        unit_circle_vector_two = (
            rod_one.position_collection[..., index_reference_opposite]
            - rod_one.position_collection[..., index_reference]
        )
        unit_circle_vector_two /= np.linalg.norm(unit_circle_vector_two)

        perpendicular_vector = np.cross(unit_circle_vector_one, unit_circle_vector_two)

        # assert (
        #     np.dot(perpendicular_vector, vector_from_rod_one_to_rod_two) == 1
        # ), " Check if ring rods are parallel  "
        assert_allclose(
            np.dot(perpendicular_vector, vector_from_rod_one_to_rod_two),
            1.0,
            atol=1e-14,
            err_msg="Check if ring rods are parallel",
        )

    # Find the all connection indexes between ring rod one and two. Number of connections depends on number of straight
    # rods (without including the center straight rod).
    index_connection_list = []
    index_connection_opposite_list = []
    index_reference_list = []
    index_reference_opposite_list = []

    n_elem_skip = int(rod_one.n_elems / n_straight_rods)
    for i in range(n_straight_rods):
        index_connection_list.append(
            (i * n_elem_skip + index_connection) % rod_one.n_elems
        )
        index_connection_opposite_list.append(
            (i * n_elem_skip + index_connection_opposite) % rod_one.n_elems
        )
        index_reference_list.append(
            (i * n_elem_skip + index_reference) % rod_one.n_elems
        )
        index_reference_opposite_list.append(
            (i * n_elem_skip + index_reference_opposite) % rod_one.n_elems
        )

    index_connection = np.array(index_connection_list, dtype=np.int)
    index_connection_opposite = np.array(index_connection_opposite_list, dtype=np.int)
    index_reference = np.array(index_reference_list, dtype=np.int)
    index_reference_opposite = np.array(index_reference_opposite_list, dtype=np.int)

    return (
        index_connection,
        index_connection_opposite,
        index_reference,
        index_reference_opposite,
    )


class OuterRingRingRodConnectionDifferentLevel(FreeJoint):
    """
    This class is for connecting neighbor outer ring rods of the arm. This class is specific for outer ring rods
    because, it uses outer straight rod for connecting ring rods. Ring rods are stack on top of each other with some
    distance and ring rods are attached to the elements of the straight rods. Thus, the distance between ring rods
    can be computed using these straight rod element positions.

    """

    def __init__(
        self,
        k,
        index_connection,
        index_connection_opposite,
        index_reference,
        index_reference_opposite,
        offset_start_idx,
        offset_end_idx,
        **kwargs
    ):
        """

        Parameters
        ----------
        k : float
            spring constant between ring rods.
        reference_index_list : np.array
            1D (4) array containing data type int.
        offset_list : list
            Position references of the straight rod nodes. This is list of lists, there are number of straight rod
            lists. Each list contains two numpy arrays. These arrays are references to the straight rod position.
            First array is the neighbor nodes of the element which ring rod_one is connected to that straight rod and
            Second array is the neighbor nodes of the element which ring rod_two is connected to that straight rod.
        radial_offset_list : list
            Radius references of the straight rod elements.  This is list of lists, there are number of straight rod
            lists. Each of these lists contains two numpy arrays. These arrays are references to the straight rod
            radius. First array corresponds to the location where ring rod_one is connected to straight rod and
            Second array corresponds to the location where ring rod_two is connected to straight rod.
        """
        super().__init__(k, nu=0)

        self.k = np.array(k).flatten()

        self.offset_start_idx = offset_start_idx[0]
        self.offset_end_idx = offset_end_idx[0]

        index_one = copy.deepcopy(index_connection)
        index_two = copy.deepcopy(index_connection)

        first_sys_idx_offset = np.array(kwargs["first_sys_idx_offset"], dtype=np.int)
        second_sys_idx_offset = np.array(kwargs["second_sys_idx_offset"], dtype=np.int)
        for i in range(len(index_connection)):
            index_one[i] += first_sys_idx_offset[i]
            index_connection_opposite[i] += first_sys_idx_offset[i]
            index_reference[i] += first_sys_idx_offset[i]
            index_reference_opposite[i] += first_sys_idx_offset[i]
            index_two[i] += second_sys_idx_offset[i]

        self.index_one = np.array(index_one, dtype=np.int).flatten()
        self.index_one_opposite = np.array(
            index_connection_opposite, dtype=np.int
        ).flatten()
        self.index_reference = np.array(index_reference, dtype=np.int).flatten()
        self.index_reference_opposite = np.array(
            index_reference_opposite, dtype=np.int
        ).flatten()
        self.index_two = np.array(index_two, dtype=np.int).flatten()

    def apply_forces(self, rod_one, index_one, rod_two, index_two):
        del index_one, index_two

        self._apply_forces(
            self.k,
            self.index_one,
            self.index_two,
            self.index_one_opposite,
            self.index_reference,
            self.index_reference_opposite,
            self.offset_start_idx,
            self.offset_end_idx,
            rod_one.radius,
            rod_two.radius,
            rod_one.position_collection,
            rod_two.position_collection,
            rod_one.external_forces,
            rod_two.external_forces,
        )

    @staticmethod
    @njit(cache=True)
    def _apply_forces(
        k,
        index_one,
        index_two,
        index_one_opposite,
        index_reference,
        index_reference_opposite,
        offset_start_idx,
        offset_end_idx,
        rod_one_radius,
        rod_two_radius,
        rod_one_position_collection,
        rod_two_position_collection,
        rod_one_external_forces,
        rod_two_external_forces,
    ):
        # Compute perpendicular and radial direction vector from ring rod one to ring rod two using the
        # reference index.
        unit_circle_vector_one = (
            rod_one_position_collection[:, index_one_opposite]
            - rod_one_position_collection[:, index_one]
        )
        unit_circle_vector_one /= _batch_norm(unit_circle_vector_one)
        unit_circle_vector_two = (
            rod_one_position_collection[:, index_reference_opposite]
            - rod_one_position_collection[:, index_reference]
        )
        unit_circle_vector_two /= _batch_norm(unit_circle_vector_two)
        # Perpendicular direction from ring rod one to ring rod two.
        perpendicular_direction = _batch_cross(
            unit_circle_vector_one, unit_circle_vector_two
        )
        # Radial direction vector is from ring rod one connected index to node at the opposite side of the ring.
        radial_direction = unit_circle_vector_one

        # Using the reference positions first compute the element position of the straight rods, then compute the
        # offset distance between ring rod one and two. Note that ring rods attached to elements of straight rods.
        start_pos = 0.5 * (
            rod_one_position_collection[:, offset_start_idx]
            + rod_one_position_collection[:, offset_start_idx + 1]
        )
        end_pos = 0.5 * (
            rod_one_position_collection[:, offset_end_idx]
            + rod_one_position_collection[:, offset_end_idx + 1]
        )
        distance_btw_straight_rod_elems = end_pos - start_pos
        # Straight rods might be tapered and banked, thus we need to compute perpendicular offset and radial offset
        # separately.
        perpendicular_offset_collection = np.abs(
            _batch_dot(distance_btw_straight_rod_elems, perpendicular_direction)
        )

        # Radial offset is nonzero only when straight rods are tapered and banked. First compute the radial distance
        # between centers of straight rod elements (due to bank angle). Second compute the radius difference between
        # straight rod elements (due to tapering).
        radial_offset = _batch_dot(distance_btw_straight_rod_elems, radial_direction)
        radial_offset += np.abs(
            rod_one_radius[offset_start_idx] - rod_one_radius[offset_end_idx]
        )

        # Add the radius difference of ring rod_one and ring rod_two. If arm is not tapered difference is zero.
        radial_offset += np.abs(rod_one_radius[index_one] - rod_two_radius[index_two])

        target_rod_two_position = (
            rod_one_position_collection[:, index_one]
            + perpendicular_direction * perpendicular_offset_collection
            + radial_offset * radial_direction
        )

        # Current position of rod two.
        current_rod_two_position = rod_two_position_collection[:, index_two]

        distance = target_rod_two_position - current_rod_two_position
        # We may need to round distance because sometimes there is small error occur due to discreatization
        # error~1e-15
        np.round_(distance, 12, distance)

        spring_force = k * distance

        for i in range(3):
            for k in range(index_one.shape[0]):
                rod_one_external_forces[i, index_one[k]] -= spring_force[i, k]
                rod_two_external_forces[i, index_two[k]] += spring_force[i, k]
