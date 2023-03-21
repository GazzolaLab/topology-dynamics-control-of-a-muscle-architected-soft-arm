__doc__ = """Numba implementation of connecting ring and helical rods"""
__all__ = [
    "get_connection_order_and_angle_btw_helical_and_ring_rods",
    "RingHelicalRodContact",
    "RingHelicalRodJoint",
]
import numpy as np
import numba
from numba import njit
from elastica.joint import FreeJoint
from elastica._elastica_numba._linalg import (
    _batch_norm,
    _batch_dot,
)
from numpy.testing import assert_allclose


def get_connection_order_and_angle_btw_helical_and_ring_rods(
    rod_one,
    rod_two,
    index_one,
    index_two,
    global_direction,
):
    """

    This function computes and returns connection order and connection indexes for connecting helical rod and ring rod.
    First variable is the connection order which is 1 or -1. If helical rod is encircling the ring rod then connection
    order is 1.
    This function also returns the rod two indexes called as index_two_hinge_side and index_two_hinge_opposite_side.
    These indexes together with index_two are used to compute the normal the plane where rod two lives. This normal
    vector later on used to compute helical rod index_one+1 node position by using rod_two information.
    You should call this function before initializing the RingHelicalRodJoint class and use the returned variables
    from this function.


    Parameters
    ----------
    rod_one : object
        Helical rod object.
    rod_two : object
        Ring rod object.
    index_one : int
        Helical rod node index which is connected to the ring rod.
    index_two : int
        Ring rod node index which is connected to the helical rod.

    Returns
    -------
    connection_order : int
        Connection order of rod one and rod two. Depending on the relative positions of these rods, this variable
        can be -1 or 1.
    index_two_hing_side : int
    index_two_hinge_opposite_side : int
    next_connection_index : int
    next_connection_index_opposite : int
    """

    center_of_ring_rod = np.mean(rod_two.position_collection, axis=1)

    distance_btw_center_and_rod_two = np.linalg.norm(
        rod_two.position_collection[..., index_two] - center_of_ring_rod
    )

    distance_btw_center_and_rod_one = np.linalg.norm(
        rod_one.position_collection[..., index_one] - center_of_ring_rod
    )

    # Connection order is used to determine if rod one is inside rod two (ring rod) or outside of rod two.
    if distance_btw_center_and_rod_one > distance_btw_center_and_rod_two:
        connection_order = 1
    else:
        connection_order = -1

    index_two_opposite = int((index_two + rod_two.n_elems / 2) % rod_two.n_elems)
    # Index hinge side and hinge opposite side are used to compute hinge_direction_vec.
    index_two_hing_side = int((index_two + rod_two.n_elems / 4) % rod_two.n_elems)
    index_two_hinge_opposite_side = int(
        (index_two + 3 * rod_two.n_elems / 4) % rod_two.n_elems
    )

    # Rod spring connection vector is the direction in which nodes of rod one and rod two are connected.
    # This vector starts from the rod two index_two node and points towards the rod one index_one node.
    rod_spring_connection_vec = (
        rod_two.position_collection[..., index_two]
        - rod_two.position_collection[..., index_two_opposite]
    )
    rod_spring_connection_vec /= np.linalg.norm(rod_spring_connection_vec)
    rod_spring_connection_vec *= connection_order

    # hinge_direction_vec and rod_spring_connection_vec together are used to compute perpendicular
    # vector to the plane rod two lives in.
    hinge_direction_vec = (
        rod_two.position_collection[..., index_two_hing_side]
        - rod_two.position_collection[..., index_two_hinge_opposite_side]
    )
    hinge_direction_vec /= np.linalg.norm(hinge_direction_vec)

    # Perpendicular direction vector is orthogonal to the plane rod two lives in.
    perpendicular_direction = np.cross(rod_spring_connection_vec, hinge_direction_vec)
    perpendicular_direction /= np.linalg.norm(perpendicular_direction)

    # This for loop iteration is choosing correct hinge index, so that cross product of rod_two_connection_vec
    # will give the global direction.
    for idx in range(2):
        # hinge_direction_vec and rod_spring_connection_vec together are used to compute perpendicular
        # vector to the plane rod two lives in.
        hinge_direction_vec = (
            rod_two.position_collection[..., index_two_hing_side]
            - rod_two.position_collection[..., index_two_hinge_opposite_side]
        )
        hinge_direction_vec /= np.linalg.norm(hinge_direction_vec)

        # Perpendicular direction vector is orthogonal to the plane rod two lives in.
        perpendicular_direction = np.cross(
            rod_spring_connection_vec, hinge_direction_vec
        )
        perpendicular_direction /= np.linalg.norm(perpendicular_direction)

        # Global direction and perpendicular_direction should be same. Otherwise swap the hinge_side and
        # hinge_opposite_side indexes and redo the calculations.
        if np.abs(global_direction - perpendicular_direction).sum() > 1e-10:
            # If tolerance is not satisfied, then swap the hinge_index,
            # and recompute angle_btw_straight_ring_rods
            index_temp = index_two_hing_side
            index_two_hing_side = index_two_hinge_opposite_side
            index_two_hinge_opposite_side = index_temp

        else:
            continue

    assert_allclose(
        global_direction,
        perpendicular_direction,
        atol=1e-10,
        err_msg=" Global direction cannot be generated using rod two indices, check your inputs.",
    )

    # We need a second set of connections between rod one and rod two. For these connections we are connecting rod one
    # index_one + 1 node with the closest rod two node. Here we cannot assume the closest node is rod_two index_two+1,
    # and we need to compute it.
    # First project rod one node index_one+1 on to the plane where rod_two lives.
    rod_one_index_one_plus_node_position_in_plane = (
        rod_one.position_collection[..., index_one + 1]
        - np.dot(
            rod_one.position_collection[..., index_one + 1], perpendicular_direction
        )
        * perpendicular_direction
    )
    rod_two_position_collection_in_plane = rod_two.position_collection - np.einsum(
        "ij,i->j", rod_two.position_collection, perpendicular_direction
    ) * perpendicular_direction.reshape(3, 1)

    distance_btw_rod_one_and_rod_two = (
        rod_one_index_one_plus_node_position_in_plane.reshape(3, 1)
        - rod_two_position_collection_in_plane
    )
    distance_btw_rod_one_and_rod_two_norm = _batch_norm(
        distance_btw_rod_one_and_rod_two
    )
    # Check the minimum distance between rod_one index_one+1 and rod_two and if it is smaller than the tolerance
    # find that node location on rod_two
    if (
        np.abs(
            np.min(distance_btw_rod_one_and_rod_two_norm)
            - (rod_two.radius[0] + rod_one.radius[index_one])
        )
        < 1
    ):
        next_connection_index = np.argmin(distance_btw_rod_one_and_rod_two_norm)
        next_connection_index_opposite = int(
            (next_connection_index + rod_two.n_elems / 2) % rod_two.n_elems
        )
    else:
        raise ValueError(
            " Connection between rod one and rod two cannot constructed, check your inputs."
        )

    # Since we find the next_connection_index of rod two, now starting from rod_two next_connection_index
    # find the position of rod_one node index_one+1.
    next_index_connection_vec = (
        rod_two.position_collection[..., next_connection_index]
        - rod_two.position_collection[..., next_connection_index_opposite]
    )
    next_index_connection_vec_in_plane = (
        next_index_connection_vec
        - np.dot(next_index_connection_vec, perpendicular_direction)
        * perpendicular_direction
    )
    next_index_connection_vec_in_plane /= np.linalg.norm(
        next_index_connection_vec_in_plane
    )

    rod_one_index_one_to_index_one_plus_one_in_plane_vec = (
        rod_one_index_one_plus_node_position_in_plane
        - np.mean(rod_two_position_collection_in_plane, axis=1)
    )
    rod_one_index_one_to_index_one_plus_one_in_plane_vec /= np.linalg.norm(
        rod_one_index_one_to_index_one_plus_one_in_plane_vec
    )

    return (
        connection_order,
        index_two_opposite,
        index_two_hing_side,
        index_two_hinge_opposite_side,
        next_connection_index,
        next_connection_index_opposite,
    )


class RingHelicalRodJoint(FreeJoint):
    """
    This class is for connecting ring and helical rods. First rod is helical rod and second rod is ring rod.
    Connection happens between helical and ring rod nodes and helical rod is encircling the ring rods.

    """

    def __init__(
        self,
        k,
        nu,
        # kt,
        connection_order,
        index_two_opposite_side,
        index_two_hing_side,
        index_two_hinge_opposite_side,
        next_connection_index,
        next_connection_index_opposite,
        ring_rod_start_idx,
        ring_rod_end_idx,
        **kwargs,
    ):
        """

        Parameters
        ----------
        k
        nu
        kt
        connection_order
        index_two_hing_side
        index_two_hinge_opposite_side
        next_connection_index
        next_connection_index_opposite
        """
        super().__init__(np.array(k), np.array(nu))
        # self.kt = np.array(kt)
        self.connection_order = np.array(connection_order, dtype=np.int)

        second_sys_idx_offset = np.array(kwargs["second_sys_idx_offset"], dtype=np.int)
        self.index_two_opposite_side = (
            np.array(index_two_opposite_side, dtype=np.int) + second_sys_idx_offset
        )
        self.index_two_hinge_side = (
            np.array(index_two_hing_side, dtype=np.int) + second_sys_idx_offset
        )
        self.index_two_hinge_opposite_side = (
            np.array(index_two_hinge_opposite_side, dtype=np.int)
            + second_sys_idx_offset
        )
        self.next_connection_index = (
            np.array(next_connection_index, dtype=np.int) + second_sys_idx_offset
        )
        self.next_connection_index_opposite = (
            np.array(next_connection_index_opposite, dtype=np.int)
            + second_sys_idx_offset
        )
        self.ring_rod_start_idx = (
            np.array(ring_rod_start_idx, dtype=np.int) + second_sys_idx_offset
        )
        self.ring_rod_end_idx = (
            np.array(ring_rod_end_idx, dtype=np.int) + second_sys_idx_offset
        )

    def apply_forces(self, rod_one, index_one, rod_two, index_two):
        self._apply_forces(
            self.k,
            self.nu,
            index_one,
            index_two,
            self.index_two_opposite_side,
            rod_one.position_collection,
            rod_two.position_collection,
            rod_one.radius,
            rod_two.radius,
            rod_one.velocity_collection,
            rod_two.velocity_collection,
            rod_one.external_forces,
            rod_two.external_forces,
            self.connection_order,
        )

    @staticmethod
    @njit(cache=True)
    def _apply_forces(
        k,
        nu,
        index_one,
        index_two,
        index_two_opposite_side,
        rod_one_position_collection,
        rod_two_position_collection,
        rod_one_radius,
        rod_two_radius,
        rod_one_velocity_collection,
        rod_two_velocity_collection,
        rod_one_external_forces,
        rod_two_external_forces,
        connection_order,
    ):
        # Spring force
        # Compute the spring force direction. Spring force direction is between two opposite side of rod two(ring rod).
        # Since rod one element and rod two node are at the same height, we just need a direction to go from rod one
        # to rod two or vice versa.
        direction = (
            rod_two_position_collection[:, index_two]
            - rod_two_position_collection[:, index_two_opposite_side]
        )
        direction /= _batch_norm(direction)
        rod_spring_connection_vec = -direction * connection_order

        # Compute the surface point on rod one where connection happens.
        rod_one_r_connection_vec = rod_spring_connection_vec * (
            rod_one_radius[index_one]
        )
        surface_point_rod_one = (
            rod_one_position_collection[:, index_one] + rod_one_r_connection_vec
        )

        # Compute the surface point on rod two where connection happens.
        rod_two_r_connection_vec = (
            -rod_spring_connection_vec * rod_two_radius[index_two]
        )
        surface_point_rod_two = (
            rod_two_position_collection[:, index_two] + rod_two_r_connection_vec
        )

        # Distance vector between connection nodes of rods two and one.
        distance_vector = surface_point_rod_two - surface_point_rod_one
        np.round_(distance_vector, 12, distance_vector)

        # Compute the connection spring force
        spring_force = k * distance_vector

        relative_velocity = (
            rod_two_velocity_collection[:, index_two]
            - rod_one_velocity_collection[:, index_one]
        )

        distance = _batch_norm(distance_vector)

        normalized_distance_vector = np.zeros(distance_vector.shape)
        non_zero_dist_idx = np.where(distance >= 1e-12)[0]
        normalized_distance_vector[:, non_zero_dist_idx] = (
            distance_vector[:, non_zero_dist_idx] / distance[non_zero_dist_idx]
        )

        normal_relative_velocity_vector = (
            _batch_dot(relative_velocity, normalized_distance_vector)
            * normalized_distance_vector
        )

        damping_force = -nu * normal_relative_velocity_vector

        # Total force by the first connection btw helical rods and ring rods
        total_force_by_first_connection = spring_force + damping_force

        for i in range(3):
            for k in range(index_one.shape[0]):
                rod_one_external_forces[
                    i, index_one[k]
                ] += total_force_by_first_connection[i, k]
                rod_two_external_forces[
                    i, index_two[k]
                ] -= total_force_by_first_connection[i, k]

    def apply_torques(self, rod_one, index_one, rod_two, index_two):
        pass


class RingHelicalRodContact(FreeJoint):
    """ """

    def __init__(
        self,
        k,
        nu,
        **kwargs,
    ):
        """

        Parameters
        ----------
        k
        nu
        kt
        connection_order
        index_two_hing_side
        index_two_hinge_opposite_side
        next_connection_index
        next_connection_index_opposite
        """
        super().__init__(np.array(k), np.array(nu))
        pass

    def apply_forces(self, rod_one, index_one, rod_two, index_two):
        self._apply_forces(
            self.k,
            self.nu,
            index_one,
            index_two,
            rod_one.position_collection,
            rod_two.position_collection,
            rod_one.radius,
            rod_two.radius,
            rod_one.velocity_collection,
            rod_two.velocity_collection,
            rod_one.external_forces,
            rod_two.external_forces,
        )

    @staticmethod
    @njit(cache=True)
    def _apply_forces(
        k,
        nu,
        index_one_array,
        index_two_array,
        rod_one_position_collection,
        rod_two_position_collection,
        rod_one_radius,
        rod_two_radius,
        rod_one_velocity_collection,
        rod_two_velocity_collection,
        rod_one_external_forces,
        rod_two_external_forces,
    ):
        for idx in range(index_one_array.shape[0]):
            index_one = index_one_array[idx]
            index_two = index_two_array[idx]

            # Distance vector between rod two (ring rod) and rod one (straight rod)
            distance = (
                rod_two_position_collection[:, index_two]
                - rod_one_position_collection[:, index_one]
            )
            # Compute the norm of distance.
            norm_distance = np.linalg.norm(distance)

            # If penetration smaller than 0 than there is contact, otherwise rods are not in contact.
            penetration = norm_distance - (
                rod_one_radius[index_one] + rod_two_radius[index_two]
            )

            penetration = round(penetration, 12)

            if penetration >= 0:
                # Rods elements are not in contact. Thus do not compute contact force
                continue

            # Compute unit distance vector
            if norm_distance <= 1e-12:
                # Distance between two rod elements are very small, division by norm can cause overflow.
                # Set directly to zero.
                unit_distance_vector = np.array([0.0, 0.0, 0.0], np.float64)
            else:
                unit_distance_vector = distance / norm_distance

            # Apply Hertzian contact model
            contact_force_mag = -k[idx] * np.abs(penetration) ** (1.5)
            contact_force = contact_force_mag * unit_distance_vector

            # Damping force
            rod_one_element_velocity = 0.5 * (
                rod_one_velocity_collection[:, index_one]
                + rod_one_velocity_collection[:, index_one + 1]
            )
            relative_velocity = (
                rod_two_velocity_collection[:, index_two] - rod_one_element_velocity
            )

            projected_relative_velocity = (
                np.dot(relative_velocity, unit_distance_vector) * unit_distance_vector
            )
            damping_force = -nu[idx] * projected_relative_velocity

            # Compute total force
            total_force = contact_force + damping_force

            # Update external forces
            rod_one_external_forces[:, index_one] += total_force
            rod_two_external_forces[:, index_two] -= total_force
