__doc__ = """Numba implementation of connecting two orthogonal rods side by side"""
__all__ = ["get_connection_order_and_angle", "OrthogonalRodsSideBySideJoint"]
import numpy as np
import numba
from numba import njit
from elastica.joint import FreeJoint
from elastica._elastica_numba._linalg import (
    _batch_matvec,
    _batch_norm,
    _batch_dot,
    _batch_cross,
)
from elastica._elastica_numba._rotations import _get_rotation_matrix
from numpy.testing import assert_allclose
from numpy import sqrt, cos, sin


@njit(cache=True)
def _get_rotation_matrix_for_1D_scale(scale, axis_collection):
    blocksize = axis_collection.shape[1]
    rot_mat = np.empty((3, 3, blocksize))

    for k in range(blocksize):
        v0 = axis_collection[0, k]
        v1 = axis_collection[1, k]
        v2 = axis_collection[2, k]

        theta = sqrt(v0 * v0 + v1 * v1 + v2 * v2)

        v0 /= theta + 1e-14
        v1 /= theta + 1e-14
        v2 /= theta + 1e-14

        theta *= scale[k]
        u_prefix = sin(theta)
        u_sq_prefix = 1.0 - cos(theta)

        rot_mat[0, 0, k] = 1.0 - u_sq_prefix * (v1 * v1 + v2 * v2)
        rot_mat[1, 1, k] = 1.0 - u_sq_prefix * (v0 * v0 + v2 * v2)
        rot_mat[2, 2, k] = 1.0 - u_sq_prefix * (v0 * v0 + v1 * v1)

        rot_mat[0, 1, k] = u_prefix * v2 + u_sq_prefix * v0 * v1
        rot_mat[1, 0, k] = -u_prefix * v2 + u_sq_prefix * v0 * v1
        rot_mat[0, 2, k] = -u_prefix * v1 + u_sq_prefix * v0 * v2
        rot_mat[2, 0, k] = u_prefix * v1 + u_sq_prefix * v0 * v2
        rot_mat[1, 2, k] = u_prefix * v0 + u_sq_prefix * v1 * v2
        rot_mat[2, 1, k] = -u_prefix * v0 + u_sq_prefix * v1 * v2

    return rot_mat


def get_connection_order_and_angle(
    rod_one, rod_two, index_one, index_two, global_direction
):
    """
    This function computes returns some variables for connecting straight and ring rods. First variable is
    connection order which is 1 or -1. If ring rod encircling the straight rod then connection order
    is -1 or if straight rod is outside of ring rod connection order is 1. Second variable is the angle between
    straight rod tangent and a vector perpendicular to the ring rod. In order to find the vector perpendicular
    to the ring rod we first compute a vector in between index two node and node opposite side of that; this vector is
    called connection vector. We also compute another vector from index_two+n_elem/4 to node at the opposite side of
    that, this vector gives the hinge vector. These two vectors are dividing the ring rod into four equal quadrants
    and they are in plane and perpendicular to each other. Cross product of connection and hinge vector gives the
    rod two perpendicular direction vector and we check the angle in between perpendicular direction vector and straight
    rod tangent.

    Parameters
    ----------
    rod_one
    rod_two
    index_one
    index_two

    Returns
    -------
    connection_order : int
        Connection order of rod one and rod two. Depending on the relative positions of these rods, this variable
        can be -1 or 1.
    angle_btw_straight_ring_rods : double
        Connection angle between straight rod tangent and plane normal where rod two lives.
    """

    center_of_ring_rod = np.mean(rod_two.position_collection, axis=1)

    distance_btw_center_and_rod_two = np.linalg.norm(
        rod_two.position_collection[..., index_two] - center_of_ring_rod
    )

    rod_one_element_position = 0.5 * (
        rod_one.position_collection[..., 1:] + rod_one.position_collection[..., :-1]
    )
    distance_btw_center_and_rod_one = np.linalg.norm(
        rod_one_element_position[..., index_one] - center_of_ring_rod
    )

    if distance_btw_center_and_rod_one > distance_btw_center_and_rod_two:
        connection_order = 1
    else:
        connection_order = -1

    index_two_opposite = int((index_two + rod_two.n_elems / 2) % rod_two.n_elems)
    index_two_hing_side = int((index_two + rod_two.n_elems / 4) % rod_two.n_elems)
    index_two_hinge_opposite_side = int(
        (index_two + 3 * rod_two.n_elems / 4) % rod_two.n_elems
    )

    # This for loop iteration is choosing correct hinge index, so that cross product of rod_two_connection_vec
    # will give the global direction.
    for idx in range(2):
        rod_two_connection_vec = (
            rod_two.position_collection[..., index_two]
            - rod_two.position_collection[..., index_two_opposite]
        )
        rod_two_connection_vec /= np.linalg.norm(rod_two_connection_vec)
        rod_two_connection_vec *= connection_order

        rod_two_hinge_vec = (
            rod_two.position_collection[..., index_two_hing_side]
            - rod_two.position_collection[..., index_two_hinge_opposite_side]
        )
        rod_two_hinge_vec /= np.linalg.norm(rod_two_hinge_vec)

        rod_two_perpendicular_direction_vec = np.cross(
            rod_two_connection_vec, rod_two_hinge_vec
        )
        rod_two_perpendicular_direction_vec /= np.linalg.norm(
            rod_two_perpendicular_direction_vec
        )
        # Global direction and rod_two_perpendicular_direction_vec should be same. Otherwise swap the hinge_side and
        # hinge_opposite_side indexes and redo the calculations.
        if np.abs(global_direction - rod_two_perpendicular_direction_vec).sum() > 1e-10:
            # If tolerance is not satisfied, then swap the hinge_index,
            # and recompute angle_btw_straight_ring_rods
            index_temp = index_two_hing_side
            index_two_hing_side = index_two_hinge_opposite_side
            index_two_hinge_opposite_side = index_temp

        else:
            continue

    assert_allclose(
        global_direction,
        rod_two_perpendicular_direction_vec,
        atol=1e-10,
        err_msg=" Global direction cannot be generated using rod two indices, check your inputs.",
    )

    # Now we need to find the angle between rod one tangent and rod_two_perpendicular_direction_vec.
    rod_one_tangent = rod_one.director_collection[..., index_one][2, :]
    angle_btw_straight_ring_rods_sign = 1
    # We need to perform iteration to make sure we get the correct angle. Note that using arccos we can get the angle
    # between two vectors but we need the correct sign so that after performing rotation around rod_two_hinge_vec
    # we get the rod_one_tangent.
    for _ in range(2):
        if (
            np.sign(
                np.cross(rod_one_tangent, rod_two_perpendicular_direction_vec)
            ).sum()
            == 0
        ):
            angle_btw_straight_ring_rods = np.arccos(
                np.dot(rod_one_tangent, rod_two_perpendicular_direction_vec)
            )
            angle_btw_straight_ring_rods *= connection_order

        else:
            angle_btw_straight_ring_rods = np.arccos(
                np.dot(rod_one_tangent, rod_two_perpendicular_direction_vec)
            )
            # For some configuration cross product can have two component and each can be +, which makes sign 2. So take
            # sign one more time.
            angle_btw_straight_ring_rods *= np.sign(
                np.sign(
                    np.cross(rod_one_tangent, rod_two_perpendicular_direction_vec)
                ).sum()
            )

        if (
            np.dot(rod_two_perpendicular_direction_vec, rod_one_tangent) == -1.0
            or np.dot(rod_two_perpendicular_direction_vec, rod_one_tangent) == 1.0
        ):
            angle_btw_straight_ring_rods = np.arccos(
                np.dot(rod_two_perpendicular_direction_vec, rod_one_tangent)
            )

        angle_btw_straight_ring_rods *= angle_btw_straight_ring_rods_sign

        target_tangent_direction = _batch_matvec(
            _get_rotation_matrix(
                angle_btw_straight_ring_rods, rod_two_hinge_vec.reshape(3, 1)
            ),
            rod_two_perpendicular_direction_vec.reshape(3, 1),
        ).reshape(3)
        np.round_(target_tangent_direction, 12, target_tangent_direction)

        # If we cannot generate the rod_one_tangent by rotating target_tangent_direction, then we should perform
        # rotation in opposite direction, so multiply angle_btw_straight_ring_rods_sign by -1.
        if np.abs(target_tangent_direction - rod_one_tangent).sum() > 1e-10:
            angle_btw_straight_ring_rods_sign = -1

    assert_allclose(
        target_tangent_direction,
        rod_one_tangent,
        atol=1e-10,
        err_msg="Connection between rod one and rod two cannot constructed, check your inputs.",
    )

    return (
        connection_order,
        angle_btw_straight_ring_rods,
        index_two_opposite,
        index_two_hing_side,
        index_two_hinge_opposite_side,
    )


class OrthogonalRodsSideBySideJoint(FreeJoint):
    """
    This class is for connecting straight rods and ring rods. First rod is always straight rod and second rod is
    ring rod. Ring and straight rods do not have to be necessarily perpendicular to each other, but straight rod
    tangent and plane normal where ring rod lives have to be in same plane.
    """

    def __init__(
        self,
        k,
        nu,
        kt,
        k_repulsive,
        surface_pressure_idx,
        connection_order,
        angle_btw_straight_ring_rods,
        index_two_opposite_side,
        index_two_hing_side,
        index_two_hinge_opposite_side,
        n_connection,
        total_contact_force,
        total_contact_force_mag,
        **kwargs,
    ):
        """"""
        super().__init__(np.array(k), np.array(nu))
        self.kt = np.array(kt)
        self.k_repulsive = np.array(k_repulsive)
        self.surface_pressure_idx = np.array(surface_pressure_idx, dtype=np.int)
        self.connection_order = np.array(connection_order)
        self.angle_btw_straight_ring_rods = np.array(angle_btw_straight_ring_rods)
        self.index_two_opposite_side = np.array(index_two_opposite_side, dtype=np.int)
        self.index_two_hing_side = np.array(index_two_hing_side, dtype=np.int)
        self.index_two_hinge_opposite_side = np.array(
            index_two_hinge_opposite_side, dtype=np.int
        )
        # Hinge direction vector is perpendicular to the connection vector and they are on the same plane.
        self.hinge_direction_vector = np.zeros((3, self.index_two_hing_side.shape[0]))
        # Perpendicular vector is the cross product of hinge and connection vector.
        self.perpendicular_direction_vector = np.zeros(
            (3, self.index_two_hing_side.shape[0])
        )
        self.total_contact_force = total_contact_force[0]
        self.total_contact_force_mag = total_contact_force_mag[0]

        second_sys_idx_offset = np.array(kwargs["second_sys_idx_offset"], dtype=np.int)
        self.index_two_opposite_side += second_sys_idx_offset
        self.index_two_hing_side += second_sys_idx_offset
        self.index_two_hinge_opposite_side += second_sys_idx_offset

        second_sys_idx = np.array(kwargs["second_sys_idx"], dtype=np.int)

        # We average the contact forces magnitude applied on the ring rod.
        # contact_forces array elements order is the same as the straight rod element order. So we need to find the
        # which ring rod element are corresponds to which contact_force elements (order is mixed for ring rods).

        # We have to check if straight rods are inside or outside of the ring rods. So for each connection, we will
        # append second_sys_idx (always the ring rod system idx) to one of the list; avg_list_in or avg_list_out.
        # If straight rods are inside of the ring rods then we append ring rod system idx in avg_list_in.
        avg_list_in = []
        avg_list_in_idx = []
        # If straight rods are outside of the ring rods then we append ring rod system idx in avg_list_out.
        avg_list_out = []
        avg_list_out_idx = []
        for i in range(second_sys_idx.shape[0]):
            if connection_order[i] == -1:
                avg_list_in.append(second_sys_idx[i])
                avg_list_in_idx.append(i)
            elif connection_order[i] == 1:
                avg_list_out.append(second_sys_idx[i])
                avg_list_out_idx.append(i)

        size_of_avg_contact_force_mag = len(list(set(avg_list_in))) + len(
            list(set(avg_list_out))
        )
        # avg_contact_force_mag_in_plane stores the average contact force acting on one ring rod. Ring rod can be
        # connected more than on straight rod. For each connection we compute the magnitude of contact force, average
        # over all the straight-ring connections of that ring rod and store in avg_contact_force_mag_in_plane.
        # Size of avg_contact_force_mag_in_plane is same as the number of ring rods.
        self.avg_contact_force_mag_in_plane = np.zeros((size_of_avg_contact_force_mag))
        # After we compute the average contact force acting on one ring rod, we distribute this force to the ring rod
        # elements that are connected with straight rods. Size of avg_contact_force_in_plane is same as number of
        # ring-straight rod connection and different than avg_contact_force_mag_in_plane.
        self.avg_contact_force_in_plane = np.zeros((3, second_sys_idx.shape[0]))

        avg_in = np.array(avg_list_in, dtype=np.int64)
        avg_out = np.array(avg_list_out, dtype=np.int64) + len(list(set(avg_list_in)))
        avg_in_idx = np.array(avg_list_in_idx, dtype=np.int64)
        avg_out_idx = np.array(avg_list_out_idx, dtype=np.int64)
        avg_in_out = np.hstack((avg_in, avg_out))
        avg_in_out_idx = np.hstack((avg_in_idx, avg_out_idx))

        # Find the unique ring rod indexes. In avg_in_out same ring rod may appear more than one depending on how many
        # connections ring rod make with straight rods. Our goal is to find system idx of each ring rod and number of
        # occurances in avg_in_out array.
        values, counts = np.unique(avg_in_out, return_counts=True)
        self.n_connection_rod_two = np.zeros((second_sys_idx.shape[0]), dtype=np.int64)

        # Each element of avg_contact_force_mag_in_plane array corresponds to a ring rod. avg_contact_idx contains the
        # element indexes of ring rods on avg_contact_force_mag_in_plane.
        self.avg_contact_idx = np.zeros(second_sys_idx.shape[0], dtype=np.int64)
        for i in range(avg_in_out_idx.shape[0]):
            self.avg_contact_idx[avg_in_out_idx[i]] = int(
                np.where(values == avg_in_out[i])[0][0]
            )
            self.n_connection_rod_two[avg_in_out_idx[i]] = int(
                counts[np.where(values == avg_in_out[i])[0]]
            )

        # self.n_connection_rod_one = np.array(n_connection)
        # self.n_connection_rod_one[np.where(self.n_connection_rod_one==2)[0]] = 1

        self.post_processing_dict_list = kwargs.get("post_processing_dict", [None])[0]
        self.step_skip = kwargs.get("step_skip", 0)[0]
        self.first_sys_idx = np.array(kwargs.get("first_sys_idx"), dtype=np.int64)
        self.second_sys_idx = np.array(kwargs.get("second_sys_idx"), dtype=np.int64)
        self.counter = 0

    def apply_forces(self, rod_one, index_one, rod_two, index_two):
        # pass

        (
            self.total_force,
            self.rod_spring_connection_vec,
            repulsive_force,
            spring_force,
            damping_force,
            contact_force,
            contact_force_in_plane_mag,
            spring_force_in_plane,
            spring_force_in_plane_mag,
            distance,
            penetration_strain,
        ) = self._apply_forces(
            self.k,
            self.nu,
            self.k_repulsive,
            index_one,
            index_two,
            self.index_two_opposite_side,
            self.index_two_hing_side,
            self.index_two_hinge_opposite_side,
            self.hinge_direction_vector,
            self.perpendicular_direction_vector,
            rod_one.position_collection,
            rod_two.position_collection,
            rod_one.radius,
            rod_two.radius,
            rod_one.velocity_collection,
            rod_two.velocity_collection,
            rod_one.external_forces,
            rod_two.external_forces,
            self.surface_pressure_idx,
            rod_one.lengths,
            self.connection_order,
            self.n_connection_rod_two,
            self.avg_contact_idx,
            self.avg_contact_force_mag_in_plane,
            self.avg_contact_force_in_plane,
            self.total_contact_force,
            self.total_contact_force_mag,
        )

        if self.counter % self.step_skip == 0:
            if (self.post_processing_dict_list) is not None:
                self.post_processing_dict_list["spring_force"].append(
                    spring_force.copy()
                )

                self.post_processing_dict_list["repulsive_force"].append(
                    repulsive_force.copy()
                )

                self.post_processing_dict_list["damping_force"].append(
                    damping_force.copy()
                )

                self.post_processing_dict_list["index_two"].append(index_two.copy())

                self.post_processing_dict_list["contact_force"].append(
                    contact_force.copy()
                )

                self.post_processing_dict_list["contact_force_in_plane_mag"].append(
                    contact_force_in_plane_mag.copy()
                )

                self.post_processing_dict_list["spring_force_in_plane"].append(
                    spring_force_in_plane.copy()
                )

                self.post_processing_dict_list["spring_force_in_plane_mag"].append(
                    spring_force_in_plane_mag.copy()
                )
                self.post_processing_dict_list["total_contact_force"].append(
                    self.total_contact_force.copy()
                )
                self.post_processing_dict_list["total_contact_force_mag"].append(
                    self.total_contact_force_mag.copy()
                )
                self.post_processing_dict_list["distance"].append(distance.copy())
                self.post_processing_dict_list["penetration_strain"].append(
                    penetration_strain.copy()
                )
                self.post_processing_dict_list["total_force"].append(
                    self.total_force.copy()
                )

                if self.counter == 0:
                    self.post_processing_dict_list["rod_two_group_idx"].append(
                        self.avg_contact_idx.copy()
                    )

        self.counter += 1

    @staticmethod
    @njit(cache=True)
    def _apply_forces(
        k,
        nu,
        k_repulsive,
        index_one,
        index_two,
        index_two_opposite_side,
        index_two_hing_side,
        index_two_hinge_opposite_side,
        hinge_direction_vector,
        perpendicular_direction_vector,
        rod_one_position_collection,
        rod_two_position_collection,
        rod_one_radius,
        rod_two_radius,
        rod_one_velocity_collection,
        rod_two_velocity_collection,
        rod_one_external_forces,
        rod_two_external_forces,
        surface_pressure_idx,
        rod_one_lengths,
        connection_order,
        n_connection_rod_two,
        avg_contact_idx,
        avg_contact_force_mag_in_plane,
        avg_contact_force_in_plane,
        total_contact_force,
        total_contact_force_mag,
    ):
        # Spring force
        # Compute the spring force direction. Spring force direction is between two opposite side of rod two(ring rod).
        # Since rod one element and rod two node are at the same height, we just need a direction to go from rod one
        # to rod two or vice versa.
        direction = -(
            rod_two_position_collection[:, index_two]
            - rod_two_position_collection[:, index_two_opposite_side]
        )
        # direction = difference_kernel_for_block_structure(rod_two_tangents, ghost_elems_idx)[:,index_two]
        direction /= _batch_norm(direction)
        rod_spring_connection_vec = direction * connection_order

        # Compute torques
        # First restrict the motion of the first rod with respect to second rod in a plane defined by the
        # hinge_direction_vec. If rod one goes out of plane, by applying torques we force rod one back to the plane.
        hinge_direction_vector[:] = (
            rod_two_position_collection[:, index_two_hing_side]
            - rod_two_position_collection[:, index_two_hinge_opposite_side]
        )
        hinge_direction_vector /= _batch_norm(hinge_direction_vector)
        np.round_(hinge_direction_vector[:], 12, hinge_direction_vector[:])
        # Second compute in plane torques. These torques are restricting the relative position of rod one and
        # rod two in plane. Rod_spring_connection_vec and hinge_direction_vec divides the ring rod 4 equal
        # quadrants and they are perpendicular to each other on the same plane. Cross-product of these two
        # vectors gives a new vector perpendicular to the plane where rod two lives. We call this vector
        # perpendicular_direction_vec. Our goal is to restrict the relative orientation of perpendicular_direction_vec
        # and rod_one tangents defined initially by angle_btw_straight_rid_rods. In order to restrict the relative
        # orientation we apply restoring torques to both rods.
        perpendicular_direction_vector[:] = _batch_cross(
            -rod_spring_connection_vec, hinge_direction_vector
        )

        # Compute the surface point on rod one where connection happens.
        # We connect rod one elements with rod two nodes, so compute rod one element position.
        rod_one_r_connection_vec = rod_spring_connection_vec * (
            rod_one_radius[index_one]
        )
        rod_one_element_position = 0.5 * (
            rod_one_position_collection[:, index_one]
            + rod_one_position_collection[:, index_one + 1]
        )
        surface_point_rod_one = rod_one_element_position + rod_one_r_connection_vec

        # Compute the surface point on rod two where connection happens.
        rod_two_r_connection_vec = (
            -rod_spring_connection_vec * rod_two_radius[index_two]
        )
        surface_point_rod_two = (
            rod_two_position_collection[:, index_two] + rod_two_r_connection_vec
        )

        # Distance vector between connection nodes of rods two and one.
        # distance_vector = surface_point_rod_two - surface_point_rod_one
        distance_vector = surface_point_rod_one - surface_point_rod_two
        np.round_(distance_vector, 12, distance_vector)

        in_plane_distance = (
            rod_one_element_position - rod_two_position_collection[:, index_two]
        )
        in_plane_distance -= (
            _batch_dot(in_plane_distance, perpendicular_direction_vector)
            * perpendicular_direction_vector
        )
        penetration_strain = (
            _batch_norm(in_plane_distance)
            / (rod_one_radius[index_one] + rod_two_radius[index_two])
            - 1
        )
        np.round_(penetration_strain, 12, penetration_strain)
        idx_penetrate = np.where(penetration_strain < 0)[0]
        k_contact = np.zeros(index_one.shape)
        k_contact_temp = k_repulsive * np.abs(penetration_strain)

        k_contact[idx_penetrate] += k_contact_temp[idx_penetrate]
        contact_area = (
            1  # (2*np.pi*rod_one_radius[index_one])*rod_one_lengths[index_one]
        )

        contact_force = k_contact * distance_vector * contact_area

        # Compute the connection spring force
        spring_force = k * contact_area * distance_vector

        # Damping force
        rod_one_element_velocity = 0.5 * (
            rod_one_velocity_collection[:, index_one]
            + rod_one_velocity_collection[:, index_one + 1]
        )

        relative_velocity = (
            rod_one_element_velocity - rod_two_velocity_collection[:, index_two]
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

        # Total force
        total_force = spring_force + damping_force

        # Compute surface pressure force
        repulsive_force = (
            contact_force  # FIXME: remove this it is for post-processing and debugging
        )
        contact_force = (
            contact_force
            - _batch_dot(contact_force, perpendicular_direction_vector)
            * perpendicular_direction_vector
        )
        contact_force_in_plane_mag = _batch_dot(
            contact_force, rod_spring_connection_vec
        )

        # Find the avg contact forces,
        # these are averaged over the ring rod elements. Then we brodcast averaged contact force on one ring to its
        # elements.
        avg_contact_force_mag_in_plane *= 0
        for i in range(contact_force_in_plane_mag.shape[0]):
            avg_contact_force_mag_in_plane[avg_contact_idx[i]] += (
                contact_force_in_plane_mag[i] / n_connection_rod_two[i]
            )

        avg_contact_force_in_plane *= 0
        for i in range(contact_force_in_plane_mag.shape[0]):
            avg_contact_force_in_plane[:, i] = (
                avg_contact_force_mag_in_plane[avg_contact_idx[i]]
                * rod_spring_connection_vec[:, i]
            )

        # Compute in plane spring force and add to contact force. Since if rods penetrated spring force has also
        # some contribution to the pressure.
        spring_force_in_plane = np.zeros((contact_force.shape))
        spring_force_in_plane_mag = np.zeros((contact_force_in_plane_mag.shape))
        spring_force_in_plane[:, idx_penetrate] = (
            spring_force
            - _batch_dot(spring_force, perpendicular_direction_vector)
            * perpendicular_direction_vector
        )[:, idx_penetrate]
        spring_force_in_plane_mag[idx_penetrate] = (
            _batch_dot(spring_force_in_plane, rod_spring_connection_vec)
        )[idx_penetrate]

        # Find the total contact force, later we will use this to compute pressure force.
        for i in range(surface_pressure_idx.shape[0]):
            total_contact_force[:, surface_pressure_idx[i]] += (
                -contact_force[:, i] - spring_force_in_plane[:, i]
            )
            total_contact_force_mag[surface_pressure_idx[i]] += (
                -contact_force_in_plane_mag[i] - spring_force_in_plane_mag[i]
            )

        for k in range(total_force.shape[1]):
            rod_two_external_forces[0, index_two[k]] += (
                total_force[0, k] + avg_contact_force_in_plane[0, k]
            )
            rod_two_external_forces[1, index_two[k]] += (
                total_force[1, k] + avg_contact_force_in_plane[1, k]
            )
            rod_two_external_forces[2, index_two[k]] += (
                total_force[2, k] + avg_contact_force_in_plane[2, k]
            )

            rod_one_external_forces[0, index_one[k]] -= 0.5 * (
                total_force[0, k] + contact_force[0, k]
            )
            rod_one_external_forces[1, index_one[k]] -= 0.5 * (
                total_force[1, k] + contact_force[1, k]
            )
            rod_one_external_forces[2, index_one[k]] -= 0.5 * (
                total_force[2, k] + contact_force[2, k]
            )

            rod_one_external_forces[0, index_one[k] + 1] -= 0.5 * (
                total_force[0, k] + contact_force[0, k]
            )
            rod_one_external_forces[1, index_one[k] + 1] -= 0.5 * (
                total_force[1, k] + contact_force[1, k]
            )
            rod_one_external_forces[2, index_one[k] + 1] -= 0.5 * (
                total_force[2, k] + contact_force[2, k]
            )

        return (
            total_force,
            -rod_spring_connection_vec,
            repulsive_force,
            spring_force,
            damping_force,
            contact_force,
            contact_force_in_plane_mag,
            spring_force_in_plane,
            spring_force_in_plane_mag,
            distance_vector,
            penetration_strain,
        )

    def apply_torques(self, rod_one, index_one, rod_two, index_two):
        # pass
        self._apply_torques(
            self.kt,
            self.angle_btw_straight_ring_rods,
            index_one,
            index_two,
            self.hinge_direction_vector,
            self.perpendicular_direction_vector,
            rod_one.position_collection,
            rod_two.position_collection,
            rod_one.radius,
            rod_two.radius,
            rod_one.lengths,
            self.rod_spring_connection_vec,
            rod_one.director_collection,
            rod_two.director_collection,
            rod_one.external_torques,
            rod_two.external_torques,
        )

    @staticmethod
    @njit(cache=True)
    def _apply_torques(
        kt,
        angle_btw_straight_ring_rods,
        index_one,
        index_two,
        hinge_direction_vector,
        perpendicular_direction_vector,
        rod_one_position_collection,
        rod_two_position_collection,
        rod_one_radius,
        rod_two_radius,
        rod_one_lengths,
        rod_spring_connection_vec,
        rod_one_director,
        rod_two_director,
        rod_one_external_torques,
        rod_two_external_torques,
    ):
        # Compute torques
        # First restrict the motion of the first rod with respect to second rod in a plane defined by the
        # hinge_direction_vec. If rod one goes out of plane, by applying torques we force rod one back to the plane.

        # hinge_direction_vec = (
        #         rod_two_position_collection[:, index_two_hing_side]
        #         - rod_two_position_collection[:, index_two_hinge_opposite_side]
        # )
        # # hinge_direction_vec = -difference_kernel_for_block_structure(rod_two_tangents, ghost_elems_idx)[:,index_two_hing_side]
        # # np.round_(direction_hinge, 12, direction_hinge)
        # hinge_direction_vec /= _batch_norm(hinge_direction_vec)

        link_direction = (
            rod_one_position_collection[:, index_one + 1]
            - rod_one_position_collection[:, index_one]
        )

        force_direction = (
            -_batch_dot(link_direction, hinge_direction_vector) * hinge_direction_vector
        )
        np.round_(force_direction, 12, force_direction)
        torque_hinge = kt * _batch_cross(link_direction / 2, force_direction)

        # Second compute in plane torques. These torques are restricting the relative position of rod one and
        # rod two in plane. Rod_spring_connection_vec and hinge_direction_vec divides the ring rod 4 equal
        # quadrants and they are perpendicular to each other on the same plane. Cross-product of these two
        # vectors gives a new vector perpendicular to the plane where rod two lives. We call this vector
        # perpendicular_direction_vec. Our goal is to restrict the relative orientation of perpendicular_direction_vec
        # and rod_one tangents defined initially by angle_btw_straight_rid_rods. In order to restrict the relative
        # orientation we apply restoring torques to both rods.
        current_position = rod_one_position_collection[:, index_one + 1]
        rod_two_connection_vec = rod_spring_connection_vec
        # perpendicular_direction = _batch_cross(
        #     rod_two_connection_vec, hinge_direction_vec
        # )
        target_tangent_direction = _batch_matvec(
            _get_rotation_matrix_for_1D_scale(
                angle_btw_straight_ring_rods, hinge_direction_vector
            ),
            perpendicular_direction_vector,
        )

        target_position = (
            rod_two_position_collection[:, index_two]
            + rod_two_connection_vec
            * (rod_one_radius[index_one] + rod_two_radius[index_two])
        ) + (0.5 * rod_one_lengths[index_one] * target_tangent_direction)
        torque_force = target_position - current_position
        np.round_(torque_force, 12, torque_force)
        torque_constrain_orientation = kt * _batch_cross(
            link_direction / 2, torque_force
        )

        # Together with hinge torque and constrain orientation torques we are restricting the relative orientations
        # of the ring and straight rod.
        total_torque = torque_hinge + torque_constrain_orientation

        torque_on_rod_one = _batch_matvec(
            rod_one_director[:, :, index_one], total_torque
        )
        torque_on_rod_two = _batch_matvec(
            rod_two_director[:, :, index_two], total_torque
        )

        for i in range(3):
            for k in range(index_one.shape[0]):
                rod_one_external_torques[i, index_one[k]] += torque_on_rod_one[i, k]
                rod_two_external_torques[i, index_two[k]] -= torque_on_rod_two[i, k]
