__all__ = ["get_connection_vector_straight_straight_rod", "SurfaceJointSideBySide"]
import numpy as np
import numba
from numba import njit
from elastica.joint import FreeJoint

# Join the two rods
from elastica._linalg import (
    _batch_norm,
    _batch_cross,
    _batch_matvec,
    _batch_dot,
    _batch_matmul,
    _batch_matrix_transpose,
)


def get_connection_vector_straight_straight_rod(
    rod_one,
    rod_two,
):
    # Compute rod element positions
    rod_one_element_position = 0.5 * (
        rod_one.position_collection[..., 1:] + rod_one.position_collection[..., :-1]
    )
    rod_two_element_position = 0.5 * (
        rod_two.position_collection[..., 1:] + rod_two.position_collection[..., :-1]
    )

    # Lets get the distance between rod elements
    distance_vector_rod_one_to_rod_two = (
        rod_two_element_position - rod_one_element_position
    )
    distance_vector_rod_one_to_rod_two_norm = _batch_norm(
        distance_vector_rod_one_to_rod_two
    )
    distance_vector_rod_one_to_rod_two /= distance_vector_rod_one_to_rod_two_norm

    distance_vector_rod_two_to_rod_one = -distance_vector_rod_one_to_rod_two

    rod_one_direction_vec_in_material_frame = _batch_matvec(
        rod_one.director_collection, distance_vector_rod_one_to_rod_two
    )
    rod_two_direction_vec_in_material_frame = _batch_matvec(
        rod_two.director_collection, distance_vector_rod_two_to_rod_one
    )

    offset_btw_rods = distance_vector_rod_one_to_rod_two_norm - (
        rod_one.radius + rod_two.radius
    )

    return (
        rod_one_direction_vec_in_material_frame,
        rod_two_direction_vec_in_material_frame,
        offset_btw_rods,
    )


class SurfaceJointSideBySide(FreeJoint):
    """"""

    def __init__(
        self,
        k,
        nu,
        # kt,
        k_repulsive,
        rod_one_direction_vec_in_material_frame,
        rod_two_direction_vec_in_material_frame,
        offset_btw_rods,
        contact_force_rod_one_idx,
        contact_force_rod_two_idx,
        total_contact_force,
        total_contact_force_mag,
        **kwargs,
    ):
        super().__init__(np.array(k), np.array(nu))
        # additional in-plane constraint through restoring torque
        # stiffness of the restoring constraint -- tuned empirically
        # self.kt = np.array(kt)
        self.k_repulsive = np.array(k_repulsive)

        self.offset_btw_rods = np.array(offset_btw_rods)
        self.contact_force_index_one = np.array(contact_force_rod_one_idx, dtype=np.int)
        self.contact_force_index_two = np.array(contact_force_rod_two_idx, dtype=np.int)
        self.total_contact_force = total_contact_force[0]
        self.total_contact_force_mag = total_contact_force_mag[0]

        self.rod_one_direction_vec_in_material_frame = np.array(
            rod_one_direction_vec_in_material_frame
        ).T
        self.rod_two_direction_vec_in_material_frame = np.array(
            rod_two_direction_vec_in_material_frame
        ).T

        self.post_processing_dict = kwargs.get("post_processing_dict", [None])[0]
        self.step_skip = kwargs.get("step_skip", 0)[0]
        self.counter = 0

    # Apply force is same as free joint
    def apply_forces(self, rod_one, index_one, rod_two, index_two):
        # TODO: documentation

        (
            self.rod_one_rd2,
            self.rod_two_rd2,
            self.spring_force,
            penetration_strain,
        ) = self._apply_forces(
            self.k,
            self.nu,
            self.k_repulsive,
            index_one,
            index_two,
            self.rod_one_direction_vec_in_material_frame,
            self.rod_two_direction_vec_in_material_frame,
            self.offset_btw_rods,
            rod_one.director_collection,
            rod_two.director_collection,
            rod_one.position_collection,
            rod_two.position_collection,
            rod_one.radius,
            rod_two.radius,
            rod_one.dilatation,
            rod_two.dilatation,
            rod_one.velocity_collection,
            rod_two.velocity_collection,
            rod_one.external_forces,
            rod_two.external_forces,
            self.contact_force_index_one,
            self.contact_force_index_two,
            self.total_contact_force,
            self.total_contact_force_mag,
        )

        if self.counter % self.step_skip == 0:
            if (self.post_processing_dict) is not None:
                self.post_processing_dict["spring_force"].append(
                    self.spring_force.copy()
                )
                self.post_processing_dict["penetration_strain"].append(
                    penetration_strain.copy()
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
        rod_one_direction_vec_in_material_frame,
        rod_two_direction_vec_in_material_frame,
        rest_offset_btw_rods,
        rod_one_director_collection,
        rod_two_director_collection,
        rod_one_position_collection,
        rod_two_position_collection,
        rod_one_radius,
        rod_two_radius,
        rod_one_dilatation,
        rod_two_dilatation,
        rod_one_velocity_collection,
        rod_two_velocity_collection,
        rod_one_external_forces,
        rod_two_external_forces,
        contact_force_index_one,
        contact_force_index_two,
        total_contact_force,
        total_contact_force_mag,
    ):
        rod_one_to_rod_two_connection_vec = _batch_matvec(
            _batch_matrix_transpose(rod_one_director_collection[:, :, index_one]),
            rod_one_direction_vec_in_material_frame,
        )
        rod_two_to_rod_one_connection_vec = _batch_matvec(
            _batch_matrix_transpose(rod_two_director_collection[:, :, index_two]),
            rod_two_direction_vec_in_material_frame,
        )

        # Compute element positions
        rod_one_element_position = 0.5 * (
            rod_one_position_collection[:, index_one]
            + rod_one_position_collection[:, index_one + 1]
        )
        rod_two_element_position = 0.5 * (
            rod_two_position_collection[:, index_two]
            + rod_two_position_collection[:, index_two + 1]
        )

        # If there is an offset between rod one and rod two surface, then it should change as a function of dilatation.
        offset_rod_one = (
            0.5 * rest_offset_btw_rods / np.sqrt(rod_one_dilatation[index_one])
        )
        offset_rod_two = (
            0.5 * rest_offset_btw_rods / np.sqrt(rod_two_dilatation[index_two])
        )

        # Compute vector r*d2 (radius * connection vector) for each rod and element
        rod_one_rd2 = rod_one_to_rod_two_connection_vec * (
            rod_one_radius[index_one] + offset_rod_one
        )
        rod_two_rd2 = rod_two_to_rod_one_connection_vec * (
            rod_two_radius[index_two] + offset_rod_two
        )

        # Compute connection points on the rod surfaces
        surface_position_rod_one = rod_one_element_position + rod_one_rd2
        surface_position_rod_two = rod_two_element_position + rod_two_rd2

        # Compute spring force between two rods
        distance_vector = surface_position_rod_two - surface_position_rod_one
        np.round_(distance_vector, 12, distance_vector)
        spring_force = k * (distance_vector)

        # Damping force
        rod_one_element_velocity = 0.5 * (
            rod_one_velocity_collection[:, index_one]
            + rod_one_velocity_collection[:, index_one + 1]
        )
        rod_two_element_velocity = 0.5 * (
            rod_two_velocity_collection[:, index_two]
            + rod_two_velocity_collection[:, index_two + 1]
        )

        relative_velocity = rod_two_element_velocity - rod_one_element_velocity

        distance = _batch_norm(distance_vector)

        normalized_distance_vector = np.zeros((relative_velocity.shape))

        idx_nonzero_distance = np.where(distance >= 1e-12)[0]

        normalized_distance_vector[..., idx_nonzero_distance] = (
            distance_vector[..., idx_nonzero_distance] / distance[idx_nonzero_distance]
        )

        normal_relative_velocity_vector = (
            _batch_dot(relative_velocity, normalized_distance_vector)
            * normalized_distance_vector
        )

        damping_force = -nu * normal_relative_velocity_vector

        # Compute the total force
        total_force = spring_force + damping_force

        # Compute contact forces. Contact forces are applied in the case one rod penetrates to the other, in that case
        # we apply a repulsive force. Later on these repulsive forces are used to move rods apart from each other and
        # as a pressure force.
        # We assume contact forces are in plane.
        center_distance = rod_two_element_position - rod_one_element_position
        center_distance_unit_vec = center_distance / _batch_norm(center_distance)
        penetration_strain = _batch_norm(center_distance) - (
            rod_one_radius[index_one]
            + offset_rod_one
            + rod_two_radius[index_two]
            + offset_rod_two
        )
        np.round_(penetration_strain, 12, penetration_strain)
        idx_penetrate = np.where(penetration_strain < 0)[0]
        k_contact = np.zeros(index_one.shape[0])
        k_contact_temp = -k_repulsive * np.abs(penetration_strain) ** (1.5)
        k_contact[idx_penetrate] += k_contact_temp[idx_penetrate]
        contact_force = k_contact * center_distance_unit_vec
        # contact_force[:,idx_penetrate] = 0.0

        # Add contact forces
        total_force += contact_force

        # Compute the spring forces in plane. If there is contact spring force is also contributing to contact force
        # so we need to compute it and add to contact_force.
        spring_force_temp_for_contact = np.zeros((3, index_one.shape[0]))
        spring_force_temp_for_contact[:, idx_penetrate] += spring_force[
            :, idx_penetrate
        ]

        contact_force += spring_force_temp_for_contact

        contact_force_mag_on_rod_one = _batch_dot(
            contact_force, rod_one_to_rod_two_connection_vec
        )
        contact_force_mag_on_rod_two = _batch_dot(
            -contact_force, rod_two_to_rod_one_connection_vec
        )
        contact_force_on_rod_one = (
            contact_force_mag_on_rod_one * rod_one_to_rod_two_connection_vec
        )
        contact_force_on_rod_two = (
            contact_force_mag_on_rod_two * rod_two_to_rod_one_connection_vec
        )

        # Re-distribute forces from elements to nodes.
        block_size = index_one.shape[0]
        for k in range(block_size):
            rod_one_external_forces[0, index_one[k]] += 0.5 * total_force[0, k]
            rod_one_external_forces[1, index_one[k]] += 0.5 * total_force[1, k]
            rod_one_external_forces[2, index_one[k]] += 0.5 * total_force[2, k]

            rod_one_external_forces[0, index_one[k] + 1] += 0.5 * total_force[0, k]
            rod_one_external_forces[1, index_one[k] + 1] += 0.5 * total_force[1, k]
            rod_one_external_forces[2, index_one[k] + 1] += 0.5 * total_force[2, k]

            rod_two_external_forces[0, index_two[k]] -= 0.5 * total_force[0, k]
            rod_two_external_forces[1, index_two[k]] -= 0.5 * total_force[1, k]
            rod_two_external_forces[2, index_two[k]] -= 0.5 * total_force[2, k]

            rod_two_external_forces[0, index_two[k] + 1] -= 0.5 * total_force[0, k]
            rod_two_external_forces[1, index_two[k] + 1] -= 0.5 * total_force[1, k]
            rod_two_external_forces[2, index_two[k] + 1] -= 0.5 * total_force[2, k]

            total_contact_force[
                0, contact_force_index_one[k]
            ] += contact_force_on_rod_one[0, k]
            total_contact_force[
                1, contact_force_index_one[k]
            ] += contact_force_on_rod_one[1, k]
            total_contact_force[
                2, contact_force_index_one[k]
            ] += contact_force_on_rod_one[2, k]

            total_contact_force[
                0, contact_force_index_two[k]
            ] += contact_force_on_rod_two[0, k]
            total_contact_force[
                1, contact_force_index_two[k]
            ] += contact_force_on_rod_two[1, k]
            total_contact_force[
                2, contact_force_index_two[k]
            ] += contact_force_on_rod_two[2, k]

            total_contact_force_mag[
                contact_force_index_one[k]
            ] += contact_force_mag_on_rod_one[k]
            total_contact_force_mag[
                contact_force_index_two[k]
            ] += contact_force_mag_on_rod_two[k]

        return (
            rod_one_rd2,
            rod_two_rd2,
            spring_force,
            penetration_strain,
        )

    def apply_torques(self, rod_one, index_one, rod_two, index_two):
        # pass

        self._apply_torques(
            self.spring_force,
            self.rod_one_rd2,
            self.rod_two_rd2,
            index_one,
            index_two,
            rod_one.director_collection,
            rod_two.director_collection,
            rod_one.external_torques,
            rod_two.external_torques,
        )

    @staticmethod
    @njit(cache=True)
    def _apply_torques(
        spring_force,
        rod_one_rd2,
        rod_two_rd2,
        index_one,
        index_two,
        rod_one_director_collection,
        rod_two_director_collection,
        rod_one_external_torques,
        rod_two_external_torques,
    ):
        # Compute torques due to the connection forces
        torque_on_rod_one = _batch_cross(rod_one_rd2, spring_force)
        torque_on_rod_two = _batch_cross(rod_two_rd2, -spring_force)

        torque_on_rod_one_material_frame = _batch_matvec(
            rod_one_director_collection[:, :, index_one], torque_on_rod_one
        )
        torque_on_rod_two_material_frame = _batch_matvec(
            rod_two_director_collection[:, :, index_two], torque_on_rod_two
        )

        blocksize = index_one.shape[0]
        for k in range(blocksize):
            rod_one_external_torques[
                0, index_one[k]
            ] += torque_on_rod_one_material_frame[0, k]
            rod_one_external_torques[
                1, index_one[k]
            ] += torque_on_rod_one_material_frame[1, k]
            rod_one_external_torques[
                2, index_one[k]
            ] += torque_on_rod_one_material_frame[2, k]

            rod_two_external_torques[
                0, index_two[k]
            ] += torque_on_rod_two_material_frame[0, k]
            rod_two_external_torques[
                1, index_two[k]
            ] += torque_on_rod_two_material_frame[1, k]
            rod_two_external_torques[
                2, index_two[k]
            ] += torque_on_rod_two_material_frame[2, k]
