__doc__ = """Numba implementation of contact model between all ring elements and straight rod"""
__all__ = ["OrthogonalRodsSideBySideContact"]
import numpy as np
from numba import njit
from elastica.joint import FreeJoint


class OrthogonalRodsSideBySideContact(FreeJoint):
    def __init__(
        self,
        k,
        nu,
        surface_pressure_idx,
        total_contact_force,
        total_contact_force_mag,
        **kwargs,
    ):
        """"""
        super().__init__(np.array(k), np.array(nu))

        self.surface_pressure_idx = np.array(surface_pressure_idx, dtype=np.int64)
        self.total_contact_force = total_contact_force[0]
        self.total_contact_force_mag = total_contact_force_mag[0]

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
            self.surface_pressure_idx,
            self.total_contact_force,
            self.total_contact_force_mag,
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
        surface_pressure_idx,
        total_contact_force,
        total_contact_force_mag,
    ):
        for idx in range(index_one_array.shape[0]):
            index_one = index_one_array[idx]
            index_two = index_two_array[idx]

            rod_one_element_position = 0.5 * (
                rod_one_position_collection[:, index_one]
                + rod_one_position_collection[:, index_one + 1]
            )

            # Distance vector between rod two (ring rod) and rod one (straight rod)
            distance = (
                rod_two_position_collection[:, index_two] - rod_one_element_position
            )
            # Compute the norm of distance.
            norm_distance = np.linalg.norm(distance)

            # If penetration smaller than 0 than there is contact, otherwise rods are not in contact.
            penetration = norm_distance - (
                rod_one_radius[index_one] + rod_two_radius[index_two]
            )

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

            # Update pressure forces
            total_contact_force[:, surface_pressure_idx[idx]] += contact_force
            total_contact_force_mag[surface_pressure_idx[idx]] += contact_force_mag

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
            rod_one_external_forces[:, index_one] += 0.5 * total_force
            rod_one_external_forces[:, index_one + 1] += 0.5 * total_force

            rod_two_external_forces[:, index_two] -= total_force
