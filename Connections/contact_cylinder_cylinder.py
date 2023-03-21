__doc__ = """Contact of two rigid cylinders for memory block connections"""
# Developer note: We are using the same function in _elastica_numba._joint ExternalContact class, but in order to use
# memory block connections wrapper, we have to change the ExternalContact class.
# Current implementation will go over rods that can contact one by one.
__all__ = ["ExternalContactCylinderCylinderForMemoryBlock"]
import numpy as np
import numba
from numba import njit
from elastica._elastica_numba._joint import (
    _norm,
    # _find_min_dist,
    _dot_product,
    ExternalContact,
    _clip,
    _out_of_bounds,
    _aabbs_not_intersecting,
)


@numba.njit(cache=True)
def _prune_using_aabbs_cylinder_cylinder(
    cylinder_one_position,
    cylinder_one_director,
    cylinder_one_radius,
    cylinder_one_length,
    cylinder_two_position,
    cylinder_two_director,
    cylinder_two_radius,
    cylinder_two_length,
):
    # Is actually Q^T * d but numba complains about performance so we do
    # d^T @ Q
    aabb_cylinder_one = np.empty((3, 2))
    cylinder_one_dimensions_in_local_FOR = np.array(
        [cylinder_one_radius, cylinder_one_radius, 0.5 * cylinder_one_length]
    )
    cylinder_one_dimensions_in_world_FOR = np.zeros_like(
        cylinder_one_dimensions_in_local_FOR
    )
    for i in range(3):
        for j in range(3):
            cylinder_one_dimensions_in_world_FOR[i] += (
                cylinder_one_director[j, i, 0] * cylinder_one_dimensions_in_local_FOR[j]
            )

    max_possible_dimension = np.abs(cylinder_one_dimensions_in_world_FOR)
    aabb_cylinder_one[..., 0] = cylinder_one_position[..., 0] - max_possible_dimension
    aabb_cylinder_one[..., 1] = cylinder_one_position[..., 0] + max_possible_dimension

    # Is actually Q^T * d but numba complains about performance so we do
    # d^T @ Q
    aabb_cylinder_two = np.empty((3, 2))
    cylinder_two_dimensions_in_local_FOR = np.array(
        [cylinder_two_radius, cylinder_two_radius, 0.5 * cylinder_two_length]
    )
    cylinder_two_dimensions_in_world_FOR = np.zeros_like(
        cylinder_two_dimensions_in_local_FOR
    )
    for i in range(3):
        for j in range(3):
            cylinder_two_dimensions_in_world_FOR[i] += (
                cylinder_two_director[j, i, 0] * cylinder_two_dimensions_in_local_FOR[j]
            )

    max_possible_dimension = np.abs(cylinder_two_dimensions_in_world_FOR)
    aabb_cylinder_two[..., 0] = cylinder_two_position[..., 0] - max_possible_dimension
    aabb_cylinder_two[..., 1] = cylinder_two_position[..., 0] + max_possible_dimension
    return _aabbs_not_intersecting(aabb_cylinder_two, aabb_cylinder_one)


@numba.njit(cache=True)
def _find_min_dist(x1, e1, x2, e2):
    e1e1 = _dot_product(e1, e1)
    e1e2 = _dot_product(e1, e2)
    e2e2 = _dot_product(e2, e2)

    x1e1 = _dot_product(x1, e1)
    x1e2 = _dot_product(x1, e2)
    x2e1 = _dot_product(e1, x2)
    x2e2 = _dot_product(x2, e2)

    s = 0.0
    t = 0.0

    parallel = abs(1.0 - e1e2**2 / (e1e1 * e2e2)) < 1e-6
    if parallel:
        # Some are parallel, so do processing
        t = (x2e1 - x1e1) / e1e1  # Comes from taking dot of e1 with a normal
        t = _clip(t, 0.0, 1.0)
        s = (x1e2 + t * e1e2 - x2e2) / e2e2  # Same as before
        s = _clip(s, 0.0, 1.0)
    else:
        # Using the Cauchy-Binet formula on eq(7) in docstring referenc
        s = (e1e1 * (x1e2 - x2e2) + e1e2 * (x2e1 - x1e1)) / (e1e1 * e2e2 - (e1e2) ** 2)
        t = (e1e2 * s + x2e1 - x1e1) / e1e1

        if _out_of_bounds(s, 0.0, 1.0) or _out_of_bounds(t, 0.0, 1.0):
            # potential_s = -100.0
            # potential_t = -100.0
            # potential_d = -100.0
            # overall_minimum_distance = 1e20

            # Fill in the possibilities
            potential_t = (x2e1 - x1e1) / e1e1
            s = 0.0
            t = _clip(potential_t, 0.0, 1.0)
            potential_d = _norm(x1 + e1 * t - x2)
            overall_minimum_distance = potential_d

            potential_t = (x2e1 + e1e2 - x1e1) / e1e1
            potential_t = _clip(potential_t, 0.0, 1.0)
            potential_d = _norm(x1 + e1 * potential_t - x2 - e2)
            if potential_d < overall_minimum_distance:
                s = 1.0
                t = potential_t
                overall_minimum_distance = potential_d

            potential_s = (x1e2 - x2e2) / e2e2
            potential_s = _clip(potential_s, 0.0, 1.0)
            potential_d = _norm(x2 + potential_s * e2 - x1)
            if potential_d < overall_minimum_distance:
                s = potential_s
                t = 0.0
                overall_minimum_distance = potential_d

            potential_s = (x1e2 + e1e2 - x2e2) / e2e2
            potential_s = _clip(potential_s, 0.0, 1.0)
            potential_d = _norm(x2 + potential_s * e2 - x1 - e1)
            if potential_d < overall_minimum_distance:
                s = potential_s
                t = 1.0

    return x2 + s * e2 - x1 - t * e1, x2 + s * e2


@numba.njit(cache=True)
def _calculate_contact_forces(
    x_cylinder_one,
    edge_cylinder_one,
    x_cylinder_two,
    edge_cylinder_two,
    radii_sum,
    length_sum,
    external_forces_cylinder_one,
    external_torques_cylinder_one,
    cylinder_one_director_collection,
    external_forces_cylinder_two,
    external_torques_cylinder_two,
    cylinder_two_director_collection,
    velocity_cylinder_one,
    velocity_cylinder_two,
    contact_k,
    contact_nu,
    x_cylinder_one_center,
    x_cylinder_two_center,
):
    del_x = x_cylinder_one - x_cylinder_two
    norm_del_x = _norm(del_x)

    # If outside then don't process
    if norm_del_x >= (radii_sum + length_sum):
        return

    # find the shortest line segment between the two centerline
    # segments : differs from normal cylinder-cylinder intersection
    # rod to cylinder
    distance_vector, x_cylinder_contact_point = _find_min_dist(
        x_cylinder_one, edge_cylinder_one, x_cylinder_two, edge_cylinder_two
    )
    # x_cylinder_contact_point = x_selected + distance_vector
    distance_vector_length = _norm(distance_vector)
    distance_vector /= distance_vector_length

    gamma = radii_sum - distance_vector_length

    # If distance is large, don't worry about it
    if gamma < -1e-5:
        return

    # CHECK FOR GAMMA > 0.0, heaviside but we need to overload it in numba
    # As a quick fix, use this instead
    mask = (gamma > 0.0) * 1.0

    contact_force = contact_k * gamma
    interpenetration_velocity = (
        velocity_cylinder_one[..., 0] - velocity_cylinder_two[..., 0]
    )
    contact_damping_force = contact_nu * _dot_product(
        interpenetration_velocity, distance_vector
    )

    # magnitude* direction
    net_contact_force = (
        0.5 * mask * ((contact_damping_force + contact_force)) * distance_vector
    )

    # Update the cylinder external forces and torques
    external_forces_cylinder_one[..., 0] -= net_contact_force
    moment_arm_cylinder_one = x_cylinder_contact_point - x_cylinder_one_center
    external_torques_cylinder_one[
        ..., 0
    ] -= cylinder_one_director_collection @ np.cross(
        moment_arm_cylinder_one, net_contact_force
    )

    external_forces_cylinder_two[..., 0] += net_contact_force
    moment_arm_cylinder_two = x_cylinder_contact_point - x_cylinder_two_center
    external_torques_cylinder_two[
        ..., 0
    ] += cylinder_two_director_collection @ np.cross(
        moment_arm_cylinder_two, net_contact_force
    )


class ExternalContactCylinderCylinderForMemoryBlock(ExternalContact):
    def __init__(
        self,
        k,
        nu,
        **kwargs,
    ):
        super().__init__(np.array(k), np.array(nu))

        self.first_sys_idx = kwargs.get("first_sys_idx")
        self.second_sys_idx = kwargs.get("second_sys_idx")
        self.first_sys_idx_offset = kwargs.get("first_sys_idx_offset")
        self.second_sys_idx_offset = kwargs.get("second_sys_idx_offset")
        self.first_sys_idx_on_block = np.array(
            kwargs["first_sys_idx_on_block"], dtype=np.int32
        ).flatten()
        self.second_sys_idx_on_block = np.array(
            kwargs["second_sys_idx_on_block"], dtype=np.int32
        ).flatten()

        # Number of contact pairs (rod-cylinder)
        self.n_contact_pairs = self.first_sys_idx_on_block.shape[0]

        a = 5

    def apply_forces(self, cylinder_one, index_one, cylinder_two, index_two):
        self._apply_forces(
            self.k,
            self.nu,
            self.n_contact_pairs,
            self.first_sys_idx_on_block,
            self.second_sys_idx_on_block,
            cylinder_one.start_idx_in_rod_elems,
            cylinder_one.end_idx_in_rod_nodes,
            cylinder_two.start_idx_in_rod_elems,
            cylinder_two.end_idx_in_rod_nodes,
            cylinder_one.position_collection,
            cylinder_one.director_collection,
            cylinder_one.radius,
            cylinder_one.length,
            cylinder_one.external_forces,
            cylinder_one.external_torques,
            cylinder_one.velocity_collection,
            cylinder_two.position_collection,
            cylinder_two.director_collection,
            cylinder_two.radius,
            cylinder_two.length,
            cylinder_two.external_forces,
            cylinder_two.external_torques,
            cylinder_two.velocity_collection,
        )

    @staticmethod
    @njit(cache=True)
    def _apply_forces(
        k,
        nu,
        n_contact_pairs,
        first_sys_idx_on_block,
        second_sys_idx_on_block,
        cylinder_one_start_idx_in_rod_elems,
        cylinder_one_end_idx_in_rod_nodes,
        cylinder_two_start_idx_in_rod_elems,
        cylinder_two_end_idx_in_rod_nodes,
        cylinder_one_position_collection,
        cylinder_one_director_collection,
        cylinder_one_radius,
        cylinder_one_length,
        cylinder_one_external_forces,
        cylinder_one_external_torques,
        cylinder_one_velocity_collection,
        cylinder_two_position_collection,
        cylinder_two_director_collection,
        cylinder_two_radius,
        cylinder_two_length,
        cylinder_two_external_forces,
        cylinder_two_external_torques,
        cylinder_two_velocity_collection,
    ):
        for loop_idx in range(n_contact_pairs):
            cylinder_one_elems_start_idx = cylinder_one_start_idx_in_rod_elems[
                first_sys_idx_on_block[loop_idx]
            ]
            cylinder_one_elems_end_idx = cylinder_one_end_idx_in_rod_nodes[
                first_sys_idx_on_block[loop_idx]
            ]
            cylinder_two_elems_start_idx = cylinder_two_start_idx_in_rod_elems[
                second_sys_idx_on_block[loop_idx]
            ]
            cylinder_two_elems_end_idx = cylinder_two_end_idx_in_rod_nodes[
                second_sys_idx_on_block[loop_idx]
            ]

            # First, check for a global AABB bounding box, and see whether that
            # intersects
            if _prune_using_aabbs_cylinder_cylinder(
                cylinder_one_position_collection[
                    :, cylinder_one_elems_start_idx:cylinder_one_elems_end_idx
                ],
                cylinder_one_director_collection[
                    :, :, cylinder_one_elems_start_idx:cylinder_one_elems_end_idx
                ],
                cylinder_one_radius[cylinder_one_elems_start_idx],
                cylinder_one_length[cylinder_one_elems_start_idx],
                cylinder_two_position_collection[
                    :, cylinder_two_elems_start_idx:cylinder_two_elems_end_idx
                ],
                cylinder_two_director_collection[
                    :, :, cylinder_two_elems_start_idx:cylinder_two_elems_end_idx
                ],
                cylinder_two_radius[cylinder_two_elems_start_idx],
                cylinder_two_length[cylinder_two_elems_start_idx],
            ):
                continue

            x_cyl_one = (
                cylinder_one_position_collection[..., cylinder_one_elems_start_idx]
                - 0.5
                * cylinder_one_length[cylinder_one_elems_start_idx]
                * cylinder_one_director_collection[2, :, cylinder_one_elems_start_idx]
            )

            x_cyl_two = (
                cylinder_two_position_collection[..., cylinder_two_elems_start_idx]
                - 0.5
                * cylinder_two_length[cylinder_two_elems_start_idx]
                * cylinder_two_director_collection[2, :, cylinder_two_elems_start_idx]
            )
            _calculate_contact_forces(
                x_cyl_one,
                cylinder_one_length[
                    cylinder_one_elems_start_idx:cylinder_one_elems_end_idx
                ]
                * cylinder_one_director_collection[2, :, cylinder_one_elems_start_idx],
                x_cyl_two,
                cylinder_two_length[
                    cylinder_two_elems_start_idx:cylinder_two_elems_end_idx
                ]
                * cylinder_two_director_collection[2, :, cylinder_two_elems_start_idx],
                cylinder_one_radius[
                    cylinder_one_elems_start_idx:cylinder_one_elems_end_idx
                ]
                + cylinder_two_radius[
                    cylinder_two_elems_start_idx:cylinder_two_elems_end_idx
                ],
                cylinder_one_length[
                    cylinder_one_elems_start_idx:cylinder_one_elems_end_idx
                ]
                + cylinder_two_length[
                    cylinder_two_elems_start_idx:cylinder_two_elems_end_idx
                ],
                cylinder_one_external_forces[
                    :, cylinder_one_elems_start_idx:cylinder_one_elems_end_idx
                ],
                cylinder_one_external_torques[
                    :, cylinder_one_elems_start_idx:cylinder_one_elems_end_idx
                ],
                cylinder_one_director_collection[:, :, cylinder_one_elems_start_idx],
                cylinder_two_external_forces[
                    :, cylinder_two_elems_start_idx:cylinder_two_elems_end_idx
                ],
                cylinder_two_external_torques[
                    :, cylinder_two_elems_start_idx:cylinder_two_elems_end_idx
                ],
                cylinder_two_director_collection[:, :, cylinder_two_elems_start_idx],
                cylinder_one_velocity_collection[
                    :, cylinder_one_elems_start_idx:cylinder_one_elems_end_idx
                ],
                cylinder_two_velocity_collection[
                    :, cylinder_two_elems_start_idx:cylinder_two_elems_end_idx
                ],
                k[loop_idx],
                nu[loop_idx],
                cylinder_one_position_collection[..., cylinder_one_elems_start_idx],
                cylinder_two_position_collection[..., cylinder_two_elems_start_idx],
            )
