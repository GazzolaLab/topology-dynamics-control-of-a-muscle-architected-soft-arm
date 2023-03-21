__doc__ = """Contact with rigid cylinder for memory block connections"""
# Developer note: We are using the same function in _elastica_numba._joint ExternalContact class, but in order to use
# memory block connections wrapper, we have to change the ExternalContact class.
# Current implementation will go over rods that can contact one by one.
__all__ = ["ExternalContactForMemoryBlock"]
import numpy as np
import numba
from numba import njit
from elastica._elastica_numba._joint import (
    _prune_using_aabbs,
    _norm,
    # _find_min_dist,
    _dot_product,
    ExternalContact,
    _clip,
    _out_of_bounds,
)


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
    x_collection_rod,
    edge_collection_rod,
    x_cylinder,
    edge_cylinder,
    radii_sum,
    length_sum,
    internal_forces_rod,
    external_forces_rod,
    external_forces_cylinder,
    external_torques_cylinder,
    cylinder_director_collection,
    velocity_rod,
    velocity_cylinder,
    contact_k,
    contact_nu,
    x_cylinder_center,
):
    # We already pass in only the first n_elem x
    n_points = x_collection_rod.shape[1]

    cylinder_total_contact_forces = np.zeros((3))
    cylinder_total_contact_torques = np.zeros((3))
    for i in range(n_points):
        # Element-wise bounding box
        x_selected = x_collection_rod[..., i]
        # x_cylinder is already a (,) array from outised
        del_x = x_selected - x_cylinder
        norm_del_x = _norm(del_x)

        # If outside then don't process
        if norm_del_x >= (radii_sum[i] + length_sum[i]):
            continue

        # find the shortest line segment between the two centerline
        # segments : differs from normal cylinder-cylinder intersection
        # rod to cylinder
        distance_vector, x_cylinder_contact_point = _find_min_dist(
            x_selected, edge_collection_rod[..., i], x_cylinder, edge_cylinder
        )
        # x_cylinder_contact_point = x_selected + distance_vector
        distance_vector_length = _norm(distance_vector)
        distance_vector /= distance_vector_length

        gamma = radii_sum[i] - distance_vector_length

        # If distance is large, don't worry about it
        if gamma < -1e-5:
            continue

        rod_elemental_forces = 0.5 * (
            external_forces_rod[..., i]
            + external_forces_rod[..., i + 1]
            + internal_forces_rod[..., i]
            + internal_forces_rod[..., i + 1]
        )
        equilibrium_forces = -rod_elemental_forces + external_forces_cylinder[..., 0]

        normal_force = _dot_product(equilibrium_forces, distance_vector)
        # Following line same as np.where(normal_force < 0.0, -normal_force, 0.0)
        normal_force = abs(min(normal_force, 0.0))

        # CHECK FOR GAMMA > 0.0, heaviside but we need to overload it in numba
        # As a quick fix, use this instead
        mask = (gamma > 0.0) * 1.0

        contact_force = contact_k * gamma
        interpenetration_velocity = (
            0.5 * (velocity_rod[..., i] + velocity_rod[..., i + 1])
            - velocity_cylinder[..., 0]
        )
        contact_damping_force = contact_nu * _dot_product(
            interpenetration_velocity, distance_vector
        )

        # magnitude* direction
        net_contact_force = (
            0.5
            * mask
            * (normal_force + (contact_damping_force + contact_force))
            * distance_vector
        )

        # Torques acting on the cylinder
        moment_arm = x_cylinder_contact_point - x_cylinder_center

        # Add it to the rods at the end of the day
        if i == 0:
            external_forces_rod[..., i] -= 2 / 3 * net_contact_force
            external_forces_rod[..., i + 1] -= 4 / 3 * net_contact_force
            cylinder_total_contact_forces += 2.0 * net_contact_force
            cylinder_total_contact_torques += np.cross(
                moment_arm, 2.0 * net_contact_force
            )
        elif i == n_points:
            external_forces_rod[..., i] -= 4 / 3 * net_contact_force
            external_forces_rod[..., i + 1] -= 2 / 3 * net_contact_force
            cylinder_total_contact_forces += 2.0 * net_contact_force
            cylinder_total_contact_torques += np.cross(
                moment_arm, 2.0 * net_contact_force
            )
        else:
            external_forces_rod[..., i] -= net_contact_force
            external_forces_rod[..., i + 1] -= net_contact_force
            cylinder_total_contact_forces += 2.0 * net_contact_force
            cylinder_total_contact_torques += np.cross(
                moment_arm, 2.0 * net_contact_force
            )

    # Update the cylinder external forces and torques
    external_forces_cylinder[..., 0] += cylinder_total_contact_forces
    external_torques_cylinder[..., 0] += (
        cylinder_director_collection @ cylinder_total_contact_torques
    )


class ExternalContactForMemoryBlock(ExternalContact):
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

    def apply_forces(self, rod_one, index_one, cylinder_two, index_two):
        self._apply_forces(
            self.k,
            self.nu,
            self.n_contact_pairs,
            self.first_sys_idx_on_block,
            self.second_sys_idx_on_block,
            rod_one.start_idx_in_rod_elems,
            rod_one.end_idx_in_rod_elems,
            rod_one.start_idx_in_rod_nodes,
            rod_one.end_idx_in_rod_nodes,
            cylinder_two.start_idx_in_rod_elems,
            cylinder_two.end_idx_in_rod_nodes,
            rod_one.position_collection,
            rod_one.tangents,
            rod_one.radius,
            rod_one.lengths,
            rod_one.internal_forces,
            rod_one.external_forces,
            rod_one.velocity_collection,
            cylinder_two.position_collection,
            cylinder_two.director_collection,
            cylinder_two.radius,
            cylinder_two.length,
            cylinder_two.external_forces,
            cylinder_two.velocity_collection,
            cylinder_two.external_torques,
        )

    @staticmethod
    @njit(cache=True)
    def _apply_forces(
        k,
        nu,
        n_contact_pairs,
        first_sys_idx_on_block,
        second_sys_idx_on_block,
        rod_one_start_idx_in_rod_elems,
        rod_one_end_idx_in_rod_elems,
        rod_one_start_idx_in_rod_nodes,
        rod_one_end_idx_in_rod_nodes,
        cylinder_two_start_idx_in_rod_elems,
        cylinder_two_end_idx_in_rod_nodes,
        rod_one_position_collection,
        rod_one_tangents,
        rod_one_radius,
        rod_one_lengths,
        rod_one_internal_forces,
        rod_one_external_forces,
        rod_one_velocity_collection,
        cylinder_two_position_collection,
        cylinder_two_director_collection,
        cylinder_two_radius,
        cylinder_two_length,
        cylinder_two_external_forces,
        cylinder_two_velocity_collection,
        cylinder_two_external_torques,
    ):
        for loop_idx in range(n_contact_pairs):
            rod_one_elems_start_idx = rod_one_start_idx_in_rod_elems[
                first_sys_idx_on_block[loop_idx]
            ]
            rod_one_elems_end_idx = rod_one_end_idx_in_rod_elems[
                first_sys_idx_on_block[loop_idx]
            ]
            rod_one_nodes_start_idx = rod_one_start_idx_in_rod_nodes[
                first_sys_idx_on_block[loop_idx]
            ]
            rod_one_nodes_end_idx = rod_one_end_idx_in_rod_nodes[
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
            if _prune_using_aabbs(
                rod_one_position_collection[
                    :, rod_one_nodes_start_idx:rod_one_nodes_end_idx
                ],
                rod_one_radius[rod_one_elems_start_idx:rod_one_elems_end_idx],
                rod_one_lengths[rod_one_elems_start_idx:rod_one_elems_end_idx],
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

            x_cyl = (
                cylinder_two_position_collection[..., cylinder_two_elems_start_idx]
                - 0.5
                * cylinder_two_length[cylinder_two_elems_start_idx]
                * cylinder_two_director_collection[2, :, cylinder_two_elems_start_idx]
            )

            rod_node_position = rod_one_position_collection[
                :, rod_one_nodes_start_idx:rod_one_nodes_end_idx
            ]
            rod_element_position = 0.5 * (
                rod_node_position[..., 1:] + rod_node_position[..., :-1]
            )
            _calculate_contact_forces(
                # rod_one_position_collection[
                #     :, rod_one_nodes_start_idx : rod_one_nodes_end_idx - 1
                # ],
                rod_element_position,
                rod_one_lengths[rod_one_elems_start_idx:rod_one_elems_end_idx]
                * rod_one_tangents[:, rod_one_elems_start_idx:rod_one_elems_end_idx],
                x_cyl,
                cylinder_two_length[
                    cylinder_two_elems_start_idx:cylinder_two_elems_end_idx
                ]
                * cylinder_two_director_collection[2, :, cylinder_two_elems_start_idx],
                rod_one_radius[rod_one_elems_start_idx:rod_one_elems_end_idx]
                + cylinder_two_radius[
                    cylinder_two_elems_start_idx:cylinder_two_elems_end_idx
                ],
                rod_one_lengths[rod_one_elems_start_idx:rod_one_elems_end_idx]
                + cylinder_two_length[
                    cylinder_two_elems_start_idx:cylinder_two_elems_end_idx
                ],
                rod_one_internal_forces[
                    :, rod_one_nodes_start_idx:rod_one_nodes_end_idx
                ],
                rod_one_external_forces[
                    :, rod_one_nodes_start_idx:rod_one_nodes_end_idx
                ],
                cylinder_two_external_forces[
                    :, cylinder_two_elems_start_idx:cylinder_two_elems_end_idx
                ],
                cylinder_two_external_torques[
                    :, cylinder_two_elems_start_idx:cylinder_two_elems_end_idx
                ],
                cylinder_two_director_collection[:, :, cylinder_two_elems_start_idx],
                rod_one_velocity_collection[
                    :, rod_one_nodes_start_idx:rod_one_nodes_end_idx
                ],
                cylinder_two_velocity_collection[
                    :, cylinder_two_elems_start_idx:cylinder_two_elems_end_idx
                ],
                k[loop_idx],
                nu[loop_idx],
                cylinder_two_position_collection[..., cylinder_two_elems_start_idx],
            )
