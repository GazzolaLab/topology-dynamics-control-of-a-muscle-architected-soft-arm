__doc__ = """ Muscular rod equations implementation for Elastica Numpy implementation. Different than Cosserat rod
equations, muscluar rod uses a nonlinear constitutive model for stress."""

__all__ = ["MuscularRod"]
import numpy as np
import functools
import numba
from elastica.rod import RodBase
from elastica._elastica_numba._linalg import (
    _batch_cross,
    _batch_norm,
    _batch_dot,
    _batch_matvec,
)
from elastica._elastica_numba._rotations import _inv_rotate
from elastica.rod.factory_function import allocate, allocate_ring_rod
from elastica._calculus import (
    quadrature_kernel_for_block_structure,
    difference_kernel_for_block_structure,
    _difference,
    _average,
)
from elastica._elastica_numba._interaction import node_to_element_pos_or_vel
from elastica.utils import Tolerance

position_difference_kernel = _difference
position_average = _average


@functools.lru_cache(maxsize=1)
def _get_z_vector():
    return np.array([0.0, 0.0, 1.0]).reshape(3, -1)


def _compute_sigma_kappa_for_blockstructure(memory_block):
    """
    This function is a wrapper to call functions which computes shear stretch, strain and bending twist and strain.

    Parameters
    ----------
    memory_block : object

    Returns
    -------

    """
    _compute_shear_stretch_strains(
        memory_block.position_collection,
        memory_block.volume,
        memory_block.lengths,
        memory_block.tangents,
        memory_block.radius,
        memory_block.rest_lengths,
        memory_block.rest_voronoi_lengths,
        memory_block.dilatation,
        memory_block.voronoi_dilatation,
        memory_block.director_collection,
        memory_block.sigma,
    )

    # Compute bending twist strains for the block
    _compute_bending_twist_strains(
        memory_block.director_collection,
        memory_block.rest_voronoi_lengths,
        memory_block.kappa,
    )


class MuscularRod(RodBase):
    def __init__(
        self,
        n_elements,
        position,
        velocity,
        omega,
        acceleration,
        angular_acceleration,
        directors,
        radius,
        mass_second_moment_of_inertia,
        inv_mass_second_moment_of_inertia,
        shear_matrix,
        bend_matrix,
        density,
        volume,
        mass,
        dissipation_constant_for_forces,
        dissipation_constant_for_torques,
        internal_forces,
        internal_torques,
        external_forces,
        external_torques,
        lengths,
        rest_lengths,
        tangents,
        dilatation,
        dilatation_rate,
        voronoi_dilatation,
        rest_voronoi_lengths,
        sigma,
        kappa,
        rest_sigma,
        rest_kappa,
        internal_stress,
        internal_couple,
        damping_forces,
        damping_torques,
        args,
        kwargs,
    ):
        self.n_elems = n_elements
        self.position_collection = position
        self.velocity_collection = velocity
        self.omega_collection = omega
        self.acceleration_collection = acceleration
        self.alpha_collection = angular_acceleration
        self.director_collection = directors
        self.radius = radius
        self.mass_second_moment_of_inertia = mass_second_moment_of_inertia
        self.inv_mass_second_moment_of_inertia = inv_mass_second_moment_of_inertia
        self.shear_matrix = shear_matrix
        self.bend_matrix = bend_matrix
        self.density = density
        self.volume = volume
        self.mass = mass
        self.dissipation_constant_for_forces = dissipation_constant_for_forces
        self.dissipation_constant_for_torques = dissipation_constant_for_torques
        self.internal_forces = internal_forces
        self.internal_torques = internal_torques
        self.external_forces = external_forces
        self.external_torques = external_torques
        self.lengths = lengths
        self.rest_lengths = rest_lengths
        self.tangents = tangents
        self.dilatation = dilatation
        self.dilatation_rate = dilatation_rate
        self.voronoi_dilatation = voronoi_dilatation
        self.rest_voronoi_lengths = rest_voronoi_lengths
        self.sigma = sigma
        self.kappa = kappa
        self.rest_sigma = rest_sigma
        self.rest_kappa = rest_kappa
        self.internal_stress = internal_stress
        self.internal_couple = internal_couple
        self.damping_forces = damping_forces
        self.damping_torques = damping_torques

        # Variables for constitutive model.
        self.fiber_activation = np.zeros((n_elements))

        # rest base area
        self.rest_area = np.pi * self.radius * self.radius

        # if kwargs.__contains__("stretch_optimal"):
        #     self.stretch_optimal = np.ones((n_elements)) * kwargs.get("stretch_optimal")
        # else:
        #     # raise AttributeError("Did you forget to input stretch_optimal in kwargs ?")
        #     self.stretch_optimal = np.ones((n_elements))

        # If n_elems_with_boundary defined and passed with kwargs, then this rod is ring and
        # n_elems_with_boundary is a member of ring rod.
        if kwargs.__contains__("ring_rod_flag"):
            self.ring_rod_flag = kwargs.get("ring_rod_flag")

        # if kwargs.__contains__("muscle_force_scale"):
        #     self.muscle_force_scale = np.ones((n_elements)) * kwargs.get(
        #         "muscle_force_scale"
        #     )
        # else:
        #     self.muscle_force_scale = np.ones((n_elements))

        if kwargs.__contains__("tension_passive_force_scale"):
            self.tension_passive_force_scale = np.ones((n_elements)) * kwargs.get(
                "tension_passive_force_scale"
            )
        else:
            self.tension_passive_force_scale = np.ones((n_elements))

        # if kwargs.__contains__("myosin_lengths"):
        #     temp_myosin_lengths = kwargs.get("myosin_lengths")
        #     assert temp_myosin_lengths.shape == (n_elements,), (
        #         "myosin_lengths shape is "
        #         + str(temp_myosin_lengths.shape)
        #         + " it should be "
        #         + str(n_elements)
        #     )
        #     for k in range(n_elements):
        #         assert temp_myosin_lengths[k] > Tolerance.atol(), (
        #             " Myosin lengths has to be greater than 0"
        #             + " Check you myosin_lengths input!"
        #         )
        #     self.myosin_lengths = temp_myosin_lengths
        # else:
        #     self.myosin_lengths = 0.7 * np.ones((n_elements))  # in micro meters
        #
        # if kwargs.__contains__("sarcomere_rest_lengths"):
        #     temp_sarcomere_rest_lengths = kwargs.get("sarcomere_rest_lengths")
        #     assert temp_sarcomere_rest_lengths.shape == (n_elements,), (
        #         "sarcomere_rest_lengths shape is "
        #         + str(temp_sarcomere_rest_lengths.shape)
        #         + " it should be "
        #         + str(n_elements)
        #     )
        #     for k in range(n_elements):
        #         assert temp_sarcomere_rest_lengths[k] > Tolerance.atol(), (
        #             " Sacromere rest lengths has to be greater than 0"
        #             + " Check you sarcomere_rest_lengths input!"
        #         )
        #     self.sarcomere_rest_lengths = temp_sarcomere_rest_lengths
        # else:
        #     self.sarcomere_rest_lengths = 1.0 * np.ones((n_elements))  # in micro meters

        if kwargs.__contains__("maximum_active_stress"):
            temp_maximum_active_stress = kwargs.get("maximum_active_stress")
            assert temp_maximum_active_stress.shape == (n_elements,), (
                "maximum_active_stress shape is "
                + str(temp_maximum_active_stress.shape)
                + " it should be "
                + str(n_elements)
            )
            for k in range(n_elements):
                assert temp_maximum_active_stress[k] > Tolerance.atol(), (
                    " Maximum active stress has to be greater than 0"
                    + " Check you maximum_active_stress input!"
                )
            self.maximum_active_stress = temp_maximum_active_stress
        else:
            maximum_active_stress_ref = 280e3
            self.maximum_active_stress = maximum_active_stress_ref * np.ones(
                (n_elements)
            )

        if kwargs.__contains__("minimum_strain_rate"):
            temp_minimum_strain_rate = kwargs.get("minimum_strain_rate")
            assert temp_minimum_strain_rate.shape == (n_elements,), (
                "minimum_strain_rate shape is "
                + str(temp_minimum_strain_rate.shape)
                + " it should be "
                + str(n_elements)
            )
            for k in range(n_elements):
                assert np.abs(temp_minimum_strain_rate[k]) > Tolerance.atol(), (
                    " Minimum strain rate has to be greater than 0"
                    + " Check you minimum_strain_rate input!"
                )
            self.minimum_strain_rate = temp_minimum_strain_rate
        else:
            minimum_strain_rate_ref = kwargs.get(
                "minimum_strain_rate_ref", -17
            )  # 1/sec
            l0_sarc_ref = kwargs.get("sacromere_reference_length", 2.37)  # micro m
            self.minimum_strain_rate = minimum_strain_rate_ref * (
                l0_sarc_ref / 1.0 * np.ones((n_elements))  # self.sarcomere_rest_lengths
            )

        # if kwargs.__contains__("normalized_active_force_y_intercept"):
        #     self.normalized_active_force_y_intercept = kwargs.get(
        #         "normalized_active_force_y_intercept"
        #     )
        # else:
        #     raise AttributeError(
        #         "Did you forget to input normalized_active_force_y_intercept in kwargs ?"
        #     )
        #
        # if kwargs.__contains__("normalized_active_force_slope"):
        #     self.normalized_active_force_slope = kwargs.get(
        #         "normalized_active_force_slope"
        #     )
        # else:
        #     raise AttributeError(
        #         "Did you forget to input normalized_active_force_slope in kwargs ?"
        #     )
        #
        # if kwargs.__contains__("normalized_active_force_break_points"):
        #     self.normalized_active_force_break_points = kwargs.get(
        #         "normalized_active_force_break_points"
        #     )
        # else:
        #     raise AttributeError(
        #         "Did you forget to input normalized_active_force_break_points in kwargs ?"
        #     )
        #
        if kwargs.__contains__("E_compression"):
            self.E_compression = np.ones((n_elements)) * kwargs.get("E_compression")
        else:
            raise AttributeError("Did you forget to input E_comp in kwargs ?")

        if kwargs.__contains__("compression_strain_limit"):
            self.compression_strain_limit = np.ones((n_elements)) * kwargs.get(
                "compression_strain_limit"
            )
        else:
            self.compression_strain_limit = np.zeros((n_elements))

        if kwargs.__contains__("extension_strain_limit"):
            self.extension_strain_limit = np.ones((n_elements)) * kwargs.get(
                "extension_strain_limit"
            )
        else:
            self.extension_strain_limit = np.zeros((n_elements))

        if kwargs.__contains__("passive_force_coefficients"):
            self.passive_force_coefficients = kwargs.get("passive_force_coefficients")
        else:
            raise AttributeError(
                "Did you forget to input passive_force_coefficients in kwargs ?"
            )

        if kwargs.__contains__("active_force_coefficients"):
            self.active_force_coefficients = kwargs.get("active_force_coefficients")
        else:
            raise AttributeError(
                "Did you forget to input active_force_coefficients in kwargs ?"
            )

        if kwargs.__contains__("force_velocity_constant"):
            self.force_velocity_constant = kwargs.get("force_velocity_constant")
        else:
            raise AttributeError(
                "Did you forget to input force_velocity_constant in kwargs ?"
            )

        # Compute shear stretch and strains.
        # _compute_shear_stretch_strains(
        #     self.position_collection,
        #     self.volume,
        #     self.lengths,
        #     self.tangents,
        #     self.radius,
        #     self.rest_lengths,
        #     self.rest_voronoi_lengths,
        #     self.dilatation,
        #     self.voronoi_dilatation,
        #     self.director_collection,
        #     self.sigma,
        # )

        # Compute bending twist strains
        # _compute_bending_twist_strains(
        #     self.director_collection, self.rest_voronoi_lengths, self.kappa
        # )

    @classmethod
    def straight_rod(
        cls,
        n_elements,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        nu,
        youngs_modulus,
        poisson_ratio,
        alpha_c=4.0 / 3.0,
        *args,
        **kwargs,
    ):
        (
            n_elements,
            position,
            velocity,
            omega,
            acceleration,
            angular_acceleration,
            directors,
            radius,
            mass_second_moment_of_inertia,
            inv_mass_second_moment_of_inertia,
            shear_matrix,
            bend_matrix,
            density,
            volume,
            mass,
            dissipation_constant_for_forces,
            dissipation_constant_for_torques,
            internal_forces,
            internal_torques,
            external_forces,
            external_torques,
            lengths,
            rest_lengths,
            tangents,
            dilatation,
            dilatation_rate,
            voronoi_dilatation,
            rest_voronoi_lengths,
            sigma,
            kappa,
            rest_sigma,
            rest_kappa,
            internal_stress,
            internal_couple,
            damping_forces,
            damping_torques,
            args,
            kwargs,
        ) = allocate(
            n_elements,
            start,
            direction,
            normal,
            base_length,
            base_radius,
            density,
            nu,
            youngs_modulus,
            poisson_ratio,
            alpha_c=4.0 / 3.0,
            *args,
            **kwargs,
        )

        return cls(
            n_elements,
            position,
            velocity,
            omega,
            acceleration,
            angular_acceleration,
            directors,
            radius,
            mass_second_moment_of_inertia,
            inv_mass_second_moment_of_inertia,
            shear_matrix,
            bend_matrix,
            density,
            volume,
            mass,
            dissipation_constant_for_forces,
            dissipation_constant_for_torques,
            internal_forces,
            internal_torques,
            external_forces,
            external_torques,
            lengths,
            rest_lengths,
            tangents,
            dilatation,
            dilatation_rate,
            voronoi_dilatation,
            rest_voronoi_lengths,
            sigma,
            kappa,
            rest_sigma,
            rest_kappa,
            internal_stress,
            internal_couple,
            damping_forces,
            damping_torques,
            args,
            kwargs,
        )

    @classmethod
    def ring_rod(
        cls,
        n_elements,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        nu,
        youngs_modulus,
        poisson_ratio,
        alpha_c=4.0 / 3.0,
        *args,
        **kwargs,
    ):
        (
            n_elements,
            position,
            velocities,
            omegas,
            accelerations,
            angular_accelerations,
            directors,
            radius,
            mass_second_moment_of_inertia,
            inv_mass_second_moment_of_inertia,
            shear_matrix,
            bend_matrix,
            density,
            volume,
            mass,
            dissipation_constant_for_forces,
            dissipation_constant_for_torques,
            internal_forces,
            internal_torques,
            external_forces,
            external_torques,
            lengths,
            rest_lengths,
            tangents,
            dilatation,
            dilatation_rate,
            voronoi_dilatation,
            rest_voronoi_lengths,
            sigma,
            kappa,
            rest_sigma,
            rest_kappa,
            internal_stress,
            internal_couple,
            damping_forces,
            damping_torques,
            args,
            kwargs,
        ) = allocate_ring_rod(
            n_elements,
            start,
            direction,
            normal,
            base_length,
            base_radius,
            density,
            nu,
            youngs_modulus,
            poisson_ratio,
            alpha_c=4.0 / 3.0,
            *args,
            **kwargs,
        )

        return cls(
            n_elements,
            position,
            velocities,
            omegas,
            accelerations,
            angular_accelerations,
            directors,
            radius,
            mass_second_moment_of_inertia,
            inv_mass_second_moment_of_inertia,
            shear_matrix,
            bend_matrix,
            density,
            volume,
            mass,
            dissipation_constant_for_forces,
            dissipation_constant_for_torques,
            internal_forces,
            internal_torques,
            external_forces,
            external_torques,
            lengths,
            rest_lengths,
            tangents,
            dilatation,
            dilatation_rate,
            voronoi_dilatation,
            rest_voronoi_lengths,
            sigma,
            kappa,
            rest_sigma,
            rest_kappa,
            internal_stress,
            internal_couple,
            damping_forces,
            damping_torques,
            args,
            kwargs,
        )

    def compute_internal_forces_and_torques(self, time):
        """
        Compute internal forces and torques. We need to compute internal forces and torques before the acceleration because
        they are used in interaction. Thus in order to speed up simulation, we will compute internal forces and torques
        one time and use them. Previously, we were computing internal forces and torques multiple times in interaction.
        Saving internal forces and torques in a variable take some memory, but we will gain speed up.
        Parameters
        ----------
        time

        Returns
        -------

        """

        _compute_internal_forces(
            self.position_collection,
            self.volume,
            self.lengths,
            self.tangents,
            self.radius,
            self.rest_lengths,
            self.rest_voronoi_lengths,
            self.dilatation,
            self.dilatation_rate,
            self.voronoi_dilatation,
            self.director_collection,
            self.sigma,
            self.rest_sigma,
            self.shear_matrix,
            self.internal_stress,
            self.velocity_collection,
            self.dissipation_constant_for_forces,
            self.damping_forces,
            self.internal_forces,
            self.fiber_activation,
            self.rest_area,
            self.tension_passive_force_scale,
            self.maximum_active_stress,
            self.minimum_strain_rate,
            self.active_force_coefficients,
            self.force_velocity_constant,
            # self.normalized_active_force_y_intercept,
            # self.normalized_active_force_slope,
            # self.normalized_active_force_break_points,
            self.E_compression,
            self.compression_strain_limit,
            self.extension_strain_limit,
            self.passive_force_coefficients,
            self.ghost_elems_idx,
        )

        _compute_internal_torques(
            self.position_collection,
            self.velocity_collection,
            self.tangents,
            self.lengths,
            self.rest_lengths,
            self.director_collection,
            self.rest_voronoi_lengths,
            self.bend_matrix,
            self.rest_kappa,
            self.kappa,
            self.voronoi_dilatation,
            self.mass_second_moment_of_inertia,
            self.omega_collection,
            self.internal_stress,
            self.internal_couple,
            self.dilatation,
            self.dilatation_rate,
            self.dissipation_constant_for_torques,
            self.damping_torques,
            self.internal_torques,
            self.ghost_elems_idx,
        )

    # Interface to time-stepper mixins (Symplectic, Explicit), which calls this method
    def update_accelerations(self, time):
        """
        This class method function is only a wrapper to call Numba njit function, which
        updates the acceleration

        Parameters
        ----------
        time

        Returns
        -------

        """
        _update_accelerations(
            self.acceleration_collection,
            self.internal_forces,
            self.external_forces,
            self.mass,
            self.alpha_collection,
            self.inv_mass_second_moment_of_inertia,
            self.internal_torques,
            self.external_torques,
            self.dilatation,
        )

    def zeroed_out_external_forces_and_torques(self, time):
        _zeroed_out_external_forces_and_torques(
            self.external_forces, self.external_torques
        )

    def compute_translational_energy(self):
        return (
            0.5
            * (
                self.mass
                * np.einsum(
                    "ij, ij-> j", self.velocity_collection, self.velocity_collection
                )
            ).sum()
        )

    def compute_rotational_energy(self):
        J_omega_upon_e = (
            _batch_matvec(self.mass_second_moment_of_inertia, self.omega_collection)
            / self.dilatation
        )
        return 0.5 * np.einsum("ik,ik->k", self.omega_collection, J_omega_upon_e).sum()

    def compute_velocity_center_of_mass(self):
        mass_times_velocity = np.einsum("j,ij->ij", self.mass, self.velocity_collection)
        sum_mass_times_velocity = np.einsum("ij->i", mass_times_velocity)

        return sum_mass_times_velocity / self.mass.sum()

    def compute_position_center_of_mass(self):
        mass_times_position = np.einsum("j,ij->ij", self.mass, self.position_collection)
        sum_mass_times_position = np.einsum("ij->i", mass_times_position)

        return sum_mass_times_position / self.mass.sum()

    def compute_bending_energy(self):
        kappa_diff = self.kappa - self.rest_kappa
        bending_internal_torques = _batch_matvec(self.bend_matrix, kappa_diff)

        return (
            0.5
            * (
                _batch_dot(kappa_diff, bending_internal_torques)
                * self.rest_voronoi_lengths
            ).sum()
        )

    def compute_shear_energy(self):
        sigma_diff = self.sigma - self.rest_sigma
        shear_internal_torques = _batch_matvec(self.shear_matrix, sigma_diff)

        return (
            0.5
            * (_batch_dot(sigma_diff, shear_internal_torques) * self.rest_lengths).sum()
        )


@numba.njit(cache=True)
def _compute_geometry_from_state(
    position_collection, volume, lengths, tangents, radius
):
    """
    Returns
    -------

    """
    # Compute eq (3.3) from 2018 RSOS paper

    # Note : we can use the two-point difference kernel, but it needs unnecessary padding
    # and hence will always be slower
    position_diff = position_difference_kernel(position_collection)
    # FIXME: Here 1E-14 is added to fix ghost lengths, which is 0, and causes division by zero error!
    lengths[:] = _batch_norm(position_diff) + 1e-14
    # _reset_scalar_ghost(lengths, ghost_elems_idx, 1.0)

    for k in range(lengths.shape[0]):
        tangents[0, k] = position_diff[0, k] / lengths[k]
        tangents[1, k] = position_diff[1, k] / lengths[k]
        tangents[2, k] = position_diff[2, k] / lengths[k]
        # resize based on volume conservation
        radius[k] = np.sqrt(volume[k] / lengths[k] / np.pi)


@numba.njit(cache=True)
def _compute_all_dilatations(
    position_collection,
    volume,
    lengths,
    tangents,
    radius,
    dilatation,
    rest_lengths,
    rest_voronoi_lengths,
    voronoi_dilatation,
):
    """
    Compute element and Voronoi region dilatations
    Returns
    -------

    """
    _compute_geometry_from_state(position_collection, volume, lengths, tangents, radius)
    # Caveat : Needs already set rest_lengths and rest voronoi domain lengths
    # Put in initialization
    for k in range(lengths.shape[0]):
        dilatation[k] = lengths[k] / rest_lengths[k]

    # Compute eq (3.4) from 2018 RSOS paper
    # Note : we can use trapezoidal kernel, but it has padding and will be slower
    voronoi_lengths = position_average(lengths)

    # Compute eq (3.45 from 2018 RSOS paper
    for k in range(voronoi_lengths.shape[0]):
        voronoi_dilatation[k] = voronoi_lengths[k] / rest_voronoi_lengths[k]


@numba.njit(cache=True)
def _compute_dilatation_rate(
    position_collection, velocity_collection, lengths, rest_lengths, dilatation_rate
):
    """

    Returns
    -------

    """
    # TODO Use the vector formula rather than separating it out
    # self.lengths = l_i = |r^{i+1} - r^{i}|
    r_dot_v = _batch_dot(position_collection, velocity_collection)
    r_plus_one_dot_v = _batch_dot(
        position_collection[..., 1:], velocity_collection[..., :-1]
    )
    r_dot_v_plus_one = _batch_dot(
        position_collection[..., :-1], velocity_collection[..., 1:]
    )

    blocksize = lengths.shape[0]

    for k in range(blocksize):
        dilatation_rate[k] = (
            (r_dot_v[k] + r_dot_v[k + 1] - r_dot_v_plus_one[k] - r_plus_one_dot_v[k])
            / lengths[k]
            / rest_lengths[k]
        )


@numba.njit(cache=True)
def _compute_shear_stretch_strains(
    position_collection,
    volume,
    lengths,
    tangents,
    radius,
    rest_lengths,
    rest_voronoi_lengths,
    dilatation,
    voronoi_dilatation,
    director_collection,
    sigma,
):
    # Quick trick : Instead of evaliation Q(et-d^3), use property that Q*d3 = (0,0,1), a constant
    _compute_all_dilatations(
        position_collection,
        volume,
        lengths,
        tangents,
        radius,
        dilatation,
        rest_lengths,
        rest_voronoi_lengths,
        voronoi_dilatation,
    )

    z_vector = np.array([0.0, 0.0, 1.0]).reshape(3, -1)
    sigma[:] = dilatation * _batch_matvec(director_collection, tangents) - z_vector


# @numba.njit(cache=True)
# def _compute_internal_shear_stretch_stresses_from_model_BLEMKER(
#     position_collection,
#     volume,
#     lengths,
#     tangents,
#     radius,
#     rest_lengths,
#     rest_voronoi_lengths,
#     dilatation,
#     dilatation_rate,
#     voronoi_dilatation,
#     director_collection,
#     sigma,
#     rest_sigma,
#     shear_matrix,
#     internal_stress,
#     fiber_force_passive,
#     fiber_force_active,
#     fiber_activation,
#     stretch_optimal,
#     stretch_critical,
#     stress_maximum,
#     P1,
#     P2,
#     P3,
#     P4,
#     rest_area,
# ):
#     """
#     OLD Constitutive model for the muscle.
#
#     References
#     ----------
#     Blemker SS, Pinsky PM, Delp SL. A 3D model of muscle reveals the causes of nonuniform strains in the
#      biceps brachii. J Biomech. 2005;38(4):657-665. doi:10.1016/j.jbiomech.2004.04.009
#
#     Returns
#     -------
#
#     """
#     _compute_shear_stretch_strains(
#         position_collection,
#         volume,
#         lengths,
#         tangents,
#         radius,
#         rest_lengths,
#         rest_voronoi_lengths,
#         dilatation,
#         voronoi_dilatation,
#         director_collection,
#         sigma,
#     )
#
#     internal_stress[:] = _batch_matvec(shear_matrix, sigma - rest_sigma)
#
#     fiber_force_passive *= 0.0
#     fiber_force_active *= 0.0
#
#     # Lambda stretch ratio in the paper is dilatation in our simulation
#     stretch = dilatation
#
#     blocksize = stretch.shape[0]
#
#     for k in range(blocksize):
#
#         # Compute passive fiber force. Table 1 in reference
#         # Case 1 stretch =< stretch_optimal, fiber_force_passive = 0 No action needed
#
#         # Case 2 stretch_optimal < stretch < stretch_critical
#         if stretch[k] > stretch_optimal and stretch[k] < stretch_critical:
#             fiber_force_passive[k] = P1 * (
#                 np.exp(P2 * (stretch[k] / stretch_optimal - 1.0)) - 1.0
#             )
#
#         # Case 3 stretch >= stretch_critical
#         elif stretch[k] >= stretch_critical:
#             fiber_force_passive[k] = P3 * stretch[k] / stretch_optimal + P4
#
#         # Compute active fiber force. Table 1 in reference
#         # Case 1 stretch <= 0.6*stretch_optimal
#         if stretch[k] <= 0.6 * stretch_optimal:
#             fiber_force_active[k] = 9.0 * (stretch[k] / stretch_optimal - 0.4) ** 2
#
#         # Case 2 stretch >= 1.4 * stretch_optimal
#         elif stretch[k] >= 1.4 * stretch_optimal:
#             fiber_force_active[k] = 9.0 * (stretch[k] / stretch_optimal - 1.6) ** 2
#
#         # Case 3 0.6*stretch_optimal< stretch <  1.4 * stretch_optimal
#         elif stretch[k] > 0.6 * stretch_optimal and stretch[k] < 1.4 * stretch_optimal:
#             fiber_force_active[k] = (
#                 1.0 - 4.0 * (1.0 - stretch[k] / stretch_optimal) ** 2
#             )
#
#         # Compute total fiber force. Eq 7 in reference
#         fiber_force_total = (
#             fiber_force_passive[k] + fiber_activation[k] * fiber_force_active[k]
#         )
#
#         # Compute total internal stress in the tangent direction. Inside the internal
#         # stress variable we are storing internal force in material frame. This is
#         # confusing, but we store n_L eqn 2.11 in RSoS 2018 paper.
#         internal_stress[2, k] = rest_area[k] * (
#             # stress_maximum * fiber_force_total * dilatation[k]*dilatation[k] / stretch_optimal
#             stress_maximum
#             * fiber_force_total
#             * stretch[k]
#             / stretch_optimal
#         )
#
#
# @numba.njit(cache=True)
# def _compute_internal_shear_stretch_stresses_from_model_VL(
#     position_collection,
#     volume,
#     lengths,
#     tangents,
#     radius,
#     rest_lengths,
#     rest_voronoi_lengths,
#     dilatation,
#     dilatation_rate,
#     voronoi_dilatation,
#     director_collection,
#     sigma,
#     rest_sigma,
#     shear_matrix,
#     internal_stress,
#     fiber_activation,
#     stretch_optimal,
#     rest_area,
#     muscle_force_scale,
#     l_myo,  # myosin_lengths
#     l0_sarc,  # sarcomere_rest_lengths
#     maximum_active_stress,
#     minimum_strain_rate,
# ):
#     """
#     Constitutive model for the muscle. The model provided in the below reference is implemented with the
#     addition of if a small passive elastic response for strain < -0.3 to keep the model stable in this regime.
#     Model parameters are taken as rough averages of the range given for squid tentacles in Table 1 of the below
#     reference.
#
#     References
#     ----------
#     Van Leeuwen, J. L., and William M. Kier. "Functional design of tentacles in squid: linking sarcomere
#     ultrastructure to gross morphological dynamics." Philosophical Transactions of the Royal Society of
#     London. Series B: Biological Sciences 352.1353 (1997): 551-571.
#
#     Returns
#     -------
#     """
#
#     _compute_shear_stretch_strains(
#         position_collection,
#         volume,
#         lengths,
#         tangents,
#         radius,
#         rest_lengths,
#         rest_voronoi_lengths,
#         dilatation,
#         voronoi_dilatation,
#         director_collection,
#         sigma,
#     )
#
#     internal_stress[:] = _batch_matvec(shear_matrix, sigma - rest_sigma)
#
#     # These are the adjustable model parameters.
#     ################################
#     # strain_rate_min = -42.0  # 1/sec
#     c3 = 1450e3
#     c4 = -625e3
#     eps_c = 0.773
#     l_bz = 0.14  # micro meter
#     l_z = 0.06  # micro meter
#     # l0_sarc = l_act + l_z + 0.5*l_bz
#     # l0_sarc = 1.0e-6
#     # l_myo = 0.7e-6
#     D_act = 0.68
#     D_myo = 1.90
#     C_myo = 0.44
#     l_min = 6e-7  # l_bz
#     # f_a = 1.0
#     # stress_max = 280e3
#     K = 0.25
#     E_comp = 2e5
#     compress_strain = -0.0  # 0.25  # -0.3
#     ################################
#
#     c2 = c3 * eps_c / (c3 * eps_c + c4)
#     c1 = c3 / (c2 * eps_c ** (c2 - 1))
#     # l_act = l0_sarc - l_z - 0.5 * l_bz
#
#     # Lambda stretch ratio in the paper is dilatation in our simulation
#     stretch = dilatation
#     # eps = dilatation/stretch_optimal - 1.0
#     # eps = sigma[2, :] - (stretch_optimal - 1.0)
#     eps = sigma[2, :]
#
#     # TODO: make this the true strain rate
#     eps_dot = dilatation_rate / minimum_strain_rate
#
#     sigma_pass = np.zeros((stretch.shape[0]))
#     f_v = np.zeros((stretch.shape[0]))
#     f_l = np.zeros((stretch.shape[0]))
#
#     blocksize = stretch.shape[0]
#
#     for k in range(blocksize):
#         if eps_dot[k] < 0:
#             f_v[k] = 1.8 - 0.8 * (1 + eps_dot[k]) / (1.0 - 7.56 * eps_dot[k] / K)
#         else:
#             f_v[k] = (1.0 - eps_dot[k]) / (1.0 + eps_dot[k] / K)
#
#         l_act = l0_sarc[k] - l_z - 0.5 * l_bz
#         l_sarc = l0_sarc[k] + eps[k] * l0_sarc[k]
#
#         if l_act + l_bz + l_z <= l_sarc and l_sarc <= l_myo[k] + l_act + l_z:
#             f_l[k] = (l_myo[k] + l_act + l_z - l_sarc) / (l_myo[k] - l_bz)
#         elif l_act + l_z <= l_sarc and l_sarc <= l_act + l_bz + l_z:
#             f_l[k] = 1.0
#         elif l_myo[k] + l_z <= l_sarc and l_sarc <= l_act + l_z:
#             f_l[k] = (l_myo[k] - l_bz - D_act * (l_act + l_z - l_sarc)) / (
#                 l_myo[k] - l_bz
#             )
#         # TODO: Compute l_min and have it correspond to the length of l_sarc when f_l = 0.
#         elif l_min <= l_sarc and l_sarc <= l_myo[k] + l_z:
#             f_l[k] = (
#                 l_myo[k]
#                 - l_bz
#                 - D_act * (l_act + l_z - l_sarc)
#                 - D_myo * (l_myo[k] + l_z - l_sarc)
#                 - C_myo * (l_myo[k] + l_z - l_sarc)
#             ) / (l_myo[k] - l_bz)
#         #     if f_l[k] < 0.0:
#         #         f_l[k] = 0.0
#         else:
#             f_l[k] = 0.0
#
#         if eps[k] <= compress_strain:
#             sigma_pass[k] = E_comp * (eps[k] - compress_strain)
#         elif eps[k] <= 0.0:
#             sigma_pass[k] = 0.0
#         elif eps[k] < eps_c:
#             sigma_pass[k] = c1 * eps[k] ** c2
#         else:
#             sigma_pass[k] = c3 * eps[k] + c4
#
#         internal_stress[2, k] = (
#             rest_area[k]
#             * (
#                 fiber_activation[k] * maximum_active_stress[k] * f_v[k] * f_l[k]
#                 + sigma_pass[k]
#             )
#             * muscle_force_scale[k]
#         )


# @numba.njit(cache=True)
# def _compute_internal_shear_stretch_stresses_from_model(
#     position_collection,
#     volume,
#     lengths,
#     tangents,
#     radius,
#     rest_lengths,
#     rest_voronoi_lengths,
#     dilatation,
#     dilatation_rate,
#     voronoi_dilatation,
#     director_collection,
#     sigma,
#     rest_sigma,
#     shear_matrix,
#     internal_stress,
#     fiber_activation,
#     rest_area,
#     tension_passive_force_scale,
#     maximum_active_stress,
#     minimum_strain_rate,
#     normalized_active_force_y_intercept,
#     normalized_active_force_slope,
#     normalized_active_force_break_points,
#     E_compression,
#     compression_strain_limit,
#     extension_strain_limit,
#     passive_force_coefficients,
#     velocity_collection,
# ):
#     """
#     Constitutive model for the muscle. The model provided in the below reference is implemented with the
#     addition of if a small passive elastic response for strain < -0.3 to keep the model stable in this regime.
#     Model parameters are taken as rough averages of the range given for squid tentacles in Table 1 of the below
#     reference.
#
#     References
#     ----------
#     Van Leeuwen, J. L., and William M. Kier. "Functional design of tentacles in squid: linking sarcomere
#     ultrastructure to gross morphological dynamics." Philosophical Transactions of the Royal Society of
#     London. Series B: Biological Sciences 352.1353 (1997): 551-571.
#
#     Returns
#     -------
#     """
#
#     _compute_shear_stretch_strains(
#         position_collection,
#         volume,
#         lengths,
#         tangents,
#         radius,
#         rest_lengths,
#         rest_voronoi_lengths,
#         dilatation,
#         voronoi_dilatation,
#         director_collection,
#         sigma,
#     )
#
#     internal_stress[:] = _batch_matvec(shear_matrix, sigma - rest_sigma)
#
#     # Compute dilatation rate when needed, dilatation itself is done before
#     # in internal_stresses
#     _compute_dilatation_rate(
#         position_collection, velocity_collection, lengths, rest_lengths, dilatation_rate
#     )
#
#     # These are the adjustable model parameters.
#     ################################
#     # c3 = 1450e3
#     # c4 = -625e3
#     # eps_c = 0.773
#     K = 0.25
#     ################################
#
#     # c2 = c3 * eps_c / (c3 * eps_c + c4)
#     # c1 = c3 / (c2 * eps_c ** (c2 - 1))
#
#     # Lambda stretch ratio in the paper is dilatation in our simulation
#     stretch = dilatation
#     eps = sigma[2, :] - rest_sigma[2,:]
#
#     # TODO: make this the true strain rate
#     eps_dot = dilatation_rate / minimum_strain_rate
#
#     sigma_pass = np.zeros((stretch.shape[0]))
#     f_v = np.zeros((stretch.shape[0]))
#     f_l = np.zeros((stretch.shape[0]))
#
#     blocksize = stretch.shape[0]
#
#     for k in range(blocksize):
#         if eps_dot[k] < 0:
#             f_v[k] = 1.8 - 0.8 * (1 + eps_dot[k]) / (1.0 - 7.56 * eps_dot[k] / K)
#         else:
#             f_v[k] = (1.0 - eps_dot[k]) / (1.0 + eps_dot[k] / K)
#
#         if (
#             eps[k] >= normalized_active_force_break_points[0, k]
#             and eps[k] <= normalized_active_force_break_points[1, k]
#         ):
#             f_l[k] = (
#                 normalized_active_force_y_intercept[0, k]
#                 + normalized_active_force_slope[0, k] * eps[k]
#             )
#
#         elif (
#             eps[k] > normalized_active_force_break_points[1, k]
#             and eps[k] <= normalized_active_force_break_points[2, k]
#         ):
#             f_l[k] = (
#                 normalized_active_force_y_intercept[1, k]
#                 + normalized_active_force_slope[1, k] * eps[k]
#             )
#
#         elif (
#             eps[k] > normalized_active_force_break_points[2, k]
#             and eps[k] <= normalized_active_force_break_points[3, k]
#         ):
#             f_l[k] = (
#                 normalized_active_force_y_intercept[2, k]
#                 + normalized_active_force_slope[2, k] * eps[k]
#             )
#         else:
#             f_l[k] = (
#                 normalized_active_force_y_intercept[3, k]
#                 + normalized_active_force_slope[3, k] * eps[k]
#             )
#
#         if f_l[k] < 0:
#             f_l[k] = 0
#
#         if eps[k] <= compression_strain_limit[k]:
#             sigma_pass[k] = E_compression[k] * (eps[k] - compression_strain_limit[k])
#         elif eps[k] <= extension_strain_limit[k]:
#             sigma_pass[k] = 0.0
#         else:
#             sigma_pass[k] = (passive_force_coefficients[0,k] * (eps[k] - extension_strain_limit[k])**3
#                             + passive_force_coefficients[1,k] * (eps[k] - extension_strain_limit[k])**2 +
#                 passive_force_coefficients[2,k] * (eps[k] - extension_strain_limit[k]) +
#                             passive_force_coefficients[3,k])*tension_passive_force_scale[k]
#         # elif eps[k] < eps_c + extension_strain_limit[k]:
#         #     sigma_pass[k] = (
#         #         c1 * (eps[k] - extension_strain_limit[k]) ** c2
#         #     ) * tension_passive_force_scale[k]
#         # else:
#         #     sigma_pass[k] = (
#         #         c3 * (eps[k] - extension_strain_limit[k]) + c4
#         #     ) * tension_passive_force_scale[k]
#
#         internal_stress[2, k] = rest_area[k] * (
#             fiber_activation[k] * maximum_active_stress[k] * f_l[k] * f_v[k]
#             + sigma_pass[k]
#         )
@numba.njit(cache=True)
def _compute_internal_shear_stretch_stresses_from_model(
    position_collection,
    volume,
    lengths,
    tangents,
    radius,
    rest_lengths,
    rest_voronoi_lengths,
    dilatation,
    dilatation_rate,
    voronoi_dilatation,
    director_collection,
    sigma,
    rest_sigma,
    shear_matrix,
    internal_stress,
    fiber_activation,
    rest_area,
    tension_passive_force_scale,
    maximum_active_stress,
    minimum_strain_rate,
    active_force_coefficients,
    force_velocity_constant,
    # normalized_active_force_y_intercept,
    # normalized_active_force_slope,
    # normalized_active_force_break_points,
    E_compression,
    compression_strain_limit,
    extension_strain_limit,
    passive_force_coefficients,
    velocity_collection,
):
    """
    Constitutive model for the muscle. The model provided in the below reference is implemented with the
    addition of if a small passive elastic response for strain < -0.3 to keep the model stable in this regime.
    Model parameters are taken as rough averages of the range given for squid tentacles in Table 1 of the below
    reference.

    References
    ----------
    Van Leeuwen, J. L., and William M. Kier. "Functional design of tentacles in squid: linking sarcomere
    ultrastructure to gross morphological dynamics." Philosophical Transactions of the Royal Society of
    London. Series B: Biological Sciences 352.1353 (1997): 551-571.

    Returns
    -------
    """

    _compute_shear_stretch_strains(
        position_collection,
        volume,
        lengths,
        tangents,
        radius,
        rest_lengths,
        rest_voronoi_lengths,
        dilatation,
        voronoi_dilatation,
        director_collection,
        sigma,
    )

    internal_stress[:] = _batch_matvec(shear_matrix, sigma - rest_sigma)

    # Compute dilatation rate when needed, dilatation itself is done before
    # in internal_stresses
    _compute_dilatation_rate(
        position_collection, velocity_collection, lengths, rest_lengths, dilatation_rate
    )

    # These are the adjustable model parameters.
    ################################
    # c3 = 1450e3
    # c4 = -625e3
    # eps_c = 0.773
    # K = 0.25
    ################################

    # c2 = c3 * eps_c / (c3 * eps_c + c4)
    # c1 = c3 / (c2 * eps_c ** (c2 - 1))

    # Lambda stretch ratio in the paper is dilatation in our simulation
    # stretch = dilatation
    eps = sigma[2, :] - rest_sigma[2, :]
    stretch = eps + 1

    # TODO: make this the true strain rate
    eps_dot = dilatation_rate / minimum_strain_rate

    sigma_pass = np.zeros((stretch.shape[0]))
    f_v = np.zeros((stretch.shape[0]))
    f_l = np.zeros((stretch.shape[0]))

    blocksize = stretch.shape[0]

    for k in range(blocksize):
        if eps_dot[k] < 0:
            f_v[k] = 1.8 - 0.8 * (1 + eps_dot[k]) / (
                1.0 - 7.56 * eps_dot[k] / force_velocity_constant[k]
            )
        else:
            f_v[k] = (1.0 - eps_dot[k]) / (
                1.0 + eps_dot[k] / force_velocity_constant[k]
            )

        f_l[k] = (
            active_force_coefficients[0, k] * stretch[k] ** 8
            + active_force_coefficients[1, k] * stretch[k] ** 7
            + active_force_coefficients[2, k] * stretch[k] ** 6
            + active_force_coefficients[3, k] * stretch[k] ** 5
            + active_force_coefficients[4, k] * stretch[k] ** 4
            + active_force_coefficients[5, k] * stretch[k] ** 3
            + active_force_coefficients[6, k] * stretch[k] ** 2
            + active_force_coefficients[7, k] * stretch[k]
            + active_force_coefficients[8, k]
        )

        if f_l[k] < 0:
            f_l[k] = 0

        # if eps[k] <= compression_strain_limit[k]:
        #     sigma_pass[k] = E_compression[k] * (eps[k] - compression_strain_limit[k])
        # elif eps[k] <= extension_strain_limit[k]:
        #     sigma_pass[k] = 0.0
        # else:
        #     sigma_pass[k] = (passive_force_coefficients[0,k] * (eps[k] - extension_strain_limit[k])**3
        #                     + passive_force_coefficients[1,k] * (eps[k] - extension_strain_limit[k])**2 +
        #         passive_force_coefficients[2,k] * (eps[k] - extension_strain_limit[k]) +
        #                     passive_force_coefficients[3,k])*tension_passive_force_scale[k]
        # elif eps[k] < eps_c + extension_strain_limit[k]:
        #     sigma_pass[k] = (
        #         c1 * (eps[k] - extension_strain_limit[k]) ** c2
        #     ) * tension_passive_force_scale[k]
        # else:
        #     sigma_pass[k] = (
        #         c3 * (eps[k] - extension_strain_limit[k]) + c4
        #     ) * tension_passive_force_scale[k]
        # sigma_pass[k] = (
        #     passive_force_coefficients[0, k]
        #     * ((eps[k] - extension_strain_limit[k])+1) ** 2
        #     + passive_force_coefficients[1, k]
        #     * ((eps[k] - extension_strain_limit[k])+1)
        #     + passive_force_coefficients[2, k]
        #     # * (eps[k] - extension_strain_limit[k])
        #     # + passive_force_coefficients[3, k]
        # ) * tension_passive_force_scale[k]
        if eps[k] <= compression_strain_limit[k]:
            sigma_pass[k] = E_compression[k] * (eps[k] - compression_strain_limit[k])
        elif eps[k] <= extension_strain_limit[k]:
            sigma_pass[k] = 0.0
        else:
            sigma_pass[k] = (
                passive_force_coefficients[0, k] * stretch[k] ** 8
                + passive_force_coefficients[1, k] * stretch[k] ** 7
                + passive_force_coefficients[2, k] * stretch[k] ** 6
                + passive_force_coefficients[3, k] * stretch[k] ** 5
                + passive_force_coefficients[4, k] * stretch[k] ** 4
                + passive_force_coefficients[5, k] * stretch[k] ** 3
                + passive_force_coefficients[6, k] * stretch[k] ** 2
                + passive_force_coefficients[7, k] * stretch[k]
                + passive_force_coefficients[8, k]
            )

            sigma_pass[k] *= maximum_active_stress[k] * tension_passive_force_scale[k]

        internal_stress[2, k] = rest_area[k] * (
            fiber_activation[k] * maximum_active_stress[k] * f_l[k] * f_v[k]
            + sigma_pass[k]
        )


@numba.njit(cache=True)
def _compute_bending_twist_strains(director_collection, rest_voronoi_lengths, kappa):
    temp = _inv_rotate(director_collection)
    blocksize = rest_voronoi_lengths.shape[0]
    for k in range(blocksize):
        kappa[0, k] = temp[0, k] / rest_voronoi_lengths[k]
        kappa[1, k] = temp[1, k] / rest_voronoi_lengths[k]
        kappa[2, k] = temp[2, k] / rest_voronoi_lengths[k]


@numba.njit(cache=True)
def _compute_internal_bending_twist_stresses_from_model(
    director_collection,
    rest_voronoi_lengths,
    internal_couple,
    bend_matrix,
    kappa,
    rest_kappa,
):
    """
    Linear force functional
    Operates on
    B : (3,3,n) tensor and curvature kappa (3,n)

    Returns
    -------

    """
    _compute_bending_twist_strains(
        director_collection, rest_voronoi_lengths, kappa
    )  # concept : needs to compute kappa

    blocksize = kappa.shape[1]
    temp = np.empty((3, blocksize))
    for i in range(3):
        for k in range(blocksize):
            temp[i, k] = kappa[i, k] - rest_kappa[i, k]

    internal_couple[:] = _batch_matvec(bend_matrix, temp)


@numba.njit(cache=True)
def _compute_damping_forces(
    damping_forces,
    velocity_collection,
    dissipation_constant_for_forces,
    lengths,
    ghost_elems_idx,
):
    # Internal damping foces.
    elemental_velocities = node_to_element_pos_or_vel(velocity_collection)

    blocksize = elemental_velocities.shape[1]
    elemental_damping_forces = np.zeros((3, blocksize))

    for i in range(3):
        for k in range(blocksize):
            elemental_damping_forces[i, k] = (
                dissipation_constant_for_forces[k]
                * elemental_velocities[i, k]
                * lengths[k]
            )

    damping_forces[:] = quadrature_kernel_for_block_structure(
        elemental_damping_forces, ghost_elems_idx
    )


@numba.njit(cache=True)
def _compute_internal_forces(
    position_collection,
    volume,
    lengths,
    tangents,
    radius,
    rest_lengths,
    rest_voronoi_lengths,
    dilatation,
    dilatation_rate,
    voronoi_dilatation,
    director_collection,
    sigma,
    rest_sigma,
    shear_matrix,
    internal_stress,
    velocity_collection,
    dissipation_constant_for_forces,
    damping_forces,
    internal_forces,
    fiber_activation,
    rest_area,
    tension_passive_force_scale,
    maximum_active_stress,
    minimum_strain_rate,
    active_force_coefficients,
    force_velocity_constant,
    # normalized_active_force_y_intercept,
    # normalized_active_force_slope,
    # normalized_active_force_break_points,
    E_compression,
    compression_strain_limit,
    extension_strain_limit,
    passive_force_coefficients,
    ghost_elems_idx,
):
    # Compute n_l and cache it using internal_stress
    # Be careful about usage though
    _compute_internal_shear_stretch_stresses_from_model(
        position_collection,
        volume,
        lengths,
        tangents,
        radius,
        rest_lengths,
        rest_voronoi_lengths,
        dilatation,
        dilatation_rate,
        voronoi_dilatation,
        director_collection,
        sigma,
        rest_sigma,
        shear_matrix,
        internal_stress,
        fiber_activation,
        rest_area,
        tension_passive_force_scale,
        maximum_active_stress,
        minimum_strain_rate,
        active_force_coefficients,
        force_velocity_constant,
        # normalized_active_force_y_intercept,
        # normalized_active_force_slope,
        # normalized_active_force_break_points,
        E_compression,
        compression_strain_limit,
        extension_strain_limit,
        passive_force_coefficients,
        velocity_collection,
    )

    # Signifies Q^T n_L / e
    # Not using batch matvec as I don't want to take directors.T here

    blocksize = internal_stress.shape[1]
    cosserat_internal_stress = np.zeros((3, blocksize))

    for i in range(3):
        for j in range(3):
            for k in range(blocksize):
                cosserat_internal_stress[i, k] += (
                    director_collection[j, i, k] * internal_stress[j, k]
                )

    cosserat_internal_stress /= dilatation

    _compute_damping_forces(
        damping_forces,
        velocity_collection,
        dissipation_constant_for_forces,
        lengths,
        ghost_elems_idx,
    )

    internal_forces[:] = (
        difference_kernel_for_block_structure(cosserat_internal_stress, ghost_elems_idx)
        - damping_forces
    )


@numba.njit(cache=True)
def _compute_damping_torques(
    damping_torques, omega_collection, dissipation_constant_for_torques, lengths
):
    blocksize = damping_torques.shape[1]
    for i in range(3):
        for k in range(blocksize):
            damping_torques[i, k] = (
                dissipation_constant_for_torques[k]
                * omega_collection[i, k]
                * lengths[k]
            )


@numba.njit(cache=True)
def _compute_internal_torques(
    position_collection,
    velocity_collection,
    tangents,
    lengths,
    rest_lengths,
    director_collection,
    rest_voronoi_lengths,
    bend_matrix,
    rest_kappa,
    kappa,
    voronoi_dilatation,
    mass_second_moment_of_inertia,
    omega_collection,
    internal_stress,
    internal_couple,
    dilatation,
    dilatation_rate,
    dissipation_constant_for_torques,
    damping_torques,
    internal_torques,
    ghost_voronoi_idx,
):
    # Compute \tau_l and cache it using internal_couple
    # Be careful about usage though
    _compute_internal_bending_twist_stresses_from_model(
        director_collection,
        rest_voronoi_lengths,
        internal_couple,
        bend_matrix,
        kappa,
        rest_kappa,
    )
    # # Compute dilatation rate when needed, dilatation itself is done before
    # # in internal_stresses
    # _compute_dilatation_rate(
    #     position_collection, velocity_collection, lengths, rest_lengths, dilatation_rate
    # )

    # FIXME: change memory overload instead for the below calls!
    voronoi_dilatation_inv_cube_cached = 1.0 / voronoi_dilatation**3
    # Delta(\tau_L / \Epsilon^3)
    bend_twist_couple_2D = difference_kernel_for_block_structure(
        internal_couple * voronoi_dilatation_inv_cube_cached, ghost_voronoi_idx
    )
    # \mathcal{A}[ (\kappa x \tau_L ) * \hat{D} / \Epsilon^3 ]
    bend_twist_couple_3D = quadrature_kernel_for_block_structure(
        _batch_cross(kappa, internal_couple)
        * rest_voronoi_lengths
        * voronoi_dilatation_inv_cube_cached,
        ghost_voronoi_idx,
    )
    # (Qt x n_L) * \hat{l}
    shear_stretch_couple = (
        _batch_cross(_batch_matvec(director_collection, tangents), internal_stress)
        * rest_lengths
    )

    # I apply common sub expression elimination here, as J w / e is used in both the lagrangian and dilatation
    # terms
    # TODO : the _batch_matvec kernel needs to depend on the representation of J, and should be coded as such
    J_omega_upon_e = (
        _batch_matvec(mass_second_moment_of_inertia, omega_collection) / dilatation
    )

    # (J \omega_L / e) x \omega_L
    # Warning : Do not do micro-optimization here : you can ignore dividing by dilatation as we later multiply by it
    # but this causes confusion and violates SRP
    lagrangian_transport = _batch_cross(J_omega_upon_e, omega_collection)

    # Note : in the computation of dilatation_rate, there is an optimization opportunity as dilatation rate has
    # a dilatation-like term in the numerator, which we cancel here
    # (J \omega_L / e^2) . (de/dt)
    unsteady_dilatation = J_omega_upon_e * dilatation_rate / dilatation

    _compute_damping_torques(
        damping_torques, omega_collection, dissipation_constant_for_torques, lengths
    )

    blocksize = internal_torques.shape[1]
    for i in range(3):
        for k in range(blocksize):
            internal_torques[i, k] = (
                bend_twist_couple_2D[i, k]
                + bend_twist_couple_3D[i, k]
                + shear_stretch_couple[i, k]
                + lagrangian_transport[i, k]
                + unsteady_dilatation[i, k]
                - damping_torques[i, k]
            )


@numba.njit(cache=True)
def _update_accelerations(
    acceleration_collection,
    internal_forces,
    external_forces,
    mass,
    alpha_collection,
    inv_mass_second_moment_of_inertia,
    internal_torques,
    external_torques,
    dilatation,
):
    blocksize_acc = internal_forces.shape[1]
    blocksize_alpha = internal_torques.shape[1]

    for i in range(3):
        for k in range(blocksize_acc):
            acceleration_collection[i, k] = (
                internal_forces[i, k] + external_forces[i, k]
            ) / mass[k]

    alpha_collection *= 0.0
    for i in range(3):
        for j in range(3):
            for k in range(blocksize_alpha):
                alpha_collection[i, k] += (
                    inv_mass_second_moment_of_inertia[i, j, k]
                    * (internal_torques[j, k] + external_torques[j, k])
                ) * dilatation[k]


@numba.njit(cache=True)
def _zeroed_out_external_forces_and_torques(external_forces, external_torques):
    """
    This function is to zeroed out external forces and torques.

    Parameters
    ----------
    external_forces
    external_torques

    Returns
    -------

    Note
    ----
    Microbenchmark results 100 elements
    python version: 3.32 µs ± 44.5 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    this version: 583 ns ± 1.94 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
    """
    n_nodes = external_forces.shape[1]
    n_elems = external_torques.shape[1]

    for i in range(3):
        for k in range(n_nodes):
            external_forces[i, k] = 0.0

    for i in range(3):
        for k in range(n_elems):
            external_torques[i, k] = 0.0
