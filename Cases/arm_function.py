__docs__ = ["Common functions used for octopus arm simulations."]
import numpy as np
from elastica import *
import numba
from numba import njit
from elastica._linalg import _batch_norm, _batch_matvec


def helical_rod_radius_function(
    r_helical_rod,
    r_center_straight_rod,
    r_outer_straight_rod,
    r_outer_ring_rod,
    area_left,
):
    length = (
        r_center_straight_rod
        + 2 * r_outer_straight_rod
        + 2 * r_outer_ring_rod
        + r_helical_rod
    )

    return (2 * np.pi * length * (2 * r_helical_rod) - area_left) ** 2


def outer_ring_rod_functions(
    r_outer_ring_rod,
    n_elem_ring_rod,
    r_center_straight_rod,
    r_outer_straight_rod,
    area_left,
):
    return (
        # Perimeter of polygon 2*n_corner*np.sin(pi/n_corner)*R
        2
        * n_elem_ring_rod
        * np.sin(np.pi / n_elem_ring_rod)
        * (r_center_straight_rod + 2 * r_outer_straight_rod + r_outer_ring_rod)
        * 2
        * r_outer_ring_rod
        - area_left
    ) ** 2


def no_offset_radius_for_straight_rods(variables, n_corner, area):
    center_straight_rod_radius, outer_straight_rod_radius = variables

    eq1 = (
        2
        * n_corner
        * (center_straight_rod_radius + outer_straight_rod_radius)
        * np.sin(np.pi / n_corner)
        - 2 * outer_straight_rod_radius * n_corner
    )

    eq2 = (
        area
        - np.pi * center_straight_rod_radius**2
        - n_corner * np.pi * outer_straight_rod_radius**2
    )

    return [eq1, eq2]


# Boundary conditions


class FixNodePosition(FreeRod):
    def __init__(self, fixed_position, fixed_directors):
        self.fixed_position = fixed_position
        self.fixed_directors = fixed_directors
        self.constrained_position_idx = 0
        self.constrained_director_idx = 0

    def constrain_values(self, rod, time):
        rod.position_collection[..., self.constrained_position_idx][
            1
        ] = self.fixed_position[1]
        rod.director_collection[
            ..., self.constrained_director_idx
        ] = self.fixed_directors

    def constrain_rates(self, rod, time):
        rod.velocity_collection[..., self.constrained_position_idx][1] = 0.0
        rod.omega_collection[..., self.constrained_director_idx] = 0.0


class ConstrainRingPositionDirectors(FreeRod):
    def __init__(self, fixed_position, fixed_directors):
        self.fixed_position = fixed_position
        self.fixed_directors = fixed_directors
        self.constrained_position_idx = 0
        self.constrained_director_idx = 0

    def constrain_values(self, rod, time):
        rod.position_collection[:] = self.fixed_position[:]
        rod.director_collection[:] = self.fixed_directors

    def constrain_rates(self, rod, time):
        rod.velocity_collection[:] = 0.0
        rod.omega_collection[:] = 0.0


class FixPositionDirectorForSomeTime(FreeRod):
    def __init__(self, fixed_position, fixed_directors, release_time):
        self.fixed_position = fixed_position
        self.fixed_directors = fixed_directors
        self.constrained_position_idx = 0
        self.constrained_director_idx = 0
        self.release_time = release_time

    def constrain_values(self, rod, time):
        if time < self.release_time:
            rod.position_collection[
                ..., self.constrained_position_idx
            ] = self.fixed_position
            rod.director_collection[
                ..., self.constrained_director_idx
            ] = self.fixed_directors

    def constrain_rates(self, rod, time):
        if time < self.release_time:
            rod.velocity_collection[..., self.constrained_position_idx] = 0
            rod.omega_collection[..., self.constrained_director_idx] = 0


class DampingFilterBC(FreeRod):
    """
    Damping filter.
    TODO expand docs if working


        Attributes
        ----------
        filter_order: int
            Order of the filter.


    """

    def __init__(self, fixed_position, fixed_directors, filter_order):
        """

        Damping Filter initializer

        Parameters
        ----------

        filter_order: int
            Order of the filter.
        """
        assert filter_order > 0 and isinstance(
            filter_order, int
        ), "invalid filter order"
        self.filter_order = filter_order

    def constrain_values(self, rod, time):
        pass

    def constrain_rates(self, rod, time):
        rod.velocity_collection[...] = rod.velocity_collection - nb_compute_filter_term(
            rate_collection=rod.velocity_collection, filter_order=self.filter_order
        )
        rod.omega_collection[...] = rod.omega_collection - nb_compute_filter_term(
            rate_collection=rod.omega_collection, filter_order=self.filter_order
        )


@njit(cache=True)
def nb_compute_filter_term(
    rate_collection: np.ndarray, filter_order: int
) -> np.ndarray:
    filter_term = rate_collection.copy()
    for i in range(filter_order):
        filter_term[..., 1:-1] = (
            -filter_term[..., 2:] - filter_term[..., :-2] + 2.0 * filter_term[..., 1:-1]
        ) / 4.0
        # dont touch boundary values
        filter_term[..., 0] = 0.0
        filter_term[..., -1] = 0.0
    return filter_term


class DampingFilterBCRingRod(FreeRod):
    """
    Damping filter.
    TODO expand docs if working


        Attributes
        ----------
        filter_order: int
            Order of the filter.


    """

    def __init__(self, fixed_position, fixed_directors, filter_order):
        """

        Damping Filter initializer

        Parameters
        ----------

        filter_order: int
            Order of the filter.
        """
        assert filter_order > 0 and isinstance(
            filter_order, int
        ), "invalid filter order"
        self.filter_order = filter_order

    def constrain_values(self, rod, time):
        pass

    def constrain_rates(self, rod, time):
        self._apply_dissipation(
            rod.velocity_collection, rod.omega_collection, self.filter_order
        )

    @staticmethod
    @numba.njit(cache=True)
    def _apply_dissipation(velocity_collection, omega_collection, filter_order):
        blocksize = velocity_collection.shape[1] + 2
        temp_velocity = np.empty((3, blocksize))
        temp_velocity[:, 1:-1] = velocity_collection[:]
        temp_velocity[:, 0] = velocity_collection[:, -1]
        temp_velocity[:, -1] = velocity_collection[:, 0]
        temp_omega = np.empty((3, blocksize))
        temp_omega[:, 1:-1] = omega_collection[:]
        temp_omega[:, 0] = omega_collection[:, -1]
        temp_omega[:, -1] = omega_collection[:, 0]

        velocity_collection[...] = (
            velocity_collection
            - nb_compute_filter_term(
                rate_collection=temp_velocity, filter_order=filter_order
            )[:, 1:-1]
        )
        omega_collection[...] = (
            omega_collection
            - nb_compute_filter_term(
                rate_collection=temp_omega, filter_order=filter_order
            )[:, 1:-1]
        )


class ExponentialDampingBC(FreeRod):
    """
    Damping filter.
    TODO expand docs if working
        Attributes
        ----------
        filter_order: int
            Order of the filter.
    """

    def __init__(self, fixed_position, fixed_directors, nu, time_step, rod):
        """
        Damping Filter initializer
        Parameters
        ----------
        filter_order: int
            Order of the filter.
        """
        self.nu_dt = nu * time_step
        nodal_mass = rod.mass.copy()
        self.translational_damping_coefficient = np.exp(-nu * time_step)

        element_mass = 0.5 * (nodal_mass[1:] + nodal_mass[:-1])
        element_mass[0] += 0.5 * nodal_mass[0]
        element_mass[-1] += 0.5 * nodal_mass[-1]

        if rod.mass.shape[0] == rod.rest_lengths.shape[0]:
            element_mass = nodal_mass

        self.rotational_damping_coefficient = np.exp(
            -nu
            * time_step
            * element_mass
            * np.diagonal(rod.inv_mass_second_moment_of_inertia).T
        )

    def constrain_rates(self, rod, time):
        # rod.velocity_collection[:] = rod.velocity_collection * np.exp(-self.nu_dt*rod.lengths/rod.mass)
        #
        # element_mass = rod.mass
        #
        # if rod.ring_rod_flag == False:
        #     element_mass = 0.5 * (rod.mass[1:]+rod.mass[:-1])
        #     element_mass[0] += 0.5*rod.mass[0]
        #     element_mass[-1] += 0.5*rod.mass[-1]
        #
        # rod.omega_collection[:] = rod.omega_collection * np.exp(-self.nu_dt * rod.lengths * rod.dilatation * np.diagonal(rod.inv_mass_second_moment_of_inertia).T / element_mass)

        rod.velocity_collection[:] = (
            rod.velocity_collection * self.translational_damping_coefficient
        )

        rod.omega_collection[:] = rod.omega_collection * np.power(
            self.rotational_damping_coefficient, rod.dilatation
        )


# Control muscle forces
class MuscleFiberForceActivation(NoForces):
    def __init__(self, ramp_up_time, activation_level=1.0):
        self.ramp_up_time = ramp_up_time
        self.activation_level = activation_level

    def apply_forces(self, system, time: np.float64 = 0.0):
        # TODO: Dont do like this separate out integration routine and change activation
        factor = min(self.activation_level, time / self.ramp_up_time)
        system.fiber_activation[:] = factor  # np.ones((system.n_elems)) * factor


class MuscleFiberForceWaveActivation(NoForces):
    def __init__(self, activation_level=1.0, beta=1, time_constant=0.01):
        self.activation_level = activation_level
        self.beta = beta
        self.tau = time_constant

    def apply_forces(self, system, time: np.float64 = 0.0):
        index = np.arange(0, system.n_elems, 1, dtype=np.int)[::-1]
        system.fiber_activation[:] = (
            self.activation_level
            * 0.5
            * (1.0 + np.tanh(self.beta * (time / self.tau - index + 0)))
        )


class MuscleFiberForceActivationStepFunction(NoForces):
    def __init__(
        self,
        activation_time,
        activation_exponent=15,
        delay=0,
        activation_factor=1.0,
    ):
        self.activation_time = activation_time  # ta
        self.activation_exponent = activation_exponent  # q
        self.delay = delay  # td
        self.activation_factor = activation_factor

    def apply_forces(self, system, time: np.float64 = 0.0):
        time = round(time, 5)

        self._apply_activation(
            time,
            self.delay,
            self.activation_time,
            self.activation_exponent,
            self.activation_factor,
            system.fiber_activation,
        )

    @staticmethod
    @njit(cache=True)
    def _apply_activation(
        time,
        delay,
        activation_time,
        activation_exponent,
        activation_factor,
        fiber_activation,
    ):
        if time <= delay:
            fiber_activation[:] = 0.0
        elif time > delay and time < delay + activation_time:
            fiber_activation[:] = activation_factor * (
                (
                    0.5
                    * (
                        1
                        + np.sin(np.pi * (time - delay) / activation_time - 0.5 * np.pi)
                    )
                )
                ** activation_exponent
            )
        elif time >= delay + activation_time:
            fiber_activation[:] = 1.0 * activation_factor


class ActivationRampUpRampDown(NoForces):
    def __init__(
        self, ramp_up_time, ramp_down_time, ramp_interval, activation_level=1.0
    ):
        self.ramp_up_time = ramp_up_time
        self.ramp_down_time = ramp_down_time
        self.ramp = ramp_interval
        self.activation_level = activation_level

    def apply_forces(self, system, time: np.float64 = 0.0):
        factor = 0.0
        time = round(time, 5)

        if time > self.ramp_up_time:
            factor = min(1.0, (time - self.ramp_up_time) / self.ramp)

        if time > self.ramp_down_time:
            factor = max(0.0, -1 / self.ramp * (time - self.ramp_down_time) + 1.0)

        activation = self.activation_level * factor
        if activation > 0.0:
            system.fiber_activation[:] = activation


class LocalActivation(NoForces):
    def __init__(
        self,
        ramp_interval,
        ramp_up_time,
        ramp_down_time,
        start_idx,
        end_idx,
        activation_level=1.0,
    ):
        self.ramp = ramp_interval
        self.ramp_up_time = ramp_up_time
        self.ramp_down_time = ramp_down_time
        self.activation_level = activation_level
        self.start_idx = int(start_idx)
        self.end_idx = int(end_idx)

    def apply_forces(self, system, time: np.float64 = 0.0):
        time = round(time, 5)
        factor = 0.0

        #        if time > self.ramp_up_time:
        #            factor = min(1.0, (time - self.ramp_up_time) / self.ramp)
        #
        #        if time > self.ramp_down_time:
        #            factor = max(0.0, -1 / self.ramp * (time - self.ramp_down_time) + 1.0)
        if (time - self.ramp_up_time) <= 0:
            factor = 0.0
        elif (time - self.ramp_up_time) > 0 and (time - self.ramp_up_time) <= self.ramp:
            factor = (
                1 + np.sin(np.pi * (time - self.ramp_up_time) / self.ramp - np.pi / 2)
            ) / 2
        elif (time - self.ramp_up_time) > 0 and (time - self.ramp_down_time) < 0:
            factor = 1.0

        elif (time - self.ramp_down_time) > 0 and (
            time - self.ramp_down_time
        ) / self.ramp < 1.0:
            factor = (
                1
                - (
                    1
                    + np.sin(
                        np.pi * (time - self.ramp_down_time) / self.ramp - np.pi / 2
                    )
                )
                / 2
            )

        activation = self.activation_level * factor
        if activation > 0.0:
            system.fiber_activation[self.start_idx : self.end_idx] = activation


class SigmoidActivationLongitudinalMuscles(NoForces):
    def __init__(
        self,
        beta,
        tau,
        start_time,
        end_time,
        start_idx,
        end_idx,
        activation_level_max=1.0,
        activation_level_end=0.0,
        activation_lower_threshold=2e-3,
    ):
        self.beta = beta
        self.tau = tau
        self.start_time = start_time
        self.end_time = end_time
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.activation_level_max = activation_level_max
        self.activation_level_end = activation_level_end
        self.activation_lower_threshold = activation_lower_threshold

    def apply_forces(self, system, time: np.float64 = 0.0):
        n_elems = self.end_idx - self.start_idx
        index = np.arange(0, n_elems, dtype=np.int64)
        fiber_activation = np.zeros((n_elems))

        time = round(time, 5)
        if time > self.start_time - 4 * self.tau / self.beta:
            fiber_activation = (
                self.activation_level_max
                * 0.5
                * (
                    1
                    + np.tanh(
                        self.beta * ((time - self.start_time) / self.tau - index + 0)
                    )
                )
            ) + (
                -(self.activation_level_max - self.activation_level_end)
                * (
                    0.5
                    * (
                        1
                        + np.tanh(
                            self.beta * ((time - self.end_time) / self.tau - index + 0)
                        )
                    )
                )
            )
        active_index = np.where(fiber_activation > self.activation_lower_threshold)[0]
        system.fiber_activation[self.start_idx + active_index] = fiber_activation[
            active_index
        ]


class SigmoidActivationTransverseMuscles(NoForces):
    def __init__(
        self,
        beta,
        tau,
        start_time,
        end_time,
        rod_idx,
        activation_level_max=1.0,
        activation_level_end=1.0,
        activation_lower_threshold=2e-3,
    ):
        self.beta = beta
        self.tau = tau
        self.start_time = start_time
        self.end_time = end_time
        self.rod_idx = rod_idx
        self.activation_level_max = activation_level_max
        self.activation_level_end = activation_level_end
        self.activation_lower_threshold = activation_lower_threshold

    def apply_forces(self, system, time: np.float64 = 0.0):
        time = round(time, 5)
        activation = 0.0
        if time > self.start_time:
            activation = self.activation_level_max * 0.5 * (
                1
                + np.tanh(
                    self.beta * ((time - self.start_time) / self.tau - self.rod_idx + 0)
                )
            ) + -(self.activation_level_max - self.activation_level_end) * 0.5 * (
                1
                + np.tanh(
                    self.beta * ((time - self.end_time) / self.tau - self.rod_idx + 0)
                )
            )

        if activation > self.activation_lower_threshold:
            system.fiber_activation[:] = activation


class LinearlyDecreasingActivationRampUpRampDown(NoForces):
    def __init__(
        self, ramp_up_time, ramp_down_time, ramp_interval, activation_level=1.0
    ):
        self.ramp_up_time = ramp_up_time
        self.ramp_down_time = ramp_down_time
        self.ramp = ramp_interval
        self.activation_level = activation_level

    def apply_forces(self, system, time: np.float64 = 0.0):
        factor = 0.0
        time = round(time, 5)

        if time > self.ramp_up_time:
            factor = min(1.0, (time - self.ramp_up_time) / self.ramp)

        if time > self.ramp_down_time:
            factor = max(0.0, -1 / self.ramp * (time - self.ramp_down_time) + 1.0)

        activation = factor * self.activation_level

        if activation > 0.0:
            system.fiber_activation[:] = activation * np.linspace(1, 0, system.n_elems)


class TravellingGaussianWaveActivationLongitudinalMuscles(NoForces):
    def __init__(
        self,
        sigma,
        tau,
        start_time,
        start_idx,
        end_idx,
        activation_level_max=1.0,
        activation_lower_threshold=2e-3,
    ):
        self.sigma = sigma
        self.tau = tau
        self.start_time = start_time
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.activation_level_max = activation_level_max
        self.activation_lower_threshold = activation_lower_threshold

    def apply_forces(self, system, time: np.float64 = 0.0):
        n_elems = self.end_idx - self.start_idx
        index = np.arange(0, n_elems, dtype=np.int64)

        time = round(time, 5)
        fiber_activation = self.activation_level_max * np.exp(
            -0.5 * ((index - (time - self.start_time) / self.tau) / self.sigma) ** 2
        )
        active_index = np.where(fiber_activation > self.activation_lower_threshold)[
            0
        ]  # Activations smaller than 2E-3 can be neglected
        system.fiber_activation[self.start_idx + active_index] = fiber_activation[
            active_index
        ]


class RampUpGaussianActivationFunction(NoForces):
    def __init__(
        self,
        start_time,
        end_time,
        ramp_interval,
        activation_level_max,
        center_idx,
        sigma,
        activation_lower_threshold=2e-3,
    ):
        self.start_time = start_time
        self.end_time = end_time
        self.ramp_interval = ramp_interval
        self.center_idx = center_idx
        self.sigma = sigma
        self.activation_lower_threshold = activation_lower_threshold
        self.activation_level_max = activation_level_max

    def apply_forces(self, system, time: np.float64 = 0.0):
        index = np.arange(0, system.n_elems, dtype=np.int64)
        factor = 0.0
        time = round(time, 5)

        if time > self.start_time:
            factor = min(1.0, (time - self.start_time) / self.ramp_interval)

        if time > self.end_time:
            factor = max(0.0, -1 / self.ramp_interval * (time - self.end_time) + 1.0)

        fiber_activation = (
            self.activation_level_max
            * np.exp(-0.5 * ((index - self.center_idx) / self.sigma) ** 2)
            * factor
        )

        active_index = np.where(fiber_activation > self.activation_lower_threshold)[
            0
        ]  # Activations smaller than 2E-3 can be neglected

        system.fiber_activation[active_index] = fiber_activation[active_index]


class SensoryFeedbackLongitudinalMuscles(NoForces):
    def __init__(
        self,
        # beta,
        # tau,
        # start_time,
        # end_time,
        # start_idx,
        # end_idx,
        lm_direction_flag,
        target,
        ramp_up_time,
        center_rod,
        activation_level_max=1.0,
        activation_level_end=0.0,
        activation_lower_threshold=2e-3,
    ):
        self.lm_direction_flag = lm_direction_flag
        self.target = target
        self.ramp_up_time = ramp_up_time
        self.center_rod = center_rod
        # self.beta = beta
        # self.tau = tau
        # self.start_time = start_time
        # self.end_time = end_time
        # self.start_idx = start_idx
        # self.end_idx = end_idx
        self.activation_level_max = activation_level_max
        self.activation_level_end = activation_level_end
        self.activation_lower_threshold = activation_lower_threshold

    def sensory_feedback(self, s):
        element_positions = 0.5 * (
            self.center_rod.position_collection[:, 1:]
            + self.center_rod.position_collection[:, :-1]
        )
        target_vector = self.target[:, None] - element_positions
        self.dist = _batch_norm(target_vector)
        normalized_target_vector = target_vector / self.dist
        self.min_idx = np.argmin(self.dist)  # -20 # -1 #
        self.s0 = s[self.min_idx]
        normal_vector = self.center_rod.director_collection[1, :, :]
        self.sin_alpha = np.einsum("in,in->n", normalized_target_vector, normal_vector)

    def apply_forces(self, system, time: np.float = 0.0):
        time = round(time, 5)

        s = np.hstack((0, np.cumsum(system.lengths)))
        self.sensory_feedback(s)

        factor = min(1.0, time / self.ramp_up_time)

        error_feedback = self.lm_direction_flag * self.sin_alpha
        activation_index = np.where(error_feedback > 0)[0]

        system.fiber_activation[activation_index] = factor * self.activation_level_max
        system.fiber_activation *= error_feedback
        system.fiber_activation[self.min_idx + 1 :] *= 0
        np.clip(system.fiber_activation, 0, self.activation_level_max)

        pass


# Drag forces
from elastica._linalg import _batch_dot
from elastica._elastica_numba._interaction import elements_to_nodes_inplace


class DragForceOnStraightRods(NoForces):
    def __init__(self, cd_perpendicular, cd_tangent, rho_water, start_time=0.0):
        self.cd_perpendicular = cd_perpendicular
        self.cd_tangent = cd_tangent
        self.rho_water = rho_water
        self.start_time = start_time

    def apply_forces(self, system, time: np.float64 = 0.0):
        if time > self.start_time:
            self._apply_forces(
                self.cd_perpendicular,
                self.cd_tangent,
                self.rho_water,
                system.radius,
                system.lengths,
                system.tangents,
                system.velocity_collection,
                system.external_forces,
            )

    @staticmethod
    @njit(cache=True)
    def _apply_forces(
        cd_perpendicular,
        cd_tangent,
        rho_water,
        radius,
        lengths,
        tangents,
        velocity_collection,
        external_forces,
    ):
        projected_area = 2 * radius * lengths
        surface_area = np.pi * projected_area

        element_velocity = 0.5 * (
            velocity_collection[:, 1:] + velocity_collection[:, :-1]
        )

        tangent_velocity = _batch_dot(element_velocity, tangents) * tangents
        perpendicular_velocity = element_velocity - tangent_velocity

        tangent_velocity_mag = _batch_norm(tangent_velocity)
        perpendicular_velocity_mag = _batch_norm(perpendicular_velocity)

        forces_in_tangent_dir = (
            0.5
            * rho_water
            * surface_area
            * cd_tangent
            * tangent_velocity_mag
            * tangent_velocity
        )
        forces_in_perpendicular_dir = (
            0.5
            * rho_water
            * projected_area
            * cd_perpendicular
            * perpendicular_velocity_mag
            * perpendicular_velocity
        )

        elements_to_nodes_inplace(-forces_in_tangent_dir, external_forces)
        elements_to_nodes_inplace(-forces_in_perpendicular_dir, external_forces)


class CenterRodCurvatureCallBack(CallBackBaseClass):
    def __init__(self, step_skip: int, callback_params: dict):
        self.every = step_skip
        self.callback_params = callback_params
        self.previous_time = 0
        self.previous_node_position = np.zeros(
            3,
        )

    def make_callback(self, system, time, current_step: int):
        max_curvature_idx = np.argmax(np.abs(system.kappa[0, :]))
        node_position = system.position_collection[:, max_curvature_idx]
        time_step = time - self.previous_time
        node_velocity = (node_position - self.previous_node_position) / time_step
        self.previous_time = time
        self.previous_node_position[:] = node_position

        self.callback_params["time"].append(time)
        self.callback_params["node_position"].append(node_position.copy())
        self.callback_params["node_velocity"].append(node_velocity.copy())


# Call back functions
class StraightRodCallBack(CallBackBaseClass):
    """
    Call back function for two arm octopus
    """

    def __init__(self, step_skip: int, callback_params: dict):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            self.callback_params["time"].append(time)
            self.callback_params["step"].append(current_step)
            self.callback_params["position"].append(system.position_collection.copy())
            self.callback_params["velocity"].append(system.velocity_collection.copy())
            self.callback_params["radius"].append(system.radius.copy())
            self.callback_params["com"].append(system.compute_position_center_of_mass())
            self.callback_params["external_forces"].append(
                system.external_forces.copy()
            )
            self.callback_params["internal_forces"].append(
                system.internal_forces.copy()
            )
            self.callback_params["tangents"].append(system.tangents.copy())
            self.callback_params["internal_stress"].append(
                system.internal_stress.copy()
            )
            self.callback_params["dilatation"].append(system.dilatation.copy())
            if current_step == 0:
                self.callback_params["lengths"].append(system.rest_lengths.copy())
            else:
                self.callback_params["lengths"].append(system.lengths.copy())

            if "fiber_activation" in system.__dict__:
                self.callback_params["activation"].append(
                    system.fiber_activation.copy()
                )
            else:
                self.callback_params["activation"].append(
                    np.zeros((system.n_elems)).copy()
                )

            self.callback_params["kappa"].append(system.kappa.copy())

            self.callback_params["directors"].append(system.director_collection.copy())

            self.callback_params["sigma"].append(system.sigma.copy())

            self.callback_params["translational_energy"].append(
                system.compute_translational_energy()
            )
            self.callback_params["rotational_energy"].append(
                system.compute_rotational_energy()
            )
            self.callback_params["total_bending_energy"].append(
                system.compute_bending_energy()
            )
            self.callback_params["shear_energy"].append(system.compute_shear_energy())

            # Compute the activation energy
            length_change = system.lengths - system.rest_lengths
            # Compute muscle forces. internal stress actually internal forces
            # divide by dilitation because we used rest_area calculating muscle internal force
            # see implementation.
            # Also this internal forces is in fiber direction and material frame.
            internal_forces = system.internal_stress[2, :] / system.dilatation
            # W = F * DX
            muscle_contraction_energy = 0.5 * (internal_forces * length_change).sum()
            self.callback_params["muscle_contraction_energy"].append(
                muscle_contraction_energy.copy()
            )

            # Compute bend energy
            kappa_diff = system.kappa - system.rest_kappa
            bending_internal_torques = _batch_matvec(system.bend_matrix, kappa_diff)
            rest_voronoi_lengths = system.rest_voronoi_lengths
            bending_energy = (
                0.5
                * np.dot(
                    kappa_diff[0, :],
                    bending_internal_torques[0, :] * rest_voronoi_lengths,
                ).sum()
            )
            bending_energy += (
                0.5
                * np.dot(
                    kappa_diff[1, :],
                    bending_internal_torques[1, :] * rest_voronoi_lengths,
                ).sum()
            )
            twist_energy = (
                0.5
                * np.dot(
                    kappa_diff[2, :],
                    bending_internal_torques[2, :] * rest_voronoi_lengths,
                ).sum()
            )

            self.callback_params["bending_energy"].append(bending_energy.copy())
            self.callback_params["twist_energy"].append(twist_energy.copy())

            return


class RingRodCallBack(CallBackBaseClass):
    def __init__(self, step_skip: int, callback_params: dict):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            self.callback_params["time"].append(time)
            self.callback_params["step"].append(current_step)
            # position_data = np.hstack(
            #     (
            #         system.position_collection,
            #         system.position_collection[..., 0].reshape(3, 1),
            #     )
            # )
            # self.callback_params["position"].append(position_data.copy())
            self.callback_params["position"].append(system.position_collection.copy())
            self.callback_params["com"].append(system.compute_position_center_of_mass())
            self.callback_params["radius"].append(system.radius.copy())
            self.callback_params["external_forces"].append(
                system.external_forces.copy()
            )
            self.callback_params["internal_forces"].append(
                system.internal_forces.copy()
            )
            self.callback_params["internal_stress"].append(
                system.internal_stress.copy()
            )
            self.callback_params["element_position"].append(
                np.cumsum(system.lengths.copy())
            )
            self.callback_params["strain"].append(system.sigma.copy())
            self.callback_params["tangents"].append(system.tangents.copy())
            self.callback_params["dilatation"].append(system.dilatation.copy())
            self.callback_params["kappa"].append(system.kappa.copy())
            if current_step == 0:
                self.callback_params["lengths"].append(system.rest_lengths.copy())
            else:
                self.callback_params["lengths"].append(system.lengths.copy())

            self.callback_params["sigma"].append(system.sigma.copy())
            self.callback_params["translational_energy"].append(
                system.compute_translational_energy()
            )
            self.callback_params["rotational_energy"].append(
                system.compute_rotational_energy()
            )
            self.callback_params["total_bending_energy"].append(
                system.compute_bending_energy()
            )
            self.callback_params["shear_energy"].append(system.compute_shear_energy())

            # Compute the activation energy
            length_change = system.lengths - system.rest_lengths
            # Compute muscle forces. internal stress actually internal forces
            # divide by dilitation because we used rest_area calculating muscle internal force
            # see implementation.
            # Also this internal forces is in fiber direction and material frame.
            internal_forces = system.internal_stress[2, :] / system.dilatation
            # W = F * DX
            muscle_contraction_energy = 0.5 * (internal_forces * length_change).sum()
            self.callback_params["muscle_contraction_energy"].append(
                muscle_contraction_energy.copy()
            )

            # Compute bend energy
            kappa_diff = system.kappa - system.rest_kappa
            bending_internal_torques = _batch_matvec(system.bend_matrix, kappa_diff)
            rest_voronoi_lengths = system.rest_voronoi_lengths
            bending_energy = (
                0.5
                * np.dot(
                    kappa_diff[0, :],
                    bending_internal_torques[0, :] * rest_voronoi_lengths,
                ).sum()
            )
            bending_energy += (
                0.5
                * np.dot(
                    kappa_diff[1, :],
                    bending_internal_torques[1, :] * rest_voronoi_lengths,
                ).sum()
            )
            twist_energy = (
                0.5
                * np.dot(
                    kappa_diff[2, :],
                    bending_internal_torques[2, :] * rest_voronoi_lengths,
                ).sum()
            )

            self.callback_params["bending_energy"].append(bending_energy.copy())
            self.callback_params["twist_energy"].append(twist_energy.copy())


class RigidCylinderCallBack(CallBackBaseClass):
    """
    Call back function for two arm octopus
    """

    def __init__(
        self, step_skip: int, callback_params: dict, resize_cylinder_elems: int
    ):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params
        self.n_elem_cylinder = resize_cylinder_elems
        self.n_node_cylinder = self.n_elem_cylinder + 1

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            self.callback_params["time"].append(time)
            self.callback_params["step"].append(current_step)

            cylinder_center_position = system.position_collection
            cylinder_length = system.length
            cylinder_direction = system.director_collection[2, :, :].reshape(3, 1)
            cylinder_radius = system.radius

            # Expand cylinder data. Create multiple points on cylinder later to use for rendering.

            start_position = (
                cylinder_center_position - cylinder_length / 2 * cylinder_direction
            )

            cylinder_position_collection = (
                start_position
                + np.linspace(0, cylinder_length[0], self.n_node_cylinder)
                * cylinder_direction
            )
            cylinder_radius_collection = (
                np.ones((self.n_elem_cylinder)) * cylinder_radius
            )
            cylinder_length_collection = (
                np.ones((self.n_elem_cylinder)) * cylinder_length
            )
            cylinder_velocity_collection = (
                np.ones((self.n_node_cylinder)) * system.velocity_collection
            )

            self.callback_params["position"].append(cylinder_position_collection.copy())
            self.callback_params["velocity"].append(cylinder_velocity_collection.copy())
            self.callback_params["radius"].append(cylinder_radius_collection.copy())
            self.callback_params["com"].append(system.compute_position_center_of_mass())

            self.callback_params["lengths"].append(cylinder_length_collection.copy())
            self.callback_params["com_velocity"].append(
                system.velocity_collection[..., 0].copy()
            )

            total_energy = (
                system.compute_translational_energy()
                + system.compute_rotational_energy()
            )
            self.callback_params["total_energy"].append(total_energy[..., 0].copy())

            return


class TargetRodCallBack(CallBackBaseClass):
    """
    Call back function for two arm octopus
    """

    def __init__(self, step_skip: int, callback_params: dict):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            self.callback_params["time"].append(time)
            self.callback_params["step"].append(current_step)
            self.callback_params["position"].append(system.position_collection.copy())
            self.callback_params["velocity"].append(system.velocity_collection.copy())
            self.callback_params["radius"].append(system.radius.copy())
            self.callback_params["com"].append(system.compute_position_center_of_mass())

            return
