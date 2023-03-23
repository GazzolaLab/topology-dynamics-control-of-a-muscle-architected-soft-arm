__all__ = ["Environment"]
from elastica._calculus import _isnan_check
from elastica.timestepper import extend_stepper_interface
from elastica import *
from elastica._elastica_numba._rod._muscular_rod import MuscularRod
from elastica._linalg import _batch_norm
from scipy.optimize import minimize_scalar, fsolve
from Cases.post_processing import (
    plot_video_with_surface,
    plot_video_activation_muscle,
)
from elastica._rotations import _get_rotation_matrix
import os
from itertools import groupby
from Cases.arm_function import (
    no_offset_radius_for_straight_rods,
    outer_ring_rod_functions,
    helical_rod_radius_function,
)
from Cases.arm_function import ConstrainRingPositionDirectors
from Cases.arm_function import (
    SigmoidActivationLongitudinalMuscles,
    ActivationRampUpRampDown,
    SigmoidActivationTransverseMuscles,
    LocalActivation,
)
from Cases.arm_function import DragForceOnStraightRods
from Cases.arm_function import (
    StraightRodCallBack,
    RingRodCallBack,
    RigidCylinderCallBack,
)
from Cases.arm_function import (
    DampingFilterBC,
    ExponentialDampingBC,
    DampingFilterBCRingRod,
)
from Connections import *


def all_equal(iterable):
    """
    Checks if all elements of list are equal.

    Parameters
    ----------
    iterable : list
        Iterable list

    Returns
    -------
        Boolean
    References
    ----------
        https://stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list-are-identical
    """
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


# Set base simulator class
class BaseSimulator(
    BaseSystemCollection, Constraints, MemoryBlockConnections, Forcing, CallBacks
):
    pass


class Environment:
    def __init__(
        self,
        final_time,
        k_straight_straight_connection_spring_scale,
        k_straight_straight_connection_contact_scale,
        k_ring_ring_spring_connection_scale,
        k_ring_straight_spring_connection_scale,
        k_ring_straight_spring_torque_connection_scale,
        k_ring_straight_contact_connection_scale,
        k_ring_helical_spring_connection_scale,
        k_ring_helical_contact_connection_scale,
        COLLECT_DATA_FOR_POSTPROCESSING=False,
    ):
        # Integrator type
        self.StatefulStepper = PositionVerlet()

        # Simulation parameters
        self.final_time = final_time
        time_step = 1e-5  # 1.25e-6 * 2 * 2 #* 2  # * 2 * 2 #* 2# (
        #     2.0e-5 / 2 / 2 / 4
        # )  # / 20 /4 / 2 * 8  * 2 * 5 #/200 # this is a stable timestep
        self.learning_step = 1
        self.total_steps = int(self.final_time / time_step / self.learning_step) + 1
        # self.time_step = np.float64(
        #     float(self.final_time) / (self.total_steps * self.learning_step)
        # )
        self.time_step = time_step
        # self.time_step = 1.000002000004e-05
        # Video speed
        self.rendering_fps = 30  # 20 * 1e1
        self.step_skip = int(1.0 / (self.rendering_fps * self.time_step))

        # Set the connection scales
        self.k_straight_straight_connection_spring_scale = (
            k_straight_straight_connection_spring_scale
        )
        self.k_straight_straight_connection_contact_scale = (
            k_straight_straight_connection_contact_scale
        )
        self.k_ring_ring_spring_connection_scale = k_ring_ring_spring_connection_scale
        self.k_ring_straight_spring_connection_scale = (
            k_ring_straight_spring_connection_scale
        )
        self.k_ring_straight_spring_torque_connection_scale = (
            k_ring_straight_spring_torque_connection_scale
        )
        self.k_ring_straight_contact_connection_scale = (
            k_ring_straight_contact_connection_scale
        )
        self.k_ring_helical_spring_connection_scale = (
            k_ring_helical_spring_connection_scale
        )
        self.k_ring_helical_contact_connection_scale = (
            k_ring_helical_contact_connection_scale
        )

        # Collect data is a boolean. If it is true callback function collects
        # rod parameters defined by user in a list.
        self.COLLECT_DATA_FOR_POSTPROCESSING = COLLECT_DATA_FOR_POSTPROCESSING

    def save_state(self, directory: str = "", time=0.0, verbose: bool = False):
        """
        Save state parameters of each rod.
        TODO : environment list variable is not uniform at the current stage of development.
        It would be nice if we have set list (like env.system) that iterates all the rods.
        Parameters
        ----------
        directory : str
            Directory path name. The path must exist.
        """
        os.makedirs(directory, exist_ok=True)
        for idx, rod in enumerate(self.rod_list):
            path = os.path.join(directory, "rod_{}.npz".format(idx))
            np.savez(path, time=time, **rod.__dict__)

        if verbose:
            print("Save complete: {}".format(directory))

    def load_state(self, directory: str = "", verbose: bool = False):
        """
        Load the rod-state.
        Compatibale with 'save_state' method.
        If the save-file does not exist, it returns error.
        Parameters
        ----------
        directory : str
            Directory path name.
        """
        time_list = []  # Simulation time of rods when they are saved.
        for idx, rod in enumerate(self.rod_list):
            path = os.path.join(directory, "rod_{}.npz".format(idx))
            data = np.load(path, allow_pickle=True)
            for key, value in data.items():
                if key == "time":
                    time_list.append(value.item())
                    continue

                if value.shape != ():
                    # Copy data into placeholders
                    getattr(rod, key)[:] = value
                else:
                    # For single-value data
                    setattr(rod, key, value)

        if all_equal(time_list) == False:
            raise ValueError(
                "Restart time of loaded rods are different, check your inputs!"
            )

        # Apply boundary conditions. Ring rods have periodic BC, so we need to update periodic elements in memory block
        self.simulator.constrain_values(0.0)
        self.simulator.constrain_rates(0.0)

        if verbose:
            print("Load complete: {}".format(directory))

        return time_list[0]

    def reset(self):
        """"""
        self.simulator = BaseSimulator()

        # setting up test params
        n_elem = 180  # * 2 * 2  # * 2  # 33
        n_elem_ring_rod = 40
        number_of_straight_rods = 9  # 13#25#7#13  # 25#25#13#25  # 49#9#13#25#9
        self.n_connections = number_of_straight_rods - 1

        start = np.zeros((3,))
        direction = np.array([0.0, 1.0, 0.0])  # rod direction
        normal = np.array([0.0, 0.0, 1.0])
        binormal = np.cross(direction, normal)
        arm_length = 200  # / 2 / 2  # 53.143*4  # mm
        mass_club = 1.8  # g
        mass_stalk = mass_club / 0.75
        # mass_total =  mass_stalk + mass_club #mass_club * (1 + 1 / 0.75)
        outer_base_radius = 12  # 12#16  # 12  # 2*3.7  # mm
        density = 1.050e-3  # *5*2*2  # # g/mm3
        nu = 250  # * 5 * 2 * 10  # * 5  # * 2 # dissipation coefficient
        E = 1e4  # Young's Modulus Pa  # Tramacere et. al. 2013 Interface, octopus dorsal tissue elastic modulus
        E *= 1  # convert Pa (kg m.s-2 / m2) to g.mm.s-2 / mm2 This is required for units to be consistent
        poisson_ratio = 0.5

        taper_ratio = 12  # Kier & Stella 2007
        taper_ratio_axial_cord = taper_ratio  # 8.42

        outer_tip_radius = outer_base_radius / taper_ratio

        # Tapered arm is like a cone. Subtracting bigger cone from smaller cone will give us the arm volume.
        volume_arm = 1 / 3 * (
            ((taper_ratio / (taper_ratio - 1)) * arm_length)
            * np.pi
            * outer_base_radius**2
        ) - 1 / 3 * (
            (1 / (taper_ratio - 1) * arm_length) * np.pi * outer_tip_radius**2
        )
        mass_total = density * volume_arm
        # taper_ratio = 1
        # taper_ratio_axial_cord = taper_ratio  # 8.42

        # Kier & Stella 2007 also Hannsy et. al. 2015
        eta_ring_rods = (
            0.17  # *5/2/2.5#0.10*4.5  # percentage area of the transverse muscles
        )
        eta_straight_rods = 0.56  # 0.52
        eta_helical_rods = 0.21  # 0.35#0.21
        eta_axial_cord = 0.06  # 0.17#0.06 #  axial nerve cord

        # Compute center rod and other straight rod radius
        # center_straight_rod_radius = np.sqrt(outer_base_radius ** 2 * eta_axial_cord)
        # outer_straight_rod_radius = np.sqrt(
        #     outer_base_radius ** 2 * eta_straight_rods / (number_of_straight_rods - 1)
        # )
        center_straight_rod_radius = np.sqrt(
            outer_base_radius**2
            * (eta_axial_cord + eta_straight_rods)
            / number_of_straight_rods
        )
        outer_straight_rod_radius = center_straight_rod_radius

        variables = fsolve(
            no_offset_radius_for_straight_rods,
            x0=(center_straight_rod_radius, outer_straight_rod_radius),
            args=(
                number_of_straight_rods - 1,
                np.pi * outer_base_radius**2 * (eta_axial_cord + eta_straight_rods),
            ),
        )
        center_straight_rod_radius, outer_straight_rod_radius = variables

        area_ring_rods = np.pi * outer_base_radius**2 * eta_ring_rods

        # outer_ring_rod_radius = minimize_scalar(
        #     outer_ring_rod_functions,
        #     args=(
        #         n_elem_ring_rod,
        #         center_straight_rod_radius,
        #         outer_straight_rod_radius,
        #         area_ring_rods,
        #     ),
        # ).x
        outer_ring_rod_radius = arm_length / n_elem / 2

        # outer_ring_rod_radius = 0.5 * (outer_base_radius - center_straight_rod_radius - 2*outer_straight_rod_radius)

        # Compute radius of helical rod. We will place helical rods at the most
        # outer layer.
        area_helical = (np.pi * outer_base_radius**2) * eta_helical_rods

        helical_rod_radius = minimize_scalar(
            helical_rod_radius_function,
            args=(
                center_straight_rod_radius,
                outer_straight_rod_radius,
                outer_ring_rod_radius,
                area_helical,
            ),
        ).x
        # FIXME: for stability we doubled the helical rod radius, but also with this change helical rod radius density
        #  is closer to the octopus arm density.
        helical_rod_radius *= 1.829478746907853  # 2#1.7197#2#5

        center_rod_radius_along_arm = center_straight_rod_radius * np.linspace(
            1, 1 / taper_ratio_axial_cord, n_elem
        )
        radius_ratio_factor_from_base_to_tip = np.linspace(1, 1 / taper_ratio, n_elem)

        outer_straight_rod_radius_along_arm = (
            outer_straight_rod_radius * radius_ratio_factor_from_base_to_tip
        )

        outer_ring_rod_radius_along_arm = outer_ring_rod_radius * np.ones(
            (n_elem)
        )  # * radius_ratio_factor_from_base_to_tip

        helical_rod_radius_along_arm = (
            helical_rod_radius * radius_ratio_factor_from_base_to_tip
        )

        # Compute the bank angle of the straight rods or longitudinal muscles. These rods are the ones except the one
        # at the center and they are not only tapered but banked.
        ratio = arm_length / (
            (center_rod_radius_along_arm[0] + outer_straight_rod_radius_along_arm[0])
            - (
                center_rod_radius_along_arm[-1]
                + outer_straight_rod_radius_along_arm[-1]
            )
        )
        bank_angle = np.arctan(ratio)

        # Van Leeuwen and Kier 1997 model
        # Compute the sacromere, myosin, maximum active stress
        l_bz = 0.14  # length of the bare zone (micro meter)

        # l_sacro_ref = 2.37  # reference sacromere length (micro meter)
        # Sarcomere lengths are changed to fit the Van Leeuwen model to the tentacle transver muscle active force
        # measurements given in  Kier and Curtin 2002,
        l_sacro_base = 1.1  # 1.3399  # sacromere length at the base (micro meter)
        l_sacro_tip = 1.1  # 1.3399#0.7276  # sacromere length at the tip  (micro meter)
        sarcomere_rest_lengths = np.linspace(
            l_sacro_base, l_sacro_tip, n_elem
        )  # np.ones((n_elem))*l_sacro_base#np.linspace(l_sacro_base, l_sacro_tip, n_elem)

        l_myo_ref = 1.58  # reference myosin length (micro meter)
        # Myosin lengths are measured in Kier and Curtin 2002.
        l_myo_base = 0.81  # 0.9707  # 7.41#6.5#0.9707  # myosin length at the base of the tentacle (micro meter)
        l_myo_tip = 0.81  # 0.9707#0.4997  # myosin length at the tip of the tentacle (micro meter)
        myosin_lengths = np.linspace(
            l_myo_base, l_myo_tip, n_elem
        )  # np.ones((n_elem))*l_myo_base#np.linspace(l_myo_base, l_myo_tip, n_elem)

        # Active maximum stress measured in Kier and Curtin 2002 is 130kPa which can be also computed
        # using the VanLeeuwen model for a given myosin length (0.81 micro m).
        maximum_active_stress_ref = 280e3  # maximum active stress reference value (Pa)
        maximum_active_stress = (
            maximum_active_stress_ref * (myosin_lengths - l_bz) / (l_myo_ref - l_bz)
        )

        # minimum_strain_rate_ref = -17  # -17  # 1/s
        # minimum_strain_rate = minimum_strain_rate_ref * (
        #         l_sacro_ref / sarcomere_rest_lengths
        # )
        # minimum_strain_rate = -1.8 * np.ones(
        #     (n_elem)
        # )  # 1/s # Kier and Curtin 2002 Squid Arm
        minimum_strain_rate_longitudinal = (
            -1.8
        )  # -0.913  # 1/s # Zullo 2022 for octopus longitudinal muscle
        minimum_strain_rate_transverse = (
            -1.8
        )  # -0.3560   # 1/s # Zullo 2022 for octopus longitudinal muscle
        # Force velocity constant is the G term in Hills equation. In Zullo 2022 paper it is not given but we found
        # the best fit for G term as 0.80. In our modeling approach we are using K=1/G to be consitend with the
        # paper of VanLeewuen and Kier . Since they provide force-velocity relation for both extending and contracting
        # muscles.
        force_velocity_constant = 0.25  # 1/0.80
        # (
        #     normalized_active_force_slope_transverse_muscles,
        #     normalized_active_force_y_intercept_transverse_muscles,
        #     normalized_active_force_break_points_transverse_muscles,
        # ) = get_active_force_piecewise_linear_function_parameters_for_VanLeeuwen_muscle_model(
        #     sarcomere_rest_lengths, myosin_lengths
        # )
        # # Load longitudinal muscle data for squid tentacle. Kier provided the measurments.
        # longitudinal_muscle_active_force_experimental_data = np.load(
        #     "kier_longitudinal_active_force_measurement.npz"
        # )
        # normalized_active_force_slope_longitudinal_muscles = (
        #     longitudinal_muscle_active_force_experimental_data["slope"]
        # )
        # normalized_active_force_y_intercept_longitudinal_muscles = (
        #     longitudinal_muscle_active_force_experimental_data["y_intercept"]
        # )
        # normalized_active_force_break_points_longitudinal_muscles = (
        #     longitudinal_muscle_active_force_experimental_data["break_points"]
        # )
        #
        # # Load passive force data. For transverse muscles we used the VanLeeuwen model. For longitudinal muscles
        # # measurments done by Udit. For passive stress Elastica implementation uses 3rd degree polynomial and
        # # requires 4 coefficients. Order of coefficients starts from highest order (cube) to lowest order of poly.
        # transverse_muscle_passive_force_coefficients = get_passive_force_cubic_function_coefficients_for_VanLeeuwen_muscle_model(
        #     strain=np.linspace(0, 0.8, 200)
        # )
        # # For the longitudinal muscle passive force we are using the coefficients measured by Udit Halder. These
        # # measurments are done by taking real octopus arm and performing stretch tests on the actual arm, under
        # # different loads. Since the fit provided is 2nd order other cofficients of cubic polynomial are zero.
        # longitudinal_muscle_passive_force_coefficients = np.array([0,112.8492e3,8.8245e3 , 0 ])
        #
        # For both longitudinal muscle active and passive force length curves we fit a polynomial. Lets read
        # the coefficients
        longitudinal_muscle_coefficient = np.load(
            "../../octopus_longitudinal_muscles_fit.npz"
        )
        # Longitudinal muscle active coefficients are computed by fitting 4th order polynomial to the data given in
        # Zullo 2022.
        longitudinal_muscle_active_force_coefficients = longitudinal_muscle_coefficient[
            "longitudinal_active_part_coefficients_with_shift"
        ]
        # Longitudinal muscle passive coefficients are computed by fitting 2nd order polynomial to the data given in
        # Zullo 2022.
        longitudinal_muscle_passive_force_coefficients = (
            longitudinal_muscle_coefficient[
                "longitudinal_passive_part_coefficients_with_shift"
            ]
        )

        # For both transverse muscle active and passive force length curves we fit a polynomial. Lets read the
        # coefficients.
        transverse_muscle_coefficient = np.load(
            "../../octopus_transverse_muscles_fit.npz"
        )
        # Transverse muscle active coefficients are computed by fitting 4th order polynomial to the data given in
        # Zullo 2022.
        transverse_muscle_active_force_coefficients = transverse_muscle_coefficient[
            "transverse_active_part_coefficients_with_shift"
        ]
        # Transverse muscle passive coefficients are computed by fitting 2nd order polynomial to the data given in
        # Zullo 2022.
        transverse_muscle_passive_force_coefficients = transverse_muscle_coefficient[
            "transverse_passive_part_coefficients_with_shift"
        ]

        direction_ring_rod = normal
        normal_ring_rod = direction

        self.rod_list = []
        self.straight_rod_list = []
        self.ring_straight_rod_connection_index_list = []
        # ring straight rod connection list is containing the list of directions for possible connections.
        # Here the idea is to connect nodes in specific position, and make sure there is symmetry in whole arm.
        self.ring_straight_rod_connection_direction_list = []

        # First straight rod is at the center, remaining ring rods are around the first ring rod.
        angle_btw_straight_rods = (
            0
            if number_of_straight_rods == 1
            else 2 * np.pi / (number_of_straight_rods - 1)
        )

        self.angle_wrt_center_rod = []
        self.bank_angle_of_straight_rods_list = []

        # First straight rod at the center
        start_rod_1 = start

        self.ring_straight_rod_connection_direction_list.append([])
        for i in range(int((number_of_straight_rods - 1))):
            # Compute the direction from center (origin) of the rod towards the other rods straight rods.
            rotation_matrix = _get_rotation_matrix(
                angle_btw_straight_rods * i, direction.reshape(3, 1)
            ).reshape(3, 3)
            direction_from_center_to_rod = rotation_matrix @ binormal

            self.ring_straight_rod_connection_direction_list[0].append(
                direction_from_center_to_rod
            )

        # Center rod has 0 bank angle and 0 angle wrt center rod. THis is center rod.
        self.angle_wrt_center_rod.append(0)
        self.bank_angle_of_straight_rods_list.append(np.pi / 2)

        base_radius_varying_center = (
            center_rod_radius_along_arm  # center_rod_radius * np.ones((n_elem))
        )

        axial_cord_length = np.ones((n_elem)) * arm_length / n_elem
        volume_axial_cord = axial_cord_length * (
            np.pi * center_rod_radius_along_arm**2
        )
        density_axial_cord = (
            density  # mass_total * eta_axial_cord / volume_axial_cord.sum()
        )

        # Arrange sizes of normalized active force piecewise function parameters, before initializing straight rods.
        # normalized_active_force_slope_straight_rods = np.ones(
        #     (n_elem)
        # ) * normalized_active_force_slope_longitudinal_muscles.reshape(4, 1)
        # normalized_active_force_y_intercept_straight_rods = np.ones(
        #     (n_elem)
        # ) * normalized_active_force_y_intercept_longitudinal_muscles.reshape((4, 1))
        # normalized_active_force_break_points_straight_rods = np.ones(
        #     (n_elem)
        # ) * normalized_active_force_break_points_longitudinal_muscles.reshape((4, 1))
        muscle_active_force_coefficients_straight_rods = np.ones(
            (n_elem)
        ) * longitudinal_muscle_active_force_coefficients.reshape(9, 1)
        straight_rods_force_velocity_constant = (
            np.ones((n_elem)) * force_velocity_constant
        )
        maximum_active_stress_straight_rods = np.ones((n_elem)) * maximum_active_stress
        minimum_strain_rate_straight_rods = (
            np.ones((n_elem)) * minimum_strain_rate_longitudinal
        )
        passive_force_coefficients_straight_rods = (
            longitudinal_muscle_passive_force_coefficients.reshape(9, 1)
            * np.ones((n_elem))
        )

        nu_rod_1 = (
            density_axial_cord
            * np.pi
            * center_rod_radius_along_arm**2
            * nu
            / 8
            / 3
            # * 2
            # * 2
            * 15
            / 4
            / 2
            / 15
            # * 2  # *5 * 5#* 10# * 2
        ) * 0

        self.shearable_rod1 = MuscularRod.straight_rod(
            n_elem,
            start_rod_1,
            direction,
            normal,
            arm_length,
            base_radius=center_rod_radius_along_arm,
            density=density_axial_cord,
            nu=nu_rod_1,
            youngs_modulus=E,
            poisson_ratio=poisson_ratio,
            maximum_active_stress=maximum_active_stress_straight_rods,
            minimum_strain_rate=minimum_strain_rate_straight_rods,
            force_velocity_constant=straight_rods_force_velocity_constant,
            # normalized_active_force_break_points=normalized_active_force_break_points_straight_rods,
            # normalized_active_force_y_intercept=normalized_active_force_y_intercept_straight_rods,
            # normalized_active_force_slope=normalized_active_force_slope_straight_rods,
            E_compression=2.5e4,
            compression_strain_limit=-0.025,
            active_force_coefficients=muscle_active_force_coefficients_straight_rods,
            # tension_passive_force_scale=1/30,#/4,
            # Active force starts to decrease around strain 0.55 so we shift passive force to strain of 0.55
            # This is also seen in Leech longitudinal muscles (Gerry & Ellebry 2011)
            extension_strain_limit=0.025,
            passive_force_coefficients=passive_force_coefficients_straight_rods,
        )
        # self.shearable_rod1.bend_matrix[2,2,:]/=4
        self.straight_rod_list.append(self.shearable_rod1)

        rod_height = np.hstack((0.0, np.cumsum(self.shearable_rod1.rest_lengths)))
        rod_non_dimensional_position = rod_height / np.sin(bank_angle)

        for i in range(number_of_straight_rods - 1):
            rotation_matrix = _get_rotation_matrix(
                angle_btw_straight_rods * i, direction.reshape(3, 1)
            ).reshape(3, 3)
            direction_from_center_to_rod = rotation_matrix @ binormal

            self.angle_wrt_center_rod.append(angle_btw_straight_rods * i)

            self.ring_straight_rod_connection_direction_list.append(
                [direction_from_center_to_rod, -direction_from_center_to_rod]
            )

            # Compute the rotation matrix, for getting the correct banked angle.
            normal_banked_rod = rotation_matrix @ normal
            # Rotate direction vector around new normal to get the new direction vector.
            # Note that we are doing ccw rotation and direction should be towards the center.
            rotation_matrix_banked_rod = _get_rotation_matrix(
                -(np.pi / 2 - bank_angle), normal_banked_rod.reshape(3, 1)
            ).reshape(3, 3)
            direction_banked_rod = rotation_matrix_banked_rod @ direction

            start_rod = start + (direction_from_center_to_rod) * (
                # center rod            # this rod
                +base_radius_varying_center[0]
                + outer_straight_rod_radius_along_arm[0]
            )

            position_collection = np.zeros((3, n_elem + 1))
            position_collection[:] = np.einsum(
                "i,j->ij", direction_banked_rod, rod_non_dimensional_position
            ) + start_rod.reshape(3, 1)

            self.bank_angle_of_straight_rods_list.append(bank_angle)

            # Now lets recompute the straight rod radius, since it is banked, elements of center and outer straight rod
            # is not touching perfectly. Lets adjust straight_rod_radius so that center and outer ones touch.
            element_position_center_rod = 0.5 * (
                self.shearable_rod1.position_collection[:, 1:]
                + self.shearable_rod1.position_collection[:, :-1]
            )
            element_position_straight_rod = 0.5 * (
                position_collection[:, 1:] + position_collection[:, :-1]
            )
            straight_rod_radius = _batch_norm(
                element_position_center_rod - element_position_straight_rod
            ) - (self.shearable_rod1.radius)

            longitudinal_muscle_length = np.ones((n_elem)) * arm_length / n_elem
            volume_longitudinal_muscle = longitudinal_muscle_length * (
                np.pi * straight_rod_radius**2
            )
            # Adjust density such that its mass is consistent with percentage area of this muscle group
            density_longitudinal_muscle = (
                mass_total
                * eta_straight_rods
                / (number_of_straight_rods - 1)
                / volume_longitudinal_muscle.sum()
            )
            # density_longitudinal_muscle = density#density_axial_cord

            # nu_rod = (
            #         density_longitudinal_muscle
            #         * np.pi
            #         * outer_straight_rod_radius_along_arm ** 2
            #         * nu
            #         / 8
            #         / 3
            # )
            nu_rod = nu_rod_1  # /10

            self.straight_rod_list.append(
                MuscularRod.straight_rod(
                    n_elem,
                    start_rod,
                    direction_banked_rod,
                    normal_banked_rod,
                    arm_length,
                    base_radius=straight_rod_radius,
                    density=density_longitudinal_muscle,
                    nu=nu_rod,
                    youngs_modulus=E,
                    poisson_ratio=poisson_ratio,
                    position=position_collection,
                    maximum_active_stress=maximum_active_stress_straight_rods,
                    minimum_strain_rate=minimum_strain_rate_straight_rods,
                    force_velocity_constant=straight_rods_force_velocity_constant,
                    # normalized_active_force_break_points=normalized_active_force_break_points_straight_rods,
                    # normalized_active_force_y_intercept=normalized_active_force_y_intercept_straight_rods,
                    # normalized_active_force_slope=normalized_active_force_slope_straight_rods,
                    E_compression=2.5e4,
                    compression_strain_limit=-0.025,
                    active_force_coefficients=muscle_active_force_coefficients_straight_rods,
                    # tension_passive_force_scale=1/30,#/4,
                    # Extension shift is to make sure at rest configuration there is no passive tension stress.
                    extension_strain_limit=0.025,
                    passive_force_coefficients=passive_force_coefficients_straight_rods,
                )
            )

        # Ring rods
        # n_elem_ring_rod = (
        #     12#number_of_straight_rods-1  # 48#12#24  # chose multiplies of 8
        # )

        total_number_of_ring_rods = n_elem  # 33
        self.ring_rod_list_outer = []
        straight_rod_element_position = 0.5 * (
            self.shearable_rod1.position_collection[..., 1:]
            + self.shearable_rod1.position_collection[..., :-1]
        )
        straight_rod_position = np.einsum(
            "ij, i->j", straight_rod_element_position, direction
        )
        ring_rod_position = straight_rod_position

        center_position_ring_rod = np.zeros((3, total_number_of_ring_rods))
        connection_idx_straight_rod = np.zeros(
            (total_number_of_ring_rods), dtype=np.int
        )
        radius_outer_ring_rod = np.zeros((total_number_of_ring_rods))
        base_length_ring_rod_outer = np.zeros((total_number_of_ring_rods))
        volume_outer_ring_rod = np.zeros((total_number_of_ring_rods))

        for i in range(total_number_of_ring_rods):
            center_position_ring_rod[..., i] = start + direction * ring_rod_position[i]
            # center_position_ring_rod[..., i] = start + direction * ring_rod_position[1]

            # Position of ring rods are defined as percentage of the center rod length. In order to place
            # ring rod node positions and center rod node positions we have to do following calculation.
            # Placing ring rod and center rod positions on the same plane simplifies the connection routines.
            center_position_ring_rod_along_direction = np.dot(
                center_position_ring_rod[..., i], direction
            )
            connection_idx_straight_rod[i] = np.argmin(
                np.abs(straight_rod_position - center_position_ring_rod_along_direction)
            )

            center_position_ring_rod[..., i] = straight_rod_element_position[
                ..., connection_idx_straight_rod[i]
            ]

            radius_outer_ring_rod[i] = outer_ring_rod_radius_along_arm[
                connection_idx_straight_rod[i]
            ]

            base_length_ring_rod_outer[i] = (
                2
                * np.pi
                * (
                    self.shearable_rod1.radius[connection_idx_straight_rod[i]]
                    + 2
                    * self.straight_rod_list[1].radius[connection_idx_straight_rod[i]]
                    + radius_outer_ring_rod[i]
                )
            )
            volume_outer_ring_rod[i] = base_length_ring_rod_outer[i] * (
                np.pi * radius_outer_ring_rod[i] ** 2
            )

        # Adjust density such that its mass is consistent with percentage area of this muscle group
        density_ring_rod = mass_total * eta_ring_rods / volume_outer_ring_rod.sum()
        # density_ring_rod = density

        for i in range(total_number_of_ring_rods):
            nu_ring_rod_outer = (
                density_ring_rod
                * np.pi
                * radius_outer_ring_rod[i] ** 2
                * nu
                / 4
                * 2
                * 2
                * 2
                * 2
                * 2
                / 8
                / 3
                # * 3
                / 4
                / 2
                * 2
                #                * 2
                # * 5 #*1.5
            )

            # myosin_lengths_ring = (
            #         np.ones((n_elem_ring_rod))
            #         * myosin_lengths[connection_idx_straight_rod[i]]
            # )
            # sarcomere_rest_lengths_ring = (
            #         np.ones((n_elem_ring_rod))
            #         * sarcomere_rest_lengths[connection_idx_straight_rod[i]]
            # )
            # maximum_active_stress_ring = (
            #         np.ones((n_elem_ring_rod))
            #         * maximum_active_stress[connection_idx_straight_rod[i]]
            # )
            # minimum_strain_rate_ring = (
            #         np.ones((n_elem_ring_rod))
            #         * minimum_strain_rate[connection_idx_straight_rod[i]]
            # )

            # myosin_lengths_ring = np.ones((n_elem_ring_rod)) * myosin_lengths[0]
            # sarcomere_rest_lengths_ring = (
            #     np.ones((n_elem_ring_rod)) * sarcomere_rest_lengths[0]
            # )
            maximum_active_stress_ring = (
                np.ones((n_elem_ring_rod)) * maximum_active_stress[0]
            )
            minimum_strain_rate_ring = (
                np.ones((n_elem_ring_rod)) * minimum_strain_rate_transverse
            )
            force_velocity_constant_ring = (
                np.ones((n_elem_ring_rod)) * force_velocity_constant
            )
            # normalized_active_force_break_points_ring = np.ones(
            #     (n_elem_ring_rod)
            # ) * normalized_active_force_break_points_transverse_muscles[
            #     :, connection_idx_straight_rod[i]
            # ].reshape(
            #     4, 1
            # )
            # normalized_active_force_y_intercept_ring = np.ones(
            #     (n_elem_ring_rod)
            # ) * normalized_active_force_y_intercept_transverse_muscles[
            #     :, connection_idx_straight_rod[i]
            # ].reshape(
            #     4, 1
            # )
            # normalized_active_force_slope_ring = np.ones(
            #     (n_elem_ring_rod)
            # ) * normalized_active_force_slope_transverse_muscles[
            #     :, connection_idx_straight_rod[i]
            # ].reshape(
            #     4, 1
            # )
            muscle_active_force_coefficients_ring_rods = np.ones(
                (n_elem_ring_rod)
            ) * transverse_muscle_active_force_coefficients.reshape(9, 1)
            passive_force_coefficients_ring_rods = (
                transverse_muscle_passive_force_coefficients.reshape(9, 1)
                * np.ones((n_elem_ring_rod))
            )

            self.ring_rod_list_outer.append(
                MuscularRod.ring_rod(
                    n_elem_ring_rod,
                    center_position_ring_rod[..., i],
                    direction_ring_rod,
                    -normal_ring_rod,
                    base_length_ring_rod_outer[i],
                    radius_outer_ring_rod[i],
                    density=density_ring_rod,
                    nu=nu_ring_rod_outer,
                    youngs_modulus=E,  # *2,#*4*4,  # *4*(2.5)**2,
                    poisson_ratio=poisson_ratio,
                    maximum_active_stress=maximum_active_stress_ring,  # *4*4,
                    minimum_strain_rate=minimum_strain_rate_ring,
                    force_velocity_constant=force_velocity_constant_ring,
                    E_compression=1e4,
                    # tension_passive_force_scale = 1 ,
                    # normalized_active_force_break_points=normalized_active_force_break_points_ring,
                    # normalized_active_force_y_intercept=normalized_active_force_y_intercept_ring,
                    # normalized_active_force_slope=normalized_active_force_slope_ring,
                    compression_strain_limit=-0.025,
                    active_force_coefficients=muscle_active_force_coefficients_ring_rods,
                    # Extension shift is to make sure at rest configuration there is no passive tension stress.
                    extension_strain_limit=0.025,
                    passive_force_coefficients=passive_force_coefficients_ring_rods,
                )
            )
        #     #self.ring_rod_list_outer[i].shear_matrix[0,0,:] *= 5*2#0
        #     #self.ring_rod_list_outer[i].shear_matrix[1,1,:] *= 5*2#0

        self.ring_rod_list = self.ring_rod_list_outer

        # Create helix rods or oblique muscles
        self.helical_rod_list = []

        direction_helical_rod = direction
        normal_helical_rod = normal
        binormal_helical_rod = np.cross(direction_helical_rod, normal_helical_rod)

        total_number_of_helical_rods = 8

        # Sum of center rod radius and ring rod diameter and helical rod radius
        helix_radius_base = (
            self.shearable_rod1.radius[0]
            + 2 * self.straight_rod_list[1].radius[0]
            + 2 * self.ring_rod_list_outer[0].radius[0]
            + helical_rod_radius_along_arm[0]
        )
        distance_btw_two_ring_rods = ring_rod_position[1] - ring_rod_position[0]
        # Helix should cover all ring rods. First term below gives the distance from first ring rod upto last ring rod
        # but we want to connect last ring rod and helical rod as well. Thus we add second term.
        helix_length_covered = (
            ring_rod_position[-1] - ring_rod_position[0]
        ) + distance_btw_two_ring_rods / 2
        # Target helix angle is 72 degrees. If we set number of ring rods covered by one helix turn to 6, helix angle
        # becomes 71.0072 degrees, which is the closest to the target angle.
        distance_in_one_turn = (20 + 5e-15) * distance_btw_two_ring_rods  # 6
        self.helix_angle = np.rad2deg(
            np.arctan(2 * np.pi * helix_radius_base / distance_in_one_turn)
        )
        n_helix_turns = helix_length_covered / distance_in_one_turn
        pitch_factor = distance_in_one_turn / (2 * np.pi)

        # Number of helix turns
        n_elem_per_turn = 40  # This should be divided by the 6 at least, # of ring rod that is passed in one turn
        n_elem_helical_rod = int(n_helix_turns * n_elem_per_turn) + 1

        # Compute the curve angle of helix for each node. Using curve angle later on we will compute the position of
        # nodes.
        curve_angle = np.linspace(
            0.0, 2 * np.pi * n_helix_turns, n_elem_helical_rod + 1
        )

        # Start helical rod at the same location as the ring rod at the base.
        start_position_of_helix = (
            np.zeros((3,)) + direction_helical_rod * ring_rod_position[0]
        )
        # Helix radius is radius of one helix turn, you can make this radius varying and let it decrease as
        # it goes along the global direction. For example for tapered arm helix radius decreases along the arm.
        # First compute the node positions of the helical rod in `direction` direction.
        position_of_helical_rod_in_global_dir = np.einsum(
            "ij, i->j",
            pitch_factor * np.einsum("i,j->ij", direction_helical_rod, curve_angle),
            direction,
        )

        # We need to find the positions of nodes where ring rod and helical rod are connected. These points are the
        # interpolation points for the spline for computing the helix radius, which is used to compute node positions.
        # first dimension is height, second is radius of helix
        helix_radius_points_for_interp = np.zeros((2, 2 * total_number_of_ring_rods))
        helical_rod_radius_points_for_interp = np.zeros((2, total_number_of_ring_rods))

        for idx, rod in enumerate(self.ring_rod_list_outer):
            # Find the center position of ring rod.
            center_position_ring_rod = np.mean(rod.position_collection, axis=1)
            center_position_ring_rod_along_direction = np.dot(
                center_position_ring_rod - start_position_of_helix, direction
            )
            # Find the node idx where helical rod node and ring rod are at the same height.
            index = np.argmin(
                np.abs(
                    position_of_helical_rod_in_global_dir
                    - center_position_ring_rod_along_direction
                )
            )
            # We will connect two pairs of nodes (2 from ring and 2 from helical rod). Thus also cache the
            # position data for these two pairs.
            helix_radius_points_for_interp[
                0, 2 * idx
            ] = position_of_helical_rod_in_global_dir[index]
            helix_radius_points_for_interp[1, 2 * idx] = (
                self.shearable_rod1.radius[idx]
                + 2 * self.straight_rod_list[1].radius[idx]
                + 2 * self.ring_rod_list_outer[idx].radius[0]
                + helical_rod_radius_along_arm[idx]
            )
            helix_radius_points_for_interp[
                0, 2 * idx + 1
            ] = position_of_helical_rod_in_global_dir[index + 1]
            helix_radius_points_for_interp[1, 2 * idx + 1] = (
                self.shearable_rod1.radius[idx]
                + 2 * self.straight_rod_list[1].radius[idx]
                + 2 * self.ring_rod_list_outer[idx].radius[0]
                + helical_rod_radius_along_arm[idx]
            )

            helical_rod_radius_points_for_interp[0, idx] = 0.5 * (
                position_of_helical_rod_in_global_dir[index]
                + position_of_helical_rod_in_global_dir[index + 1]
            )
            helical_rod_radius_points_for_interp[1, idx] = helical_rod_radius_along_arm[
                idx
            ]

        from scipy.interpolate import interp1d

        # Generate the interpolation function using the data points computed above.
        helix_radius_interp_func = interp1d(
            helix_radius_points_for_interp[0, :],
            helix_radius_points_for_interp[1, :],
            fill_value="extrapolate",
        )
        # Compute the helix radius, which has a dimension same as number of nodes.
        helix_radius = helix_radius_interp_func(position_of_helical_rod_in_global_dir)

        # Compute the helical rod radius. This is the radius of the rod and if arm is tapered it decreases from base
        # to tip.
        helical_rod_radius_interp_func = interp1d(
            helical_rod_radius_points_for_interp[0, :],
            helical_rod_radius_points_for_interp[1, :],
            fill_value="extrapolate",
        )
        helical_rod_radius = helical_rod_radius_interp_func(
            0.5
            * (
                position_of_helical_rod_in_global_dir[:-1]
                + position_of_helical_rod_in_global_dir[1:]
            )
        )

        # normalized_active_force_break_points_helical_rods = np.ones(
        #     (n_elem_helical_rod)
        # ) * normalized_active_force_break_points_longitudinal_muscles.reshape((4, 1))
        # normalized_active_force_y_intercept_helical_rods = np.ones(
        #     (n_elem_helical_rod)
        # ) * normalized_active_force_y_intercept_longitudinal_muscles.reshape((4, 1))
        # normalized_active_force_slope_helical_rods = np.ones(
        #     (n_elem_helical_rod)
        # ) * normalized_active_force_slope_longitudinal_muscles.reshape((4, 1))
        maximum_active_stress_helical_rods = (
            np.ones((n_elem_helical_rod)) * maximum_active_stress[0]
        )
        minimum_strain_rate_helical_rods = (
            np.ones((n_elem_helical_rod)) * minimum_strain_rate_longitudinal
        )
        force_velocity_constant_helical_rods = (
            np.ones((n_elem_helical_rod)) * force_velocity_constant
        )
        muscle_active_force_coefficients_helical_rods = np.ones(
            (n_elem_helical_rod)
        ) * longitudinal_muscle_active_force_coefficients.reshape((9, 1))
        passive_force_coefficients_helical_rods = np.ones(
            (n_elem_helical_rod)
        ) * longitudinal_muscle_passive_force_coefficients.reshape(9, 1)

        for idx in range(0, int(total_number_of_helical_rods / 2)):
            # Offset angle changes the start angle of the helix
            offset_angle = np.pi / 2 * idx

            # Compute node positions of helical rod, using parameters.
            position = np.zeros((3, n_elem_helical_rod + 1))
            for i in range(n_elem_helical_rod + 1):
                position[..., i] = (
                    (
                        helix_radius[i]
                        * (
                            np.cos(curve_angle[i] + offset_angle) * binormal_helical_rod
                            + np.sin(curve_angle[i] + offset_angle) * normal_helical_rod
                        )
                    )
                    + start_position_of_helix
                    + pitch_factor * curve_angle[i] * direction_helical_rod
                )

            start = position[..., 0]

            # Compute rod tangents using positions
            position_for_difference = position
            position_diff = (
                position_for_difference[..., 1:] - position_for_difference[..., :-1]
            )
            rest_lengths = _batch_norm(position_diff)
            tangents = position_diff / rest_lengths
            base_length_helical_rod = rest_lengths.sum()

            # Compute normal, binormal and director collection
            # We need to compute directors because helix is not a straight rod and it has a complex shape.
            normal_collection = np.zeros((3, n_elem_helical_rod))
            binormal_collection = np.zeros((3, n_elem_helical_rod))
            director_collection = np.zeros((3, 3, n_elem_helical_rod))

            for i in range(n_elem_helical_rod):
                # Compute the normal vector at each element. Since we allow helix radius to vary, we need to compute
                # vectors creating normal for each element.
                vector_one = helix_radius[i] * (
                    np.sin(curve_angle[i] + offset_angle) * binormal_helical_rod
                    - np.cos(curve_angle[i] + offset_angle) * normal_helical_rod
                )
                vector_two = helix_radius[i + 1] * (
                    np.sin(curve_angle[i + 1] + offset_angle) * binormal_helical_rod
                    - np.cos(curve_angle[i + 1] + offset_angle) * normal_helical_rod
                )
                normal_collection[..., i] = 0.5 * (vector_two - vector_one)
                normal_collection[..., i] /= np.linalg.norm(normal_collection[..., i])

                binormal_collection[..., i] = np.cross(
                    tangents[..., i], normal_collection[..., i]
                )
                director_collection[..., i] = np.vstack(
                    (
                        normal_collection[..., i],
                        binormal_collection[..., i],
                        tangents[..., i],
                    )
                )

            volume_helical_rod = (
                rest_lengths * (np.pi * helical_rod_radius**2)
            ).sum()
            density_helical_rod = (
                mass_total
                * eta_helical_rods
                / total_number_of_helical_rods
                / volume_helical_rod
            )

            nu_helical_rod = (
                density_helical_rod
                * np.pi
                * helical_rod_radius**2
                * nu
                / 2
                / 3
                / 2
                # * 4
            )  # *0

            self.helical_rod_list.append(
                MuscularRod.straight_rod(
                    n_elem_helical_rod,
                    start,
                    direction_helical_rod,
                    normal_helical_rod,
                    base_length_helical_rod,
                    helical_rod_radius,
                    density=density_helical_rod,  # density,#density_helical_rod,
                    nu=nu_helical_rod,
                    youngs_modulus=E,
                    poisson_ratio=poisson_ratio,
                    position=position,
                    directors=director_collection,
                    maximum_active_stress=maximum_active_stress_helical_rods,
                    minimum_strain_rate=minimum_strain_rate_helical_rods,
                    force_velocity_constant=force_velocity_constant_helical_rods,
                    active_force_coefficients=muscle_active_force_coefficients_helical_rods,
                    # normalized_active_force_break_points=normalized_active_force_break_points_helical_rods,
                    # normalized_active_force_y_intercept=normalized_active_force_y_intercept_helical_rods,
                    # normalized_active_force_slope=normalized_active_force_slope_helical_rods,
                    E_compression=2.5e4,
                    # tension_passive_force_scale=1,
                    compression_strain_limit=-0.025,
                    # Extension shift is to make sure at rest configuration there is no passive tension stress.
                    extension_strain_limit=0.025,
                    passive_force_coefficients=passive_force_coefficients_helical_rods,
                )
            )

        # Create helical rods rotates in opposite direction. These helical rods starts at the same point as the
        # ones above. However, helix rotation direction is opposite. If above rods rotate clockwise these ones
        # will rotate counter clockwise. This helps the whole structure to be stable.
        # Because during expansion, due to the helix rotation direction, whole structure rotates, but we dont
        # want that because helical muscles are not activated. Also in real squid or octopus arm there are two
        # layers of helical rods rotating in opposite directions.
        for idx in range(
            int(total_number_of_helical_rods / 2), total_number_of_helical_rods
        ):
            # Offset angle changes the start angle of the helix
            offset_angle = np.pi / 2 * idx  # + np.pi/2

            # Compute node positions of helical rod, using parameters.
            position = np.zeros((3, n_elem_helical_rod + 1))
            for i in range(n_elem_helical_rod + 1):
                position[..., i] = (
                    (
                        helix_radius[i]
                        * (
                            np.cos(-curve_angle[i] + offset_angle)
                            * binormal_helical_rod
                            + np.sin(-curve_angle[i] + offset_angle)
                            * normal_helical_rod
                        )
                    )
                    + start_position_of_helix
                    + pitch_factor * curve_angle[i] * direction_helical_rod
                )

            start = position[..., 0]

            # Compute rod tangents using positions
            position_for_difference = position
            position_diff = (
                position_for_difference[..., 1:] - position_for_difference[..., :-1]
            )
            rest_lengths = _batch_norm(position_diff)
            tangents = position_diff / rest_lengths
            base_length_helical_rod = rest_lengths.sum()

            # Compute normal, binormal and director collection of elements
            # We need to compute directors because helix is not a straight rod and it has a complex shape.
            normal_collection = np.zeros((3, n_elem_helical_rod))
            binormal_collection = np.zeros((3, n_elem_helical_rod))
            director_collection = np.zeros((3, 3, n_elem_helical_rod))

            for i in range(n_elem_helical_rod):
                # Compute the normal vector at each element. Since we allow helix radius to vary, we need to compute
                # vectors creating normal for each element.
                vector_one = helix_radius[i] * (
                    np.sin(-curve_angle[i] + offset_angle) * binormal_helical_rod
                    - np.cos(-curve_angle[i] + offset_angle) * normal_helical_rod
                )
                vector_two = helix_radius[i + 1] * (
                    np.sin(-curve_angle[i + 1] + offset_angle) * binormal_helical_rod
                    - np.cos(-curve_angle[i + 1] + offset_angle) * normal_helical_rod
                )
                normal_collection[..., i] = 0.5 * (vector_two - vector_one)
                normal_collection[..., i] /= np.linalg.norm(normal_collection[..., i])

                binormal_collection[..., i] = np.cross(
                    tangents[..., i], normal_collection[..., i]
                )
                director_collection[..., i] = np.vstack(
                    (
                        normal_collection[..., i],
                        binormal_collection[..., i],
                        tangents[..., i],
                    )
                )

            volume_helical_rod = (
                rest_lengths * (np.pi * helical_rod_radius**2)
            ).sum()
            density_helical_rod = (
                mass_total
                * eta_helical_rods
                / total_number_of_helical_rods
                / volume_helical_rod
            )

            nu_helical_rod = (
                density_helical_rod
                * np.pi
                * helical_rod_radius**2
                * nu
                / 2
                / 3
                / 2
                # * 4
            )  # *0

            self.helical_rod_list.append(
                MuscularRod.straight_rod(
                    n_elem_helical_rod,
                    start,
                    direction_helical_rod,
                    normal_helical_rod,
                    base_length_helical_rod,
                    helical_rod_radius,
                    density=density_helical_rod,  # density,#density_helical_rod,
                    nu=nu_helical_rod,
                    youngs_modulus=E,
                    poisson_ratio=poisson_ratio,
                    position=position,
                    directors=director_collection,
                    maximum_active_stress=maximum_active_stress_helical_rods,
                    minimum_strain_rate=minimum_strain_rate_helical_rods,
                    force_velocity_constant=force_velocity_constant_helical_rods,
                    active_force_coefficients=muscle_active_force_coefficients_helical_rods,
                    # normalized_active_force_break_points=normalized_active_force_break_points_helical_rods,
                    # normalized_active_force_y_intercept=normalized_active_force_y_intercept_helical_rods,
                    # normalized_active_force_slope=normalized_active_force_slope_helical_rods,
                    E_compression=2.5e4,
                    # tension_passive_force_scale=1,
                    compression_strain_limit=-0.025,
                    # Extension shift is to make sure at rest configuration there is no passive tension stress.
                    extension_strain_limit=0.025,
                    passive_force_coefficients=passive_force_coefficients_helical_rods,
                )
            )

        self.rod_list = (
            self.ring_rod_list + self.straight_rod_list + self.helical_rod_list
        )

        # TODO: remove below lines, these are for checking the masses
        test_total_ring_rod_mass = 0.0
        for _, rod in enumerate(self.ring_rod_list):
            test_total_ring_rod_mass += rod.mass.sum()

        test_total_straight_rod_mass = 0.0
        for _, rod in enumerate(self.straight_rod_list):
            test_total_straight_rod_mass += rod.mass.sum()

        test_total_helical_rod_mass = 0.0
        for _, rod in enumerate(self.helical_rod_list):
            test_total_helical_rod_mass += rod.mass.sum()

        print("total ring rod mass " + str(test_total_ring_rod_mass))
        print("total straight rod mass " + str(test_total_straight_rod_mass))
        print("total helical rod mass " + str(test_total_helical_rod_mass))
        print(
            "total mass "
            + str(
                test_total_ring_rod_mass
                + test_total_straight_rod_mass
                + test_total_helical_rod_mass
            )
        )
        print("Correct total mass " + str(mass_total))

        # Now rod is ready for simulation, append rod to simulation
        for _, rod in enumerate(self.rod_list):
            self.simulator.append(rod)

        # Add cylinders
        # Create cylinder as a rigid body
        self.cylinder_list = []
        cylinder_height = 280
        cylinder_radius = 10
        cylinder_start = (
            start
            + 1.5 * (outer_base_radius + cylinder_radius) * binormal
            + 115 * direction
            - normal * cylinder_height / 2
        )
        cylinder_direction = direction  # normal
        cylinder_normal = normal  # direction
        cylinder_binormal = np.cross(direction, normal)

        # self.cylinder = Cylinder(
        #     start=cylinder_start,
        #     direction=cylinder_direction,
        #     normal=cylinder_normal,
        #     base_length=cylinder_height,
        #     base_radius=cylinder_radius,
        #     density=density,
        # )
        # self.cylinder_list.append(self.cylinder)

        # Second cylinder
        # cylinder_two_start = (
        #     start
        #     + 1.5 * (outer_base_radius + cylinder_radius) * binormal
        #     + 45 * direction
        #     - normal * cylinder_height / 2
        # )
        # self.cylinder_two = Cylinder(
        #     start=cylinder_two_start,
        #     direction=cylinder_direction,
        #     normal=cylinder_normal,
        #     base_length=cylinder_height,
        #     base_radius=cylinder_radius,
        #     density=density,
        # )
        # self.cylinder_list.append(self.cylinder_two)

        # Add target
        rod_idx = np.argmin(
            np.abs((self.straight_rod_list[6].position_collection[1, :] - 0))
        )
        target_height = 80
        target_radius = 12 / 2
        target_start = (
            # start
            self.straight_rod_list[6].position_collection[..., rod_idx]
            - (
                3.0 * target_radius
                + self.straight_rod_list[6].radius[rod_idx]
                + 2 * self.helical_rod_list[0].radius[0]
                + 2 * self.ring_rod_list[0].radius[0]
            )
            * (binormal + normal)
            / np.linalg.norm(
                binormal + normal
            )  # 0.5 * (outer_base_radius + target_radius) * binormal
            - 12 * direction
            #            - normal * target_height / 2
            #            - normal * target_radius
        )

        #        target_start = np.array([ -8.33492496,   0.        , -24.56222838])
        angle = np.deg2rad(30)
        first_cylinder_direction = cylinder_direction * np.cos(
            angle
        ) + cylinder_binormal * np.sin(angle)
        first_cylinder_direction /= np.linalg.norm(first_cylinder_direction)
        first_cylinder_start = (
            target_start
            + (cylinder_direction - first_cylinder_direction) * target_height / 2
        )

        self.target_cylinder = Cylinder(
            start=first_cylinder_start,  # target_start,
            direction=first_cylinder_direction,
            normal=cylinder_normal,
            base_length=target_height,
            base_radius=target_radius,
            density=density / 100,
        )
        self.cylinder_list.append(self.target_cylinder)

        # Add second cylinder
        target_start_two = (
            target_start
            + cylinder_direction * target_height / 2
            - cylinder_binormal * target_height / 2
        )
        angle = np.deg2rad(-30)
        second_cylinder_direction = cylinder_binormal * np.cos(
            angle
        ) + cylinder_normal * np.sin(angle)
        second_cylinder_direction /= np.linalg.norm(second_cylinder_direction)
        target_start_two += (
            (cylinder_binormal - second_cylinder_direction) * target_height / 2
        )

        print("target_two_start " + str(target_start_two))
        self.target_cylinder_two = Cylinder(
            start=target_start_two,
            direction=second_cylinder_direction,  # cylinder_binormal,#cylinder_direction,
            normal=cylinder_direction,  # cylinder_normal,
            base_length=target_height,
            base_radius=target_radius,
            density=density / 100,
        )
        self.cylinder_list.append(self.target_cylinder_two)

        # Add third cylinder
        target_start_third = (
            target_start + cylinder_direction * target_height / 2
        )  # - cylinder_normal * target_height/2
        #        print("target_third_start "+str(target_start_two))
        #        print("target_third direction " + str(cylinder_normal))
        #        print("target_third_normal " + str(cylinder_direction))
        angle = np.deg2rad(-30)
        third_cylinder_direction = cylinder_normal * np.cos(
            angle
        ) + cylinder_direction * np.sin(angle)
        third_cylinder_direction /= np.linalg.norm(third_cylinder_direction)
        target_start_third -= third_cylinder_direction * target_height / 2
        self.target_cylinder_third = Cylinder(
            start=target_start_third,
            direction=third_cylinder_direction,  # cylinder_normal*np.cos(angle) + cylinder_direction*np.sin(angle),#cylinder_direction,
            normal=cylinder_binormal,  # cylinder_normal,
            base_length=target_height,
            base_radius=target_radius,
            density=density / 100,
        )
        self.cylinder_list.append(self.target_cylinder_third)

        for _, cylinder in enumerate(self.cylinder_list):
            self.simulator.append(cylinder)

        # Constrain the rods
        for i, rod in enumerate(self.straight_rod_list):
            if i == 0:
                self.simulator.constrain(rod).using(
                    OneEndFixedRod,  # FixNodePosition,
                    constrained_position_idx=(0,),
                    constrained_director_idx=(0,),
                )
            else:
                # pass
                self.simulator.constrain(rod).using(
                    OneEndFixedRod,  # FixNodePosition,
                    constrained_position_idx=(0,),
                    constrained_director_idx=(0,),
                )

        # for _, rod in enumerate(self.straight_rod_list):
        #     self.simulator.constrain(rod).using(
        #         OneEndFixedRod,  # FixNodePosition,
        #         constrained_position_idx=(0,),
        #         constrained_director_idx=(0,),
        #     )

        for _, rod in enumerate(self.helical_rod_list):
            self.simulator.constrain(rod).using(
                OneEndFixedRod,  # FixNodePosition,
                constrained_position_idx=(0,),
                constrained_director_idx=(0,),
            )

        if not len(self.ring_rod_list) == 0:
            self.simulator.constrain(self.ring_rod_list[0]).using(
                ConstrainRingPositionDirectors,
                fixed_position=self.ring_rod_list[0].position_collection.copy(),
                fixed_directors=self.ring_rod_list[0].director_collection.copy(),
            )

        for _, rod in enumerate(self.straight_rod_list):
            self.simulator.constrain(rod).using(
                DampingFilterBC,
                constrained_position_idx=(0,),
                constrained_director_idx=(0,),
                filter_order=5,  # 10,
            )
        for _, rod in enumerate(self.ring_rod_list):
            self.simulator.constrain(rod).using(
                DampingFilterBCRingRod,  # DampingFilterBC,
                constrained_position_idx=(0,),
                constrained_director_idx=(0,),
                filter_order=5,  # 10,
            )
            # self.simulator.constrain(rod).using(
            #     ExponentialDampingBC,
            #     constrained_position_idx=(0,),
            #     constrained_director_idx=(0,),
            #     time_step = self.time_step,
            #     nu = nu_ring_rod_outer,
            # )
        for _, rod in enumerate(self.helical_rod_list):
            self.simulator.constrain(rod).using(
                DampingFilterBC,
                constrained_position_idx=(0,),
                constrained_director_idx=(0,),
                filter_order=5,  # 10,
            )

        class RigidCylinderGlueBC(NoForces):
            def __init__(
                self,
                ramp_up_time,
                ramp_interval,
                ramp_down_time,
                k_spring,
                target_direction,
                target_position,
            ):
                self.ramp_up_time = ramp_up_time
                self.ramp_interval = ramp_interval
                self.ramp_down_time = ramp_down_time
                self.k_spring = k_spring
                self.target_direction = target_direction
                self.target_position = target_position

            def apply_forces(self, system, time: np.float64 = 0.0):
                factor = 0.0

                if (time - self.ramp_up_time) <= 0:
                    factor = 0.0
                elif (time - self.ramp_up_time) > 0 and (
                    time - self.ramp_up_time
                ) <= self.ramp_interval:
                    factor = (
                        1
                        + np.sin(
                            np.pi * (time - self.ramp_up_time) / self.ramp_interval
                            - np.pi / 2
                        )
                    ) / 2
                elif (time - self.ramp_up_time) > 0 and (
                    time - self.ramp_down_time
                ) < 0:
                    factor = 1.0

                elif (time - self.ramp_down_time) > 0 and (
                    time - self.ramp_down_time
                ) / self.ramp_interval < 1.0:
                    factor = (
                        1
                        - (
                            1
                            + np.sin(
                                np.pi
                                * (time - self.ramp_down_time)
                                / self.ramp_interval
                                - np.pi / 2
                            )
                        )
                        / 2
                    )

                k_effective = self.k_spring * factor

                system.external_forces[:, 0] += k_effective * (
                    self.target_position - system.position_collection[:, 0]
                )

                moment_arm = system.length / 2
                current_direction = system.director_collection[2, :, 0]
                # Forcing is defined between current to target position. Since we only one node we can simplify
                # difference as the difference between directions.
                torque_force = k_effective * (
                    moment_arm * (self.target_direction - current_direction)
                )
                torque = moment_arm * np.cross(current_direction, torque_force)
                system.external_torques[:, 0] += (
                    system.director_collection[..., 0] @ torque
                )

        class ExponentialDampingBCRigidBodies(FreeRod):
            """
            Damping filter.
            TODO expand docs if working
                Attributes
                ----------
                filter_order: int
                    Order of the filter.
            """

            def __init__(
                self, fixed_position=(0,), fixed_directors=(0,), nu=0, time_step=0
            ):
                """
                Damping Filter initializer
                Parameters
                ----------
                filter_order: int
                    Order of the filter.
                """
                self.nu_dt = nu * time_step

            def constrain_rates(self, rod, time):
                rod.velocity_collection[:] = rod.velocity_collection * np.exp(
                    -self.nu_dt
                )

                rod.omega_collection[:] = rod.omega_collection * np.exp(
                    -self.nu_dt
                    * np.diagonal(rod.inv_mass_second_moment_of_inertia).T
                    * rod.mass
                )

        for _, rod in enumerate(self.cylinder_list):
            #            self.simulator.constrain(rod).using(
            #                ExponentialDampingBCRigidBodies,
            #                nu = 50,
            #                time_step = self.time_step
            #            )
            #            self.simulator.add_forcing_to(rod).using(
            #                RigidCylinderGlueBC,
            #                ramp_up_time=0,
            #                ramp_interval=1.0,
            #                ramp_down_time=7.0,
            #                k_spring=1e5,
            #                target_direction=rod.director_collection[2, :, 0].copy(),
            #                target_position=rod.position_collection[:, 0].copy(),
            #            )
            self.simulator.constrain(rod).using(
                OneEndFixedRod,  # FixNodePosition,
                constrained_position_idx=(0,),
                constrained_director_idx=(0,),
            )

        # Control muscle forces
        for idx, rod in enumerate(self.straight_rod_list):
            # Apply Forces
            self.simulator.add_forcing_to(rod).using(
                ActivationRampUpRampDown,
                ramp_up_time=0.0,
                ramp_down_time=220.0,
                ramp_interval=0.01,
                activation_level=0.001,
            )
            #            if idx == 1 or idx == 2 or idx == 8:
            if idx == 4 or idx == 5 or idx == 6:
                #                self.simulator.add_forcing_to(rod).using(SigmoidActivationLongitudinalMuscles, start_time=0,
                #                                                     end_time=0 + 50 * 0.01, start_idx=0, end_idx=180, beta=1, tau=0.01,
                #                                                     activation_level_max=1.0, activation_level_end=1.0)
                #                self.simulator.add_forcing_to(rod).using(
                #                    LocalActivation, ramp_interval = 1.0, ramp_up_time=0.0, ramp_down_time = 5.0, start_idx = 0, end_idx = 150, activation_level=0.25,
                #                )
                self.simulator.add_forcing_to(rod).using(
                    SigmoidActivationLongitudinalMuscles,
                    start_time=1,
                    end_time=15,
                    start_idx=0,
                    end_idx=170,
                    beta=1,
                    tau=0.02,
                    activation_level_max=0.3,
                    activation_level_end=0.3,
                )
        #            if idx == 1 or idx == 2 or idx == 8:
        #                self.simulator.add_forcing_to(rod).using(
        #                     ActivationRampUpRampDown,
        #                     ramp_up_time=1.5,
        #                     ramp_down_time=220.0,
        #                     ramp_interval=4.0,
        #                     activation_level=0.08,
        #                 )

        # for idx, rod in enumerate(self.ring_rod_list_outer[:]):
        #     self.simulator.add_forcing_to(rod).using(
        #        ActivationRampUpRampDown,
        #        ramp_up_time=0.0,
        #        ramp_down_time=150.0,
        #        ramp_interval=0.01,
        #        activation_level=0.001,
        #     )
        #        for idx, rod in enumerate(self.ring_rod_list_outer[:170]):
        #            self.simulator.add_forcing_to(rod).using(
        #               ActivationRampUpRampDown,
        #               ramp_up_time=23.5,
        #               ramp_down_time=150.0,
        #               ramp_interval=1.0,
        #               activation_level=0.4 * (170 - idx) / 170,
        #            )
        #         for idx, rod in enumerate(self.ring_rod_list_outer[:60]):
        #             self.simulator.add_forcing_to(rod).using(
        #                 ActivationRampUpRampDown,
        #                 ramp_up_time=0.0,
        #                 ramp_down_time=150.0,
        #                 ramp_interval=1.0,
        #                 activation_level=0.1,
        #             )
        #        for idx, rod in enumerate(self.ring_rod_list_outer[60:120]):
        #            self.simulator.add_forcing_to(rod).using(
        #                ActivationRampUpRampDown,
        #                ramp_up_time=0.0,
        #                ramp_down_time=150.0,
        #                ramp_interval=1.0,
        #                activation_level=0.15,
        #            )
        for idx, rod in enumerate(self.ring_rod_list_outer[:]):
            self.simulator.add_forcing_to(rod).using(
                ActivationRampUpRampDown,
                ramp_up_time=0.0,
                ramp_down_time=150.0,
                ramp_interval=0.01,
                activation_level=0.001,
            )
        #         for idx, rod in enumerate(self.ring_rod_list[:1]):
        #             if idx < 60:
        #                 self.simulator.add_forcing_to(rod).using(SigmoidActivationTransverseMuscles, rod_idx = idx, start_time = 0.0, end_time=15, beta=1, tau=0.1, activation_level_max=0.01, activation_level_end=0.01) #1.0
        #             # elif idx>=60 and idx <=120:
        #             #     self.simulator.add_forcing_to(rod).using(SigmoidActivationTransverseMuscles, rod_idx = idx, start_time = 0.0, end_time=15, beta=1, tau=0.01, activation_level_max=0.2, activation_level_end=0.20) #0.5
        #             # else:
        #             #     self.simulator.add_forcing_to(rod).using(SigmoidActivationTransverseMuscles, rod_idx = idx, start_time = 0.0, end_time=15, beta=1, tau=0.01, activation_level_max=0.03, activation_level_end=0.03) #0.5

        # for idx, rod in enumerate(self.helical_rod_list[:]):
        #     # Apply Forces
        #     self.simulator.add_forcing_to(rod).using(
        #         ActivationRampUpRampDown,
        #         ramp_up_time=0.0,
        #         ramp_down_time=22.0,
        #         ramp_interval=1.0,
        #         activation_level=1.0,#0.001,
        #     )
        #        for idx, rod in enumerate(self.helical_rod_list[4:]):
        #            # Apply Forces
        #            self.simulator.add_forcing_to(rod).using(
        #                ActivationRampUpRampDown,
        #                ramp_up_time=0.0,
        #                ramp_down_time=220.0,
        #                ramp_interval=0.1,
        #                activation_level=0.001,
        #            )
        #
        #        for idx, rod in enumerate(self.helical_rod_list[4:]):
        #            self.simulator.add_forcing_to(rod).using(LocalActivation, ramp_interval = 0.5, ramp_up_time=0.0, ramp_down_time = 3.0, start_idx = 0, end_idx =359, activation_level=0.05,)
        #
        #        for idx, rod in enumerate(self.helical_rod_list[:4]):
        #            self.simulator.add_forcing_to(rod).using(LocalActivation, ramp_interval = 1.0, ramp_up_time=5.0, ramp_down_time = 120.5, start_idx = 0, end_idx =359, activation_level=0.05,)
        for idx, rod in enumerate(self.helical_rod_list[4:]):
            self.simulator.add_forcing_to(rod).using(
                LocalActivation,
                ramp_interval=0.5,
                ramp_up_time=0.0,
                ramp_down_time=8.0,
                start_idx=0,
                end_idx=150,
                activation_level=0.20,
            )
        #            self.simulator.add_forcing_to(rod).using(
        #                    LocalActivation, ramp_interval = 0.5, ramp_up_time=10.0, ramp_down_time = 15.0, start_idx = 300, end_idx = 356, activation_level=0.05,
        #                )

        # Connect straight rods with a surface offset
        k_connection_btw_straight_straight = (
            arm_length / n_elem * E
        )  # (r1+r2)*l1 /(r1+r2) * E
        nu_connection_btw_straight_straight = (
            nu
            * (
                density
                * np.pi
                * outer_straight_rod_radius_along_arm
                * arm_length
                / n_elem
            )
            / n_elem
        )

        self.straight_straight_rod_connection_list = []
        # max_offset_btw_straight_rods = base_radius_inner_ring_rod[0] + 1e-4
        max_offset_btw_straight_rods = center_straight_rod_radius + 1e-4

        for i, rod_one in enumerate(self.straight_rod_list[:]):
            for j in range(i + 1, len(self.straight_rod_list[:])):
                rod_two = self.straight_rod_list[j]

                # Compute the distance between rod bases and if it is smaller than tolerance connect.
                distance_btw_rods = np.linalg.norm(
                    rod_one.position_collection[..., 0]
                    - rod_two.position_collection[..., 0]
                ) - (rod_one.radius[0] + rod_two.radius[0])
                if distance_btw_rods > max_offset_btw_straight_rods:
                    continue

                if not i == j:
                    (
                        rod_one_direction_vec_in_material_frame,
                        rod_two_direction_vec_in_material_frame,
                        offset_btw_rods,
                    ) = get_connection_vector_straight_straight_rod(rod_one, rod_two)

                    if (
                        offset_btw_rods[0] <= max_offset_btw_straight_rods
                    ):  # just check the base
                        self.straight_straight_rod_connection_list.append(
                            [
                                rod_one,
                                rod_two,
                                rod_one_direction_vec_in_material_frame.copy(),
                                rod_two_direction_vec_in_material_frame.copy(),
                                offset_btw_rods.copy(),
                                i,
                                j,
                            ]
                        )

        self.straight_rod_total_contact_force = np.zeros(
            (3, len(self.straight_rod_list) * n_elem)
        )
        self.straight_rod_total_contact_force_mag = np.zeros(
            len(self.straight_rod_list) * n_elem
        )
        self.straight_straight_rod_connection_post_processing_dict = defaultdict(list)

        for i, my_connection in enumerate(self.straight_straight_rod_connection_list):
            rod_one = my_connection[0]
            rod_two = my_connection[1]
            rod_one_direction_vec_in_material_frame = my_connection[2]
            rod_two_direction_vec_in_material_frame = my_connection[3]
            offset_btw_rods = my_connection[4]
            rod_one_list_idx = my_connection[5]
            rod_two_list_idx = my_connection[6]

            assert (
                rod_one.n_elems == rod_two.n_elems
            ), "number of elements are not same. Change the number of element for these rods."

            for k in range(rod_one.n_elems):
                k_conn = (
                    rod_one.radius[k]
                    * rod_two.radius[k]
                    / (rod_one.radius[k] + rod_two.radius[k])
                    * arm_length
                    / n_elem
                    * E
                    / (rod_one.radius[k] + rod_two.radius[k])
                )
                self.simulator.connect(
                    first_rod=rod_one,
                    second_rod=rod_two,
                    first_connect_idx=k,
                    second_connect_idx=k,
                ).using(
                    SurfaceJointSideBySide,
                    #                    k=k_connection_btw_straight_straight *self.k_straight_straight_connection_spring_scale,
                    k=k_conn * self.k_straight_straight_connection_spring_scale,
                    # / 2
                    # * 12,  # * 2,  # * 2#  / 2 #/30  # *4, # for extension
                    nu=nu_connection_btw_straight_straight[k] * 0,
                    #                    k_repulsive=k_connection_btw_straight_straight*self.k_straight_straight_connection_contact_scale,
                    k_repulsive=k_conn
                    * self.k_straight_straight_connection_contact_scale,
                    # / 50
                    # * 100e4
                    # / 1e4,  # * 4,  # *2,#*2,#*5*2,#/30*1E6,
                    rod_one_direction_vec_in_material_frame=rod_one_direction_vec_in_material_frame[
                        ..., k
                    ],
                    rod_two_direction_vec_in_material_frame=rod_two_direction_vec_in_material_frame[
                        ..., k
                    ],
                    offset_btw_rods=offset_btw_rods[k],
                    contact_force_rod_one_idx=rod_one_list_idx * n_elem + k,
                    contact_force_rod_two_idx=rod_two_list_idx * n_elem + k,
                    total_contact_force=self.straight_rod_total_contact_force,
                    total_contact_force_mag=self.straight_rod_total_contact_force_mag,
                    post_processing_dict=self.straight_straight_rod_connection_post_processing_dict,
                    step_skip=self.step_skip,
                )

        # Connect neighbor outer ring rods with each other.
        # outer_ring_ring_rod_offset_distance_list is empty list, but
        # they are filled later on after finalize call.
        number_of_straight_rods_except_center = number_of_straight_rods - 1

        self.outer_ring_ring_rod_connection_offset_start_idx = np.zeros(
            ((total_number_of_ring_rods - 1) * number_of_straight_rods_except_center),
            dtype=np.int,
        )
        self.outer_ring_ring_rod_connection_offset_end_idx = np.zeros(
            ((total_number_of_ring_rods - 1) * number_of_straight_rods_except_center),
            dtype=np.int,
        )

        for idx in range(len(self.ring_rod_list_outer) - 1):
            rod_one = self.ring_rod_list_outer[idx]
            rod_two = self.ring_rod_list_outer[idx + 1]
            k_connection_btw_ring_ring = (
                2
                * rod_two.radius[0]
                * rod_two.rest_lengths[0]
                * n_elem_ring_rod
                / self.n_connections
                * E
                / (arm_length / n_elem)
                * self.k_ring_ring_spring_connection_scale
            )  # * 2 * 2

            # Reference index list is used to compute a vector pointing from rod one to rod two.
            (
                index_connection,
                index_connection_opposite,
                index_reference,
                index_reference_opposite,
            ) = get_ring_ring_connection_reference_index(
                rod_one, rod_two, number_of_straight_rods_except_center
            )
            k_connection_btw_ring_ring = k_connection_btw_ring_ring * np.ones(
                index_connection.shape
            )

            self.simulator.connect(rod_one, rod_two).using(
                OuterRingRingRodConnectionDifferentLevel,
                k=k_connection_btw_ring_ring,
                index_connection=index_connection,
                index_connection_opposite=index_connection_opposite,
                index_reference=index_reference,
                index_reference_opposite=index_reference_opposite,
                offset_start_idx=self.outer_ring_ring_rod_connection_offset_start_idx,
                offset_end_idx=self.outer_ring_ring_rod_connection_offset_end_idx,
            )

        # Connect ring and helical rods
        self.ring_helical_rod_connection_list = []
        for i, my_helical_rod in enumerate(self.helical_rod_list):
            if len(self.ring_rod_list) == 0:
                # If there is no ring rods don't do anything.
                continue

            connection_idx_helical_rod = np.zeros(
                (total_number_of_ring_rods), dtype=np.int
            )

            helical_rod_position = np.einsum(
                "ij, i->j", my_helical_rod.position_collection, direction
            )

            for idx, rod in enumerate(self.ring_rod_list_outer):
                center_position_ring_rod = np.mean(rod.position_collection, axis=1)
                center_position_ring_rod_along_direction = np.dot(
                    center_position_ring_rod, direction
                )
                connection_idx_helical_rod[idx] = np.argmin(
                    np.abs(
                        helical_rod_position - center_position_ring_rod_along_direction
                    )
                )

                ring_rod_radius = np.mean(rod.radius)
                ring_rod_radius_length = np.mean(rod.rest_lengths)

                # Compute spring and damping constants at the connections
                k_connection_btw_helix_and_ring = (
                    np.pi * ring_rod_radius * E / 30
                )  # total_number_of_ring_rods
                nu_connection_btw_helix_and_ring = (
                    nu
                    * (density * np.pi * ring_rod_radius * ring_rod_radius_length)
                    # / 10
                    # / 5
                    * 0
                )

                position = my_helical_rod.position_collection[
                    ..., connection_idx_helical_rod[idx]
                ]

                distance = position.reshape(3, 1) - rod.position_collection
                distance_norm = _batch_norm(distance)

                # Check the distance between helical rod and ring rod; if distance is smaller than tolerance, rods are
                # connected.
                if (
                    np.abs(
                        np.min(distance_norm)
                        - (
                            ring_rod_radius
                            + my_helical_rod.radius[connection_idx_helical_rod[idx]]
                        )
                    )
                    > 1e-12
                ):
                    continue

                connection_idx_ring_rod = np.argmin(distance_norm)

                self.ring_helical_rod_connection_list.append(
                    [
                        my_helical_rod,
                        rod,
                        connection_idx_helical_rod[idx],
                        connection_idx_ring_rod,
                        k_connection_btw_helix_and_ring,
                        nu_connection_btw_helix_and_ring,
                    ]
                )

        for idx, my_connection in enumerate(self.ring_helical_rod_connection_list):
            rod_one = my_connection[0]  # helical rod
            rod_two = my_connection[1]  # ring rod
            index_one = my_connection[2]
            index_two = my_connection[3]

            k_connection = my_connection[4]
            nu_connection = my_connection[5]
            #            k_connection = (2*rod_one.radius[index_one]*rod_one.rest_lengths[index_one])*E/(rod_two.radius[index_two]+rod_one.radius[index_one])/15
            k_connection = (
                (2 * rod_two.radius[index_two] * rod_two.rest_lengths[index_two])
                * E
                / (rod_two.radius[index_two] + rod_one.radius[index_one])
                * self.k_ring_helical_spring_connection_scale
            )

            (
                connection_order,
                index_two_opposite_side,
                index_two_hing_side,
                index_two_hinge_opposite_side,
                next_connection_index,
                next_connection_index_opposite,
            ) = get_connection_order_and_angle_btw_helical_and_ring_rods(
                rod_one, rod_two, index_one, index_two, direction
            )

            self.simulator.connect(
                first_rod=rod_one,
                second_rod=rod_two,
                first_connect_idx=index_one,
                second_connect_idx=index_two,
            ).using(
                RingHelicalRodJoint,
                k=k_connection,
                nu=nu_connection,
                connection_order=connection_order,
                index_two_opposite_side=index_two_opposite_side,
                index_two_hing_side=index_two_hing_side,
                index_two_hinge_opposite_side=index_two_hinge_opposite_side,
                next_connection_index=next_connection_index,
                next_connection_index_opposite=next_connection_index_opposite,
                ring_rod_start_idx=0,
                ring_rod_end_idx=rod_two.n_elems,
            )
            k_ring_helical_contact = (
                k_connection * self.k_ring_helical_contact_connection_scale
            )
            self.simulator.connect(
                first_rod=rod_one,
                second_rod=rod_two,
                first_connect_idx=index_one,
                second_connect_idx=index_two,
            ).using(
                RingHelicalRodContact,
                k=k_ring_helical_contact,
                nu=nu_connection,
            )

            # Compute the tangent directions of both helical rod and ring rod for the index_one and index_two.
            # If tangent directions are pointing same direction  then both of the rod elements numbers are increasing in
            # same direction. You can think as helical rod and ring are both rotating in counter clockwise direction.
            # Otherwise if tangent directions are pointing opposite direction then ring rod elements and helical rod
            # elements are increasing in different directions. You can think as helical rod rotating clockwise and
            # ring rod rotating counter clockwise.
            # Depending on rotation of helical rod, possible nodes of helical rod that can go into contact with ring
            # can change. For simplicity we will assume ring elements are always increases in counter clockwise
            # direction. However, you can change current setup of ring rods and following code should still work.
            # First compute tangent direction of ring rod
            ring_rod_tangent_direction = (
                rod_two.position_collection[..., index_two + 1]
                - rod_two.position_collection[..., index_two]
            )
            ring_rod_tangent_direction /= np.linalg.norm(ring_rod_tangent_direction)
            # Compute helical rod tangent direction. Helical rod height increases between its nodes, but we want to
            # compute in plane tangent direction of helical rod element, so we will project it to in plane.
            helical_rod_tangent_direction = (
                rod_one.position_collection[..., index_one + 1]
                - rod_one.position_collection[..., index_one]
            )
            helical_rod_tangent_direction_in_plane = (
                helical_rod_tangent_direction
                - np.dot(helical_rod_tangent_direction, direction) * direction
            )
            helical_rod_tangent_direction_in_plane /= np.linalg.norm(
                helical_rod_tangent_direction_in_plane
            )
            # Compute dot product of ring and helical rod tangents. If number is positive than both rods rotate in
            # same direction, otherwise they rotate in opposite direction.
            ring_helical_rod_rotate_same_direction = (
                True
                if np.dot(
                    ring_rod_tangent_direction, helical_rod_tangent_direction_in_plane
                )
                > 0
                else False
            )

            # Extra contact between ring rod elements and helical rod element.
            # Number of elements of ring rods that is possible to contact with helical rods.
            n_elem_possible_contact = 4
            for idx in range(n_elem_possible_contact):
                if ring_helical_rod_rotate_same_direction:
                    # Both ring and helical rod elements are increasing in same direction.
                    if idx < n_elem_possible_contact / 2:
                        # index right of index one
                        index_one_possible_contact = index_one + 1
                        if index_one == rod_one.n_elems - 1:
                            # Last node of helical rod does not have a neighbor at index_one + 1, so continue
                            continue
                        # indexes right of index two
                        index_two_possible_contact = index_two + idx + 1

                    else:
                        # index left of index one
                        index_one_possible_contact = index_one - 1
                        if index_one == 0:
                            # First node of helical rod does not have a neighbor at index_one-1, so continue
                            continue
                        # indexes left of index two
                        index_two_possible_contact = index_two - (
                            idx + 1 - n_elem_possible_contact / 2
                        )

                else:
                    # Ring and helical rod elements are increasing in opposite direction.
                    if idx < n_elem_possible_contact / 2:
                        # index left of index one
                        index_one_possible_contact = index_one - 1
                        if index_one == 0:
                            # First node of helical rod does not have a neighbor at index_one-1, so continue
                            continue
                        # indexes right of index two
                        index_two_possible_contact = index_two + idx + 1
                    else:
                        # index right of index one
                        index_one_possible_contact = index_one + 1
                        if index_one == rod_one.n_elems - 1:
                            # Last node of helical rod does not have a neighbor at index_one + 1, so continue
                            continue
                        # indexes left of index two
                        index_two_possible_contact = index_two - (
                            idx + 1 - n_elem_possible_contact / 2
                        )

                # take the mod, to make sure index is between 0 and n_elem_ring_rod
                index_two_possible_contact = int(
                    index_two_possible_contact % n_elem_ring_rod
                )

                self.simulator.connect(
                    first_rod=rod_one,
                    second_rod=rod_two,
                    first_connect_idx=index_one_possible_contact,
                    second_connect_idx=index_two_possible_contact,
                ).using(
                    RingHelicalRodContact,
                    k=k_ring_helical_contact,
                    nu=nu_connection,
                )

        # Connect straight and ring rods
        self.ring_straight_rod_connection_list = []
        self.surface_pressure_force_list = []
        self.surface_pressure_force_idx = 0

        for i, my_straight_rod in enumerate(self.straight_rod_list):
            if len(self.ring_rod_list) == 0:
                # If there is no ring rods don't do anything.
                continue

            connection_idx_straight_rod = np.zeros(
                (total_number_of_ring_rods), dtype=np.int
            )
            connection_pressure_mag_scale = np.zeros((total_number_of_ring_rods))
            direction_list = self.ring_straight_rod_connection_direction_list[i]

            straight_rod_element_position = 0.5 * (
                my_straight_rod.position_collection[..., 1:]
                + my_straight_rod.position_collection[..., :-1]
            )
            straight_rod_position = np.einsum(
                "ij, i->j", straight_rod_element_position, direction
            )

            for idx, rod in enumerate(self.ring_rod_list_outer):
                center_position_ring_rod = np.mean(rod.position_collection, axis=1)
                center_position_ring_rod_along_direction = np.dot(
                    center_position_ring_rod, direction
                )
                connection_idx_straight_rod[idx] = np.argmin(
                    np.abs(
                        straight_rod_position - center_position_ring_rod_along_direction
                    )
                )
                ring_rod_radius = np.mean(rod.radius)
                ring_rod_length = np.mean(rod.rest_lengths)

                # Compute spring and damping constants at the connections
                k_connection_btw_straight_ring = (
                    2
                    * ring_rod_radius
                    * ring_rod_length
                    * n_elem_ring_rod
                    / self.n_connections
                    * E
                    / (
                        ring_rod_radius
                        + my_straight_rod.radius[connection_idx_straight_rod[idx]]
                    )
                )
                nu_connection_btw_straight_ring = (
                    nu
                    * (density * np.pi * ring_rod_radius * ring_rod_length)
                    # / 10
                    # / 5
                    / 90
                )

                for _, my_direction in enumerate(direction_list):
                    position = straight_rod_element_position[
                        ..., connection_idx_straight_rod[idx]
                    ] + my_direction * (
                        my_straight_rod.radius[connection_idx_straight_rod[idx]]
                        + ring_rod_radius
                    )

                    distance = position.reshape(3, 1) - rod.position_collection
                    distance_norm = _batch_norm(distance)

                    if np.min(distance_norm) > 1e-5:
                        continue

                    connection_idx_ring_rod = np.argmin(distance_norm)

                    self.ring_straight_rod_connection_list.append(
                        [
                            my_straight_rod,
                            rod,
                            connection_idx_straight_rod[idx],
                            connection_idx_ring_rod,
                            k_connection_btw_straight_ring,
                            nu_connection_btw_straight_ring,
                            len(direction_list),
                        ]
                    )
                    connection_pressure_mag_scale[idx] += 1

            self.surface_pressure_force_list.append(
                [
                    np.array(
                        connection_idx_straight_rod.copy()
                        + self.surface_pressure_force_idx
                    ).copy(),
                    # np.zeros((total_number_of_ring_rods)),
                    connection_idx_straight_rod.copy(),
                    connection_pressure_mag_scale.copy(),
                ]
            )
            self.surface_pressure_force_idx += connection_idx_straight_rod[-1] + 1

        # This is the block, we save the surface pressure for all straight rods.
        self.surface_pressure_array = np.zeros((len(self.straight_rod_list) * n_elem))
        self.straight_ring_rod_total_contact_force = np.zeros(
            (3, len(self.straight_rod_list) * n_elem)
        )
        self.straight_ring_rod_total_contact_force_mag = np.zeros(
            (len(self.straight_rod_list) * n_elem)
        )
        self.ring_straight_rod_connection_post_processing_dict = defaultdict(list)

        for idx, my_connection in enumerate(self.ring_straight_rod_connection_list):
            rod_one = my_connection[0]  # straight rod
            rod_two = my_connection[1]  # ring rod
            index_one = my_connection[2]
            index_two = my_connection[3]

            # k_connection = my_connection[4]
            # nu_connection = my_connection[5]
            n_connection = my_connection[6]

            k_connection = (
                2
                * rod_two.radius[index_two]
                * rod_two.rest_lengths[index_two]
                * rod_two.n_elems
                / n_connection
                * E
                / (rod_two.radius[index_two] + rod_one.radius[index_one])
                * self.k_ring_straight_spring_connection_scale
            )
            k_repulsive = k_connection * self.k_ring_straight_contact_connection_scale
            kt_connection = (
                rod_two.rest_lengths[index_two]
                * rod_two.n_elems
                / n_connection
                * E
                / 480
                * self.k_ring_straight_spring_torque_connection_scale
            )
            nu_connection = (
                nu
                * (
                    density
                    * np.pi
                    * rod_two.radius[index_two]
                    * rod_two.lengths[index_two]
                )
                / 90
            )

            (
                connection_order,
                angle_btw_straight_ring_rods,
                index_two_opposite_side,
                index_two_hing_side,
                index_two_hinge_opposite_side,
            ) = get_connection_order_and_angle(
                rod_one, rod_two, index_one, index_two, direction
            )

            for list_idx, my_straight_rod in enumerate(self.straight_rod_list):
                if my_connection[0] == my_straight_rod:  #
                    # if my_connection[1] == my_straight_rod:
                    my_surface_pressure_idx = np.where(
                        self.surface_pressure_force_list[list_idx][1]
                        == index_one  # index_two
                    )[0][0]
                    # surface_force_rod_two = np.ndarray.view(
                    #     self.surface_pressure_force_list[list_idx][0][
                    #         my_surface_pressure_idx : my_surface_pressure_idx + 1
                    #     ]
                    # )
                    surface_pressure_idx = self.surface_pressure_force_list[list_idx][
                        0
                    ][my_surface_pressure_idx]
                    scale = self.surface_pressure_force_list[list_idx][-1][0]

            self.simulator.connect(
                first_rod=rod_one,
                second_rod=rod_two,
                first_connect_idx=index_one,
                second_connect_idx=index_two,
            ).using(
                OrthogonalRodsSideBySideJoint,
                k=k_connection,  # / 10 / 2,#/8/2,  # /48,#/8,#/12,  # * 8 * 2,
                nu=nu_connection,
                kt=kt_connection,  # * 2,  # *2,# / 2,
                k_repulsive=k_repulsive * 0.0,  # * 5,# *5,
                surface_pressure_idx=surface_pressure_idx,
                surface_pressure=self.surface_pressure_array,
                connection_order=connection_order,
                angle_btw_straight_ring_rods=angle_btw_straight_ring_rods,
                index_two_opposite_side=index_two_opposite_side,
                index_two_hing_side=index_two_hing_side,
                index_two_hinge_opposite_side=index_two_hinge_opposite_side,
                post_processing_dict=self.ring_straight_rod_connection_post_processing_dict,
                step_skip=self.step_skip,
                n_connection=n_connection,
                total_contact_force=self.straight_rod_total_contact_force,
                total_contact_force_mag=self.straight_rod_total_contact_force_mag,
            )

            self.simulator.connect(
                first_rod=rod_one,
                second_rod=rod_two,
                first_connect_idx=index_one,
                second_connect_idx=index_two,
            ).using(
                OrthogonalRodsSideBySideContact,
                k=k_repulsive,  # / 10 / 2,#/8/2,  # /48,#/8,#/12,  # * 8 * 2,
                nu=nu_connection,
                surface_pressure_idx=surface_pressure_idx,
                total_contact_force=self.straight_rod_total_contact_force,
                total_contact_force_mag=self.straight_rod_total_contact_force_mag,
            )

            # Extra contact between ring rod elements and straight rod element.
            # Number of elements of ring rods that is possible to contact with straight rods
            n_elem_possible_contact = 8
            for idx in range(n_elem_possible_contact):
                if idx < n_elem_possible_contact / 2:
                    # indexes left of index two
                    index_two_possible_contact = index_two + idx + 1
                else:
                    # indexes right of index two
                    index_two_possible_contact = index_two - (
                        idx + 1 - n_elem_possible_contact / 2
                    )

                # take the mod, to make sure index is between 0 and n_elem_ring_rod
                index_two_possible_contact = int(
                    index_two_possible_contact % n_elem_ring_rod
                )

                self.simulator.connect(
                    first_rod=rod_one,
                    second_rod=rod_two,
                    first_connect_idx=index_one,
                    second_connect_idx=index_two_possible_contact,
                ).using(
                    OrthogonalRodsSideBySideContact,
                    k=k_repulsive,  # / 10 / 2,#/8/2,  # /48,#/8,#/12,  # * 8 * 2,
                    nu=nu_connection,
                    surface_pressure_idx=surface_pressure_idx,
                    total_contact_force=self.straight_rod_total_contact_force,
                    total_contact_force_mag=self.straight_rod_total_contact_force_mag,
                )

        # Set pressure forces
        # Scale the pressure forces with number of connections. If one rod have 4 connections, and other have 2
        # both have to have same pressure.
        self.post_processing_list_for_pressure_force_dicts = []
        for idx, straight_rod in enumerate(self.straight_rod_list):
            n_elem = straight_rod.n_elems

            start_idx = idx * n_elem
            end_idx = (idx + 1) * n_elem
            # control_points_idx = my_surface_pressure_list[1]
            total_contact_force = np.ndarray.view(
                self.straight_rod_total_contact_force[:, start_idx:end_idx]
            )
            total_contact_force_mag = np.ndarray.view(
                self.straight_rod_total_contact_force_mag[start_idx:end_idx]
            )

            self.post_processing_list_for_pressure_force_dicts.append(defaultdict(list))
            self.simulator.add_forcing_to(self.straight_rod_list[idx]).using(
                PressureForce,
                total_contact_force=total_contact_force,
                total_contact_force_mag=total_contact_force_mag,
                step_skip=self.step_skip,
                pressure_profile_recorder=self.post_processing_list_for_pressure_force_dicts[
                    idx
                ],
            )

        # Drag force
        rho_water = 1.050e-3  # g/mm3
        factor = 1  # 5.0 # FIXME: We shouldnt need this scaling!
        cd_perpendicular = 1.013 * factor
        cd_tangent = 0.0256 * factor
        for idx, rod in enumerate(
            self.straight_rod_list[1:]
        ):  # Do not include axial nerve cord
            self.simulator.add_forcing_to(rod).using(
                DragForceOnStraightRods,
                cd_perpendicular=cd_perpendicular,
                cd_tangent=cd_tangent,
                rho_water=rho_water,
                start_time=0.0,
            )

        # Cylinder rod contact
        # Add cylinder and rod contact.
        for idx_cylinder, cylinder in enumerate(self.cylinder_list):
            for idx, rod in enumerate(
                self.helical_rod_list
                # self.straight_rod_list[1:2]
            ):  # (self.straight_rod_list[1:]):
                self.simulator.connect(rod, cylinder).using(
                    ExternalContactForMemoryBlock, k=1e2, nu=1.0  # 0.3
                )
                # self.simulator.connect(rod, cylinder).using(
                #    ExternalContactWithFrictionForMemoryBlock,
                #    k=1e2,
                #    nu=1.0,
                #    velocity_damping_coefficient=1e5,
                #    friction_coefficient=0.1,
                # )

        # Sucker forces and activations
        self.activation_array_block = np.zeros(
            (len(self.helical_rod_list), n_elem_helical_rod)
        )

        class SigmoidSuckerActivation(NoForces):
            def __init__(
                self,
                beta,
                tau,
                start_time,
                end_time,
                start_idx,
                end_idx,
                activation_array,
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
                self.activation_array = activation_array

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
                                self.beta
                                * ((time - self.start_time) / self.tau - index + 0)
                            )
                        )
                    ) + (
                        -(self.activation_level_max - self.activation_level_end)
                        * (
                            0.5
                            * (
                                1
                                + np.tanh(
                                    self.beta
                                    * ((time - self.end_time) / self.tau - index + 0)
                                )
                            )
                        )
                    )
                active_index = np.where(
                    fiber_activation > self.activation_lower_threshold
                )[0]
                self.activation_array[self.start_idx + active_index] = fiber_activation[
                    active_index
                ]

        #        self.sucker_activation_array_list = []
        #        for idx, rod in enumerate(self.helical_rod_list[:]):
        #
        #            self.sucker_activation_array_list.append( np.ndarray.view(self.activation_array_block[idx,:]) )
        #
        #            self.simulator.add_forcing_to(rod).using(SigmoidSuckerActivation, start_time=5,
        #                                                                     activation_array = self.sucker_activation_array_list[idx] ,
        #                                                     end_time=5+100*0.002, start_idx=75, end_idx=350,
        #                                                     beta=1, tau=0.002, activation_level_max=1.0, activation_level_end=1.00,
        #                                                        activation_lower_threshold = 0.0
        #                                                                     )
        #
        ##        direction_of_suckers = self.target_cylinder.position_collection[...,0] - self.shearable_rod1.position_collection[...,0]
        #        direction_of_suckers =  self.straight_rod_list[5].position_collection[...,0] - self.straight_rod_list[0].position_collection[...,0]
        #        direction_of_suckers /= np.linalg.norm(direction_of_suckers)
        #        direction_of_suckers -= np.dot(direction_of_suckers, direction) * direction
        ##        print("direction of suckers : ", direction_of_suckers)
        #
        ##        sucker_activation = np.ndarray.view(self.activation_array_block[1,:])
        #
        #        from elastica._elastica_numba._linalg import _batch_matvec
        #
        #        for idx_cylinder, cylinder in enumerate(self.cylinder_list):
        #            for idx, rod in enumerate(
        #                 self.helical_rod_list
        #                #self.straight_rod_list[1:2]
        #            ):  # (self.straight_rod_list[1:]):
        #
        #                sucker_direction_in_material_frame = np.ones((3, n_elem_helical_rod)) * direction_of_suckers.reshape(3,1)
        #                sucker_direction_in_material_frame = _batch_matvec(rod.director_collection, sucker_direction_in_material_frame)
        #                self.simulator.connect(rod, cylinder).using(
        #                    SuckerForcesConnectForMemoryBlock, k=1E4,  sucker_threshold=0.2, sucker_activation=self.activation_array_block,
        #                    rod_one_sucker_direction_in_material_frame = sucker_direction_in_material_frame,
        #                    static_friction_coefficient=10,
        #                    kinetic_friction_coefficient=10,
        #                )

        # Add call backs
        if self.COLLECT_DATA_FOR_POSTPROCESSING:
            # Collect data using callback function for postprocessing
            self.post_processing_dict_list = []

            # for idx, rod in enumerate(self.rod_list):
            for idx, rod in enumerate(
                self.straight_rod_list
                + self.ring_rod_list_outer
                + self.helical_rod_list
            ):
                self.post_processing_dict_list.append(defaultdict(list))

                if hasattr(rod, "ring_rod_flag"):
                    self.simulator.collect_diagnostics(rod).using(
                        RingRodCallBack,
                        step_skip=self.step_skip,
                        callback_params=self.post_processing_dict_list[idx],
                    )
                else:
                    self.simulator.collect_diagnostics(rod).using(
                        StraightRodCallBack,
                        step_skip=self.step_skip,
                        callback_params=self.post_processing_dict_list[idx],
                    )
            # Add cylinders
            n_rods_in_post_processing_dict = len(self.post_processing_dict_list)
            self.resize_cylinder_elems = 100
            for idx, cylinder in enumerate(self.cylinder_list):
                self.post_processing_dict_list.append(defaultdict(list))
                self.simulator.collect_diagnostics(cylinder).using(
                    RigidCylinderCallBack,
                    step_skip=self.step_skip,
                    callback_params=self.post_processing_dict_list[
                        n_rods_in_post_processing_dict + idx
                    ],
                    resize_cylinder_elems=self.resize_cylinder_elems,
                )

        # Finalize simulation environment. After finalize, you cannot add
        # any forcing, constrain or call back functions
        self.simulator.finalize()

        # Since helix rod is not straight, rest sigma and kappa is not equal to straight rod values. We need to update
        # them
        for _, helix_rod in enumerate(self.helical_rod_list):
            helix_rod.rest_kappa[:] = helix_rod.kappa[:]
            helix_rod.rest_sigma[:] = helix_rod.sigma[:]

        # Following the finalize now lets set the references to the straight position. We have to set these references
        # after the finalize call because memory block containing rods constructed after finalize call. So any
        # references set before breaks.

        my_reference_straight_rod = self.straight_rod_list[0]
        straight_rod_element_position = 0.5 * (
            my_reference_straight_rod.position_collection[..., 1:]
            + my_reference_straight_rod.position_collection[..., :-1]
        )
        straight_rod_position = np.einsum(
            "ij, i->j", straight_rod_element_position, direction
        )

        for idx in range(len(self.ring_rod_list_outer) - 1):
            # First ring rod
            rod_one = self.ring_rod_list_outer[idx]

            center_position_ring_rod = np.mean(rod_one.position_collection, axis=1)
            center_position_ring_rod_along_direction = np.dot(
                center_position_ring_rod, direction
            )
            straight_rod_element_start_idx = np.argmin(
                np.abs(straight_rod_position - center_position_ring_rod_along_direction)
            )

            # second ring rod
            rod_two = self.ring_rod_list_outer[idx + 1]

            center_position_ring_rod = np.mean(rod_two.position_collection, axis=1)
            center_position_ring_rod_along_direction = np.dot(
                center_position_ring_rod, direction
            )
            straight_rod_element_end_idx = np.argmin(
                np.abs(straight_rod_position - center_position_ring_rod_along_direction)
            )

            # This list contains the index of ring rod where it touch straight rod. So later on using this list
            # we will sort offset distance list.
            ring_rod_element_idx_list = list()

            # Discard the first straight rod and create references to the straight rod lengths and fill the list.
            for i, my_straight_rod in enumerate(self.straight_rod_list[1:]):
                system_idx = self.simulator._get_sys_idx_if_valid(my_straight_rod)
                for _, my_memory_block in enumerate(self.simulator._memory_blocks):
                    if (
                        np.where(system_idx == my_memory_block.system_idx_list)[0].size
                        == 1
                    ):
                        memory_block_sys_idx = np.where(
                            system_idx == my_memory_block.system_idx_list
                        )[0]
                        index_offset_for_memory_block = (
                            my_memory_block.start_idx_in_rod_nodes[
                                memory_block_sys_idx
                            ][0]
                        )

                self.outer_ring_ring_rod_connection_offset_start_idx[
                    i + (idx * number_of_straight_rods_except_center)
                ] = (straight_rod_element_start_idx + index_offset_for_memory_block)
                self.outer_ring_ring_rod_connection_offset_end_idx[
                    i + (idx * number_of_straight_rods_except_center)
                ] = (straight_rod_element_end_idx + index_offset_for_memory_block)

                ring_rod_radius = np.mean(rod_one.radius)
                # Find the index where ring and straight rod touch each other. Later on depending on that
                # index we will sort the list containing references to straight rod lengths.
                # We are assuming all inner ring rods have same number of elements and length.
                direction_list = self.ring_straight_rod_connection_direction_list[i + 1]
                for _, my_direction in enumerate(direction_list):
                    position = 0.5 * (
                        my_straight_rod.position_collection[
                            ..., straight_rod_element_start_idx
                        ]
                        + my_straight_rod.position_collection[
                            ..., straight_rod_element_start_idx + 1
                        ]
                    ) + my_direction * (
                        my_straight_rod.radius[straight_rod_element_start_idx]
                        + ring_rod_radius
                    )

                    distance = position.reshape(3, 1) - rod_one.position_collection
                    distance_norm = _batch_norm(distance)

                    if np.min(distance_norm) > ring_rod_radius:
                        continue

                    ring_rod_element_idx_list.append(np.argmin(distance_norm))

            self.outer_ring_ring_rod_connection_offset_start_idx[
                number_of_straight_rods_except_center
                * idx : number_of_straight_rods_except_center
                * (idx + 1)
            ] = [
                x
                for _, x in sorted(
                    zip(
                        ring_rod_element_idx_list,
                        self.outer_ring_ring_rod_connection_offset_start_idx[
                            number_of_straight_rods_except_center
                            * idx : number_of_straight_rods_except_center
                            * (idx + 1)
                        ],
                    )
                )
            ]
            self.outer_ring_ring_rod_connection_offset_end_idx[
                number_of_straight_rods_except_center
                * idx : number_of_straight_rods_except_center
                * (idx + 1)
            ] = [
                x
                for _, x in sorted(
                    zip(
                        ring_rod_element_idx_list,
                        self.outer_ring_ring_rod_connection_offset_end_idx[
                            number_of_straight_rods_except_center
                            * idx : number_of_straight_rods_except_center
                            * (idx + 1)
                        ],
                    )
                )
            ]

        # do_step, stages_and_updates will be used in step function
        self.do_step, self.stages_and_updates = extend_stepper_interface(
            self.StatefulStepper, self.simulator
        )

        systems = []

        return self.total_steps, systems

    def step(self, activation_array_list, time):
        # Activation array contains lists for activation in different directions
        # assign correct activation arrays to correct directions.
        # self.activation_arr_rod1[:] = activation_array_list[0]
        # self.activation_arr_rod3[:] = activation_array_list[1]

        # Do 200 time step of simulation. Here we are doing multiple Elastica time-steps, because
        # time-step of elastica is much slower than the time-step of control or learning algorithm.
        for _ in range(self.learning_step):
            time = self.do_step(
                self.StatefulStepper,
                self.stages_and_updates,
                self.simulator,
                time,
                self.time_step,
            )

        systems = []

        """ Done is a boolean to reset the environment before episode is completed """
        done = False
        # Position of the rod cannot be NaN, it is not valid, stop the simulation
        for idx, rod in enumerate(self.rod_list):
            if _isnan_check(rod.position_collection) == True:
                print(" Nan detected, exiting simulation now")
                print("rod " + str(idx))
                done = True
        """ Done is a boolean to reset the environment before episode is completed """

        return time, systems, done

    def clear_callback(self):
        for ddict in self.post_processing_dict_list:
            ddict.clear()
        for ddict in self.post_processing_list_for_pressure_force_dicts:
            ddict.clear()

    def post_processing(self, filename_video, **kwargs):
        """
        Make video 3D rod movement in time.
        Parameters
        ----------
        filename_video
        Returns
        -------

        """

        if self.COLLECT_DATA_FOR_POSTPROCESSING:
            import os

            current_path = os.getcwd()
            current_path = kwargs.get("folder_name", current_path)

            if len(self.post_processing_list_for_pressure_force_dicts) != 0:
                plot_video_activation_muscle(
                    self.post_processing_list_for_pressure_force_dicts[0],
                    video_name="pressure_straight_rod_one.mp4",
                    fps=self.rendering_fps,
                    step=1,
                    **kwargs,
                )
            if len(self.post_processing_list_for_pressure_force_dicts) > 1:
                plot_video_activation_muscle(
                    self.post_processing_list_for_pressure_force_dicts[1],
                    video_name="pressure_straight_rod_two.mp4",
                    fps=self.rendering_fps,
                    step=1,
                    **kwargs,
                )

            plot_video_with_surface(
                self.post_processing_dict_list,
                video_name=filename_video,
                fps=self.rendering_fps,
                step=1,
                **kwargs,
            )

            # plot_video_with_surface(
            #     self.post_processing_dict_list[: len(self.straight_rod_list)],
            #     video_name="straight_rods.mp4",
            #     fps=self.rendering_fps / 10 * 4,
            #     step=1,
            #     **kwargs,
            # )
            #
            # plot_video_with_surface(
            #     self.post_processing_dict_list[:]
            #     + self.post_processing_dict_list[
            #         len(self.straight_rod_list) : len(self.straight_rod_list)
            #         + len(self.ring_rod_list_outer)
            #     ],
            #     video_name="ring_rods.mp4",
            #     fps=self.rendering_fps / 10 * 4,
            #     step=1,
            #     **kwargs,
            # )

            # import os

            save_folder = os.path.join(current_path, "data")
            os.makedirs(save_folder, exist_ok=True)

            time = np.array(self.post_processing_dict_list[0]["time"])

            number_of_inner_ring_rods = 0
            number_of_straight_rods = len(self.straight_rod_list)
            if number_of_straight_rods > 0:
                n_elems_straight_rods = self.straight_rod_list[0].n_elems
                straight_rods_position_history = np.zeros(
                    (
                        number_of_straight_rods,
                        time.shape[0],
                        3,
                        n_elems_straight_rods + 1,
                    )
                )
                straight_rods_velocity_history = np.zeros(
                    (
                        number_of_straight_rods,
                        time.shape[0],
                        3,
                        n_elems_straight_rods + 1,
                    )
                )
                straight_rods_radius_history = np.zeros(
                    (number_of_straight_rods, time.shape[0], n_elems_straight_rods)
                )
                straight_rods_length_history = np.zeros(
                    (number_of_straight_rods, time.shape[0], n_elems_straight_rods)
                )
                straight_rods_external_forces_history = np.zeros(
                    (
                        number_of_straight_rods,
                        time.shape[0],
                        3,
                        n_elems_straight_rods + 1,
                    )
                )
                straight_rods_internal_forces_history = np.zeros(
                    (
                        number_of_straight_rods,
                        time.shape[0],
                        3,
                        n_elems_straight_rods + 1,
                    )
                )
                straight_rods_tangents_history = np.zeros(
                    (
                        number_of_straight_rods,
                        time.shape[0],
                        3,
                        n_elems_straight_rods,
                    )
                )
                straight_rods_internal_stress_history = np.zeros(
                    (
                        number_of_straight_rods,
                        time.shape[0],
                        3,
                        n_elems_straight_rods,
                    )
                )
                straight_rods_dilatation_history = np.zeros(
                    (number_of_straight_rods, time.shape[0], n_elems_straight_rods)
                )
                straight_rods_kappa_history = np.zeros(
                    (
                        number_of_straight_rods,
                        time.shape[0],
                        3,
                        n_elems_straight_rods - 1,
                    )
                )
                straight_rods_director_history = np.zeros(
                    (
                        number_of_straight_rods,
                        time.shape[0],
                        3,
                        3,
                        n_elems_straight_rods,
                    )
                )
                straight_rods_activation_history = np.zeros(
                    (number_of_straight_rods, time.shape[0], n_elems_straight_rods)
                )
                straight_rods_sigma_history = np.zeros(
                    (
                        number_of_straight_rods,
                        time.shape[0],
                        3,
                        n_elems_straight_rods,
                    )
                )
                for i in range(number_of_straight_rods):
                    straight_rods_position_history[i, :, :, :] = np.array(
                        self.post_processing_dict_list[i + number_of_inner_ring_rods][
                            "position"
                        ]
                    )
                    straight_rods_velocity_history[i, :, :, :] = np.array(
                        self.post_processing_dict_list[i + number_of_inner_ring_rods][
                            "velocity"
                        ]
                    )
                    straight_rods_radius_history[i, :, :] = np.array(
                        self.post_processing_dict_list[i + number_of_inner_ring_rods][
                            "radius"
                        ]
                    )
                    straight_rods_length_history[i, :, :] = np.array(
                        self.post_processing_dict_list[i + number_of_inner_ring_rods][
                            "lengths"
                        ]
                    )
                    straight_rods_external_forces_history[i, :, :, :] = np.array(
                        self.post_processing_dict_list[i + number_of_inner_ring_rods][
                            "external_forces"
                        ]
                    )
                    straight_rods_internal_forces_history[i, :, :, :] = np.array(
                        self.post_processing_dict_list[i + number_of_inner_ring_rods][
                            "internal_forces"
                        ]
                    )
                    straight_rods_tangents_history[i, :, :, :] = np.array(
                        self.post_processing_dict_list[i + number_of_inner_ring_rods][
                            "tangents"
                        ]
                    )
                    straight_rods_internal_stress_history[i, :, :, :] = np.array(
                        self.post_processing_dict_list[i + number_of_inner_ring_rods][
                            "internal_stress"
                        ]
                    )
                    straight_rods_dilatation_history[i, :, :] = np.array(
                        self.post_processing_dict_list[i + number_of_inner_ring_rods][
                            "dilatation"
                        ]
                    )
                    straight_rods_kappa_history[i, :, :, :] = np.array(
                        self.post_processing_dict_list[i + number_of_inner_ring_rods][
                            "kappa"
                        ]
                    )
                    straight_rods_director_history[i, :, :, :, :] = np.array(
                        self.post_processing_dict_list[i + number_of_inner_ring_rods][
                            "directors"
                        ]
                    )
                    straight_rods_activation_history[i, :, :] = np.array(
                        self.post_processing_dict_list[i + number_of_inner_ring_rods][
                            "activation"
                        ]
                    )
                    straight_rods_sigma_history[i, :, :, :] = np.array(
                        self.post_processing_dict_list[i + number_of_inner_ring_rods][
                            "sigma"
                        ]
                    )

            else:
                straight_rods_position_history = None
                straight_rods_radius_history = None
                straight_rods_length_history = None
                straight_rods_external_forces_history = None
                straight_rods_internal_forces_history = None
                straight_rods_tangents_history = None
                straight_rods_internal_stress_history = None
                straight_rods_dilatation_history = None
                straight_rods_kappa_history = None
                straight_rods_director_history = None
                straight_rods_activation_history = None
                straight_rods_sigma_history = None

            number_of_outer_ring_rods = len(self.ring_rod_list_outer)
            if number_of_outer_ring_rods > 0:
                n_elems_outer_ring_rods = self.ring_rod_list_outer[0].n_elems
                outer_ring_rods_position_history = np.zeros(
                    (
                        number_of_outer_ring_rods,  # number of rods
                        time.shape[0],  # time
                        3,  # direction
                        n_elems_outer_ring_rods,  # number of nodes
                    )
                )
                outer_ring_rods_radius_history = np.zeros(
                    # number of rods            # time         # number of elems
                    (number_of_outer_ring_rods, time.shape[0], n_elems_outer_ring_rods)
                )
                outer_ring_rods_length_history = np.zeros(
                    # number of rods            # time         # number of elems
                    (number_of_outer_ring_rods, time.shape[0], n_elems_outer_ring_rods)
                )

                for i in range(
                    number_of_outer_ring_rods,
                ):
                    outer_ring_rods_position_history[i, :, :, :] = np.array(
                        self.post_processing_dict_list[
                            i + number_of_inner_ring_rods + number_of_straight_rods
                        ]["position"]
                    )
                    outer_ring_rods_radius_history[i, :, :] = np.array(
                        self.post_processing_dict_list[
                            i + number_of_inner_ring_rods + number_of_straight_rods
                        ]["radius"]
                    )
                    outer_ring_rods_length_history[i, :, :] = np.array(
                        self.post_processing_dict_list[
                            i + number_of_inner_ring_rods + number_of_straight_rods
                        ]["lengths"]
                    )

            else:
                outer_ring_rods_position_history = None
                outer_ring_rods_radius_history = None
                outer_ring_rods_length_history = None

            number_of_helical_rods = len(self.helical_rod_list)
            if number_of_helical_rods > 0:
                n_elems_helical_rods = self.helical_rod_list[0].n_elems
                helical_rods_position_history = np.zeros(
                    (
                        number_of_helical_rods,
                        time.shape[0],
                        3,
                        n_elems_helical_rods + 1,
                    )
                )
                helical_rods_radius_history = np.zeros(
                    (number_of_helical_rods, time.shape[0], n_elems_helical_rods)
                )
                helical_rods_length_history = np.zeros(
                    (number_of_helical_rods, time.shape[0], n_elems_helical_rods)
                )
                for i in range(number_of_helical_rods):
                    helical_rods_position_history[i, :, :, :] = np.array(
                        self.post_processing_dict_list[
                            i
                            + number_of_inner_ring_rods
                            + number_of_straight_rods
                            + number_of_outer_ring_rods
                        ]["position"]
                    )
                    helical_rods_radius_history[i, :, :] = np.array(
                        self.post_processing_dict_list[
                            i
                            + number_of_inner_ring_rods
                            + number_of_straight_rods
                            + number_of_outer_ring_rods
                        ]["radius"]
                    )
                    helical_rods_length_history[i, :, :] = np.array(
                        self.post_processing_dict_list[
                            i
                            + number_of_inner_ring_rods
                            + number_of_straight_rods
                            + number_of_outer_ring_rods
                        ]["radius"]
                    )
            else:
                helical_rods_position_history = None
                helical_rods_radius_history = None
                helical_rods_length_history = None

            number_of_cylinders = len(self.cylinder_list)
            n_elem_cylinder = self.resize_cylinder_elems
            if number_of_cylinders > 0:
                cylinders_position_history = np.zeros(
                    (number_of_cylinders, time.shape[0], 3, n_elem_cylinder + 1)
                )
                cylinders_radius_history = np.zeros(
                    (number_of_cylinders, time.shape[0], n_elem_cylinder)
                )

                for i in range(number_of_cylinders):
                    cylinders_position_history[i, :, :, :] = np.array(
                        self.post_processing_dict_list[
                            i
                            + number_of_inner_ring_rods
                            + number_of_straight_rods
                            + number_of_outer_ring_rods
                            + number_of_helical_rods
                        ]["position"]
                    )
                    cylinders_radius_history[i, :, :] = np.array(
                        self.post_processing_dict_list[
                            i
                            + number_of_inner_ring_rods
                            + number_of_straight_rods
                            + number_of_outer_ring_rods
                            + number_of_helical_rods
                        ]["radius"]
                    )
            else:
                cylinders_position_history = None
                cylinders_radius_history = None

            np.savez(
                os.path.join(save_folder, "octopus_arm_test.npz"),
                time=time,
                straight_rods_position_history=straight_rods_position_history,
                straight_rods_radius_history=straight_rods_radius_history,
                straight_rods_length_history=straight_rods_length_history,
                straight_rods_velocity_history=straight_rods_velocity_history,
                # straight_rods_external_forces_history=straight_rods_external_forces_history,
                # straight_rods_internal_forces_history=straight_rods_internal_forces_history,
                straight_rods_tangents_history=straight_rods_tangents_history,
                # straight_rods_internal_stress_history=straight_rods_internal_stress_history,
                straight_rods_dilatation_history=straight_rods_dilatation_history,
                straight_rods_kappa_history=straight_rods_kappa_history,
                straight_rods_director_history=straight_rods_director_history,
                straight_rods_activation_history=straight_rods_activation_history,
                straight_rods_sigma_history=straight_rods_sigma_history,
                outer_ring_rods_position_history=outer_ring_rods_position_history,
                outer_ring_rods_radius_history=outer_ring_rods_radius_history,
                outer_ring_rods_length_history=outer_ring_rods_length_history,
                helical_rods_position_history=helical_rods_position_history,
                helical_rods_radius_history=helical_rods_radius_history,
                helical_rods_length_history=helical_rods_length_history,
                cylinders_position_history=cylinders_position_history,
                cylinders_radius_history=cylinders_radius_history,
            )

            # import pickle
            #
            # my_data_for_rhino = {}
            # my_data_for_rhino[
            #     "straight_rods_position_history"
            # ] = straight_rods_position_history.tolist()
            # my_data_for_rhino[
            #     "straight_rods_radius_history"
            # ] = straight_rods_radius_history.tolist()
            # my_data_for_rhino["inner_ring_rods_position_history"] = []
            # my_data_for_rhino["inner_ring_rods_radius_history"] = []
            # my_data_for_rhino[
            #     "outer_ring_rods_position_history"
            # ] = outer_ring_rods_position_history.tolist()
            # my_data_for_rhino[
            #     "outer_ring_rods_radius_history"
            # ] = outer_ring_rods_radius_history.tolist()
            # my_data_for_rhino[
            #     "helical_rods_position_history"
            # ] = helical_rods_position_history.tolist()
            # my_data_for_rhino[
            #     "helical_rods_radius_history"
            # ] = helical_rods_radius_history.tolist()
            #
            # with open(
            #         os.path.join(save_folder, "octopus_arm_test.pkl"), "wb"
            # ) as file:
            #     pickle.dump(my_data_for_rhino, file, protocol=2)

        else:
            raise RuntimeError(
                "call back function is not called anytime during simulation, "
                "change COLLECT_DATA=True"
            )
