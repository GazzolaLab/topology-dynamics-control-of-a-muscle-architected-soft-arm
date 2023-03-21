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
import numba
from numba import njit
from elastica._rotations import _get_rotation_matrix
from Connections import *
from Cases.post_processing import (
    plot_tentacle_length_vs_time,
    plot_tentacle_velocity_vs_time,
)
from .muscle_model_piecewise_fit import (
    get_active_force_piecewise_linear_function_parameters_for_VanLeeuwen_muscle_model,
    get_passive_force_cubic_function_coefficients_for_VanLeeuwen_muscle_model,
)


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
        time_step = 1.25e-6  # this is a stable timestep
        self.learning_step = 1
        self.total_steps = int(self.final_time / time_step / self.learning_step)
        self.time_step = np.float64(
            float(self.final_time) / (self.total_steps * self.learning_step)
        )
        # Video speed
        self.rendering_fps = 20 * 1e2 / 2
        self.step_skip = int(1.0 / (self.rendering_fps * self.time_step))

        # Collect data is a boolean. If it is true callback function collects
        # rod parameters defined by user in a list.
        self.COLLECT_DATA_FOR_POSTPROCESSING = COLLECT_DATA_FOR_POSTPROCESSING

    def reset(self):
        """"""
        self.simulator = BaseSimulator()

        # setting up test params
        n_elem = 96
        n_elem_ring_rod = 24
        number_of_straight_rods = 13
        self.n_connections = number_of_straight_rods - 1

        start = np.zeros((3,))
        direction = np.array([0.0, 1.0, 0.0])  # rod direction
        normal = np.array([0.0, 0.0, 1.0])
        binormal = np.cross(direction, normal)
        arm_length = 53.143  # mm
        mass_club = 1.8  # g
        mass_stalk = mass_club / 0.75
        mass_total = mass_stalk + mass_club  # mass_club * (1 + 1 / 0.75)
        outer_base_radius = 3.7  # mm
        density = 1.050e-3  # # g/mm3
        nu = 10 * 5 * 5 * 2 * 10 * 5  # dissipation coefficient
        E = 50 * 7.5e3  # 2e5  # Young's Modulus Pa
        E *= 1  # convert Pa (kg m.s-2 / m2) to g.mm.s-2 / mm2 This is required for units to be consistent
        poisson_ratio = 0.5

        # Compute the club length using the data given.
        self.club_length = mass_club / (density * (np.pi * outer_base_radius**2))
        stretch_optimal = 1.2

        eta_ring_rods = 0.70  # percentage area of the transverse muscles
        eta_straight_rods = (
            0.20 * (number_of_straight_rods - 1) / number_of_straight_rods
        )  # 0.15
        eta_helical_rods = 0.10
        eta_axial_cord = 0.20 / number_of_straight_rods

        # Compute center rod and other straight rod radius
        center_straight_rod_radius = np.sqrt(outer_base_radius**2 * eta_axial_cord)
        outer_straight_rod_radius = np.sqrt(
            outer_base_radius**2 * eta_straight_rods / (number_of_straight_rods - 1)
        )

        def compute_straight_rod_areas(variables, number_of_straight_rods, total_area):
            """
            This function computes inner and outer straight rod radius using the inputs.
            Final radius of rods are distributed such that outer rods form a polygon around the inner rod.

            Parameters
            ----------
            variables
            number_of_straight_rods
            total_area

            Returns
            -------

            """
            r_center, r_outer = variables

            eq1 = (
                np.pi * r_center**2
                + number_of_straight_rods * np.pi * r_outer**2
                - total_area
            )
            eq2 = 2 * number_of_straight_rods * np.sin(
                np.pi / number_of_straight_rods
            ) * (r_center + r_outer) - number_of_straight_rods * (2 * r_outer)

            return eq1, eq2

        area_straight_rods = (
            np.pi * outer_base_radius**2 * (eta_axial_cord + eta_straight_rods)
        )
        center_straight_rod_radius, outer_straight_rod_radius = fsolve(
            compute_straight_rod_areas,
            x0=[center_straight_rod_radius, outer_straight_rod_radius],
            args=(number_of_straight_rods - 1, area_straight_rods),
        )

        eta_one_longitudinal_muscle_rod = (
            np.pi * outer_straight_rod_radius**2 / area_straight_rods
        )
        eta_one_axial_cord = (
            np.pi * center_straight_rod_radius**2 / area_straight_rods
        )

        def compute_outer_ring_rod_radius(
            r_outer_ring_rod,
            n_elem_ring_rod,
            r_center_straight_rod,
            r_outer_straight_rod,
            area_ring,
        ):
            """
            This function computes the outer ring rod radius such that ring encircles the straight rods and also
            user defined area is satisfied.

            Parameters
            ----------
            r_outer_ring_rod
            n_elem_ring_rod
            r_center_straight_rod
            r_outer_straight_rod
            area_ring

            Returns
            -------

            """
            return (
                # Perimeter of polygon 2*n_corner*np.sin(pi/n_corner)*R
                2
                * n_elem_ring_rod
                * np.sin(np.pi / n_elem_ring_rod)
                * (r_center_straight_rod + 2 * r_outer_straight_rod + r_outer_ring_rod)
                * 2
                * r_outer_ring_rod
                - area_ring
            ) ** 2

        area_ring_rods = np.pi * outer_base_radius**2 * eta_ring_rods

        outer_ring_rod_radius = minimize_scalar(
            compute_outer_ring_rod_radius,
            args=(
                n_elem_ring_rod,
                center_straight_rod_radius,
                outer_straight_rod_radius,
                area_ring_rods,
            ),
        ).x

        # Compute radius of helical rod. We will place helical rods at the most
        # outer layer.
        area_helical = (np.pi * outer_base_radius**2) * eta_helical_rods

        def compute_helical_rod_radius(
            r_helical_rod,
            r_center_straight_rod,
            r_outer_straight_rod,
            r_outer_ring_rod,
            area_left,
        ):
            """
            This function computes the radius of helical rods such that helical rods encircles the ring rods.

            Parameters
            ----------
            r_helical_rod
            r_center_straight_rod
            r_outer_straight_rod
            r_outer_ring_rod
            area_left

            Returns
            -------

            """

            length = (
                r_center_straight_rod
                + 2 * r_outer_straight_rod
                + 2 * r_outer_ring_rod
                + r_helical_rod
            )

            return (2 * np.pi * length * (2 * r_helical_rod) - area_left) ** 2

        helical_rod_radius = minimize_scalar(
            compute_helical_rod_radius,
            args=(
                center_straight_rod_radius,
                outer_straight_rod_radius,
                outer_ring_rod_radius,
                area_helical,
            ),
        ).x
        # This scale is for making sure that helical rod mass is correct. Our goal is to match helical rod
        # density equal to the density of stalk. I am lazy and hard coded this number instead of writting a
        # function. Which computes helix radius, helical rod length, radius etc, and using the final helical rod
        # volume compute the density to match with the helical rod mass.
        helical_rod_radius *= 1.682429536679109

        center_rod_radius_along_arm = center_straight_rod_radius * np.ones((n_elem))

        outer_straight_rod_radius_along_arm = outer_straight_rod_radius * np.ones(
            (n_elem)
        )

        outer_ring_rod_radius_along_arm = outer_ring_rod_radius * np.ones((n_elem))

        helical_rod_radius_along_arm = helical_rod_radius * np.ones((n_elem))

        # Compute the sacromere, myosin, maximum active stress
        l_bz = 0.14  # length of the bare zone (micro meter)

        l_sacro_ref = 2.37  # reference sacromere length (micro meter)
        l_sacro_base = 1.3399  # sacromere length at the base (micro meter)
        l_sacro_tip = 0.7276  # sacromere length at the tip  (micro meter)
        sarcomere_rest_lengths = np.linspace(l_sacro_base, l_sacro_tip, n_elem)

        l_myo_ref = 1.58  # reference myosin length (micro meter)
        l_myo_base = 0.9707  # myosin length at the base of the tentacle (micro meter)
        l_myo_tip = 0.4997  # myosin length at the tip of the tentacle (micro meter)
        myosin_lengths = np.linspace(l_myo_base, l_myo_tip, n_elem)

        maximum_active_stress_ref = 280e3  # maximum active stress reference value (Pa)
        maximum_active_stress = (
            maximum_active_stress_ref * (myosin_lengths - l_bz) / (l_myo_ref - l_bz)
        )

        minimum_strain_rate_ref = -17  # -17  # 1/s
        minimum_strain_rate = minimum_strain_rate_ref * (
            l_sacro_ref / sarcomere_rest_lengths
        )

        (
            normalized_active_force_slope,
            normalized_active_force_y_intercept,
            normalized_active_force_break_points,
        ) = get_active_force_piecewise_linear_function_parameters_for_VanLeeuwen_muscle_model(
            sarcomere_rest_lengths, myosin_lengths
        )
        # Load passive stress data. For transverse muscles we used the VanLeeuwen model. For passive stress Elastica
        # implementation uses 3rd degree polynomial and requires 4 coefficients. Order of coefficients starts from
        # highest order (cube) to lowest order of poly.
        muscle_passive_force_coefficients = (
            get_passive_force_cubic_function_coefficients_for_VanLeeuwen_muscle_model()
        )

        passive_force_coefficients_straight_rods = np.ones(
            (n_elem)
        ) * muscle_passive_force_coefficients.reshape(4, 1)

        # For both longitudinal muscle active and passive force length curves we fit a polynomial. Lets read
        # the coefficients
        longitudinal_muscle_coefficient = np.load("squid_longitudinal_muscles_fit.npz")
        # Longitudinal muscle active coefficients are computed by fitting 8th order polynomial to the data given in
        # Zullo 2022.
        longitudinal_muscle_active_force_coefficients = longitudinal_muscle_coefficient[
            "longitudinal_active_part_coefficients"
        ]
        # Longitudinal muscle passive coefficients are computed by fitting 2nd order polynomial to the data given in
        # Zullo 2022.
        longitudinal_muscle_passive_force_coefficients = (
            longitudinal_muscle_coefficient["longitudinal_passive_part_coefficients"]
        )

        # For both transverse muscle active and passive force length curves we fit a polynomial. Lets read the
        # coefficients.
        transverse_muscle_coefficient = np.load("squid_transverse_muscles_fit.npz")
        # Transverse muscle active coefficients are computed by fitting 4th order polynomial to the data given in
        # Zullo 2022.
        transverse_muscle_active_force_coefficients = transverse_muscle_coefficient[
            "transverse_active_part_coefficients"
        ]
        # Transverse muscle passive coefficients are computed by fitting 2nd order polynomial to the data given in
        # Zullo 2022.
        transverse_muscle_passive_force_coefficients = transverse_muscle_coefficient[
            "transverse_passive_part_coefficients"
        ]

        force_velocity_constant = 0.25  # 1/0.80

        direction_ring_rod = normal
        normal_ring_rod = direction

        self.rod_list = []
        self.straight_rod_list = []
        # ring straight rod connection direction list is containing the list of directions for possible connections.
        # Here the idea is to connect nodes in specific position, and make sure there is symmetry in whole arm.
        self.ring_straight_rod_connection_direction_list = []

        # First straight rod is at the center, remaining ring rods are around the first ring rod.
        angle_btw_straight_rods = (
            0
            if number_of_straight_rods == 1
            else 2 * np.pi / (number_of_straight_rods - 1)
        )

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

        muscle_active_force_coefficients_straight_rods = np.ones(
            (n_elem)
        ) * longitudinal_muscle_active_force_coefficients.reshape(9, 1)
        straight_rods_force_velocity_constant = (
            np.ones((n_elem)) * force_velocity_constant
        )

        passive_force_coefficients_straight_rods = (
            longitudinal_muscle_passive_force_coefficients.reshape(9, 1)
            * np.ones((n_elem))
        )

        nu_rod_1 = (
            density
            * np.pi
            * center_rod_radius_along_arm**2
            * nu
            / 50
            / 2
            / 400
            * 400
            / 8
            * 4
        ) * 0

        self.shearable_rod1 = MuscularRod.straight_rod(
            n_elem,
            start_rod_1,
            direction,
            normal,
            arm_length,
            base_radius=center_rod_radius_along_arm,
            density=density,
            nu=nu_rod_1,
            youngs_modulus=E,
            poisson_ratio=poisson_ratio,
            force_velocity_constant=straight_rods_force_velocity_constant,
            # normalized_active_force_break_points=normalized_active_force_break_points_straight_rods,
            # normalized_active_force_y_intercept=normalized_active_force_y_intercept_straight_rods,
            # normalized_active_force_slope=normalized_active_force_slope_straight_rods,
            E_compression=1e5,
            compression_strain_limit=0,  # -0.025,
            active_force_coefficients=muscle_active_force_coefficients_straight_rods,
            tension_passive_force_scale=1 / 2,
            # Active force starts to decrease around strain 0.55 so we shift passive force to strain of 0.55
            # This is also seen in Leech longitudinal muscles (Gerry & Ellebry 2011)
            extension_strain_limit=0,  # 0.025,
            passive_force_coefficients=passive_force_coefficients_straight_rods,
        )
        # Add the mass of the club
        # Distribute the club mass proportional to the cross-sectional area of rod.
        self.shearable_rod1.mass[-1] += mass_club * eta_one_axial_cord
        self.shearable_rod1.bend_matrix *= 5 * 10
        self.straight_rod_list.append(self.shearable_rod1)

        for i in range(number_of_straight_rods - 1):
            rotation_matrix = _get_rotation_matrix(
                angle_btw_straight_rods * i, direction.reshape(3, 1)
            ).reshape(3, 3)
            direction_from_center_to_rod = rotation_matrix @ binormal

            self.ring_straight_rod_connection_direction_list.append(
                [direction_from_center_to_rod, -direction_from_center_to_rod]
            )
            start_rod = start + (direction_from_center_to_rod) * (
                # center rod            # this rod
                +center_rod_radius_along_arm[0]
                + outer_straight_rod_radius_along_arm[0]
            )

            nu_rod = nu_rod_1

            self.straight_rod_list.append(
                MuscularRod.straight_rod(
                    n_elem,
                    start_rod,
                    direction,
                    normal,
                    arm_length,
                    base_radius=outer_straight_rod_radius_along_arm,
                    density=density,
                    nu=nu_rod,
                    youngs_modulus=E,
                    poisson_ratio=poisson_ratio,
                    force_velocity_constant=straight_rods_force_velocity_constant,
                    # normalized_active_force_break_points=normalized_active_force_break_points_straight_rods,
                    # normalized_active_force_y_intercept=normalized_active_force_y_intercept_straight_rods,
                    # normalized_active_force_slope=normalized_active_force_slope_straight_rods,
                    E_compression=1e5,
                    compression_strain_limit=0,  # -0.025,
                    active_force_coefficients=muscle_active_force_coefficients_straight_rods,
                    tension_passive_force_scale=1 / 2,  # /4,
                    # Extension shift is to make sure at rest configuration there is no passive tension stress.
                    extension_strain_limit=0,  # 0.025,
                    passive_force_coefficients=passive_force_coefficients_straight_rods,
                )
            )
            # Add the mass of the club
            # Distribute the club mass proportional to the cross-sectional area of rod.
            self.straight_rod_list[i + 1].mass[-1] += (
                mass_club * eta_one_longitudinal_muscle_rod
            )
            self.straight_rod_list[i + 1].bend_matrix *= 5 * 10

        # Ring rods
        total_number_of_ring_rods = n_elem
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

        # Adjust density such that its mass is consistent with percentage area of this muscle group.
        # Ring rods have circular cross-sectional area. Percentage area of ring rods on stalk cross-section is used to
        # compute the ring rod radius. Thus if we use the stalk density we can assign larger or smaller mass for the
        # ring rods. So we are computing density here again to make sure ring rod mass is correct.
        density_ring_rod = mass_stalk * eta_ring_rods / volume_outer_ring_rod.sum()

        for i in range(total_number_of_ring_rods):
            nu_ring_rod_outer = (
                density_ring_rod * np.pi * radius_outer_ring_rod[i] ** 2 * nu / 200
            )
            maximum_active_stress_ring_rods = (
                np.ones((n_elem_ring_rod))
                * maximum_active_stress[connection_idx_straight_rod[i]]
            )
            minimum_strain_rate_ring_rods = (
                np.ones((n_elem_ring_rod))
                * minimum_strain_rate[connection_idx_straight_rod[i]]
            )
            normalized_active_force_break_points_ring_rods = np.ones(
                (n_elem_ring_rod)
            ) * normalized_active_force_break_points[
                :, connection_idx_straight_rod[i]
            ].reshape(
                4, 1
            )
            normalized_active_force_y_intercept_ring_rods = np.ones(
                (n_elem_ring_rod)
            ) * normalized_active_force_y_intercept[
                :, connection_idx_straight_rod[i]
            ].reshape(
                4, 1
            )
            normalized_active_force_slope_ring_rods = np.ones(
                (n_elem_ring_rod)
            ) * normalized_active_force_slope[
                :, connection_idx_straight_rod[i]
            ].reshape(
                4, 1
            )
            passive_force_coefficients_ring_rods = (
                muscle_passive_force_coefficients.reshape(4, 1)
                * np.ones((n_elem_ring_rod))
            )

            force_velocity_constant_ring = (
                np.ones((n_elem_ring_rod)) * force_velocity_constant
            )
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
                    nu=nu_ring_rod_outer * 0,
                    youngs_modulus=E,  # *2,#*4*4,  # *4*(2.5)**2,
                    poisson_ratio=poisson_ratio,
                    maximum_active_stress=maximum_active_stress_ring_rods,  # *4*4,
                    minimum_strain_rate=minimum_strain_rate_ring_rods,
                    force_velocity_constant=force_velocity_constant_ring,
                    E_compression=2e5 * 2.6,  # *4*4,
                    # tension_passive_force_scale = 1 ,
                    # normalized_active_force_break_points=normalized_active_force_break_points_ring,
                    # normalized_active_force_y_intercept=normalized_active_force_y_intercept_ring,
                    # normalized_active_force_slope=normalized_active_force_slope_ring,
                    compression_strain_limit=0,  # -0.025,
                    active_force_coefficients=muscle_active_force_coefficients_ring_rods,
                    # Extension shift is to make sure at rest configuration there is no passive tension stress.
                    extension_strain_limit=0,  # 0.025,
                    passive_force_coefficients=passive_force_coefficients_ring_rods,
                )
            )

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
        distance_in_one_turn = (6 + 5e-15) * distance_btw_two_ring_rods  # 6
        self.helix_angle = np.rad2deg(
            np.arctan(2 * np.pi * helix_radius_base / distance_in_one_turn)
        )
        n_helix_turns = helix_length_covered / distance_in_one_turn
        pitch_factor = distance_in_one_turn / (2 * np.pi)

        # Number of helix turns
        n_elem_per_turn = 24  # This should be divided by the 6 at least, # of ring rod that is passed in one turn
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

        normalized_active_force_break_points_helical_rods = np.ones(
            (n_elem_helical_rod)
        ) * normalized_active_force_break_points[:, 0].reshape((4, 1))
        normalized_active_force_y_intercept_helical_rods = np.ones(
            (n_elem_helical_rod)
        ) * normalized_active_force_y_intercept[:, 0].reshape((4, 1))
        normalized_active_force_slope_helical_rods = np.ones(
            (n_elem_helical_rod)
        ) * normalized_active_force_slope[:, 0].reshape((4, 1))
        passive_force_coefficients_helical_rods = (
            muscle_passive_force_coefficients.reshape(4, 1)
            * np.ones((n_elem_helical_rod))
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
                mass_stalk
                * eta_helical_rods
                / total_number_of_helical_rods
                / volume_helical_rod
            )

            nu_helical_rod = (
                density_helical_rod * np.pi * helical_rod_radius**2 * nu / 200 * 2
            )

            self.helical_rod_list.append(
                MuscularRod.straight_rod(
                    n_elem_helical_rod,
                    start,
                    direction_helical_rod,
                    normal_helical_rod,
                    base_length_helical_rod,
                    helical_rod_radius,
                    density=density_helical_rod,
                    nu=nu_helical_rod * 0,
                    youngs_modulus=E,
                    poisson_ratio=poisson_ratio,
                    position=position,
                    directors=director_collection,
                    E_compression=1e5,
                    tension_passive_force_scale=1 / 2,
                    force_velocity_constant=force_velocity_constant_helical_rods,
                    active_force_coefficients=muscle_active_force_coefficients_helical_rods,
                    # normalized_active_force_break_points=normalized_active_force_break_points_helical_rods,
                    # normalized_active_force_y_intercept=normalized_active_force_y_intercept_helical_rods,
                    # normalized_active_force_slope=normalized_active_force_slope_helical_rods,
                    compression_strain_limit=0,  # -0.025,
                    # Extension shift is to make sure at rest configuration there is no passive tension stress.
                    extension_strain_limit=0,  # 0.025,
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
                mass_stalk
                * eta_helical_rods
                / total_number_of_helical_rods
                / volume_helical_rod
            )

            self.helical_rod_list.append(
                MuscularRod.straight_rod(
                    n_elem_helical_rod,
                    start,
                    direction_helical_rod,
                    normal_helical_rod,
                    base_length_helical_rod,
                    helical_rod_radius,
                    density=density_helical_rod,
                    nu=nu_helical_rod * 0,
                    youngs_modulus=E,
                    poisson_ratio=poisson_ratio,
                    position=position,
                    directors=director_collection,
                    E_compression=1e5,
                    tension_passive_force_scale=1 / 2,
                    force_velocity_constant=force_velocity_constant_helical_rods,
                    active_force_coefficients=muscle_active_force_coefficients_helical_rods,
                    # normalized_active_force_break_points=normalized_active_force_break_points_helical_rods,
                    # normalized_active_force_y_intercept=normalized_active_force_y_intercept_helical_rods,
                    # normalized_active_force_slope=normalized_active_force_slope_helical_rods,
                    compression_strain_limit=0,  # -0.025,
                    # Extension shift is to make sure at rest configuration there is no passive tension stress.
                    extension_strain_limit=0,  # 0.025,
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

        # Constrain the rods
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

        self.simulator.constrain(self.shearable_rod1).using(
            OneEndFixedRod,
            constrained_position_idx=(0,),
            constrained_director_idx=(0,),
        )

        for _, rod in enumerate(self.straight_rod_list[1:]):
            self.simulator.constrain(rod).using(
                FixNodePosition,  # FixNodePosition,
                constrained_position_idx=(0,),
                constrained_director_idx=(0,),
            )

        for _, rod in enumerate(self.helical_rod_list):
            self.simulator.constrain(rod).using(
                FixNodePosition,  # FixNodePosition,
                constrained_position_idx=(0,),
                constrained_director_idx=(0,),
            )

        from Cases.arm_function import (
            DampingFilterBC,
            DampingFilterBCRingRod,
            ExponentialDampingBC,
        )

        for _, rod in enumerate(self.straight_rod_list):
            self.simulator.constrain(rod).using(
                DampingFilterBC,
                constrained_position_idx=(0,),
                constrained_director_idx=(0,),
                filter_order=5,  # 10,
            )
            self.simulator.constrain(rod).using(
                ExponentialDampingBC,
                constrained_position_idx=(0,),
                constrained_director_idx=(0,),
                time_step=self.time_step,
                nu=50 * 1.2 * 1.4,  # 20*2,#5*5*2,
                rod=rod,
            )
        # for _, rod in enumerate(self.ring_rod_list):
        #     # self.simulator.constrain(rod).using(
        #     #     DampingFilterBCRingRod,#DampingFilterBC,
        #     #     constrained_position_idx=(0,),
        #     #     constrained_director_idx=(0,),
        #     #     filter_order=6,  # 10,
        #     # )
        #     self.simulator.constrain(rod).using(
        #         ExponentialDampingBC,
        #         constrained_position_idx=(0,),
        #         constrained_director_idx=(0,),
        #         time_step = self.time_step,
        #         nu = 5*2,
        #         rod=rod,
        #     )
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
            self.simulator.constrain(rod).using(
                ExponentialDampingBC,
                constrained_position_idx=(0,),
                constrained_director_idx=(0,),
                time_step=self.time_step,
                nu=400 / 10 / 10 * 10,  # 5*5*2,
                rod=rod,
            )

        # Control muscle forces
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

            def apply_forces(self, system, time: np.float = 0.0):
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
                                + np.sin(
                                    np.pi * (time - delay) / activation_time
                                    - 0.5 * np.pi
                                )
                            )
                        )
                        ** activation_exponent
                    )
                elif time >= delay + activation_time:
                    fiber_activation[:] = 1.0 * activation_factor

        for idx, rod in enumerate(self.ring_rod_list_outer[:]):
            self.simulator.add_forcing_to(rod).using(
                MuscleFiberForceActivationStepFunction,
                activation_time=38e-3,
                activation_factor=1.0,
            )

        # Connect straight rods with a surface offset
        k_connection_btw_straight_straight = (
            np.pi * outer_straight_rod_radius_along_arm * E / n_elem
        )  # 50  # 1e5
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
        ) * 0

        self.straight_straight_rod_connection_list = []
        max_offset_btw_straight_rods = outer_straight_rod_radius + 1e-4

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
                self.simulator.connect(
                    first_rod=rod_one,
                    second_rod=rod_two,
                    first_connect_idx=k,
                    second_connect_idx=k,
                ).using(
                    SurfaceJointSideBySide,
                    k=k_connection_btw_straight_straight[k]
                    * 100
                    * 2
                    * 2
                    * 2
                    * 2
                    * 2
                    * 2
                    / 50,  #  / 2 #/30  # *4, # for extension
                    nu=nu_connection_btw_straight_straight[k] * 1.5,
                    k_repulsive=k_connection_btw_straight_straight[k]
                    / 50
                    * 100e4
                    / 1e2
                    * 2
                    * 2
                    * 2
                    * 2
                    * 4
                    * 2
                    * 2
                    / 50,  # *2,#*5*2,#/30*1E6,
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
                np.pi
                * rod_two.radius[0]
                * E
                / n_elem_ring_rod
                * rod_two.rest_lengths[0]
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
                k=k_connection_btw_ring_ring * 10 * 3 / 50,
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
                    np.pi * ring_rod_radius * E
                )  # total_number_of_ring_rods
                nu_connection_btw_helix_and_ring = nu * (
                    density * np.pi * ring_rod_radius * ring_rod_radius_length
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
                k=k_connection / 100 / 50 * 3,
                nu=nu_connection / 1000 * 0,
                connection_order=connection_order,
                index_two_opposite_side=index_two_opposite_side,
                index_two_hing_side=index_two_hing_side,
                index_two_hinge_opposite_side=index_two_hinge_opposite_side,
                next_connection_index=next_connection_index,
                next_connection_index_opposite=next_connection_index_opposite,
                ring_rod_start_idx=0,
                ring_rod_end_idx=rod_two.n_elems,
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
                ring_rod_radius_length = np.mean(rod.rest_lengths)

                # Compute spring and damping constants at the connections
                k_connection_btw_straight_circular = (
                    np.pi * ring_rod_radius * E * ring_rod_radius_length
                )
                nu_connection_btw_straight_circular = (
                    nu
                    * (density * np.pi * ring_rod_radius * ring_rod_radius_length)
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
                            k_connection_btw_straight_circular,
                            nu_connection_btw_straight_circular,
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

            k_connection = my_connection[4]
            nu_connection = my_connection[5]
            n_connection = my_connection[6]

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
                    my_surface_pressure_idx = np.where(
                        self.surface_pressure_force_list[list_idx][1]
                        == index_one  # index_two
                    )[0][0]
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
                k=k_connection
                / 10
                * 5
                * 2
                * 2
                * 2
                / 10
                * 10
                * 6
                / 2
                * 2
                * 2
                * 4
                / 50
                / 20
                * 2
                / 50,  # *2*10,# / 10 / 2,#/8/2,  # /48,#/8,#/12,  # * 8 * 2,
                nu=nu_connection / 6 * 0,
                kt=k_connection / 50 * 3 * 6 * 30 / 20 * 3 / 50,  # * 2,  # *2,# / 2,
                k_repulsive=k_connection
                * 4e6
                / 4e7
                * 5
                * 2
                * 2
                * 2
                / 10
                * 10
                * 2
                * 2
                * 2
                * 2
                * 2
                * 2
                * 2
                * 2
                / 60
                * 30
                * 6
                / 6
                / 50,  # * 5,# *5,
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

        # Add call backs
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
                    self.callback_params["position"].append(
                        system.position_collection.copy()
                    )
                    self.callback_params["velocity"].append(
                        system.velocity_collection.copy()
                    )
                    self.callback_params["radius"].append(system.radius.copy())
                    self.callback_params["com"].append(
                        system.compute_position_center_of_mass()
                    )
                    if current_step == 0:
                        self.callback_params["lengths"].append(
                            system.rest_lengths.copy()
                        )
                    else:
                        self.callback_params["lengths"].append(system.lengths.copy())

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
                    self.callback_params["position"].append(
                        system.position_collection.copy()
                    )
                    self.callback_params["com"].append(
                        system.compute_position_center_of_mass()
                    )
                    self.callback_params["radius"].append(system.radius.copy())
                    self.callback_params["element_position"].append(
                        np.cumsum(system.lengths.copy())
                    )
                    if current_step == 0:
                        self.callback_params["lengths"].append(
                            system.rest_lengths.copy()
                        )
                    else:
                        self.callback_params["lengths"].append(system.lengths.copy())

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
            self.tentacle_length_exp = np.loadtxt(
                "tentacle_data/tentacle_length_vs_time_exp_Kier.txt"
            )
            self.tentacle_length_sim = np.loadtxt(
                "tentacle_data/tentacle_length_vs_time_sim_Kier.txt"
            )
            self.tentacle_velocity_exp = np.loadtxt(
                "tentacle_data/tentacle_velocity_vs_time_exp_Kier.txt"
            )
            self.tentacle_velocity_sim = np.loadtxt(
                "tentacle_data/tentacle_velocity_vs_time_sim_Kier.txt"
            )

            plot_tentacle_length_vs_time(
                self.post_processing_dict_list[0],
                club_length=self.club_length,
                tentacle_length_exp=self.tentacle_length_exp,
                tentacle_length_sim=self.tentacle_length_sim,
            )

            plot_tentacle_velocity_vs_time(
                self.post_processing_dict_list[0],
                tentacle_velocity_exp=self.tentacle_velocity_exp,
                tentacle_velocity_sim=self.tentacle_velocity_sim,
            )

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

            import os

            save_folder = os.path.join(os.getcwd(), "data")
            os.makedirs(save_folder, exist_ok=True)

            time = np.array(self.post_processing_dict_list[0]["time"])

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
                straight_rods_radius_history = np.zeros(
                    (number_of_straight_rods, time.shape[0], n_elems_straight_rods)
                )
                straight_rods_length_history = np.zeros(
                    (number_of_straight_rods, time.shape[0], n_elems_straight_rods)
                )
                for i in range(number_of_straight_rods):
                    straight_rods_position_history[i, :, :, :] = np.array(
                        self.post_processing_dict_list[i]["position"]
                    )
                    straight_rods_radius_history[i, :, :] = np.array(
                        self.post_processing_dict_list[i]["radius"]
                    )
                    straight_rods_length_history[i, :, :] = np.array(
                        self.post_processing_dict_list[i]["lengths"]
                    )

            else:
                straight_rods_position_history = None
                straight_rods_radius_history = None
                straight_rods_length_history = None

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
                        self.post_processing_dict_list[i + number_of_straight_rods][
                            "position"
                        ]
                    )
                    outer_ring_rods_radius_history[i, :, :] = np.array(
                        self.post_processing_dict_list[i + number_of_straight_rods][
                            "radius"
                        ]
                    )
                    outer_ring_rods_length_history[i, :, :] = np.array(
                        self.post_processing_dict_list[i + number_of_straight_rods][
                            "lengths"
                        ]
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
                            i + number_of_straight_rods + number_of_outer_ring_rods
                        ]["position"]
                    )
                    helical_rods_radius_history[i, :, :] = np.array(
                        self.post_processing_dict_list[
                            i + number_of_straight_rods + number_of_outer_ring_rods
                        ]["radius"]
                    )
                    helical_rods_length_history[i, :, :] = np.array(
                        self.post_processing_dict_list[
                            i + number_of_straight_rods + number_of_outer_ring_rods
                        ]["radius"]
                    )
            else:
                helical_rods_position_history = None
                helical_rods_radius_history = None
                helical_rods_length_history = None

            np.savez(
                os.path.join(save_folder, "squid_tentacle_test.npz"),
                time=time,
                straight_rods_position_history=straight_rods_position_history,
                straight_rods_radius_history=straight_rods_radius_history,
                straight_rods_length_history=straight_rods_length_history,
                outer_ring_rods_position_history=outer_ring_rods_position_history,
                outer_ring_rods_radius_history=outer_ring_rods_radius_history,
                outer_ring_rods_length_history=outer_ring_rods_length_history,
                helical_rods_position_history=helical_rods_position_history,
                helical_rods_radius_history=helical_rods_radius_history,
                helical_rods_length_history=helical_rods_length_history,
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
            #     os.path.join(save_folder, "squid_tentacle_test.pkl"), "wb"
            # ) as file:
            #     pickle.dump(my_data_for_rhino, file, protocol=2)

            # Save the tentacle length and tentacle velocity as txt files for post-processing later
            tentacle_time = np.array(self.post_processing_dict_list[0]["time"]) * 1e3
            tentacle_length_hist = (
                np.array(self.post_processing_dict_list[0]["position"])[:, 1, -1]
                + self.club_length
            )
            tentacle_velocity_hist = (
                np.array(self.post_processing_dict_list[0]["velocity"])[:, 1, -1] * 1e-3
            )  # mm to m

            np.savetxt(
                os.path.join(save_folder, "tentacle_length_vs_time_simulation.txt"),
                np.transpose([tentacle_time, tentacle_length_hist]),
                delimiter="   ",
            )
            np.savetxt(
                os.path.join(save_folder, "tentacle_velocity_vs_time_simulation.txt"),
                np.transpose([tentacle_time, tentacle_velocity_hist]),
                delimiter="   ",
            )

        else:
            raise RuntimeError(
                "call back function is not called anytime during simulation, "
                "change COLLECT_DATA=True"
            )
