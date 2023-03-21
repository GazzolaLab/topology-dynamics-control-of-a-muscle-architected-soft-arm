__doc__ = """ Muscular rod equations implementation for Elastica Numpy implementation. Different than Cosserat rod
equations, muscluar rod uses a nonlinear constitutive model for stress."""

__all__ = ["MuscularRod"]

import numpy as np
from elastica._elastica_numpy._linalg import _batch_matvec

from elastica.rod.factory_function import allocate
from elastica._elastica_numpy._rod._cosserat_rod import CosseratRod


class MuscularRod(CosseratRod):
    def __init__(
        self,
        n_elements,
        _vector_states,
        _matrix_states,
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
        super(MuscularRod, self).__init__(
            n_elements,
            _vector_states,
            _matrix_states,
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
        )

        # Variables for constitutive model
        self.fiber_force_passive = np.zeros((n_elements))
        self.fiber_force_active = np.zeros((n_elements))
        self.fiber_activation = np.zeros((n_elements))

        # rest base area
        self.rest_area = np.pi * self.radius * self.radius

        if kwargs.__contains__("fiber_activation"):
            self.fiber_activation = kwargs.get("fiber_activation")
            #     (
            #     fiber_activation_temp
            #     if hasattr(fiber_activation_temp, "__call__")
            #     else lambda time_v: fiber_activation_temp
            # )

        if kwargs.__contains__("P1"):
            self.P1 = kwargs.get("P1")

        else:
            raise AttributeError("Did you forget to input P1 in kwargs ?")

        if kwargs.__contains__("P2"):
            self.P2 = kwargs.get("P2")
        else:
            raise AttributeError("Did you forget to input P2 in kwargs ?")

        if kwargs.__contains__("P3"):
            self.P3 = kwargs.get("P3")
        else:
            raise AttributeError("Did you forget to input P3 in kwargs ?")

        if kwargs.__contains__("P4"):
            self.P4 = kwargs.get("P4")
        else:
            raise AttributeError("Did you forget to input P4 in kwargs ?")

        if kwargs.__contains__("stretch_optimal"):
            self.stretch_optimal = kwargs.get("stretch_optimal")
        else:
            raise AttributeError("Did you forget to input stretch_optimal in kwargs ?")

        if kwargs.__contains__("stretch_critical"):
            self.stretch_critical = kwargs.get("stretch_critical")
        else:
            raise AttributeError("Did you forget to input stretch_critical in kwargs ?")

        if kwargs.__contains__("stress_maximum"):
            self.stress_maximum = kwargs.get("stress_maximum")
        else:
            raise AttributeError("Did you forget to input stress_maximum in kwargs ?")

        if kwargs.__contains__("G1"):
            self.G1 = kwargs.get("G1")
        else:
            raise AttributeError("Did you forget to input G1 in kwargs ?")

        if kwargs.__contains__("G2"):
            self.G2 = kwargs.get("G2")
        else:
            raise AttributeError("Did you forget to input G2 in kwargs ?")

        self._compute_shear_stretch_strains()
        self._compute_bending_twist_strains()

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
            _vector_states,
            _matrix_states,
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
            _vector_states,
            _matrix_states,
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

    def _compute_internal_shear_stretch_stresses_from_model(self):
        # TODO: Change the code here and make it same as the Numba version
        """
        Constitutive model for the muscle.

        References
        ----------
        Blemker SS, Pinsky PM, Delp SL. A 3D model of muscle reveals the causes of nonuniform strains in the
         biceps brachii. J Biomech. 2005;38(4):657-665. doi:10.1016/j.jbiomech.2004.04.009

        Returns
        -------

        """
        self._compute_shear_stretch_strains()  # concept : needs to compute sigma

        # Computes internal stress all components. But change the third component with the new stress
        # computed by muscle constitutive model.
        self.internal_stress = _batch_matvec(
            self.shear_matrix, self.sigma - self.rest_sigma
        )

        self.fiber_force_passive *= 0.0
        self.fiber_force_active *= 0.0

        # Lambda stretch ratio in the paper is dilatation in our simulation
        stretch = self.dilatation

        # Compute passive fiber force. Table 1 in reference
        # Case 1 stretch =< stretch_optimal, fiber_force_passive = 0 No action needed

        # Case 2 stretch_optimal < stretch < stretch_critical
        idx_case_two_passive = np.where(
            (stretch > self.stretch_optimal) & (stretch < self.stretch_critical)
        )[0]
        # Case 3 stretch >= stretch_critical
        idx_case_three_passive = np.where(stretch >= self.stretch_critical)[0]

        self.fiber_force_passive[idx_case_two_passive] = self.P1 * (
            np.exp(
                self.P2 * (stretch[idx_case_two_passive] / self.stretch_optimal - 1.0)
            )
            - 1.0
        )

        self.fiber_force_passive[idx_case_three_passive] = (
            self.P3 * stretch[idx_case_three_passive] / self.stretch_optimal + self.P4
        )

        # Compute active fiber force. Table 1 in reference
        # Case 1 stretch <= 0.6*stretch_optimal
        idx_case_one_active = np.where(stretch <= 0.6 * self.stretch_optimal)[0]
        # Case 2 stretch >= 1.4 * stretch_optimal
        idx_case_two_active = np.where(stretch >= 1.4 * self.stretch_optimal)[0]
        # Case 3 0.6*stretch_optimal< stretch <  1.4 * stretch_optimal
        idx_case_three_active = np.where(
            (stretch > 0.6 * self.stretch_optimal)
            & (stretch < 1.4 * self.stretch_optimal)
        )[0]

        self.fiber_force_active[idx_case_one_active] = (
            9.0 * (stretch[idx_case_one_active] / self.stretch_optimal - 0.4) ** 2
        )
        self.fiber_force_active[idx_case_two_active] = (
            9.0 * (stretch[idx_case_two_active] / self.stretch_optimal - 1.6) ** 2
        )
        self.fiber_force_active[idx_case_three_active] = (
            1.0
            - 4.0 * (1.0 - stretch[idx_case_three_active] / self.stretch_optimal) ** 2
        )

        # Compute total fiber force. Eq 7 in reference
        fiber_force_total = (
            self.fiber_force_passive + self.fiber_activation * self.fiber_force_active
        )

        # Compute total internal stress in the tangent direction. Inside the internal
        # stress variable we are storing internal force in material frame. This is
        # confusing, but we store n_L eqn 2.11 in RSoS 2018 paper.
        self.internal_stress[2] = self.rest_area * (
            self.stress_maximum * fiber_force_total * stretch / self.stretch_optimal
        )
