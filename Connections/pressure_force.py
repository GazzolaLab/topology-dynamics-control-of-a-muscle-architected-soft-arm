__doc__ = """"""
__all__ = ["PressureForce"]
import numpy as np
import numba
from numba import njit
from elastica.external_forces import NoForces
from elastica._elastica_numba._external_forces import inplace_addition
from elastica._calculus import difference_kernel
from elastica._linalg import _batch_norm


@numba.njit(cache=True)
def _filter_rate_of_change(signal, input_signal, signal_rate_of_change):
    """
    This function filters the rate of change of input signal. Input signal cannot change more than the
    max_signal_rate_of_change in one function call. This function can be used to filter rapid changes of
    input signal.

    Parameters
    ----------
    signal : numpy.ndarray
        1D (blocksize) array containing data with 'float' type.
    input_signal : : numpy.ndarray
        1D (blocksize) array containing data with 'float' type.
    signal_rate_of_change : float

    Returns
    -------

    """
    signal_difference = input_signal - signal
    signal += np.sign(signal_difference) * np.minimum(
        signal_rate_of_change, np.abs(signal_difference)
    )


class PressureForce(NoForces):
    def __init__(self, total_contact_force, total_contact_force_mag, **kwargs):
        super(PressureForce, self).__init__()

        self.total_contact_force = total_contact_force
        self.total_contact_force_mag = total_contact_force_mag

        self.pressure_profile_recorder = kwargs.get("pressure_profile_recorder", None)
        self.step_skip = kwargs.get("step_skip", 0)
        self.counter = 0

    def apply_forces(self, system, time: np.float = 0.0):
        pressure = self._apply_forces(
            system.radius,
            system.lengths,
            system.tangents,
            self.total_contact_force,
            self.total_contact_force_mag,
            system.external_forces,
        )

        if self.counter % self.step_skip == 0:
            if self.pressure_profile_recorder is not None:
                self.pressure_profile_recorder["time"].append(time)

                self.pressure_profile_recorder["pressure_mag"].append(pressure)

                area = np.pi * system.radius**2
                pressure_force = pressure * area * system.tangents

                self.pressure_profile_recorder["external_forces"].append(
                    difference_kernel(pressure_force).copy()
                )
                cumulative_lengths = np.cumsum(system.lengths)

                node_position = np.hstack((np.array([0.0]), cumulative_lengths))

                element_position = 0.5 * (node_position[1:] + node_position[:-1])

                self.pressure_profile_recorder["element_position"].append(
                    element_position.copy()
                )
                control_points = np.vstack((element_position, pressure))
                self.pressure_profile_recorder["control_points"].append(control_points)
                self.pressure_profile_recorder["cross_sectional_area"].append(
                    area.copy()
                )

                surface_area = 2 * np.pi * system.radius * system.lengths
                self.pressure_profile_recorder["surface_area"].append(
                    surface_area.copy()
                )

        self.counter += 1

    @staticmethod
    @njit(cache=True)
    def _apply_forces(
        radius,
        lengths,
        tangents,
        total_contact_force,
        total_contact_force_mag,
        external_forces,
    ):
        # FIXME surface area or contact area should be adjusted.
        surface_area = (2 * np.pi * radius) * (lengths)

        radial_pressure_force = -(
            np.abs(total_contact_force_mag) - _batch_norm(total_contact_force)
        )

        pressure = -radial_pressure_force / surface_area

        cross_sectional_area = np.pi * radius**2

        axial_pressure_force = pressure * cross_sectional_area * tangents

        inplace_addition(external_forces, difference_kernel(-axial_pressure_force))

        total_contact_force *= 0
        total_contact_force_mag *= 0

        return pressure
