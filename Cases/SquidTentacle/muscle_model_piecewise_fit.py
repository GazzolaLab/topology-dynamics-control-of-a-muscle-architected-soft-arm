from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np


class PiecewiseLinear:
    def __init__(self, break_points):
        self.break_points = np.array(break_points)

    def __call__(self, x, y0, y1, y2, y3, k0, k1, k2, k3, *args, **kwargs):
        blocksize = x.shape[0]
        y = np.zeros((blocksize))

        for i in range(blocksize):
            if x[i] >= self.break_points[0] and x[i] <= self.break_points[1]:
                y[i] = y0 + k0 * x[i]
            elif x[i] > self.break_points[1] and x[i] <= self.break_points[2]:
                y[i] = y1 + k1 * x[i]
            elif x[i] > self.break_points[2] and x[i] <= self.break_points[3]:
                y[i] = y2 + k2 * x[i]
            else:
                y[i] = y3 + k3 * x[i]

        return y


def get_active_force_piecewise_linear_function_parameters_for_VanLeeuwen_muscle_model(
    sarcomere_rest_length, myosin_lenths, **kwargs
):
    """
    This function by using the input parameters fits four piecewise function and returns slope,
    y-intercept and break points. Muscle model used is from VanLeeuwen and Kier 1997.

    Parameters
    ----------
    sarcomere_rest_length : numpy.ndarray
        An array containing float.
    myosin_lenths : numpy.ndarray
        An array containing float.
    kwargs

    Returns
    -------

    """
    l_bz = kwargs.get("l_bz", 0.14)
    l_z = kwargs.get("l_z", 0.06)
    D_act = kwargs.get("D_act", 0.68)
    D_myo = kwargs.get("D_myo", 1.90)
    C_myo = kwargs.get("C_myo", 0.44)
    l_min = kwargs.get("l_min", 6e-7)
    n_strain = 200
    eps = kwargs.get("strain", np.linspace(0.1, 2.7, n_strain) - 1)

    blocksize = sarcomere_rest_length.shape[0]

    slope = np.zeros((4, blocksize))
    y_intercept = np.zeros((4, blocksize))
    break_points = np.zeros((4, blocksize))

    f_l = np.zeros((n_strain))

    for k in range(blocksize):
        f_l *= 0
        l0_sarc = sarcomere_rest_length[k]
        l_myo = myosin_lenths[k]
        l_act = l0_sarc - l_z - 0.5 * l_bz
        # Compute the filamentary overlap function for the given sarcomere lengths.
        for i in range(n_strain):
            eps_r = eps[i]

            if (l_act + l_bz + l_z - l0_sarc) / l0_sarc <= eps_r:
                f_l[i] = (l_myo + l_act + l_z - l0_sarc - eps_r * l0_sarc) / (
                    l_myo - l_bz
                )

            elif (l_act + l_z - l0_sarc) / l0_sarc <= eps_r and eps_r <= (
                l_act + l_bz + l_z - l0_sarc
            ) / l0_sarc:
                f_l[i] = 1

            elif (l_myo + l_z - l0_sarc) / l0_sarc <= eps_r and eps_r <= (
                l_act + l_z - l0_sarc
            ) / l0_sarc:
                f_l[i] = (
                    l_myo - l_bz - D_act * (l_act + l_z - l0_sarc - eps_r * l0_sarc)
                ) / (l_myo - l_bz)

            elif (l_min - l0_sarc) / l0_sarc <= eps_r and eps_r <= (
                l_myo + l_z - l0_sarc
            ) / l0_sarc:
                f_l[i] = (
                    l_myo
                    - l_bz
                    - D_act * (l_act + l_z - l0_sarc - eps_r * l0_sarc)
                    - D_myo * (l_myo + l_z - l0_sarc - eps_r * l0_sarc)
                    - C_myo * (l_myo + l_z - l0_sarc - eps_r * l0_sarc)
                ) / (l_myo - l_bz)

        break_points[0, k] = (l_min - l0_sarc) / l0_sarc
        break_points[1, k] = (l_myo + l_z - l0_sarc) / l0_sarc
        break_points[2, k] = (l_act + l_z - l0_sarc) / l0_sarc
        break_points[3, k] = (l_act + l_bz + l_z - l0_sarc) / l0_sarc

        vanleeuwen_muslce_piecewise_obj = PiecewiseLinear(break_points[:, k])
        fit_parameters, _ = optimize.curve_fit(
            vanleeuwen_muslce_piecewise_obj, eps, f_l
        )

        y_intercept[:, k] = fit_parameters[:4]
        slope[:, k] = fit_parameters[4:]

    return slope, y_intercept, break_points


class CubicFunction:
    def __init__(self, extension_strain_limit):
        self.extension_strain_limit = extension_strain_limit

    def __call__(self, x, c1, c2, c3, c4, *args, **kwargs):
        y = np.zeros((x.shape))

        for i in range(x.shape[0]):
            if x[i] < self.extension_strain_limit:
                y[i] = 0
            else:
                y[i] = (
                    c1 * (x[i] - self.extension_strain_limit) ** 3
                    + c2 * (x[i] - self.extension_strain_limit) ** 2
                    + c3 * (x[i] - self.extension_strain_limit)
                    + c4
                )

        return y


def get_passive_force_cubic_function_coefficients_for_VanLeeuwen_muscle_model(
    **kwargs,
):
    c3 = kwargs.get("c3", 1450e3)
    c4 = kwargs.get("c4", -625e3)

    eps_c = kwargs.get("eps_c", 0.773)
    c2 = c3 * eps_c / (c3 * eps_c + c4)
    c1 = c3 / (c2 * eps_c ** (c2 - 1))
    extension_strain_limit = kwargs.get("extension_strain_limit", 0.0)

    eps = kwargs.get("strain", np.linspace(0.1, 2.0, 200) - 1)

    blocksize = eps.shape[0]

    sigma_passive = np.zeros((blocksize))

    for k in range(blocksize):
        if eps[k] < extension_strain_limit:
            sigma_passive[k] = 0

        elif eps[k] < eps_c + extension_strain_limit:
            sigma_passive[k] = c1 * (eps[k] - extension_strain_limit) ** c2

        else:
            sigma_passive[k] = c3 * (eps[k] - extension_strain_limit) + c4

    cubic_function = CubicFunction(extension_strain_limit)
    coefficients, error = optimize.curve_fit(cubic_function, eps, sigma_passive)

    return coefficients
