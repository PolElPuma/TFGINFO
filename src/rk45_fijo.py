from types import SimpleNamespace
import numpy as np
from numba import njit


def make_rk45_solver(func):
    @njit(fastmath=True)
    def _solve_compiled(t0, tf, y0, dt, args):
        n_steps = int(np.round((tf - t0) / dt))
        n_vars = y0.shape[0]
        t = np.linspace(t0, tf, n_steps + 1)

        y = np.empty((n_vars, n_steps + 1), dtype=np.float64)
        curr_y = np.empty(n_vars, dtype=np.float64)
        y_tmp = np.empty(n_vars, dtype=np.float64)

        for j in range(n_vars):
            y[j, 0] = y0[j]
            curr_y[j] = y0[j]

        # Coeficientes Dormand-Prince (orden 5)
        c2, c3, c4, c5 = 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0

        a21 = 1.0 / 5.0
        a31, a32 = 3.0 / 40.0, 9.0 / 40.0
        a41, a42, a43 = 44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0
        a51, a52, a53, a54 = 19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0
        a61, a62, a63, a64, a65 = 9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0

        b1, b3, b4, b5, b6 = 35.0 / 384.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0

        for i in range(n_steps):
            ti = t[i]
            k1 = func(ti, curr_y, *args)

            for j in range(n_vars):
                y_tmp[j] = curr_y[j] + dt * (a21 * k1[j])
            k2 = func(ti + c2 * dt, y_tmp, *args)

            for j in range(n_vars):
                y_tmp[j] = curr_y[j] + dt * (a31 * k1[j] + a32 * k2[j])
            k3 = func(ti + c3 * dt, y_tmp, *args)

            for j in range(n_vars):
                y_tmp[j] = curr_y[j] + dt * (a41 * k1[j] + a42 * k2[j] + a43 * k3[j])
            k4 = func(ti + c4 * dt, y_tmp, *args)

            for j in range(n_vars):
                y_tmp[j] = curr_y[j] + dt * (a51 * k1[j] + a52 * k2[j] + a53 * k3[j] + a54 * k4[j])
            k5 = func(ti + c5 * dt, y_tmp, *args)

            for j in range(n_vars):
                y_tmp[j] = curr_y[j] + dt * (a61 * k1[j] + a62 * k2[j] + a63 * k3[j] + a64 * k4[j] + a65 * k5[j])
            k6 = func(ti + dt, y_tmp, *args)

            for j in range(n_vars):
                val = curr_y[j] + dt * (b1 * k1[j] + b3 * k3[j] + b4 * k4[j] + b5 * k5[j] + b6 * k6[j])
                curr_y[j] = val
                y[j, i + 1] = val

        return t, y

    def solve(t_span, y0, dt, args=()):
        y0_arr = np.asarray(y0, dtype=np.float64)
        t, y = _solve_compiled(float(t_span[0]), float(t_span[1]), y0_arr, float(dt), args)
        return SimpleNamespace(t=t, y=y)

    return solve