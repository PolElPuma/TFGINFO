from types import SimpleNamespace
import numpy as np

#QUIERO QUE ESTA FUNCION ESTE LO MÁS OPTIMIZADA POSIBLE, PODEMOS USAR NUMBA


#@njit(fastmath=True)
def solve(func, t_span, y0, dt, args=()):
    t0, tf = t_span
    n_steps = int(np.round((tf - t0) / dt))
    t = np.linspace(t0, tf, n_steps + 1)

    y = np.empty((len(y0), n_steps + 1), dtype=np.float64)
    y[:, 0] = y0
    curr_y = np.asarray(y0, dtype=np.float64)

    # Coeficientes Dormand-Prince (orden 5)
    c2, c3, c4, c5 = 1/5, 3/10, 4/5, 8/9

    a21 = 1/5
    a31, a32 = 3/40, 9/40
    a41, a42, a43 = 44/45, -56/15, 32/9
    a51, a52, a53, a54 = 19372/6561, -25360/2187, 64448/6561, -212/729
    a61, a62, a63, a64, a65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656

    b1, b3, b4, b5, b6 = 35/384, 500/1113, 125/192, -2187/6784, 11/84

    for i in range(n_steps):
        ti = t[i]
        k1 = np.asarray(func(ti, curr_y, *args), dtype=np.float64)
        k2 = np.asarray(func(ti + c2 * dt, curr_y + dt * (a21 * k1), *args), dtype=np.float64)
        k3 = np.asarray(func(ti + c3 * dt, curr_y + dt * (a31 * k1 + a32 * k2), *args), dtype=np.float64)
        k4 = np.asarray(func(ti + c4 * dt, curr_y + dt * (a41 * k1 + a42 * k2 + a43 * k3), *args), dtype=np.float64)
        k5 = np.asarray(func(ti + c5 * dt, curr_y + dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4), *args), dtype=np.float64)
        k6 = np.asarray(func(ti + dt,      curr_y + dt * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5), *args), dtype=np.float64)

        curr_y = curr_y + dt * (b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6)
        y[:, i + 1] = curr_y

    return SimpleNamespace(t=t, y=y)