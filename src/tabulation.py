import math
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numba import njit


@njit(fastmath=True)
def alpha_m(V):
    x = (-V - 40.0) / 10.0
    if abs(x) < 1e-7:
        return 1.0
    return x / math.expm1(x)


@njit(fastmath=True)
def beta_m(V):
    return 4.0 * math.exp((-V - 65.0) / 18.0)


@njit(fastmath=True)
def alpha_h(V):
    return 0.07 * math.exp((-V - 65.0) / 20.0)


@njit(fastmath=True)
def beta_h(V):
    return 1.0 / (math.exp((-V - 35.0) / 10.0) + 1.0)


@njit(fastmath=True)
def alpha_n(V):
    x = (-V - 55.0) / 10.0
    if abs(x) < 1e-7:
        return 0.1
    return 0.1 * x / math.expm1(x)


@njit(fastmath=True)
def beta_n(V):
    return 0.125 * math.exp((-V - 65.0) / 80.0)


def tabulate_function(func, V_range, resolution):
    V_min, V_max = float(V_range[0]), float(V_range[1])
    V_values = np.linspace(V_min, V_max, resolution)

    table = np.array([func(float(v)) for v in V_values], dtype=np.float64)
    inv_dV = (resolution - 1) / (V_max - V_min)
    max_idx = resolution - 1

    @njit(fastmath=True)
    def lookup(V):
        idx = int((V - V_min) * inv_dV + 0.5)
        if idx < 0:
            idx = 0
        elif idx > max_idx:
            idx = max_idx
        return table[idx]

    return lookup


def tabulate_function_interp(func, V_range, resolution):
    V_min, V_max = float(V_range[0]), float(V_range[1])
    V_values = np.linspace(V_min, V_max, resolution)

    table = np.array([func(float(v)) for v in V_values], dtype=np.float64)
    inv_dV = (resolution - 1) / (V_max - V_min)
    max_idx = resolution - 2

    @njit(fastmath=True)
    def lookup(V):
        pos = (V - V_min) * inv_dV
        idx = int(pos)
        if idx < 0:
            idx = 0
        elif idx > max_idx:
            idx = max_idx
        frac = pos - idx
        return table[idx] + frac * (table[idx + 1] - table[idx])

    return lookup


def tabulate_hh_functions(resolution, V_range=(-80.0, 55.0)):
    return (
        tabulate_function(alpha_m, V_range, resolution),
        tabulate_function(beta_m, V_range, resolution),
        tabulate_function(alpha_h, V_range, resolution),
        tabulate_function(beta_h, V_range, resolution),
        tabulate_function(alpha_n, V_range, resolution),
        tabulate_function(beta_n, V_range, resolution),
    )


def tabulate_hh_functions_interp(resolution, V_range=(-80.0, 55.0)):
    return (
        tabulate_function_interp(alpha_m, V_range, resolution),
        tabulate_function_interp(beta_m, V_range, resolution),
        tabulate_function_interp(alpha_h, V_range, resolution),
        tabulate_function_interp(beta_h, V_range, resolution),
        tabulate_function_interp(alpha_n, V_range, resolution),
        tabulate_function_interp(beta_n, V_range, resolution),
    )


if __name__ == "__main__":
    res10K = tabulate_function(alpha_m, V_range=(-80, 55), resolution=10000)
    res10K_interp = tabulate_function_interp(alpha_m, V_range=(-80, 55), resolution=10000)
    res5K = tabulate_function(alpha_m, V_range=(-80, 55), resolution=5000)
    res5K_interp = tabulate_function_interp(alpha_m, V_range=(-80, 55), resolution=5000)
    res1K = tabulate_function(alpha_m, V_range=(-80, 55), resolution=1000)
    res1K_interp = tabulate_function_interp(alpha_m, V_range=(-80, 55), resolution=1000)

    @njit(fastmath=True)
    def measure_execution_time_jit(func, inputs):
        acc = 0.0
        for i in range(inputs.shape[0]):
            acc += func(inputs[i])
        return acc

    def measure_execution_time(func, inputs):
        measure_execution_time_jit(func, inputs[:10])  # Warmup JIT
        start_time = time.perf_counter()
        measure_execution_time_jit(func, inputs)
        return time.perf_counter() - start_time

    inputs = np.random.uniform(-80, 55, size=100000)
    time_original = measure_execution_time(alpha_m, inputs)
    time_res10K = measure_execution_time(res10K, inputs)
    time_res10K_interp = measure_execution_time(res10K_interp, inputs)
    time_res5K = measure_execution_time(res5K, inputs)
    time_res5K_interp = measure_execution_time(res5K_interp, inputs)
    time_res1K = measure_execution_time(res1K, inputs)
    time_res1K_interp = measure_execution_time(res1K_interp, inputs)

    graph_data = {
        'Function': ['Original', 'Tabulated 10K', 'Interpolated 10K', 'Tabulated 5K', 'Interpolated 5K', 'Tabulated 1K', 'Interpolated 1K'],
        'Execution Time (s)': [time_original, time_res10K, time_res10K_interp, time_res5K, time_res5K_interp, time_res1K, time_res1K_interp]
    }
    df = pd.DataFrame(graph_data)
    ax = df.plot.bar(x='Function', y='Execution Time (s)', legend=False, color=['blue', 'orange', 'green', 'red', 'purple'])
    ax.bar_label(ax.containers[0], fmt='%.6f', padding=3, fontsize=9)
    ax.set_ylabel('Execution Time (s)')
    ax.set_title('Execution Time Comparison of alpha_m Function Variants (Numba)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('execution_time_comparison.png')