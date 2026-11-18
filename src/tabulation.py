import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd

#Solver adaptativo con tabulacion sin interpolar -> MAL

def alpha_m(V):
    x = (-V - 40.0) / 10.0
    if abs(x) < 1e-7:
        return 1.0
    return x / np.expm1(x)

def beta_m(V):
    return 4.0 * np.exp((-V - 65.0) / 18.0)

def alpha_h(V):
    return 0.07 * np.exp((-V - 65.0) / 20.0)

def beta_h(V):
    return 1.0 / (np.exp((-V - 35.0) / 10.0) + 1.0)

def alpha_n(V):
    x = (-V - 55.0) / 10.0
    if abs(x) < 1e-7:
        return 0.1
    return 0.1 * x / np.expm1(x)

def beta_n(V):
    return 0.125 * np.exp((-V - 65.0) / 80.0)

#Tabula una función redondeando a la resolución más cercana. Devuelve una función de búsqueda que devuelve el valor tabulado más cercano para un valor de entrada dado.
def tabulate_function(func, V_range, resolution):
    V_min, V_max = float(V_range[0]), float(V_range[1])
    V_values = np.linspace(V_min, V_max, resolution)
    
    table = [func(float(v)) for v in V_values]
    inv_dV = (resolution - 1) / (V_max - V_min)
    max_idx = resolution - 1

    def lookup(V):
        idx = int((V - V_min) * inv_dV + 0.5)
        return table[0 if idx < 0 else (max_idx if idx > max_idx else idx)]

    return lookup

#Tabula una función interpolando linealmente entre los valores tabulados. Devuelve una función de búsqueda que devuelve el valor interpolado para un valor de entrada dado.
def tabulate_function_interp(func, V_range, resolution):
    V_min, V_max = float(V_range[0]), float(V_range[1])
    V_values = np.linspace(V_min, V_max, resolution)
    table = [func(float(v)) for v in V_values]
    
    inv_dV = (resolution - 1) / (V_max - V_min)
    max_idx = resolution - 2

    def lookup(V):
        pos = (V - V_min) * inv_dV
        idx = int(pos)
        idx = 0 if idx < 0 else (max_idx if idx > max_idx else idx)
        frac = pos - idx
        return table[idx] + frac * (table[idx + 1] - table[idx])

    return lookup

res10K = tabulate_function(alpha_m, V_range=(-80, 55), resolution=10000)
res10K_interp = tabulate_function_interp(alpha_m, V_range=(-80, 55), resolution=10000)
res5K = tabulate_function(alpha_m, V_range=(-80, 55), resolution=5000)
res5K_interp = tabulate_function_interp(alpha_m, V_range=(-80, 55), resolution=5000)
res1K = tabulate_function(alpha_m, V_range=(-80, 55), resolution=1000)
res1K_interp = tabulate_function_interp(alpha_m, V_range=(-80, 55), resolution=1000)


def measure_execution_time(func, inputs):
    import time
    start_time = time.time()
    for V in inputs:
        func(V)
    end_time = time.time()
    return end_time - start_time
inputs = np.random.uniform(-80, 55, size=10000)
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
ax.bar_label(ax.containers[0], fmt='%.4f', padding=3, fontsize=9)
ax.set_ylabel('Execution Time (s)')
ax.set_title('Execution Time Comparison of alpha_m Function Variants')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('execution_time_comparison.png')


def tabulate_hh_functions(resolution):
    V_range = (-80, 55)
    return [
        tabulate_function(alpha_m, V_range, resolution),
        tabulate_function(beta_m, V_range, resolution),
        tabulate_function(alpha_h, V_range, resolution),
        tabulate_function(beta_h, V_range, resolution),
        tabulate_function(alpha_n, V_range, resolution),
        tabulate_function(beta_n, V_range, resolution)
    ]
def tabulate_hh_functions_interp(resolution):
    V_range = (-80, 55)
    return [
        tabulate_function_interp(alpha_m, V_range, resolution),
        tabulate_function_interp(beta_m, V_range, resolution),
        tabulate_function_interp(alpha_h, V_range, resolution),
        tabulate_function_interp(beta_h, V_range, resolution),
        tabulate_function_interp(alpha_n, V_range, resolution),
        tabulate_function_interp(beta_n, V_range, resolution)
    ]

import sys

def measure_memory_usage(func, V_range, resolutions):
    memory_usage = []
    for res in resolutions:
        tabulated_func = tabulate_function(func, V_range, res)
        
        free_vars = tabulated_func.__code__.co_freevars
        cells = tabulated_func.__closure__
        table = next(c.cell_contents for name, c in zip(free_vars, cells) if name == 'table')
        if isinstance(table, list):
            total_bytes = sys.getsizeof(table) + sum(sys.getsizeof(item) for item in table)
        else:
            total_bytes = sys.getsizeof(table)
            
        memory_usage.append(total_bytes)
    return memory_usage

resolutions = [1000, 5000, 10000, 20000, 50000, 100000]
memory_usage = measure_memory_usage(alpha_m, V_range=(-80, 55), resolutions=resolutions)
memory_kb = [mem / 1024 for mem in memory_usage]  
plt.figure(figsize=(8, 5))
plt.plot(resolutions, memory_kb, marker='o', linestyle='-', color='tab:blue', linewidth=2, markersize=6)
for res, mem in zip(resolutions, memory_kb):
    plt.annotate(
        f'{mem:.1f} KB',
        xy=(res, mem),
        xytext=(0, 8),
        textcoords='offset points',
        ha='center',
        fontsize=9
    )

plt.xlabel('Resolution (number of grid points)')
plt.ylabel('Memory Usage (KB)')
plt.title('Memory Usage vs. Resolution for Tabulated alpha_m')
plt.grid(True, linestyle='--', alpha=0.5)
plt.margins(y=0.15)  
plt.tight_layout()
plt.savefig('memory_usage_comparison.png', dpi=150)




