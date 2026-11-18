import math
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from rk45_fijo import solve
from tabulation import tabulate_hh_functions, tabulate_hh_functions_interp

output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# --- CONFIGURACIÓN DEL EXPERIMENTO ---
resolutions = [500, 1000, 5000, 10000, 20000, 50000, 100000]
ref_resolution = 10000  # Resolución usada para la comparativa directa de tiempos
y0 = [-65.0, 0.053, 0.6, 0.318]
t_span = (0.0, 100.0)
dT = 0.01
n_steps = int(np.round((t_span[1] - t_span[0]) / dT))

# Parámetros del modelo Hodgkin-Huxley
C_m = 1.0        # uF/cm^2
V_Na = 50.0      # mV
V_K = -77.0      # mV
V_L = -54.387    # mV
g_Na = 120.0     # mS/cm^2
g_K = 36.0       # mS/cm^2
g_L = 0.3        # mS/cm^2
area = 7.854e-3  # cm^2

# Funciones continuas (escalares con math para referencia exacta)
def alpha_m(V):
    x = (-V - 40.0) / 10.0
    return 1.0 if abs(x) < 1e-7 else x / math.expm1(x)

def beta_m(V):
    return 4.0 * math.exp((-V - 65.0) / 18.0)

def alpha_h(V):
    return 0.07 * math.exp((-V - 65.0) / 20.0)

def beta_h(V):
    return 1.0 / (math.exp((-V - 35.0) / 10.0) + 1.0)

def alpha_n(V):
    x = (-V - 55.0) / 10.0
    return 0.1 if abs(x) < 1e-7 else 0.1 * x / math.expm1(x)

def beta_n(V):
    return 0.125 * math.exp((-V - 65.0) / 80.0)

def I_ext_1(t):
    return 0.1 if 0.0 <= t <= 50.0 else 0.0

def hh_model(t, state, I_func, hh_functions):
    V, m, h, n = state
    J_ext = I_func(t) / area
    am, bm, ah, bh, an, bn = hh_functions

    dVdt = (J_ext - g_L*(V - V_L) - g_Na*(m**3)*h*(V - V_Na) - g_K*(n**4)*(V - V_K)) / C_m
    dmdt = am(V)*(1.0 - m) - bm(V)*m
    dhdt = ah(V)*(1.0 - h) - bh(V)*h
    dndt = an(V)*(1.0 - n) - bn(V)*n

    return np.array([dVdt, dmdt, dhdt, dndt], dtype=np.float64)

def measure_hh_memory_kb(funcs_tuple):
    """Calcula la memoria total (en KB) ocupada por las 6 tablas dentro de los closures."""
    total_bytes = 0
    for fn in funcs_tuple:
        if fn.__closure__:
            for cell in fn.__closure__:
                val = cell.cell_contents
                if isinstance(val, list):
                    total_bytes += sys.getsizeof(val) + sum(sys.getsizeof(x) for x in val)
                elif isinstance(val, np.ndarray):
                    total_bytes += sys.getsizeof(val)
    return total_bytes / 1024.0

def benchmark_step_us(funcs_tuple, runs=5):
    """Mide el tiempo mínimo por paso en microsegundos sobre varias repeticiones."""
    best_total = min(
        (lambda t0=time.perf_counter(): (solve(hh_model, t_span, y0, dT, args=(I_ext_1, funcs_tuple)), time.perf_counter() - t0)[1])()
        for _ in range(runs)
    )
    return (best_total / n_steps) * 1e6

# 1. Simulación de referencia (continua)
cont_functions = (alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n)
sol_cont = solve(hh_model, t_span, y0, dT, args=(I_ext_1, cont_functions))
step_us_cont = benchmark_step_us(cont_functions)

# 2. Barrido de resoluciones para error y memoria
results = []
solutions_nearest = {}
solutions_interp = {}

for res in resolutions:
    tab_funcs = tabulate_hh_functions(resolution=res)
    mem_kb = measure_hh_memory_kb(tab_funcs)
    sol_tab = solve(hh_model, t_span, y0, dT, args=(I_ext_1, tab_funcs))
    err_tab = np.max(np.abs(sol_cont.y[0] - sol_tab.y[0]))
    solutions_nearest[res] = sol_tab

    interp_funcs = tabulate_hh_functions_interp(resolution=res)
    sol_interp = solve(hh_model, t_span, y0, dT, args=(I_ext_1, interp_funcs))
    err_interp = np.max(np.abs(sol_cont.y[0] - sol_interp.y[0]))
    solutions_interp[res] = sol_interp

    if res == ref_resolution:
        step_us_tab = benchmark_step_us(tab_funcs)
        step_us_interp = benchmark_step_us(interp_funcs)

    results.append({
        'Resolución': res,
        'Error Tab (mV)': err_tab,
        'Error Interp (mV)': err_interp,
        'Memoria (KB)': mem_kb,
    })

# 3. Resumen en consola
df_results = pd.DataFrame(results)
print("Tiempo por paso RK45:")
print(f"  - Continua (Real):  {step_us_cont:.2f} µs")
print(f"  - Tabulada:         {step_us_tab:.2f} µs (speedup: {step_us_cont / step_us_tab:.2f}x)")
print(f"  - Interpolada:      {step_us_interp:.2f} µs (speedup: {step_us_cont / step_us_interp:.2f}x)\n")
print(df_results.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

# 4. Gráfica 1: Comparación de trayectorias V(t)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

ax1.plot(sol_cont.t, sol_cont.y[0], label='Continua (Exacta)', color='black', linewidth=2)
for res, sol in solutions_nearest.items():
    ax1.plot(sol.t, sol.y[0], label=f'Tabulada ({res})', linestyle='--', linewidth=1.2)
ax1.set_title('Potencial de membrana V(t): Continua vs. Vecino más cercano')
ax1.set_ylabel('V (mV)')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right', ncol=2)

ax2.plot(sol_cont.t, sol_cont.y[0], label='Continua (Exacta)', color='black', linewidth=2)
for res, sol in solutions_interp.items():
    ax2.plot(sol.t, sol.y[0], label=f'Interpolada ({res})', linestyle='--', linewidth=1.2)
ax2.set_title('Potencial de membrana V(t): Continua vs. Interpolación lineal')
ax2.set_xlabel('Tiempo (ms)')
ax2.set_ylabel('V (mV)')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right', ncol=2)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'hh_soluciones_comparativa.png'), dpi=200)

# 5. Gráfica 2: Convergencia del error máximo vs. resolución
plt.figure(figsize=(8, 5))
plt.loglog(df_results['Resolución'], df_results['Error Tab (mV)'],
           marker='o', linewidth=2, label='Vecino más cercano $O(\\Delta V)$')
plt.loglog(df_results['Resolución'], df_results['Error Interp (mV)'],
           marker='s', linewidth=2, label='Interpolación lineal $O(\\Delta V^2)$')
plt.xlabel('Resolución de la tabla (puntos)')
plt.ylabel('Error máximo absoluto en V (mV)')
plt.title('Convergencia del error según la resolución de tabulación')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'hh_error_convergencia.png'), dpi=200)

# 6. Gráfica 3: Comparativa directa de tiempo por paso (3 métodos)
fig, ax = plt.subplots(figsize=(7, 5))
methods = ['Continua (Real)', 'Tabulada', 'Tabulada + Interp.']
times_us = [step_us_cont, step_us_tab, step_us_interp]
colors = ['tab:gray', 'tab:blue', 'tab:orange']

bars = ax.bar(methods, times_us, color=colors, width=0.5)
ax.bar_label(bars, fmt='%.2f µs', padding=4, fontsize=10, fontweight='bold')

ax.set_ylabel('Tiempo por paso RK45 (µs)')
ax.set_title(f'Coste temporal por paso de integración (Tabla {ref_resolution} pts)')
ax.margins(y=0.18)
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'hh_tiempo_por_paso.png'), dpi=200)

# 7. Gráfica 4: Uso de memoria total (6 tablas) vs. resolución
plt.figure(figsize=(8, 5))
plt.plot(df_results['Resolución'], df_results['Memoria (KB)'],
         marker='o', color='tab:green', linewidth=2)

for res, mem in zip(df_results['Resolución'], df_results['Memoria (KB)']):
    plt.annotate(f'{mem:.1f} KB', (res, mem), textcoords='offset points', xytext=(0, 8), ha='center', fontsize=9)

plt.xlabel('Resolución de la tabla (puntos por función)')
plt.ylabel('Memoria total de las 6 tablas (KB)')
plt.title('Consumo de memoria vs. Resolución')
plt.margins(y=0.15)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'hh_memoria_comparativa.png'), dpi=200)