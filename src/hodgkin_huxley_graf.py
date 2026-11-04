import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd

# Parámetros
C_m = 1.0        # uF/cm^2
V_Na = 50.0      # mV
V_K = -77.0      # mV
V_L = -54.387    # mV
g_Na = 120.0     # mS/cm^2
g_K = 36.0       # mS/cm^2
g_L = 0.3        # mS/cm^2
area = 7.854e-3  # cm^2

# Funciones ((5) -- (10)) -> comprobación de límites para evitar división por cero
def alpha_m(V):
    x = (-V - 40.0) / 10.0
    return np.where(np.abs(x) < 1e-5, 1.0, 0.1 * (-V - 40.0) / (np.exp(x) - 1.0))

def beta_m(V):
    return 4.0 * np.exp((-V - 65.0) / 18.0)

def alpha_h(V):
    return 0.07 * np.exp((-V - 65.0) / 20.0)

def beta_h(V):
    return 1.0 / (np.exp((-V - 35.0) / 10.0) + 1.0)

def alpha_n(V):
    x = (-V - 55.0) / 10.0
    return np.where(np.abs(x) < 1e-5, 0.1, 0.01 * (-V - 55.0) / (np.exp(x) - 1.0))

def beta_n(V):
    return 0.125 * np.exp((-V - 65.0) / 80.0)

# Funciones de corriente 
def I_ext_1(t): return 0.0
def I_ext_2(t): return 0.1 if 0 <= t <= 50 else 0.0

# Sistema de ecuaciones del modelo Hodgkin-Huxley
def hh_model(t, state, I_func):
    V, m, h, n = state
    # Análisis dimensional
    J_ext = I_func(t) / area
    
    dVdt = (J_ext - g_L*(V - V_L) - g_Na*(m**3)*h*(V - V_Na) - g_K*(n**4)*(V - V_K)) / C_m
    dmdt = alpha_m(V)*(1.0 - m) - beta_m(V)*m
    dhdt = alpha_h(V)*(1.0 - h) - beta_h(V)*h
    dndt = alpha_n(V)*(1.0 - n) - beta_n(V)*n
    
    return [dVdt, dmdt, dhdt, dndt]

# Condiciones iniciales
y0 = [-65.0, 0.053, 0.6, 0.318]
t_span = (0, 100)
t_eval = np.linspace(0, 100, 10000)

# Método de integración numérica -> USANDO BIBLIOTECA EXTERNA, BDF (necesario para stiff problems)
sol1 = solve_ivp(hh_model, t_span, y0, args=(I_ext_1,), t_eval=t_eval, method='BDF')
sol2 = solve_ivp(hh_model, t_span, y0, args=(I_ext_2,), t_eval=t_eval, method='BDF', max_step=0.05)

#Voy a generar las gráficas de las soluciones obtenidas
#Representar en una gr´afica V frente a t y en otra m, h, n frente a t.

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
# Gráfica de V frente a t
ax1.plot(sol1.t, sol1.y[0], label='I_ext = 0.0', color='blue', linewidth=1)
ax1.plot(sol2.t, sol2.y[0], label='I_ext = 0.1 (0 <= t <= 50)', color='red', linewidth=1)
ax1.set_title('Hodgkin-Huxley Model: Membrane Potential V(t)')
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('Membrane Potential V (mV)')
ax1.legend()
# Gráfica de m, h, n frente a t
ax2.plot(sol1.t, sol1.y[1], label='m (I_ext = 0.0)', color='blue', linewidth=1)
ax2.plot(sol1.t, sol1.y[2], label='h (I_ext = 0.0)', color='green', linewidth=1)
ax2.plot(sol1.t, sol1.y[3], label='n (I_ext = 0.0)', color='orange', linewidth=1)
ax2.plot(sol2.t, sol2.y[1], label='m (I_ext = 0.1)', color='red', linewidth=1, linestyle='--')
ax2.plot(sol2.t, sol2.y[2], label='h (I_ext = 0.1)', color='green', linewidth=1, linestyle='--')
ax2.plot(sol2.t, sol2.y[3], label='n (I_ext = 0.1)', color='orange', linewidth=1, linestyle='--')
ax2.set_title('Hodgkin-Huxley Model: Gating Variables m, h, n')
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Gating Variables')
ax2.legend()
plt.tight_layout()
plt.savefig('hodgkin_huxley.png', dpi=300)
print("Gráfica guardada con éxito como 'hodgkin_huxley.png'")

data1 = np.column_stack((sol1.t, sol1.y[0], sol1.y[1], sol1.y[2], sol1.y[3]))
data2 = np.column_stack((sol2.t, sol2.y[0], sol2.y[1], sol2.y[2], sol2.y[3]))
df1 = pd.DataFrame(data1, columns=['Time', 'V', 'm', 'h', 'n'])
df2 = pd.DataFrame(data2, columns=['Time', 'V', 'm', 'h', 'n'])
df1.to_csv('hodgkin_huxley_solution1.csv', index=False)
df2.to_csv('hodgkin_huxley_solution2.csv', index=False)
print("Datos de las soluciones guardados con éxito en 'hodgkin_huxley_solution1.csv' y 'hodgkin_huxley_solution2.csv'")


