import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parámetros 
mu = 0.0021
S = 4.0
e_reg = 3.0
e_irreg = 3.281

# Sistema de ecuaciones del modelo Hindmarsh-Rose
def hindmarsh_rose(t, state, e, mu, S):
    x, y, z = state
    
    dx = y + 3*x**2 - x**3 - z + e
    dy = 1 - 5*x**2 - y
    dz = mu * (-z + S * (x + 1.6))
    return [dx, dy, dz]

# Condiciones iniciales (arbitrarias)
state0 = [0, 0.0, 0.0]

t_span = (0, 15000)
t_eval = np.linspace(10000, 15000, 50000) 

# Integración numérica -> USANDO BIBLIOTECA EXTERNA, RK45
sol_reg = solve_ivp(hindmarsh_rose, t_span, state0, args=(e_reg, mu, S), t_eval=t_eval, method='RK45')
sol_irreg = solve_ivp(hindmarsh_rose, t_span, state0, args=(e_irreg, mu, S), t_eval=t_eval, method='RK45')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

ax1.plot(sol_reg.t, sol_reg.y[0], color='black', linewidth=0.5)
ax1.set_title('Single HR neuron regular (e=3.0)')
ax1.set_xlim(10000, 15000)
ax1.set_ylim(-1.5, 2.0)
ax1.set_ylabel('x(t)')

ax2.plot(sol_irreg.t, sol_irreg.y[0], color='black', linewidth=0.5)
ax2.set_title('Single HR neuron non-regular (e=3.281)')
ax2.set_xlim(10000, 15000)
ax2.set_ylim(-1.5, 2.0)
ax2.set_xlabel('Time')
ax2.set_ylabel('x(t)')

plt.tight_layout()
plt.savefig('hindmarsh_rose.png', dpi=300)
print("Gráfica guardada con éxito como 'hindmarsh_rose.png'")