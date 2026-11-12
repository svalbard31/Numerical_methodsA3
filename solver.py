import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

g=9.81
l=1
y1_0=np.pi*0.5
y2_0=4.0
A=10
d=1.0
y = np.array([y1_0, y2_0])

def theta(t,y):
    y1 = y[0]
    y2 = y[1]
    return [y2, A*np.sin(t)-d*y2-g/l*np.sin(y1)]

def energy(t,y):
    y1 = y[0]
    y2 = y[1]
    return 0.5*l**2*y2**2 + g*l*(1-np.cos(y1))

time = np.linspace(0, 100, 2000)
solution = solve_ivp(
    theta, [0, 100], [y1_0, y2_0], t_eval=time,
    rtol=1e-2, atol=1e-4, method='RK45', max_step=0.02
)
solution_strict = solve_ivp(
    theta, [0, 100], [y1_0, y2_0], t_eval=time,
    rtol=1e-11, atol=1e-13, method='RK45', max_step=0.02
)

energy_change = abs(energy(solution.t[-1], solution.y[:,-1]) - energy(solution.t[0], solution.y[:,0]))/energy(solution.t[0], solution.y[:,0])
#print(f"Energy change (loose): {energy_change:.2e}")

energy_change_strict = abs(energy(solution_strict.t[-1], solution_strict.y[:,-1]) - energy(solution_strict.t[0], solution_strict.y[:,0]))/energy(solution_strict.t[0], solution_strict.y[:,0])
print(f"Energy change (strict): {energy_change_strict:.2e}")
#plt.plot(solution.t, solution.y[0])
#plt.plot(solution_strict.t, solution_strict.y[0], '--')
plt.plot(solution_strict.y[0], solution_strict.y[1])
#plt.plot(solution.y[0], solution.y[1], '--')
plt.xlabel(r"Angle $\theta$ (rad)")
plt.ylabel(r"Angular velocity $\dot{\theta}$ (rad/s)")
plt.grid()
plt.show()