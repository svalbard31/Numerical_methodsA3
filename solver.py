import scipy as sp 


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# parameters (replace with your assignment's values)
g, l = 9.81, 1.0
d, A = 0.0, 0.0          # part 2: conservative case
y0 = [np.pi*0.5, 0.0]          # [theta(0), omega(0)] placeholder
t_span = (0.0, 20.0)

def rhs(t, y):
    y1, y2 = y
    return [y2, -d*y2 - (g/l)*np.sin(y1) + A*np.sin(t)]

def energy(y1, y2):
    return 0.5*l**2*y2**2 + g*l*(1 - np.cos(y1))

t_eval = np.linspace(*t_span, 2001)
tols = {
    "reasonable (rtol=1e-6, atol=1e-9)":   dict(rtol=1e-6,  atol=1e-9),
    "tight (rtol=1e-12, atol=1e-14)":      dict(rtol=1e-12, atol=1e-14),
}

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for label, tol in tols.items():
    sol = solve_ivp(rhs, t_span, y0, method="RK45", t_eval=t_eval, **tol)
    E = energy(*sol.y)
    err = np.abs(E - E[0]) / E[0]
    print(f"{label}: nfev={sol.nfev}, max rel. energy error={err.max():.2e}")
    ax[0].plot(sol.t, sol.y[0], label=label)
    ax[1].semilogy(sol.t, err + 1e-18, label=label)

ax[0].set(xlabel="t", ylabel="θ(t)"); ax[0].legend()
ax[1].set(xlabel="t", ylabel="|E-E0|/E0"); ax[1].legend()
plt.tight_layout(); plt.show()