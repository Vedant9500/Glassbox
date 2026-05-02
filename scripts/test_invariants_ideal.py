import numpy as np

def compute_differential_invariants(y, dx=1.0):
    n = len(y)
    if n < 7:
        return [0, 0, 0, 0]
        
    k1 = np.array([-3, -2, -1, 0, 1, 2, 3]) / (28.0 * dx)
    k2 = np.array([5, 0, -3, -4, -3, 0, 5]) / (42.0 * dx**2)
    k3 = np.array([-1, 1, 1, 0, -1, -1, 1]) / (6.0 * dx**3)
    
    inv_exps, inv_sins, inv_pows, inv_rats = [], [], [], []
    
    for i in range(3, n - 3):
        window = y[i-3:i+4]
        dy = np.sum(window * k1)
        ddy = np.sum(window * k2)
        dddy = np.sum(window * k3)
        
        if np.abs(dy) < 1e-3:
            continue
            
        inv_exp = ddy / dy
        inv_sin = dddy / dy
        inv_pow = (dy * dddy) / (ddy**2) if np.abs(ddy) > 1e-3 else 0
        inv_rat = inv_sin - 1.5 * (inv_exp**2)
        
        inv_exps.append(inv_exp)
        inv_sins.append(inv_sin)
        inv_pows.append(inv_pow)
        inv_rats.append(np.abs(inv_rat))
        
    if len(inv_exps) < 2:
        return [0, 0, 0, 0]
        
    return [
        np.var(inv_exps),
        np.var(inv_sins),
        np.var(inv_pows),
        np.mean(inv_rats)
    ]

x = np.linspace(-2, 2, 500)
dx = x[1] - x[0]

funcs = {
    "exp(2x)": np.exp(2 * x),
    "sin(3x)": np.sin(3 * x),
    "power(x^3)": (x+10)**3, # shifted to avoid 0 derivative
    "power(x^2)": (x+10)**2,
    "rational(1/x)": 1 / (x + 10),
    "rational(2x+1/3x-2)": (2*x + 1) / (3*x - 20)
}

print(f"{'Function':20s} | {'Var_Exp':>10s} | {'Var_Sin':>10s} | {'Var_Pow':>10s} | {'Mean_Rat':>10s}")
print("-" * 70)
for name, y in funcs.items():
    invs = compute_differential_invariants(y, dx)
    print(f"{name:20s} | {invs[0]:10.3e} | {invs[1]:10.3e} | {invs[2]:10.3e} | {invs[3]:10.3e}")
