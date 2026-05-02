import numpy as np
import time

def extract_loop(y, dx=1.0):
    n = len(y)
    k1 = np.array([-3, -2, -1, 0, 1, 2, 3]) / (28.0 * dx)
    k2 = np.array([5, 0, -3, -4, -3, 0, 5]) / (42.0 * dx**2)
    k3 = np.array([-1, 1, 1, 0, -1, -1, 1]) / (6.0 * dx**3)
    
    inv_exps, inv_sins, inv_pows, inv_rats = [], [], [], []
    for i in range(3, n - 3):
        window = y[i-3:i+4]
        dy = np.sum(window * k1)
        ddy = np.sum(window * k2)
        dddy = np.sum(window * k3)
        
        if np.abs(dy) < 1e-3: continue
            
        inv_exp = ddy / dy
        inv_sin = dddy / dy
        inv_pow = (dy * dddy) / (ddy**2) if np.abs(ddy) > 1e-3 else 0
        inv_rat = inv_sin - 1.5 * (inv_exp**2)
        
        inv_exps.append(inv_exp)
        inv_sins.append(inv_sin)
        inv_pows.append(inv_pow)
        inv_rats.append(np.abs(inv_rat))
        
    return np.array([np.var(inv_exps), np.var(inv_sins), np.var(inv_pows), np.mean(inv_rats)])


def extract_vec(y, dx=1.0):
    n = len(y)
    if n < 7: return np.array([0.0, 0.0, 0.0, 0.0])
        
    k1 = np.array([3, 2, 1, 0, -1, -2, -3]) / (28.0 * dx)
    k2 = np.array([5, 0, -3, -4, -3, 0, 5]) / (42.0 * dx**2)
    k3 = np.array([1, -1, -1, 0, 1, 1, -1]) / (6.0 * dx**3)
    
    dy = np.convolve(y, k1, mode='valid')
    ddy = np.convolve(y, k2, mode='valid')
    dddy = np.convolve(y, k3, mode='valid')
    
    mask = np.abs(dy) >= 1e-3
    if np.sum(mask) < 2: return np.array([0.0, 0.0, 0.0, 0.0])
        
    dy_v = dy[mask]
    ddy_v = ddy[mask]
    dddy_v = dddy[mask]
    
    inv_exp = ddy_v / dy_v
    inv_sin = dddy_v / dy_v
    
    mask_p = np.abs(ddy_v) > 1e-3
    inv_pow = np.zeros_like(dy_v)
    inv_pow[mask_p] = (dy_v[mask_p] * dddy_v[mask_p]) / (ddy_v[mask_p]**2)
    
    inv_rat = inv_sin - 1.5 * (inv_exp**2)
    
    return np.array([np.var(inv_exp), np.var(inv_sin), np.var(inv_pow), np.mean(np.abs(inv_rat))])


y = np.sin(np.linspace(0, 10, 128)) + np.random.randn(128)*0.01
out1 = extract_loop(y)
out2 = extract_vec(y)

print(out1)
print(out2)
print("Difference:", np.abs(out1 - out2).max())

# Benchmark
y_test = np.random.randn(1000, 128)
t0 = time.time()
for yt in y_test: extract_loop(yt)
print("Loop time:", time.time() - t0)

t0 = time.time()
for yt in y_test: extract_vec(yt)
print("Vec time:", time.time() - t0)
