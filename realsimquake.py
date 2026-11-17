import numpy as np

def spectrum(w_g, w, z, S_o, w_l=1.0, z_l=0.7):
    kt = ((w_g**4 + 4 * z**2 * w_g**2 * w**2) / ((w_g**2 - w**2)**2 + 4 * z**2 * w_g**2 * w**2)) * S_o
    cp = (w/w_l)**4 / ((1 - (w/w_l)**2)**2 + (2*z_l*w/w_l)**2)
    S = kt * cp
    return S

def envelope(t, alpha):
    return alpha * np.e * t * np.exp(-t * alpha)

def simulate_earthquake(w_g=15.6, z=0.6, S_o=1, w_max=2*np.pi*25, M=1000,
                    duration=30.0, dt=0.01, alpha=0.25, pga_g=0.3, seed=0):

    dw = w_max/M
    w_n = (np.arange(M) + 0.5)*dw
    
    S = spectrum(w_g, w_n, z, S_o)
    
    P = S * dw
    
    np.random.seed(seed)
    
    Amp = np.sqrt(2*P)
    phi = np.array([np.random.uniform(0, 2*np.pi) for m in range(M)])
    
    t = np.arange(0, duration, dt)
    
    wave_sum = np.zeros_like(t)
    
    for i in range(M):
        wave_sum += Amp[i]*np.cos(w_n[i]*t + phi[i])
    
    E = envelope(t, alpha)
    a_s = wave_sum * E
    
    a = a_s * (pga_g*9.81) / np.max(np.abs(a_s))  
    return t, a


def load_real_quake(file):
    with open(file) as f:
        lines = f.read().splitlines()
    header = lines[3].split()
    npts = int(header[1].strip(','))
    dt = float(header[3])
    t = np.arange(npts)*dt
    vals = []
    for ln in lines[4:]:
        vals = vals + ln.split() 
    real_a = np.array([float(x) for x in vals])*9.81
    return t, real_a
           