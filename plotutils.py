
import numpy as np

def ellipsoid_surface_2D(P, n=100):
    lam, V = np.linalg.eig(P)
    phi = np.linspace(0, 2 * np.pi, n)
    a = (V @ np.diag(np.sqrt(lam))) @ np.vstack([np.cos(phi), np.sin(phi)])
    return a
