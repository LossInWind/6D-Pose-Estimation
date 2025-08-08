import numpy as np
from typing import Tuple


def ura_steering_vector(phi: float, theta: float, Nx: int, Ny: int, d_over_lambda: float = 0.5) -> np.ndarray:
    # azimuth phi, elevation theta; unit normal n = [cos phi sin theta, sin phi sin theta, cos theta]
    # For URA on XY-plane with spacing d, wavenumber k = 2π/λ -> phase π * sin(theta) * (m_y cos phi + m_x sin phi) if d=λ/2
    # Generalize with d_over_lambda factor
    sin_t = np.sin(theta)
    sx = sin_t * np.sin(phi)  # aligns with a1 index (x-axis along y? choose consistent with paper: a1 uses sin(theta) sin(phi)
    sy = sin_t * np.cos(phi)
    mx = np.arange(Nx, dtype=float)
    my = np.arange(Ny, dtype=float)
    phase_x = 2.0 * np.pi * d_over_lambda * np.outer(mx, sx)
    phase_y = 2.0 * np.pi * d_over_lambda * np.outer(my, sy)
    ax = np.exp(1j * phase_x)
    ay = np.exp(1j * phase_y)
    a = np.kron(ay, ax).reshape(Nx * Ny)
    return a / np.sqrt(Nx * Ny)


def ura_shape(N: int) -> Tuple[int, int]:
    # factor N into near-square Nx, Ny
    Ny = int(np.floor(np.sqrt(N)))
    while N % Ny != 0 and Ny > 1:
        Ny -= 1
    Nx = N // Ny
    return Nx, Ny