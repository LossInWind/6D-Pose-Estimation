import numpy as np
from typing import Tuple
from .array import ura_steering_vector
from .projective import C0
from .channel import OFDMSpec, ArraySpec, PathParam


def _stack_real_imag(H: np.ndarray) -> np.ndarray:
    return np.concatenate([H.real.ravel(), H.imag.ravel()], axis=0)


def simulate_single_path(ofdm: OFDMSpec, tx: ArraySpec, rx: ArraySpec, path: PathParam) -> np.ndarray:
    Nr = rx.N
    Nt = tx.N
    Nf = ofdm.num_subcarriers
    Nx_t, Ny_t = tx.shape
    Nx_r, Ny_r = rx.shape
    at = ura_steering_vector(path.aod_phi, path.aod_theta, Nx_t, Ny_t, d_over_lambda=tx.d_over_lambda)
    ar = ura_steering_vector(path.aoa_phi, path.aoa_theta, Nx_r, Ny_r, d_over_lambda=rx.d_over_lambda)
    n = np.arange(Nf)
    f = ofdm.f0 + (n - (Nf-1)/2.0) * ofdm.delta_f
    phase = np.exp(-1j * 2.0 * np.pi * f * path.delay_s)
    H = (path.gain * np.outer(ar, at.conj()))[:, :, None] * phase[None, None, :]
    return H


def numerical_crb(ofdm: OFDMSpec, tx: ArraySpec, rx: ArraySpec, path: PathParam, sigma2: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # parameters: [aod_phi, aod_theta, aoa_phi, aoa_theta, tau, Re(g), Im(g)]
    theta0 = np.array([path.aod_phi, path.aod_theta, path.aoa_phi, path.aoa_theta, path.delay_s, path.gain.real, path.gain.imag], dtype=float)
    eps = np.array([1e-4, 1e-4, 1e-4, 1e-4, 1e-10, 1e-3, 1e-3], dtype=float)
    H0 = simulate_single_path(ofdm, tx, rx, path)
    y0 = _stack_real_imag(H0)
    J = []
    for k in range(len(theta0)):
        th_p = theta0.copy(); th_p[k] += eps[k]
        th_m = theta0.copy(); th_m[k] -= eps[k]
        p_p = PathParam(th_p[0], th_p[1], th_p[2], th_p[3], th_p[4], th_p[5] + 1j*th_p[6])
        p_m = PathParam(th_m[0], th_m[1], th_m[2], th_m[3], th_m[4], th_m[5] + 1j*th_m[6])
        y_p = _stack_real_imag(simulate_single_path(ofdm, tx, rx, p_p))
        y_m = _stack_real_imag(simulate_single_path(ofdm, tx, rx, p_m))
        jac_k = (y_p - y_m) / (2.0 * eps[k])
        J.append(jac_k)
    J = np.stack(J, axis=1)  # shape (M, P)
    # FIM (assuming i.i.d. Gaussian noise with variance sigma2 per real component)
    FIM = (1.0 / sigma2) * (J.T @ J)
    try:
        CRB = np.linalg.inv(FIM)
    except np.linalg.LinAlgError:
        CRB = np.linalg.pinv(FIM)
    var = np.diag(CRB)
    return var, FIM, CRB