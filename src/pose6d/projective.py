import numpy as np
from numpy.linalg import svd, norm
from .se3 import normalize_vector


# Speed of light (m/s)
C0 = 299792458.0


def sph_to_cart(phi: float, theta: float) -> np.ndarray:
    # phi: azimuth, theta: elevation (paper convention)
    return np.array([
        np.cos(phi) * np.sin(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(theta)
    ], dtype=float)


def cart_to_sph(n: np.ndarray) -> tuple:
    x, y, z = n
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.arctan2(y, x)
    return phi, theta


def ao_to_virtual_point(phi: float, theta: float) -> np.ndarray:
    # Eq. (19) in paper: v_bar = [cos phi tan theta, sin phi tan theta, 1]^T
    if np.isclose(np.cos(theta), 0.0):
        # near 90 deg elevation, clip
        theta = np.nextafter(theta, 0.0)
    vx = np.cos(phi) * np.tan(theta)
    vy = np.sin(phi) * np.tan(theta)
    return np.array([vx, vy], dtype=float)


def virtual_point_to_bearing(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(2)
    # Homogeneous [vx, vy, 1] gives bearing up to scale; we normalize direction
    b = np.array([v[0], v[1], 1.0], dtype=float)
    return normalize_vector(b)


def build_epipolar_matrix(vs: np.ndarray, nus: np.ndarray) -> np.ndarray:
    # vs, nus: (N,2) virtual plane coords. Build A for A e = 0
    assert vs.shape == nus.shape and vs.shape[1] == 2
    N = vs.shape[0]
    A = np.zeros((N, 9), dtype=float)
    for i in range(N):
        u = np.array([vs[i, 0], vs[i, 1], 1.0])
        up = np.array([nus[i, 0], nus[i, 1], 1.0])
        A[i] = [
            up[0]*u[0], up[0]*u[1], up[0],
            up[1]*u[0], up[1]*u[1], up[1],
            u[0],      u[1],      1.0
        ]
    return A


def essential_from_correspondences(vs: np.ndarray, nus: np.ndarray) -> np.ndarray:
    A = build_epipolar_matrix(vs, nus)
    _, _, VT = svd(A)
    E = VT[-1].reshape(3, 3)
    # enforce rank-2
    U, S, VT2 = svd(E)
    S = np.array([(S[0]+S[1])/2.0, (S[0]+S[1])/2.0, 0.0])
    E_rank2 = U @ np.diag(S) @ VT2
    return E_rank2


def decompose_essential(E: np.ndarray) -> list:
    # Returns candidate (R, t_dir) with det(R)>0, t_dir unit
    U, _, VT = svd(E)
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    if np.linalg.det(VT) < 0:
        VT[-1, :] *= -1
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    R1 = U @ W @ VT
    R2 = U @ W.T @ VT
    t_hat = U[:, 2]
    candidates = []
    for R in (R1, R2):
        if np.linalg.det(R) < 0:
            R = -R
        for t in (t_hat, -t_hat):
            candidates.append((R, normalize_vector(t)))
    return candidates


def triangulate_midpoint(c1: np.ndarray, d1: np.ndarray, c2: np.ndarray, d2: np.ndarray) -> tuple:
    # Lines: p1=c1 + a d1, p2=c2 + b d2; return midpoint and (a,b)
    d1 = normalize_vector(d1)
    d2 = normalize_vector(d2)
    r = c2 - c1
    a11 = np.dot(d1, d1)
    a22 = np.dot(d2, d2)
    a12 = -np.dot(d1, d2)
    b1 = np.dot(r, d1)
    b2 = np.dot(r, d2)
    A = np.array([[a11, a12], [a12, a22]], dtype=float)
    b = np.array([b1, b2], dtype=float)
    try:
        x = np.linalg.solve(A, b)
        a, bpar = x[0], x[1]
    except np.linalg.LinAlgError:
        # nearly parallel; fallback
        a = b1 / max(a11, 1e-12)
        bpar = b2 / max(a22, 1e-12)
    p1 = c1 + a * d1
    p2 = c2 + bpar * d2
    return 0.5 * (p1 + p2), a, bpar