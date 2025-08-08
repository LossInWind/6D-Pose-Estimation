import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from .array import ura_steering_vector, ura_shape
from .projective import C0


@dataclass
class ArraySpec:
    N: int
    d_over_lambda: float = 0.5

    @property
    def shape(self) -> Tuple[int, int]:
        return ura_shape(self.N)


@dataclass
class OFDMSpec:
    num_subcarriers: int
    delta_f: float  # Hz
    f0: float       # center Hz (used for lambda)


@dataclass
class PathParam:
    aod_phi: float
    aod_theta: float
    aoa_phi: float
    aoa_theta: float
    delay_s: float
    gain: complex


@dataclass
class ChannelSnapshot:
    Hn: np.ndarray  # shape (N_r, N_t, N_f)
    ofdm: OFDMSpec
    tx: ArraySpec
    rx: ArraySpec


def simulate_snapshot(paths: List[PathParam], ofdm: OFDMSpec, tx: ArraySpec, rx: ArraySpec, snr_db: float = 30.0, rng: np.random.Generator = np.random.default_rng()) -> ChannelSnapshot:
    Nr = rx.N
    Nt = tx.N
    Nf = ofdm.num_subcarriers
    H = np.zeros((Nr, Nt, Nf), dtype=complex)
    Nx_t, Ny_t = tx.shape
    Nx_r, Ny_r = rx.shape
    for p in paths:
        at = ura_steering_vector(p.aod_phi, p.aod_theta, Nx_t, Ny_t, d_over_lambda=tx.d_over_lambda)
        ar = ura_steering_vector(p.aoa_phi, p.aoa_theta, Nx_r, Ny_r, d_over_lambda=rx.d_over_lambda)
        n = np.arange(Nf)
        f = ofdm.f0 + (n - (Nf-1)/2.0) * ofdm.delta_f
        phase = np.exp(-1j * 2.0 * np.pi * f * p.delay_s)
        H += (p.gain * np.outer(ar, at.conj()))[:, :, None] * phase[None, None, :]
    sig_pow = np.mean(np.abs(H)**2) + 1e-12
    noise_pow = sig_pow / (10**(snr_db/10.0))
    noise = (np.sqrt(noise_pow/2) * (rng.normal(size=H.shape) + 1j*rng.normal(size=H.shape))).astype(complex)
    Hn = H + noise
    return ChannelSnapshot(Hn=Hn, ofdm=ofdm, tx=tx, rx=rx)


def estimate_params(snapshot: ChannelSnapshot, grid_size: int = 37) -> List[Tuple[float, float, float, float, float, complex]]:
    out = estimate_multipath_params(snapshot, max_paths=1, zpf=4, grid_size=grid_size)
    return [out[0]]


def _parabolic_peak_refine(y: np.ndarray, k: int) -> float:
    k = int(np.clip(k, 1, len(y)-2))
    ym1, y0, yp1 = y[k-1], y[k], y[k+1]
    denom = (ym1 - 2*y0 + yp1) + 1e-18
    delta = 0.5 * (ym1 - yp1) / denom
    delta = float(np.clip(delta, -1.0, 1.0))
    return k + delta


def _quad2d_refine(P: np.ndarray, i: int, j: int, dphi: float, dth: float) -> Tuple[float, float]:
    # Fit 2D quadratic z = ax^2 + by^2 + cxy + dx + ey + f on 3x3 neighborhood
    h, w = P.shape
    i = int(np.clip(i, 1, h-2))
    j = int(np.clip(j, 1, w-2))
    Z = P[i-1:i+2, j-1:j+2]
    xs, ys = np.meshgrid([-dphi, 0.0, dphi], [-dth, 0.0, dth], indexing='ij')
    A = np.stack([xs.ravel()**2, ys.ravel()**2, (xs*ys).ravel(), xs.ravel(), ys.ravel(), np.ones(9)], axis=1)
    coef, *_ = np.linalg.lstsq(A, Z.ravel(), rcond=None)
    a, b, c, d, e, f0 = coef
    # Stationary point solves gradient = 0
    # [2a x + c y + d = 0, c x + 2b y + e = 0]
    M = np.array([[2*a, c], [c, 2*b]])
    v = -np.array([d, e])
    try:
        sol = np.linalg.solve(M, v)
        dx, dy = sol[0], sol[1]
        dx = float(np.clip(dx, -dphi, dphi))
        dy = float(np.clip(dy, -dth, dth))
    except np.linalg.LinAlgError:
        dx, dy = 0.0, 0.0
    return dx, dy


def _music_peak(R: np.ndarray, Nx: int, Ny: int, d_over_lambda: float, grid_size: int = 31, refine: bool = True) -> Tuple[float, float]:
    eigvals, eigvecs = np.linalg.eigh(R)
    idx = np.argsort(eigvals)
    eigvecs = eigvecs[:, idx]
    En = eigvecs[:, :-1]
    phis = np.linspace(-np.pi, np.pi, grid_size)
    thetas = np.linspace(1e-3, np.pi-1e-3, grid_size)
    P = np.zeros((grid_size, grid_size))
    for i, phi in enumerate(phis):
        for j, th in enumerate(thetas):
            a = ura_steering_vector(phi, th, Nx, Ny, d_over_lambda)
            denom = np.linalg.norm(En.conj().T @ a)**2 + 1e-12
            P[i, j] = 1.0 / denom
    imax, jmax = np.unravel_index(np.argmax(P), P.shape)
    phi_hat, th_hat = phis[imax], thetas[jmax]
    if refine:
        dphi = phis[1] - phis[0]
        dth = thetas[1] - thetas[0]
        dx, dy = _quad2d_refine(P, imax, jmax, dphi, dth)
        phi_hat += dx
        th_hat += dy
    return phi_hat, th_hat


def _refine_tau_phase_slope(H: np.ndarray, ofdm: OFDMSpec, aod_phi: float, aod_theta: float, aoa_phi: float, aoa_theta: float, tx: ArraySpec, rx: ArraySpec) -> float:
    Nr, Nt, Nf = H.shape
    Nx_t, Ny_t = tx.shape
    Nx_r, Ny_r = rx.shape
    at = ura_steering_vector(aod_phi, aod_theta, Nx_t, Ny_t, tx.d_over_lambda)
    ar = ura_steering_vector(aoa_phi, aoa_theta, Nx_r, Ny_r, rx.d_over_lambda)
    # Project per subcarrier onto rank-1 steering
    y = np.array([np.vdot(np.outer(ar, at.conj()), H[:, :, n]) for n in range(Nf)])
    # Unwrap phase vs frequency and fit slope
    n = np.arange(Nf)
    f = ofdm.f0 + (n - (Nf-1)/2.0) * ofdm.delta_f
    ang = np.unwrap(np.angle(y))
    # LS slope: minimize ||ang - (-2π τ f + b)||
    A = np.stack([-2*np.pi*f, np.ones_like(f)], axis=1)
    coef, *_ = np.linalg.lstsq(A, ang, rcond=None)
    tau = coef[0]
    return float(tau)


def estimate_multipath_params(snapshot: ChannelSnapshot, max_paths: int = 3, zpf: int = 8, grid_size: int = 37,
                               peak_prominence: float = 0.2, use_music: bool = True, num_subbands: int = 4) -> List[Tuple[float, float, float, float, float, complex]]:
    H = snapshot.Hn
    Nr, Nt, Nf = H.shape
    Nx_t, Ny_t = snapshot.tx.shape
    Nx_r, Ny_r = snapshot.rx.shape
    ofdm = snapshot.ofdm

    Nfft = zpf * Nf
    H_time = np.fft.ifft(H, n=Nfft, axis=2)
    M_tau = H_time.sum(axis=(0, 1))
    mag = np.abs(M_tau)
    thr = mag.max() * peak_prominence
    peak_idxs = np.argpartition(mag, -max_paths)[-max_paths:]
    peak_idxs = peak_idxs[np.argsort(mag[peak_idxs])[::-1]]
    peak_idxs = [idx for idx in peak_idxs if mag[idx] >= thr]
    if len(peak_idxs) == 0:
        peak_idxs = [int(np.argmax(mag))]
    paths_out = []

    # Subband partitions for multi-snapshot covariance
    n = np.arange(Nf)
    f = ofdm.f0 + (n - (Nf-1)/2.0) * ofdm.delta_f
    splits = np.array_split(np.arange(Nf), num_subbands) if num_subbands > 1 else [np.arange(Nf)]

    for idx in peak_idxs:
        k_ref = _parabolic_peak_refine(mag, idx)
        tau_est = k_ref / (Nfft * ofdm.delta_f)
        Rs = np.zeros((Nr, Nr), dtype=complex)
        Ts = np.zeros((Nt, Nt), dtype=complex)
        Ys = []
        for band in splits:
            fb = ofdm.f0 + (band - (Nf-1)/2.0) * ofdm.delta_f
            phase = np.exp(1j * 2.0 * np.pi * fb * tau_est)
            Yb = (H[:, :, band] * phase[None, None, :]).sum(axis=2)
            Ys.append(Yb)
            Rs += (Yb @ Yb.conj().T)
            Ts += (Yb.conj().T @ Yb)
        Rs /= max(len(splits), 1)
        Ts /= max(len(splits), 1)
        if use_music and min(Nr, Nt) >= 3:
            aoa_phi, aoa_theta = _music_peak(Rs, Nx_r, Ny_r, snapshot.rx.d_over_lambda, grid_size=max(31, grid_size))
            aod_phi, aod_theta = _music_peak(Ts, Nx_t, Ny_t, snapshot.tx.d_over_lambda, grid_size=max(31, grid_size))
        else:
            Y = np.mean(np.stack(Ys, axis=0), axis=0)
            U, S, Vh = np.linalg.svd(Y, full_matrices=False)
            ar_hat = U[:, 0]
            at_hat = Vh.conj().T[:, 0]
            phis = np.linspace(-np.pi, np.pi, grid_size)
            thetas = np.linspace(1e-3, np.pi-1e-3, grid_size)
            best_r = (-np.inf, 0.0, 0.0)
            for phi_r in phis:
                for th_r in thetas:
                    ar = ura_steering_vector(phi_r, th_r, Nx_r, Ny_r, snapshot.rx.d_over_lambda)
                    score = np.abs(np.vdot(ar, ar_hat))
                    if score > best_r[0]:
                        best_r = (score, phi_r, th_r)
            best_t = (-np.inf, 0.0, 0.0)
            for phi_t in phis:
                for th_t in thetas:
                    at = ura_steering_vector(phi_t, th_t, Nx_t, Ny_t, snapshot.tx.d_over_lambda)
                    score = np.abs(np.vdot(at, at_hat))
                    if score > best_t[0]:
                        best_t = (score, phi_t, th_t)
            _, aoa_phi, aoa_theta = best_r
            _, aod_phi, aod_theta = best_t
        # refine tau via phase slope using angle estimates
        tau_est = _refine_tau_phase_slope(H, ofdm, aod_phi, aod_theta, aoa_phi, aoa_theta, snapshot.tx, snapshot.rx)
        # gain LS using the average Y at refined tau
        phase = np.exp(1j * 2.0 * np.pi * f * tau_est)
        Y = (H * phase[None, None, :]).sum(axis=2)
        at = ura_steering_vector(aod_phi, aod_theta, Nx_t, Ny_t, snapshot.tx.d_over_lambda)
        ar = ura_steering_vector(aoa_phi, aoa_theta, Nx_r, Ny_r, snapshot.rx.d_over_lambda)
        A = np.outer(ar, at.conj())
        g_hat = np.vdot(A, Y) / (np.vdot(A, A) + 1e-18)
        paths_out.append((aod_phi, aod_theta, aoa_phi, aoa_theta, tau_est, g_hat))
    return paths_out