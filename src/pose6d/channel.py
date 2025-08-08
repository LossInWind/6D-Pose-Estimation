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
    lamb = C0 / ofdm.f0
    for p in paths:
        at = ura_steering_vector(p.aod_phi, p.aod_theta, Nx_t, Ny_t, d_over_lambda=tx.d_over_lambda)
        ar = ura_steering_vector(p.aoa_phi, p.aoa_theta, Nx_r, Ny_r, d_over_lambda=rx.d_over_lambda)
        # frequency response per subcarrier
        n = np.arange(Nf)
        f = ofdm.f0 + (n - (Nf-1)/2.0) * ofdm.delta_f
        phase = np.exp(-1j * 2.0 * np.pi * f * p.delay_s)
        H += (p.gain * np.outer(ar, at.conj()))[:, :, None] * phase[None, None, :]
    # add complex Gaussian noise per entry
    sig_pow = np.mean(np.abs(H)**2) + 1e-12
    noise_pow = sig_pow / (10**(snr_db/10.0))
    noise = (np.sqrt(noise_pow/2) * (rng.normal(size=H.shape) + 1j*rng.normal(size=H.shape))).astype(complex)
    Hn = H + noise
    return ChannelSnapshot(Hn=Hn, ofdm=ofdm, tx=tx, rx=rx)


def estimate_params(snapshot: ChannelSnapshot, grid_size: int = 37) -> List[Tuple[float, float, float, float, float, complex]]:
    # Backward compatibility: single path estimate via coarse grid
    out = estimate_multipath_params(snapshot, max_paths=1, zpf=4, grid_size=grid_size)
    return [out[0]]


def estimate_multipath_params(snapshot: ChannelSnapshot, max_paths: int = 3, zpf: int = 8, grid_size: int = 37,
                               peak_prominence: float = 0.2) -> List[Tuple[float, float, float, float, float, complex]]:
    H = snapshot.Hn
    Nr, Nt, Nf = H.shape
    Nx_t, Ny_t = snapshot.tx.shape
    Nx_r, Ny_r = snapshot.rx.shape
    ofdm = snapshot.ofdm

    # Time-domain via zero-padded IFFT along subcarriers per antenna pair
    Nfft = zpf * Nf
    H_time = np.fft.ifft(H, n=Nfft, axis=2)
    # Sum across antenna pairs coherently
    M_tau = H_time.sum(axis=(0, 1))  # shape (Nfft,)
    mag = np.abs(M_tau)
    # Peak picking: find up to max_paths peaks above threshold
    thr = mag.max() * peak_prominence
    peak_idxs = np.argpartition(mag, -max_paths)[-max_paths:]
    peak_idxs = peak_idxs[np.argsort(mag[peak_idxs])[::-1]]
    # refine by keeping those above threshold
    peak_idxs = [idx for idx in peak_idxs if mag[idx] >= thr]
    if len(peak_idxs) == 0:
        peak_idxs = [int(np.argmax(mag))]
    paths_out = []
    # Angle grids for matching
    phis = np.linspace(-np.pi, np.pi, grid_size)
    thetas = np.linspace(1e-3, np.pi-1e-3, grid_size)
    n = np.arange(Nf)
    f = ofdm.f0 + (n - (Nf-1)/2.0) * ofdm.delta_f

    for idx in peak_idxs:
        tau_est = idx / (Nfft * ofdm.delta_f)
        # Form rank-1 matrix at this delay by coherent sum across subcarriers
        phase = np.exp(1j * 2.0 * np.pi * f * tau_est)
        Y = (H * phase[None, None, :]).sum(axis=2)
        # SVD to get steering vectors
        U, S, Vh = np.linalg.svd(Y, full_matrices=False)
        ar_hat = U[:, 0]
        at_hat = Vh.conj().T[:, 0]
        # Angle matching by maximizing |a^H v|
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
        # gain LS
        at = ura_steering_vector(aod_phi, aod_theta, Nx_t, Ny_t, snapshot.tx.d_over_lambda)
        ar = ura_steering_vector(aoa_phi, aoa_theta, Nx_r, Ny_r, snapshot.rx.d_over_lambda)
        A = np.outer(ar, at.conj())
        g_hat = np.vdot(A, Y) / np.vdot(A, A)
        paths_out.append((aod_phi, aod_theta, aoa_phi, aoa_theta, tau_est, g_hat))
    return paths_out