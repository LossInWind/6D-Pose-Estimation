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
    sig_pow = np.mean(np.abs(H)**2)
    noise_pow = sig_pow / (10**(snr_db/10.0))
    noise = (np.sqrt(noise_pow/2) * (rng.normal(size=H.shape) + 1j*rng.normal(size=H.shape))).astype(complex)
    Hn = H + noise
    return ChannelSnapshot(Hn=Hn, ofdm=ofdm, tx=tx, rx=rx)


def estimate_params(snapshot: ChannelSnapshot, grid_size: int = 37) -> List[Tuple[float, float, float, float, float, complex]]:
    # Coarse grid search on AoD (phi,theta), AoA (phi,theta), and delay from FFT peak.
    H = snapshot.Hn
    Nr, Nt, Nf = H.shape
    Nx_t, Ny_t = snapshot.tx.shape
    Nx_r, Ny_r = snapshot.rx.shape
    ofdm = snapshot.ofdm
    # Delay via IFFT of average spatially matched energy (coarse)
    # Aggregate over antennas by Frobenius norm across a coarse spatial match (identity)
    Havg = H.reshape(Nr*Nt, Nf)
    # Simple delay spectrum
    delay_spec = np.abs(np.fft.ifft(Havg.mean(axis=0)))
    k_tau = int(np.argmax(delay_spec))
    tau_est = k_tau / (Nf * ofdm.delta_f)

    # Angular grid
    phis = np.linspace(-np.pi, np.pi, grid_size)
    thetas = np.linspace(1e-3, np.pi-1e-3, grid_size)
    best = (-np.inf, 0, 0, 0, 0)
    for ip, phi_t in enumerate(phis):
        for it, th_t in enumerate(thetas):
            at = ura_steering_vector(phi_t, th_t, Nx_t, Ny_t, snapshot.tx.d_over_lambda)
            for jp, phi_r in enumerate(phis):
                for jt, th_r in enumerate(thetas):
                    ar = ura_steering_vector(phi_r, th_r, Nx_r, Ny_r, snapshot.rx.d_over_lambda)
                    # Matched filter across antennas at estimated delay bin
                    # Project H onto ar * at^H at subcarrier index corresponding to tau_est phase
                    n = np.arange(Nf)
                    f = ofdm.f0 + (n - (Nf-1)/2.0) * ofdm.delta_f
                    phase = np.exp(1j * 2.0 * np.pi * f * tau_est)
                    score = np.vdot(np.outer(ar, at.conj())[:, :, None] * phase[None, None, :], H).real
                    if score > best[0]:
                        best = (score, phi_t, th_t, phi_r, th_r)
    _, aod_phi, aod_theta, aoa_phi, aoa_theta = best
    # gain via LS at chosen angles
    at = ura_steering_vector(aod_phi, aod_theta, Nx_t, Ny_t, snapshot.tx.d_over_lambda)
    ar = ura_steering_vector(aoa_phi, aoa_theta, Nx_r, Ny_r, snapshot.rx.d_over_lambda)
    n = np.arange(Nf)
    f = ofdm.f0 + (n - (Nf-1)/2.0) * ofdm.delta_f
    phase = np.exp(1j * 2.0 * np.pi * f * tau_est)
    A = (np.outer(ar, at.conj())[:, :, None] * phase[None, None, :]).reshape(-1)
    g_hat = np.vdot(A, H.reshape(-1)) / np.vdot(A, A)
    return [(aod_phi, aod_theta, aoa_phi, aoa_theta, tau_est, g_hat)]