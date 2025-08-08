import numpy as np
from dataclasses import dataclass
from typing import Tuple, List
from .se3 import rotation_from_rvec, normalize_vector
from .projective import sph_to_cart, cart_to_sph, ao_to_virtual_point, C0


@dataclass
class BS:
    position_w: np.ndarray  # (3,)
    rotation_w: np.ndarray  # (3,3)


@dataclass
class UE:
    position_w: np.ndarray  # (3,)
    rotation_w: np.ndarray  # (3,3)


@dataclass
class PathMeasurement:
    aod_phi: float
    aod_theta: float
    aoa_phi: float
    aoa_theta: float
    delay_s: float


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    # Uniform random rotation using random axis-angle
    axis = normalize_vector(rng.normal(size=3))
    angle = rng.uniform(0, np.pi)
    return rotation_from_rvec(axis * angle)


def simulate_single_bs_scene(rng: np.random.Generator, num_paths: int = 6, include_los: bool = True,
                             bs_pose: Tuple[np.ndarray, np.ndarray] = (np.zeros(3), np.eye(3)),
                             ue_box: Tuple[np.ndarray, np.ndarray] = (np.array([-5, -5, 0.5]), np.array([5, 5, 3.0])),
                             scat_box: Tuple[np.ndarray, np.ndarray] = (np.array([-8, -8, 0.0]), np.array([8, 8, 4.0])),
                             clock_bias_s: float = 5e-7) -> Tuple[BS, UE, List[np.ndarray], List[PathMeasurement]]:
    r_bs, R_bs = bs_pose
    bs = BS(position_w=r_bs.copy(), rotation_w=R_bs.copy())
    ue_pos = rng.uniform(ue_box[0], ue_box[1])
    ue_rot = random_rotation(rng)
    ue = UE(position_w=ue_pos, rotation_w=ue_rot)

    scatterers = []
    meas = []

    if include_los:
        d_bs_to_ue = ue.position_w - bs.position_w
        # AoD at BS (BS local frame)
        d_bs_local = (R_bs.T @ normalize_vector(d_bs_to_ue))
        phi_bs, theta_bs = cart_to_sph(d_bs_local)
        # AoA at UE (UE local frame; incoming from BS)
        d_ue_in = normalize_vector(ue.position_w - bs.position_w)
        d_ue_local = ue.rotation_w.T @ d_ue_in
        phi_ue, theta_ue = cart_to_sph(d_ue_local)
        delay = clock_bias_s + np.linalg.norm(d_bs_to_ue) / C0
        meas.append(PathMeasurement(phi_bs, theta_bs, phi_ue, theta_ue, delay))

    for _ in range(num_paths - int(include_los)):
        p = rng.uniform(scat_box[0], scat_box[1])
        scatterers.append(p)
        # AoD at BS
        d_out = normalize_vector(p - bs.position_w)
        d_out_local = R_bs.T @ d_out
        phi_bs, theta_bs = cart_to_sph(d_out_local)
        # AoA at UE (incoming from scatterer)
        d_in = normalize_vector(ue.position_w - p)
        d_in_local = ue.rotation_w.T @ d_in
        phi_ue, theta_ue = cart_to_sph(d_in_local)
        delay = clock_bias_s + (np.linalg.norm(p - bs.position_w) + np.linalg.norm(ue.position_w - p)) / C0
        meas.append(PathMeasurement(phi_bs, theta_bs, phi_ue, theta_ue, delay))

    return bs, ue, scatterers, meas


def add_noise(meas: List[PathMeasurement], rng: np.random.Generator,
              angle_noise_std_rad: float = 0.0, delay_noise_std_s: float = 0.0) -> List[PathMeasurement]:
    noisy = []
    for m in meas:
        phi_bs = m.aod_phi + rng.normal(scale=angle_noise_std_rad)
        theta_bs = m.aod_theta + rng.normal(scale=angle_noise_std_rad)
        phi_ue = m.aoa_phi + rng.normal(scale=angle_noise_std_rad)
        theta_ue = m.aoa_theta + rng.normal(scale=angle_noise_std_rad)
        tau = m.delay_s + rng.normal(scale=delay_noise_std_s)
        noisy.append(PathMeasurement(phi_bs, theta_bs, phi_ue, theta_ue, tau))
    return noisy


def simulate_multi_bs_aoa_scene(rng: np.random.Generator, num_bs: int = 4,
                                bs_box: Tuple[np.ndarray, np.ndarray] = (np.array([-10, -10, 1.0]), np.array([10, 10, 5.0])),
                                ue_box: Tuple[np.ndarray, np.ndarray] = (np.array([-5, -5, 0.5]), np.array([5, 5, 3.0]))
                                ) -> Tuple[List[BS], UE, List[Tuple[int, float, float]]]:
    bs_list: List[BS] = []
    for _ in range(num_bs):
        pos = rng.uniform(bs_box[0], bs_box[1])
        Rw = np.eye(3)  # assume BS local = world for simplicity
        bs_list.append(BS(position_w=pos, rotation_w=Rw))
    ue_pos = rng.uniform(ue_box[0], ue_box[1])
    ue_rot = random_rotation(rng)
    ue = UE(position_w=ue_pos, rotation_w=ue_rot)

    aoa_list: List[Tuple[int, float, float]] = []
    for i, bs in enumerate(bs_list):
        d_in = normalize_vector(ue.position_w - bs.position_w)
        d_in_local = ue.rotation_w.T @ d_in
        phi, theta = cart_to_sph(d_in_local)
        aoa_list.append((i, phi, theta))
    return bs_list, ue, aoa_list