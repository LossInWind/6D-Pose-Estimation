import numpy as np
from typing import List, Tuple, Optional
from scipy.optimize import least_squares

from .projective import ao_to_virtual_point, essential_from_correspondences, decompose_essential, virtual_point_to_bearing, triangulate_midpoint, C0
from .se3 import normalize_vector


def estimate_relative_pose_from_aod_aoa(aod_list: List[Tuple[float, float]], aoa_list: List[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
    assert len(aod_list) == len(aoa_list)
    vs = np.array([ao_to_virtual_point(phi, theta) for (phi, theta) in aoa_list], dtype=float)
    nus = np.array([ao_to_virtual_point(phi, theta) for (phi, theta) in aod_list], dtype=float)
    E = essential_from_correspondences(vs, nus)
    candidates = decompose_essential(E)
    # pick the candidate with most consistent cheirality (both depths positive) for a few pairs
    best = candidates[0]
    best_score = -np.inf
    for (R, tdir) in candidates:
        score = 0
        for i in range(min(8, len(vs))):
            b1 = virtual_point_to_bearing(nus[i])  # BS frame
            b2 = R @ virtual_point_to_bearing(vs[i])  # UE bearing mapped to BS frame
            # With c1=0, c2=t; check if along same half-space
            if np.dot(b1, tdir) > 0 and np.dot(b2, -tdir) > 0:
                score += 1
        if score > best_score:
            best_score = score
            best = (R, tdir)
    return best


def triangulate_scatterers(R: np.ndarray, t_dir: np.ndarray,
                           aod_list: List[Tuple[float, float]], aoa_list: List[Tuple[float, float]],
                           scale: float) -> List[np.ndarray]:
    c1 = np.zeros(3)
    c2 = t_dir * scale
    points = []
    for (aod, aoa) in zip(aod_list, aoa_list):
        b1 = virtual_point_to_bearing(ao_to_virtual_point(*aod))  # BS frame
        b2_ue = virtual_point_to_bearing(ao_to_virtual_point(*aoa))
        b2 = R @ b2_ue  # map to BS frame
        p, _, _ = triangulate_midpoint(c1, b1, c2, b2)
        points.append(p)
    return points


def residual_scale_bias(x: np.ndarray, R: np.ndarray, t_dir: np.ndarray,
                        aod_list: List[Tuple[float, float]], aoa_list: List[Tuple[float, float]],
                        delays: np.ndarray) -> np.ndarray:
    scale, B = x[0], x[1]
    c1 = np.zeros(3)
    c2 = t_dir * scale
    res = []
    for i, (aod, aoa) in enumerate(zip(aod_list, aoa_list)):
        b1 = virtual_point_to_bearing(ao_to_virtual_point(*aod))
        b2 = R @ virtual_point_to_bearing(ao_to_virtual_point(*aoa))
        p, _, _ = triangulate_midpoint(c1, b1, c2, b2)
        d = np.linalg.norm(p - c1) + np.linalg.norm(p - c2)
        tau_hat = B + d / C0
        res.append(tau_hat - delays[i])
    return np.array(res, dtype=float)


def solve_single_bs_slam(aod_list: List[Tuple[float, float]], aoa_list: List[Tuple[float, float]], delays: List[float],
                         s0: Optional[float] = None, B0: float = 0.0) -> Tuple[np.ndarray, np.ndarray, float, float, List[np.ndarray]]:
    R, t_dir = estimate_relative_pose_from_aod_aoa(aod_list, aoa_list)
    delays_arr = np.array(delays, dtype=float)
    if s0 is None:
        # heuristic: assume mean total path length ~ c * (median(delay) - min(delay))
        s0 = max(C0 * (np.min(delays_arr) - B0), 1.0)
    x0 = np.array([s0, B0], dtype=float)
    res = least_squares(residual_scale_bias, x0, args=(R, t_dir, aod_list, aoa_list, delays_arr), bounds=(0, np.inf), max_nfev=200)
    scale_opt, B_opt = res.x[0], res.x[1]
    points = triangulate_scatterers(R, t_dir, aod_list, aoa_list, scale_opt)
    return R, t_dir, scale_opt, B_opt, points