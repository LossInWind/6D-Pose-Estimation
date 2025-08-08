import numpy as np
from typing import List, Tuple, Optional
from scipy.optimize import least_squares

from .projective import ao_to_virtual_point, essential_from_correspondences, decompose_essential, virtual_point_to_bearing, triangulate_midpoint, C0
from .se3 import normalize_vector


def sampson_error(E: np.ndarray, v: np.ndarray, nu: np.ndarray) -> float:
    u = np.array([v[0], v[1], 1.0])
    up = np.array([nu[0], nu[1], 1.0])
    Exu = E @ u
    Etx_up = E.T @ up
    upT_E_u = up.T @ E @ u
    denom = Exu[0]**2 + Exu[1]**2 + Etx_up[0]**2 + Etx_up[1]**2
    denom = max(denom, 1e-12)
    return (upT_E_u**2) / denom


def ransac_essential(vs: np.ndarray, nus: np.ndarray, iters: int = 1000, thresh: float = 1e-3, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    N = vs.shape[0]
    best_inliers = None
    best_E = None
    for _ in range(iters):
        idx = rng.choice(N, size=min(8, N), replace=False)
        E = essential_from_correspondences(vs[idx], nus[idx])
        errs = np.array([sampson_error(E, vs[i], nus[i]) for i in range(N)])
        inliers = errs < thresh
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_E = E
    # re-estimate on inliers
    if best_inliers is None or best_inliers.sum() < 8:
        return essential_from_correspondences(vs, nus), np.ones(N, dtype=bool)
    E_refined = essential_from_correspondences(vs[best_inliers], nus[best_inliers])
    return E_refined, best_inliers


def estimate_relative_pose_from_aod_aoa(aod_list: List[Tuple[float, float]], aoa_list: List[Tuple[float, float]], seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert len(aod_list) == len(aoa_list)
    vs = np.array([ao_to_virtual_point(phi, theta) for (phi, theta) in aoa_list], dtype=float)
    nus = np.array([ao_to_virtual_point(phi, theta) for (phi, theta) in aod_list], dtype=float)
    E, inliers = ransac_essential(vs, nus, seed=seed)
    candidates = decompose_essential(E)
    # pick the candidate with most consistent cheirality for inliers
    best = candidates[0]
    best_score = -np.inf
    for (R, tdir) in candidates:
        score = 0
        for i in np.where(inliers)[0][:min(16, inliers.sum())]:
            b1 = virtual_point_to_bearing(nus[i])
            b2 = R @ virtual_point_to_bearing(vs[i])
            if np.dot(b1, tdir) > 0 and np.dot(b2, -tdir) > 0:
                score += 1
        if score > best_score:
            best_score = score
            best = (R, tdir)
    return best[0], best[1], inliers


def detect_los_paths(R: np.ndarray, t_dir: np.ndarray, aod_list: List[Tuple[float, float]], aoa_list: List[Tuple[float, float]], angle_thresh_deg: float = 2.0) -> np.ndarray:
    los = []
    thresh = np.deg2rad(angle_thresh_deg)
    for (aod, aoa) in zip(aod_list, aoa_list):
        b1 = virtual_point_to_bearing(ao_to_virtual_point(*aod))
        b2 = R @ virtual_point_to_bearing(ao_to_virtual_point(*aoa))
        ang = np.arccos(np.clip(np.dot(b1, -b2), -1.0, 1.0))
        # additionally ensure agreement with baseline direction
        if ang < thresh and (np.dot(b1, t_dir) > 0 and np.dot(-b2, t_dir) > 0):
            los.append(True)
        else:
            los.append(False)
    return np.array(los, dtype=bool)


def triangulate_scatterers(R: np.ndarray, t_dir: np.ndarray,
                           aod_list: List[Tuple[float, float]], aoa_list: List[Tuple[float, float]],
                           scale: float, los_flags: Optional[np.ndarray] = None) -> List[Optional[np.ndarray]]:
    c1 = np.zeros(3)
    c2 = t_dir * scale
    points = []
    for i, (aod, aoa) in enumerate(zip(aod_list, aoa_list)):
        if los_flags is not None and los_flags[i]:
            points.append(None)
            continue
        b1 = virtual_point_to_bearing(ao_to_virtual_point(*aod))  # BS frame
        b2_ue = virtual_point_to_bearing(ao_to_virtual_point(*aoa))
        b2 = R @ b2_ue  # map to BS frame
        p, _, _ = triangulate_midpoint(c1, b1, c2, b2)
        points.append(p)
    return points


def residual_scale_bias(x: np.ndarray, R: np.ndarray, t_dir: np.ndarray,
                        aod_list: List[Tuple[float, float]], aoa_list: List[Tuple[float, float]],
                        delays: np.ndarray, los_flags: np.ndarray) -> np.ndarray:
    scale, B = x[0], x[1]
    c1 = np.zeros(3)
    c2 = t_dir * scale
    res = []
    for i, (aod, aoa) in enumerate(zip(aod_list, aoa_list)):
        if los_flags[i]:
            d = np.linalg.norm(c2 - c1)
        else:
            b1 = virtual_point_to_bearing(ao_to_virtual_point(*aod))
            b2 = R @ virtual_point_to_bearing(ao_to_virtual_point(*aoa))
            p, _, _ = triangulate_midpoint(c1, b1, c2, b2)
            d = np.linalg.norm(p - c1) + np.linalg.norm(p - c2)
        tau_hat = B + d / C0
        res.append(tau_hat - delays[i])
    return np.array(res, dtype=float)


def solve_single_bs_slam(aod_list: List[Tuple[float, float]], aoa_list: List[Tuple[float, float]], delays: List[float],
                         s0: Optional[float] = None, B0: float = 0.0, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, float, float, List[Optional[np.ndarray]], np.ndarray]:
    R, t_dir, inliers = estimate_relative_pose_from_aod_aoa(aod_list, aoa_list, seed=seed)
    delays_arr = np.array(delays, dtype=float)
    if s0 is None:
        s0 = max(C0 * np.percentile(delays_arr, 10), 1.0)
    los_flags = detect_los_paths(R, t_dir, aod_list, aoa_list)
    x0 = np.array([s0, B0], dtype=float)
    res = least_squares(residual_scale_bias, x0, args=(R, t_dir, aod_list, aoa_list, delays_arr, los_flags),
                        bounds=(0, np.inf), loss='soft_l1', f_scale=1e-9, max_nfev=300)
    scale_opt, B_opt = res.x[0], res.x[1]
    points = triangulate_scatterers(R, t_dir, aod_list, aoa_list, scale_opt, los_flags=los_flags)
    return R, t_dir, scale_opt, B_opt, points, los_flags