import numpy as np
from typing import List, Tuple
from scipy.optimize import least_squares

from .projective import ao_to_virtual_point
from .se3 import world_to_cam, rotation_from_rvec


def reprojection_residual(params: np.ndarray, bs_positions: np.ndarray, v_obs: np.ndarray) -> np.ndarray:
    rvec = params[:3]
    r_cam = params[3:6]
    R_c_w = rotation_from_rvec(rvec)
    X_cam = world_to_cam(R_c_w, r_cam, bs_positions)
    vx = X_cam[:, 0] / X_cam[:, 2]
    vy = X_cam[:, 1] / X_cam[:, 2]
    res = np.stack([vx - v_obs[:, 0], vy - v_obs[:, 1]], axis=1).reshape(-1)
    return res


def solve_aoa_only_pose(bs_positions: np.ndarray, aoa_list: List[Tuple[int, float, float]],
                         num_restarts: int = 8, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    # bs_positions: (M,3); aoa_list: list of (bs_idx, phi, theta)
    # Returns (R_c_w, r_cam_w)
    idxs = [i for (i, _, _) in aoa_list]
    pts3d = bs_positions[idxs]
    v_obs = np.array([ao_to_virtual_point(phi, theta) for (_, phi, theta) in aoa_list], dtype=float)

    best_cost = np.inf
    best_params = None
    rng = np.random.default_rng(seed)

    # initial guesses: place UE near centroid, random small rotations
    centroid = pts3d.mean(axis=0)
    for _ in range(max(1, num_restarts)):
        rvec0 = rng.normal(scale=0.2, size=3)
        r_cam0 = centroid + rng.normal(scale=1.0, size=3)
        x0 = np.hstack([rvec0, r_cam0])
        res = least_squares(reprojection_residual, x0, args=(pts3d, v_obs), method='lm', max_nfev=200)
        cost = np.sum(res.fun**2)
        if cost < best_cost:
            best_cost = cost
            best_params = res.x

    rvec_opt = best_params[:3]
    r_cam = best_params[3:6]
    R_c_w = rotation_from_rvec(rvec_opt)
    return R_c_w, r_cam