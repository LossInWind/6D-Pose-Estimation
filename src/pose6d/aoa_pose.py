import numpy as np
from typing import List, Tuple
from scipy.optimize import least_squares

from .projective import ao_to_virtual_point, sph_to_cart
from .se3 import world_to_cam, rotation_from_rvec, normalize_vector

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


def reprojection_residual_virtual(params: np.ndarray, bs_positions: np.ndarray, v_obs: np.ndarray) -> np.ndarray:
    rvec = params[:3]
    r_cam = params[3:6]
    R_c_w = rotation_from_rvec(rvec)
    X_cam = world_to_cam(R_c_w, r_cam, bs_positions)
    vx = X_cam[:, 0] / X_cam[:, 2]
    vy = X_cam[:, 1] / X_cam[:, 2]
    res = np.stack([vx - v_obs[:, 0], vy - v_obs[:, 1]], axis=1).reshape(-1)
    return res


def reprojection_residual_bearing(params: np.ndarray, bs_positions: np.ndarray, d_obs_cam: np.ndarray) -> np.ndarray:
    rvec = params[:3]
    r_cam = params[3:6]
    R_c_w = rotation_from_rvec(rvec)
    rays = (R_c_w @ (bs_positions - r_cam.reshape(1, 3)).T).T
    d_pred = normalize_vector(rays)
    res = (d_pred - d_obs_cam).reshape(-1)
    return res


def solve_aoa_only_pose_lm(bs_positions: np.ndarray, v_or_d_obs: np.ndarray, num_restarts: int = 8, seed: int = 0, residual: str = 'bearing') -> Tuple[np.ndarray, np.ndarray]:
    best_cost = np.inf
    best_params = None
    rng = np.random.default_rng(seed)
    centroid = bs_positions.mean(axis=0)
    for _ in range(max(1, num_restarts)):
        rvec0 = rng.normal(scale=0.2, size=3)
        r_cam0 = centroid + rng.normal(scale=1.0, size=3)
        x0 = np.hstack([rvec0, r_cam0])
        if residual == 'virtual':
            fun = lambda x: reprojection_residual_virtual(x, bs_positions, v_or_d_obs)
        else:
            fun = lambda x: reprojection_residual_bearing(x, bs_positions, v_or_d_obs)
        res = least_squares(fun, x0, method='lm', max_nfev=300)
        cost = np.sum(res.fun**2)
        if cost < best_cost:
            best_cost = cost
            best_params = res.x
    rvec_opt = best_params[:3]
    r_cam = best_params[3:6]
    R_c_w = rotation_from_rvec(rvec_opt)
    return R_c_w, r_cam


def solve_aoa_only_pose_pnp(bs_positions: np.ndarray, v_obs: np.ndarray, use_ransac: bool = True, ransac_reproj_err: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    if cv2 is None:
        return solve_aoa_only_pose_lm(bs_positions, v_obs, residual='virtual')
    obj = bs_positions.astype(np.float64)
    img = v_obs.astype(np.float64).reshape(-1, 1, 2)
    K = np.eye(3, dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)
    if use_ransac:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(obj, img, K, dist, flags=cv2.SOLVEPNP_EPNP, reprojectionError=ransac_reproj_err, iterationsCount=300)
        if not ok or inliers is None or len(inliers) < 4:
            return solve_aoa_only_pose_lm(bs_positions, v_obs, residual='virtual')
    else:
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_EPNP)
        if not ok:
            return solve_aoa_only_pose_lm(bs_positions, v_obs, residual='virtual')
    def fun(x):
        r = x[:3].reshape(3, 1)
        t = x[3:].reshape(3, 1)
        proj, _ = cv2.projectPoints(obj, r, t, K, dist)
        proj = proj.reshape(-1, 2)
        return (proj - v_obs).reshape(-1)
    x0 = np.hstack([rvec.reshape(3), tvec.reshape(3)])
    res = least_squares(fun, x0, method='lm', max_nfev=300)
    rvec_opt = res.x[:3]
    t_opt = res.x[3:6]
    R_c_w, r_cam = rotation_from_rvec(rvec_opt), t_opt
    return R_c_w, r_cam


def solve_aoa_only_pose(bs_positions: np.ndarray, aoa_list: List[Tuple[int, float, float]],
                         num_restarts: int = 8, seed: int = 0, solver: str = 'auto', residual: str = 'bearing') -> Tuple[np.ndarray, np.ndarray]:
    idxs = [i for (i, _, _) in aoa_list]
    pts3d = bs_positions[idxs]
    v_obs = np.array([ao_to_virtual_point(phi, theta) for (_, phi, theta) in aoa_list], dtype=float)
    if solver == 'auto' and cv2 is not None:
        R, t = solve_aoa_only_pose_pnp(pts3d, v_obs, use_ransac=True)
        if R is not None:
            return R, t
    if residual == 'bearing':
        d_obs = np.array([sph_to_cart(phi, theta) for (_, phi, theta) in aoa_list], dtype=float)
        return solve_aoa_only_pose_lm(pts3d, d_obs, num_restarts=num_restarts, seed=seed, residual='bearing')
    if solver == 'pnp' and cv2 is not None:
        return solve_aoa_only_pose_pnp(pts3d, v_obs, use_ransac=True)
    return solve_aoa_only_pose_lm(pts3d, v_obs, num_restarts=num_restarts, seed=seed, residual='virtual')