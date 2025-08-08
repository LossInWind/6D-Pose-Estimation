import numpy as np
from typing import List, Tuple, Optional
from scipy.optimize import least_squares

from .projective import ao_to_virtual_point, essential_from_correspondences, decompose_essential, virtual_point_to_bearing, triangulate_midpoint, C0
from .se3 import normalize_vector

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


def _enforce_rank2(E: np.ndarray, vs: Optional[np.ndarray] = None, nus: Optional[np.ndarray] = None) -> np.ndarray:
    if E.shape == (3, 3):
        U, S, Vt = np.linalg.svd(E)
        S[2] = 0.0
        return U @ np.diag(S) @ Vt
    if vs is not None and nus is not None:
        return essential_from_correspondences(vs, nus)
    return E[:3, :3]


def sampson_error(E: np.ndarray, v: np.ndarray, nu: np.ndarray) -> float:
    u = np.array([v[0], v[1], 1.0])
    up = np.array([nu[0], nu[1], 1.0])
    Exu = E @ u
    Etx_up = E.T @ up
    upT_E_u = up.T @ E @ u
    denom = Exu[0]**2 + Exu[1]**2 + Etx_up[0]**2 + Etx_up[1]**2
    denom = max(denom, 1e-12)
    return (upT_E_u**2) / denom


def _rotation_align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # Find rotation R such that R @ a = b
    a = normalize_vector(a)
    b = normalize_vector(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)
    if s < 1e-8:
        # aligned or anti-aligned
        if c > 0:
            return np.eye(3)
        # 180-deg: pick any axis orthogonal to a
        axis = normalize_vector(np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0]))
        v = normalize_vector(np.cross(a, axis))
        Vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + 2 * Vx @ Vx  # 180deg rotation
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))
    return R


def prefilter_rotation_inliers(aod_list: List[Tuple[float, float]], aoa_list: List[Tuple[float, float]], iters: int = 500, thresh_deg: float = 5.0, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    # Hypothesize R from single path aligning d_ue to -d_bs; count consensus
    rng = np.random.default_rng(seed)
    N = len(aod_list)
    d_bs_list = [virtual_point_to_bearing(ao_to_virtual_point(*aod)) for aod in aod_list]
    d_ue_list = [virtual_point_to_bearing(ao_to_virtual_point(*aoa)) for aoa in aoa_list]
    best_inl = None
    best_count = -1
    for _ in range(max(iters, N)):
        k = int(rng.integers(N))
        R_h = _rotation_align(d_ue_list[k], -d_bs_list[k])
        # measure geodesic distance between implied R_i and R_h
        cnt = 0
        inl = []
        for i in range(N):
            R_i = _rotation_align(d_ue_list[i], -d_bs_list[i])
            dR = R_h @ R_i.T
            cosang = (np.trace(dR) - 1.0) / 2.0
            cosang = np.clip(cosang, -1.0, 1.0)
            ang = np.degrees(np.arccos(cosang))
            ok = ang < thresh_deg
            inl.append(ok)
            cnt += 1 if ok else 0
        if cnt > best_count:
            best_count = cnt
            best_inl = np.array(inl, dtype=bool)
    # refine R as average of inlier rotations
    R_sum = np.zeros((3, 3))
    for i, ok in enumerate(best_inl):
        if ok:
            R_sum += _rotation_align(d_ue_list[i], -d_bs_list[i])
    U, _, Vt = np.linalg.svd(R_sum)
    R0 = U @ Vt
    if np.linalg.det(R0) < 0:
        U[:, -1] *= -1
        R0 = U @ Vt
    return best_inl, R0


def ransac_essential_weighted(vs: np.ndarray, nus: np.ndarray, weights: np.ndarray, iters: int = 3000, thresh: float = 1e-3, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    if cv2 is not None and vs.shape[0] >= 5:
        pts1 = vs.astype(np.float64).reshape(-1, 1, 2)
        pts2 = nus.astype(np.float64).reshape(-1, 1, 2)
        E, inliers = cv2.findEssentialMat(pts1, pts2, focal=1.0, pp=(0.0, 0.0), method=cv2.RANSAC, prob=0.999, threshold=thresh)
        if E is not None and inliers is not None and inliers.sum() >= 5:
            inl = inliers.ravel().astype(bool)
            idx = np.where(inl)[0]
            idx_sorted = idx[np.argsort(weights[idx])[::-1]]
            keep = np.zeros_like(inl)
            keep[idx_sorted[:max(5, len(idx_sorted)//1)]] = True
            E3 = _enforce_rank2(E, vs[keep], nus[keep])
            return E3, keep
    rng = np.random.default_rng(seed)
    N = vs.shape[0]
    best_score = -np.inf
    best_E = None
    best_inl = None
    sample_size = min(8, N)
    for _ in range(iters):
        idx = rng.choice(N, size=sample_size, replace=False)
        E = essential_from_correspondences(vs[idx], nus[idx])
        errs = np.array([sampson_error(E, vs[i], nus[i]) for i in range(N)])
        inliers = errs < thresh
        score = weights[inliers].sum()
        if score > best_score:
            best_score = score
            best_E = E
            best_inl = inliers
    if best_inl is None or best_inl.sum() < 5:
        return _enforce_rank2(essential_from_correspondences(vs, nus)), np.ones(N, dtype=bool)
    E_refined = essential_from_correspondences(vs[best_inl], nus[best_inl])
    return _enforce_rank2(E_refined), best_inl


def _residual_scale_bias_weighted(x: np.ndarray, R: np.ndarray, t_dir: np.ndarray,
                                  aod_list: List[Tuple[float, float]], aoa_list: List[Tuple[float, float]],
                                  delays: np.ndarray, weights: np.ndarray, los_flags: np.ndarray) -> np.ndarray:
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
            p, a1, b2p = triangulate_midpoint(c1, b1, c2, b2)
            d = np.linalg.norm(p - c1) + np.linalg.norm(p - c2)
        tau_hat = B + d / C0
        res.append(np.sqrt(max(weights[i], 1e-6)) * (tau_hat - delays[i]))
    return np.array(res, dtype=float)


def _fit_scale_bias_quick(R: np.ndarray, t_dir: np.ndarray,
                          aod_list: List[Tuple[float, float]], aoa_list: List[Tuple[float, float]],
                          delays: np.ndarray, weights: np.ndarray, los_flags: np.ndarray,
                          s0: float, B0: float) -> Tuple[float, float, float]:
    x0 = np.array([s0, B0], dtype=float)
    res = least_squares(_residual_scale_bias_weighted, x0,
                        args=(R, t_dir, aod_list, aoa_list, delays, weights, los_flags),
                        bounds=(0, np.inf), loss='huber', f_scale=1e-9, max_nfev=100)
    scale_opt, B_opt = res.x[0], res.x[1]
    cost = float(np.sum(res.fun**2))
    return scale_opt, B_opt, cost


def estimate_relative_pose_from_aod_aoa(aod_list: List[Tuple[float, float]], aoa_list: List[Tuple[float, float]], weights: Optional[np.ndarray] = None, delays_for_scoring: Optional[np.ndarray] = None, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert len(aod_list) == len(aoa_list)
    vs = np.array([ao_to_virtual_point(phi, theta) for (phi, theta) in aoa_list], dtype=float)
    nus = np.array([ao_to_virtual_point(phi, theta) for (phi, theta) in aod_list], dtype=float)
    if weights is None:
        weights = np.ones(len(vs))
    # geometric rotation prefilter
    rot_inl, R0 = prefilter_rotation_inliers(aod_list, aoa_list, seed=seed)
    # apply E estimation on prefiltered set
    mask = rot_inl
    E, inliers_e = ransac_essential_weighted(vs[mask], nus[mask], weights[mask], seed=seed)
    # map back to full mask: mark prefilter outliers as false
    inliers = np.zeros(len(vs), dtype=bool)
    idxs = np.where(mask)[0]
    inliers[idxs[inliers_e]] = True
    # candidates
    candidates = []
    if cv2 is not None:
        pts1 = vs[inliers].astype(np.float64).reshape(-1, 1, 2)
        pts2 = nus[inliers].astype(np.float64).reshape(-1, 1, 2)
        _ok, R0cv, t0cv, _ = cv2.recoverPose(E, pts1, pts2, focal=1.0, pp=(0.0, 0.0))
        if R0cv is not None and t0cv is not None:
            candidates.append((R0cv.astype(float), normalize_vector(t0cv.reshape(3))))
    if not candidates:
        candidates = decompose_essential(E)
    # score
    best = candidates[0]
    best_cost = np.inf
    delays = delays_for_scoring if delays_for_scoring is not None else np.zeros(len(aod_list))
    for (R, tdir) in candidates:
        los_flags = detect_los_paths(R, tdir, aod_list, aoa_list)
        s_opt, B_opt, cost_sb = _fit_scale_bias_quick(R, tdir, aod_list, aoa_list, delays, weights, los_flags, s0=1.0, B0=0.0)
        # angular + sampson cost on inliers
        cost_ang = 0.0
        for i in np.where(inliers)[0]:
            w = weights[i]
            b1 = virtual_point_to_bearing(nus[i])
            b2 = R @ virtual_point_to_bearing(vs[i])
            ang = np.arccos(np.clip(np.dot(b1, -b2), -1.0, 1.0))
            cost_ang += w * ang**2
            cost_ang += w * sampson_error(E, vs[i], nus[i])
        # positive depth count
        pos_count = 0
        c1 = np.zeros(3); c2 = tdir * s_opt
        for (aod, aoa) in zip(aod_list, aoa_list):
            b1 = virtual_point_to_bearing(ao_to_virtual_point(*aod))
            b2 = R @ virtual_point_to_bearing(ao_to_virtual_point(*aoa))
            _, a1, b2p = triangulate_midpoint(c1, b1, c2, b2)
            if a1 > 0 and b2p > 0:
                pos_count += 1
        # total cost combines residuals and rewards positive depth
        total = cost_sb + cost_ang - 0.1 * pos_count
        if total < best_cost:
            best_cost = total
            best = (R, tdir)
    return best[0], best[1], inliers


def detect_los_paths(R: np.ndarray, t_dir: np.ndarray, aod_list: List[Tuple[float, float]], aoa_list: List[Tuple[float, float]], angle_thresh_deg: float = 2.0) -> np.ndarray:
    los = []
    thresh = np.deg2rad(angle_thresh_deg)
    for (aod, aoa) in zip(aod_list, aoa_list):
        b1 = virtual_point_to_bearing(ao_to_virtual_point(*aod))
        b2 = R @ virtual_point_to_bearing(ao_to_virtual_point(*aoa))
        ang = np.arccos(np.clip(np.dot(b1, -b2), -1.0, 1.0))
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
        b1 = virtual_point_to_bearing(ao_to_virtual_point(*aod))
        b2_ue = virtual_point_to_bearing(ao_to_virtual_point(*aoa))
        b2 = R @ b2_ue
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
                         s0: Optional[float] = None, B0: float = 0.0, seed: int = 0, weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, float, float, List[Optional[np.ndarray]], np.ndarray]:
    delays_arr = np.array(delays, dtype=float)
    R, t_dir, inliers = estimate_relative_pose_from_aod_aoa(aod_list, aoa_list, weights=weights, delays_for_scoring=delays_arr, seed=seed)
    if s0 is None:
        s0 = max(C0 * np.percentile(delays_arr, 10), 1.0)
    los_flags = detect_los_paths(R, t_dir, aod_list, aoa_list)
    x0 = np.array([s0, B0], dtype=float)
    res = least_squares(residual_scale_bias, x0, args=(R, t_dir, aod_list, aoa_list, delays_arr, los_flags),
                        bounds=(0, np.inf), loss='soft_l1', f_scale=1e-9, max_nfev=800)
    scale_opt, B_opt = res.x[0], res.x[1]
    points = triangulate_scatterers(R, t_dir, aod_list, aoa_list, scale_opt, los_flags=los_flags)
    return R, t_dir, scale_opt, B_opt, points, los_flags