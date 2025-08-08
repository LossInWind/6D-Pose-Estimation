import numpy as np
from scipy.spatial.transform import Rotation as R


def skew(vector: np.ndarray) -> np.ndarray:
    v = np.asarray(vector).reshape(3)
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=float)


def rotation_from_rvec(rvec: np.ndarray) -> np.ndarray:
    return R.from_rotvec(np.asarray(rvec).reshape(3)).as_matrix()


def rvec_from_rotation(Rm: np.ndarray) -> np.ndarray:
    return R.from_matrix(np.asarray(Rm).reshape(3, 3)).as_rotvec()


def normalize_vector(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.maximum(n, eps)
    return v / n


def compose_pose(R_w_c: np.ndarray, t_w_c: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = R_w_c
    T[:3, 3] = t_w_c.reshape(3)
    return T


def transform_points(T_w_c: np.ndarray, points_w: np.ndarray) -> np.ndarray:
    Rwc = T_w_c[:3, :3]
    twc = T_w_c[:3, 3]
    return (Rwc @ points_w.T).T + twc


def world_to_cam(R_c_w: np.ndarray, r_cam_w: np.ndarray, points_w: np.ndarray) -> np.ndarray:
    # X_cam = R_c_w @ (X_w - r_cam_w)
    pts = np.asarray(points_w, dtype=float)
    return (R_c_w @ (pts - r_cam_w.reshape(1, 3)).T).T


def cam_to_world(R_c_w: np.ndarray, r_cam_w: np.ndarray, points_c: np.ndarray) -> np.ndarray:
    # X_w = R_w_c @ X_c + r_cam_w
    R_w_c = R_c_w.T
    pts = np.asarray(points_c, dtype=float)
    return (R_w_c @ pts.T).T + r_cam_w.reshape(1, 3)