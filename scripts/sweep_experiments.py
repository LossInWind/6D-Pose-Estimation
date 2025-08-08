import os
import sys
import json
import math
import numpy as np
from tqdm import tqdm

CUR_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(CUR_DIR, '..', 'src'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from pose6d.simulate import simulate_single_bs_scene, simulate_channel_measurements
from pose6d.slam import solve_single_bs_slam
from pose6d.channel import OFDMSpec, ArraySpec


def rot_angle_deg(R1, R2):
    R = R1 @ R2.T
    cosang = (np.trace(R) - 1.0) / 2.0
    cosang = np.clip(cosang, -1.0, 1.0)
    return math.degrees(math.acos(cosang))


def sweep_snr_and_paths(trials=100, snr_list=(10,15,20,25,30), path_list=(3,5,8), include_los=True, seed=0):
    rng = np.random.default_rng(seed)
    ofdm = OFDMSpec(num_subcarriers=64, delta_f=240e3, f0=60e9)
    tx = ArraySpec(16)
    rx = ArraySpec(16)
    results = []
    for snr_db in snr_list:
        for num_paths in path_list:
            pos_err = []
            rot_err = []
            scale_rel = []
            bias_ns = []
            for _ in tqdm(range(trials), desc=f"SNR={snr_db}dB, paths={num_paths}"):
                bs, ue_gt, scat_gt, _ = simulate_single_bs_scene(rng, num_paths=num_paths, include_los=include_los)
                meas = simulate_channel_measurements(rng, bs, ue_gt, scat_gt, include_los, ofdm, tx, rx, snr_db=snr_db, max_paths=num_paths)
                aod = [(m.aod_phi, m.aod_theta) for m in meas]
                aoa = [(m.aoa_phi, m.aoa_theta) for m in meas]
                delays = [m.delay_s for m in meas]
                weights = np.array([m.weight for m in meas], dtype=float)
                R_rel, t_dir, scale, B, _points, _los = solve_single_bs_slam(aod, aoa, delays, seed=rng.integers(1e9), weights=weights)
                r_est_bs = t_dir * scale
                pos_err.append(np.linalg.norm(r_est_bs - (ue_gt.position_w - bs.position_w)))
                rot_err.append(rot_angle_deg(R_rel, ue_gt.rotation_w @ bs.rotation_w.T))
                if include_los:
                    s_true = np.linalg.norm(ue_gt.position_w - bs.position_w)
                    scale_rel.append(abs(scale - s_true) / max(s_true, 1e-6))
                    bias_ns.append(abs(B - (s_true/299792458.0)) * 1e9)
            results.append({
                'snr_db': snr_db,
                'num_paths': num_paths,
                'position_rmse_m': float(np.sqrt(np.mean(np.array(pos_err)**2))),
                'rotation_mae_deg': float(np.mean(np.array(rot_err))),
                'scale_rel_mae': float(np.mean(scale_rel)) if include_los else None,
                'clock_bias_mae_ns': float(np.mean(bias_ns)) if include_los else None,
            })
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/sweep_snr_paths.json', 'w') as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == '__main__':
    sweep_snr_and_paths(trials=20)