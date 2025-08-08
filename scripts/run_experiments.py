import os
import sys
import json
import math
import click
import numpy as np
from tqdm import trange

# Make src importable
CUR_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(CUR_DIR, '..', 'src'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from pose6d.simulate import simulate_multi_bs_aoa_scene, simulate_single_bs_scene, add_noise
from pose6d.aoa_pose import solve_aoa_only_pose
from pose6d.slam import solve_single_bs_slam
from pose6d.se3 import rvec_from_rotation


def rot_angle_deg(R1, R2):
    R = R1 @ R2.T
    cosang = (np.trace(R) - 1.0) / 2.0
    cosang = np.clip(cosang, -1.0, 1.0)
    return math.degrees(math.acos(cosang))


@click.group()
def cli():
    pass


@cli.command()
@click.option('--trials', default=100, type=int)
@click.option('--num-bs', default=6, type=int)
@click.option('--angle-noise-deg', default=0.5, type=float)
@click.option('--seed', default=0, type=int)
@click.option('--solver', default='auto', type=click.Choice(['auto', 'pnp', 'lm']))
def aoa_pose(trials, num_bs, angle_noise_deg, seed, solver):
    rng = np.random.default_rng(seed)
    angle_noise = math.radians(angle_noise_deg)
    pos_err = []
    rot_err = []
    for _ in trange(trials):
        bs_list, ue_gt, aoa = simulate_multi_bs_aoa_scene(rng, num_bs=num_bs)
        # add noise on angles
        noisy_aoa = []
        for (i, phi, theta) in aoa:
            phi += rng.normal(scale=angle_noise)
            theta += rng.normal(scale=angle_noise)
            noisy_aoa.append((i, phi, theta))
        bs_positions = np.array([b.position_w for b in bs_list], dtype=float)
        R_est, r_est = solve_aoa_only_pose(bs_positions, noisy_aoa, seed=rng.integers(1e9), solver=solver)
        pos_err.append(np.linalg.norm(r_est - ue_gt.position_w))
        rot_err.append(rot_angle_deg(R_est, ue_gt.rotation_w))

    out = {
        'task': 'aoa_pose',
        'trials': trials,
        'num_bs': num_bs,
        'angle_noise_deg': angle_noise_deg,
        'solver': solver,
        'position_rmse_m': float(np.sqrt(np.mean(np.array(pos_err)**2))),
        'rotation_mae_deg': float(np.mean(np.array(rot_err)))
    }
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/aoa_pose_results.json', 'w') as f:
        json.dump(out, f, indent=2)
    click.echo(json.dumps(out, indent=2, ensure_ascii=False))


@cli.command()
@click.option('--trials', default=50, type=int)
@click.option('--num-paths', default=8, type=int)
@click.option('--include-los', is_flag=True, default=True)
@click.option('--angle-noise-deg', default=0.5, type=float)
@click.option('--delay-noise-ns', default=2.0, type=float)
@click.option('--seed', default=0, type=int)
def slam(trials, num_paths, include_los, angle_noise_deg, delay_noise_ns, seed):
    rng = np.random.default_rng(seed)
    angle_noise = math.radians(angle_noise_deg)
    delay_noise = delay_noise_ns * 1e-9

    pos_err = []
    rot_err = []
    scale_rel_err = []
    bias_err_ns = []

    for _ in trange(trials):
        bs, ue_gt, scat_gt, meas = simulate_single_bs_scene(rng, num_paths=num_paths, include_los=include_los)
        noisy = add_noise(meas, rng, angle_noise_std_rad=angle_noise, delay_noise_std_s=delay_noise)
        # Build lists aligned (drop LoS from triangulation consistency if needed)
        aod = [(m.aod_phi, m.aod_theta) for m in noisy]
        aoa = [(m.aoa_phi, m.aoa_theta) for m in noisy]
        delays = [m.delay_s for m in noisy]
        R_rel, t_dir, scale, B, points_bs, los_flags = solve_single_bs_slam(aod, aoa, delays, seed=rng.integers(1e9))
        # Reconstruct UE position in BS frame (BS at origin, identity)
        r_est_bs = t_dir * scale
        pos_err.append(np.linalg.norm(r_est_bs - (ue_gt.position_w - bs.position_w)))
        rot_err.append(rot_angle_deg(R_rel, ue_gt.rotation_w @ bs.rotation_w.T))
        # True scale is ||ue - bs|| when LoS present; otherwise approximate with mean total path length/2
        if include_los:
            scale_true = np.linalg.norm(ue_gt.position_w - bs.position_w)
            scale_rel_err.append(abs(scale - scale_true) / max(scale_true, 1e-6))
        bias_err_ns.append(abs(B - (noisy[0].delay_s - np.linalg.norm(ue_gt.position_w - bs.position_w)/299792458.0)) * 1e9 if include_los else abs(B) * 1e9)

    out = {
        'task': 'single_bs_slam',
        'trials': trials,
        'num_paths': num_paths,
        'include_los': include_los,
        'angle_noise_deg': angle_noise_deg,
        'delay_noise_ns': delay_noise_ns,
        'position_rmse_m': float(np.sqrt(np.mean(np.array(pos_err)**2))),
        'rotation_mae_deg': float(np.mean(np.array(rot_err))),
        'scale_rel_mae': float(np.mean(scale_rel_err)) if include_los else None,
        'clock_bias_mae_ns': float(np.mean(bias_err_ns))
    }
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/slam_results.json', 'w') as f:
        json.dump(out, f, indent=2)
    click.echo(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    cli()