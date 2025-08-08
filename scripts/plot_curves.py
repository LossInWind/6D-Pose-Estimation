import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

CUR_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(CUR_DIR, '..', 'src'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def load_results(path='outputs/sweep_snr_paths.json'):
    with open(path, 'r') as f:
        return json.load(f)


def plot_vs_snr(results, num_paths_list=(3,5,8,12)):
    sns = sorted(set(r['snr_db'] for r in results))
    for metric, ylabel in [
        ('position_rmse_m', 'Position RMSE (m)'),
        ('rotation_mae_deg', 'Rotation MAE (deg)'),
        ('scale_rel_mae', 'Scale Rel. MAE'),
        ('clock_bias_mae_ns', 'Clock Bias MAE (ns)'),
    ]:
        plt.figure(figsize=(7,5))
        for npth in num_paths_list:
            ys = []
            for snr in sns:
                rows = [r for r in results if r['num_paths']==npth and r['snr_db']==snr]
                if rows:
                    ys.append(rows[0][metric] if rows[0][metric] is not None else np.nan)
                else:
                    ys.append(np.nan)
            plt.plot(sns, ys, marker='o', label=f'paths={npth}')
        plt.xlabel('SNR (dB)')
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.legend()
        os.makedirs('outputs/figs', exist_ok=True)
        plt.savefig(f'outputs/figs/{metric}_vs_snr.png', dpi=150, bbox_inches='tight')
        plt.close()

    # CRB overlay for aoa/ aod/ tau variance at num_paths irrelevant (we take first row per SNR)
    plt.figure(figsize=(7,5))
    var_tau = []
    for snr in sns:
        rows = [r for r in results if r['snr_db']==snr and r['crb'] is not None]
        if rows:
            var_tau.append(rows[0]['crb']['var_tau'])
        else:
            var_tau.append(np.nan)
    plt.plot(sns, np.array(var_tau), marker='x', color='k', label='CRB(var_tau)')
    plt.xlabel('SNR (dB)')
    plt.ylabel('CRB var(tau) (s^2)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('outputs/figs/crb_tau_vs_snr.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    res = load_results()
    plot_vs_snr(res)