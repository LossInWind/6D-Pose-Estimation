# 6D-Pose-Estimation: 论文复现

本仓库用于复现论文《A Projective Geometric View for 6D Pose Estimation in mmWave MIMO Systems (arXiv:2302.00227v2)》的两个核心问题：

- 问题1（单基站 mmWave SLAM）：基于单基站的 AoD/AoA 与时延，利用投影几何与极线模型估计 UE 的 6D 姿态及散射点位置，并联合估计时钟偏差。
- 问题2（AoA-only Pose）：在多个单天线基站场景，仅根据 AoA 估计 UE 的 6D 姿态（位置+朝向），通过投影模型转化为 3D-2D 对应，使用 PnP/非线性最小二乘求解。

## 目录结构

```
src/pose6d/
  se3.py               # 旋转/位姿与常用几何工具
  projective.py        # 投影几何、虚拟平面映射、极线(E)矩阵工具
  simulate.py          # 场景仿真：BS/UE/散射点、AoA/AoD/时延生成
  aoa_pose.py          # 问题2：AoA-only 6D 姿态估计算法
  slam.py              # 问题1：单BS mmWave SLAM（E矩阵+三角化+时延拟合）
scripts/run_experiments.py  # 复现实验脚本与CLI
```

## 环境安装

```bash
python -m pip install -r requirements.txt
```

## 快速开始

- 复现问题2（AoA-only Pose）：
```bash
python3 -m scripts.run_experiments aoa-pose --num-bs 6 --angle-noise-deg 0.5 --trials 200
```
- 复现问题1（单BS SLAM）：
```bash
python3 -m scripts.run_experiments slam --num-paths 8 --include-los --angle-noise-deg 0.5 --delay-noise-ns 2.0 --trials 100
```

脚本会输出平均位置误差、姿态误差（角度），以及在 SLAM 任务中时钟偏差与尺度估计误差等统计结果，并保存到 `outputs/`。

## 主要实现要点（与论文对应）
- 引入“虚拟平面”将 AoA/AoD 映射为 2D 点，使其满足透视投影模型；
- 问题2：将 (BS位置, AoA→虚拟点) 作为 3D-2D 对应，使用 PnP 或直接最小化重投影误差求解 UE 位姿；
- 问题1：将 (AoD@BS ↔ AoA@UE) 作为两“相机”之间的匹配点，估计本质矩阵 E=[t]_x R，分解得到相对位姿方向；再通过两视几何三角化得到散射点方向，并联合时延最小二乘求解尺度与时钟偏差；
- 提供 Monte Carlo 仿真以评估误差与鲁棒性。

## 许可

见 `LICENSE`。
