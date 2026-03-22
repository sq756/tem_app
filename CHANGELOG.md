# DeepDiffra AI 更新日志

## [2026-03-22] - Initial Setup and Task 2.1

### Added
- 初始化 `core/physics_engine.py`。
- 定义 `KinematicDiffractionModel` 类。
- 注册 10 个可导物理参数为 `nn.Parameter`：
  - `cell_lengths`: (a, b, c)
  - `cell_angles`: (alpha, beta, gamma) [弧度制]
  - `euler_angles`: (theta, phi, psi) [弧度制]
  - `s_max`: 激发误差阈值。

### Changed
- 强制实施“度转弧度”策略，确保 PyTorch 三角函数兼容性。
- 强制使用 `torch.float32` (默认) 以平衡精度与计算开销。
