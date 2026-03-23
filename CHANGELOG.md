# Changelog

## [0.1.0] - 2026-03-23
### Added
- **Core Physics Engine**: Initial implementation of `KinematicDiffractionModel`.
    - Differentiable lattice matrix construction.
    - 3D reciprocal grid generation and Euler angle rotation.
    - Ewald sphere truncation using Sinc smoothing.
- **Perception Layer**: Image processing utilities in `core/perception.py`.
    - FFT preprocessing and log-scale magnitude spectrum.
    - Peak detection using Gaussian blur and local maxima.
    - Reciprocal space alignment (pixel to 1/Angstrom).
- **Optimizer Layer**: Differentiable loss and optimization loop.
    - Weighted Chamfer Distance loss for point cloud matching.
    - Physical constraint penalty for lattice parameters.
    - Adam-based optimization loop implementation.
- **Tests**: 
    - `test_physics.py` for verifying forward pass and gradients.
    - `test_e2e.py` for end-to-end synthetic data optimization.
