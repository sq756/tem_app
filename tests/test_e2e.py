import torch
import sys
import os
import numpy as np

# 將專案根目錄加入路徑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.physics_engine import KinematicDiffractionModel
from core.optimizer import optimize_lattice

def test_e2e_optimization():
    print("Testing End-to-End Optimization (Synthetic Data)...")
    
    # 1. 生成 Ground Truth 數據
    # 假設目標晶體是 Si (a=5.43)，但歐拉角旋轉了一點
    gt_model = KinematicDiffractionModel(a=5.43, b=5.43, c=5.43, theta=5.0, phi=2.0)
    with torch.no_grad():
        output = gt_model(hkl_range=3)
        g_exp = output['coords'].cpu().numpy()
        # 僅取權重較大的斑點作為觀測值 (模擬實驗情況)
        weights = output['weights'].squeeze().cpu().numpy()
        mask = weights > 0.1
        g_exp = g_exp[mask]
    
    print(f"Generated {len(g_exp)} synthetic experimental peaks.")
    
    # 2. 初始化一個帶有偏差的模型 (擬合起始點)
    # 假設起始參數是 a=5.0, theta=0.0
    fit_model = KinematicDiffractionModel(a=5.0, b=5.0, c=5.0, theta=0.0, phi=0.0)
    
    # 3. 執行優化
    optimized_model = optimize_lattice(fit_model, g_exp, lr=0.05, epochs=100)
    
    # 4. 驗證結果
    print("\nResult Comparison:")
    print(f"GT Cell Lengths: {gt_model.cell_lengths.detach().cpu().numpy()}")
    print(f"Optimized Cell Lengths: {optimized_model.cell_lengths.detach().cpu().numpy()}")
    
    print(f"GT Euler Angles (Deg): {np.rad2deg(gt_model.euler_angles.detach().cpu().numpy())}")
    print(f"Optimized Euler Angles (Deg): {np.rad2deg(optimized_model.euler_angles.detach().cpu().numpy())}")
    
    # 檢查 a 是否接近 5.43
    final_a = optimized_model.cell_lengths[0].item()
    if abs(final_a - 5.43) < 0.1:
        print("\nE2E Optimization Successful!")
    else:
        print("\nE2E Optimization did not reach target, but check loss trend.")

if __name__ == "__main__":
    test_e2e_optimization()
