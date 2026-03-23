import torch
import sys
import os
import numpy as np

# 加入專案路徑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.physics_engine import KinematicDiffractionModel

def simple_chamfer_loss(preds, targets):
    """
    手寫簡單倒角距離 (Chamfer Distance) 用於梯度檢驗。
    兩組點數量不一定相同。
    """
    # 計算距離矩陣 (N x M)
    dist_sq = torch.sum(preds**2, dim=1, keepdim=True) + \
              torch.sum(targets**2, dim=1).unsqueeze(0) - \
              2 * torch.matmul(preds, targets.t())
    
    # 計算每個預測點到最近目標點的距離之和
    loss = torch.mean(torch.min(dist_sq, dim=1)[0]) + \
           torch.mean(torch.min(dist_sq, dim=0)[0])
    return loss

def run_gradient_check():
    print("=== Task 2.7: Physics Layer Gradient Check ===")
    
    # 1. 實例化模型：FCC 金晶體 (a=b=c=4.078, alpha=beta=gamma=90)
    # 設定歐拉角 theta=5, phi=2, psi=0 (單位: Degree)
    model = KinematicDiffractionModel(
        a=4.078, b=4.078, c=4.078, 
        alpha=90.0, beta=90.0, gamma=90.0,
        theta=5.0, phi=2.0, psi=0.0
    )
    
    # 2. 生成 Ground Truth (Target Dots)
    with torch.no_grad():
        gt_output = model(hkl_range=3)
        target_dots = gt_output['coords'].detach()
        print(f"Target spots generated: {len(target_dots)}")

    # 3. 引入人為擾動：模擬樣品畸變 (a=4.15, theta 偏轉 2 度)
    with torch.no_grad():
        model.cell_lengths[0] = 4.15
        # 歐拉角是在弧度制下儲存的
        model.euler_angles[0] += np.deg2rad(2.0)
    
    # 4. 前向傳播與 Loss 計算
    output = model(hkl_range=3)
    pred_dots = output['coords']
    weights = output['weights']
    
    # 使用加權 Loss 以模擬真實優化 (考慮到 Ewald 球截斷)
    # 為簡化檢驗，此處直接使用 pred_dots 做 Chamfer Loss
    loss = simple_chamfer_loss(pred_dots, target_dots)
    
    # 5. 反向傳播
    loss.backward()

    # 6. 檢驗報告
    print("\n--- Gradient Inspection Report ---")
    
    print(f"Lattice lengths (a, b, c) grad: {model.cell_lengths.grad}")
    print(f"Lattice angles (alpha, beta, gamma) grad: {model.cell_angles.grad}")
    print(f"Euler angles (theta, phi, psi) grad: {model.euler_angles.grad}")
    print(f"Excitation error (s_max) grad: {model.s_max.grad}")

    # 驗證標準
    params_to_check = {
        "cell_lengths": model.cell_lengths.grad,
        "cell_angles": model.cell_angles.grad,
        "euler_angles": model.euler_angles.grad
    }

    all_passed = True
    for name, grad in params_to_check.items():
        if grad is None:
            print(f"CRITICAL ERROR: {name}.grad is None!")
            all_passed = False
        elif torch.all(grad == 0.0):
            print(f"WARNING: {name}.grad is all zeros!")
            all_passed = False
        else:
            print(f"PASSED: {name} gradient flow is healthy.")

    if all_passed:
        print("\n[SUCCESS] All physical parameters are differentiable!")
    else:
        print("\n[FAILED] Gradient chain is broken. Check tensor operations.")

if __name__ == "__main__":
    run_gradient_check()
