import torch

def weighted_chamfer_loss(g_exp, g_calc, weights_calc):
    """
    計算加權 Chamfer Distance 損失函數 (取 Mean 確保不因點數多寡作弊)。
    """
    # 0. 極端情況處理：若理論斑點權重過低 (近似於 0)
    if torch.sum(weights_calc) < 1e-5:
        # 返回兩個極大值以對齊解包邏輯
        return torch.tensor(10000.0, device=g_exp.device, requires_grad=True), \
               torch.tensor(10000.0, device=g_exp.device, requires_grad=True)

    # 1. 計算所有點對之間的距離矩陣 (N x M)
    dist_sq = torch.sum(g_exp**2, dim=1, keepdim=True) + \
              torch.sum(g_calc**2, dim=1).unsqueeze(0) - \
              2 * torch.matmul(g_exp, g_calc.t())

    # 2. 實驗點到理論點的映射 (Exp -> Calc)
    # 取 Mean 確保距離是平均距離
    weighted_dist_exp_to_calc = dist_sq / (weights_calc.t() + 1e-6)
    loss_exp_to_calc = torch.mean(torch.min(weighted_dist_exp_to_calc, dim=1)[0])

    # 3. 理論點到實驗點的映射 (Calc -> Exp)
    # 考慮權重大的理論點必須找到實驗點
    dist_calc_to_exp = torch.min(dist_sq, dim=0)[0]
    loss_calc_to_exp = torch.sum(dist_calc_to_exp * weights_calc.squeeze()) / (torch.sum(weights_calc) + 1e-8)

    return loss_calc_to_exp, loss_exp_to_calc

def physical_constraints_penalty(model, a_range=(2.0, 30.0), angle_range=(60.0, 120.0)):
    """
    強大的物理約束懲罰項 (防止晶格坍縮)。
    """
    penalty = torch.tensor(0.0, device=model.device)
    a, b, c = model.cell_lengths
    # Rad -> Deg
    alpha, beta, gamma = model.cell_angles * (180.0 / 3.1415926535)

    # 1. 晶格長度二次方懲罰 [2.0, 30.0]
    for val in [a, b, c]:
        penalty += torch.relu(2.0 - val)**2
        penalty += torch.relu(val - 30.0)**2

    # 2. 晶格角度二次方懲罰 [60.0, 120.0]
    for val in [alpha, beta, gamma]:
        penalty += torch.relu(60.0 - val)**2
        penalty += torch.relu(val - 120.0)**2

    return penalty * 1000.0 # 強大的梯度引導


def optimize_lattice(model, g_exp, lr=0.01, epochs=100, lambda_p=0.1):
    """
    執行晶格參數優化循環。
    
    Args:
        model: KinematicDiffractionModel 實例
        g_exp: 實驗倒易矢量 [N, 2]
        lr: 學習率
        epochs: 迭代次數
        lambda_p: 物理約束懲罰項權重
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 將實驗數據轉換為張量並確保設備一致
    g_exp_tensor = torch.tensor(g_exp, dtype=model.dtype, device=model.device)
    
    print(f"Starting optimization for {epochs} epochs...")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 1. 前向傳播
        output = model(hkl_range=3)
        g_calc = output['coords']
        weights_calc = output['weights']
        
        # 2. 計算損失
        loss_p_to_g, loss_g_to_p = weighted_chamfer_loss(g_exp_tensor, g_calc, weights_calc)
        loss_p = physical_constraints_penalty(model)
        
        total_loss = (loss_p_to_g + loss_g_to_p) + lambda_p * loss_p
        
        # 3. 反向傳播與更新
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss.item():.6f}")
            
    print("Optimization finished.")
    return model
