# 🔬 DeepDiffra: Differentiable Physics Analysis for TEM

DeepDiffra 是一個基於 PyTorch 可微物理引擎的透射電子顯微鏡 (TEM) 衍射分析工具。它利用梯度下降算法自動擬合晶格參數，實現從 HRTEM 圖像到物理結構參數的自動化對齊。

## 📍 Phase 1: MVP & 核心物理閉環 (✅ 已完成)
- [x] **Perception Layer:** 基於 `peak_local_max` 的 FFT 衍射斑點粗提取。
- [x] **Physics Layer:** PyTorch 可微運動學衍射引擎 (10維參數: a,b,c, α,β,γ, θ,φ,ψ, scale)。
- [x] **Optimization:** 雙向 Chamfer Loss 與物理邊界懲罰項 (防止晶格坍縮)。
- [x] **UI & Viz:** 基於 Streamlit 的交互式 Web 介面與動態擬合監控。
- [x] **Human-in-the-loop:** UI 初始平面旋轉角 (In-plane Rotation) 滑塊與正交晶系剛性鎖定。

## 📍 Phase 2: 工業級自動化與魯棒性 (🚀 當前階段)
- [x] **Task 5.1:** 引入 OpenCV + OCR 實現 TEM 物理標尺 (Scale Bar) 自動識別。
- [ ] **Task 5.2:** 升級 Perception 算法，引入質心法 (Center of Mass) 或 2D 高斯擬合實現**亞像素級 (Sub-pixel)** 斑點定位。
- [ ] **Task 5.3:** 動態背景扣除 (Background Subtraction) 壓制中心透射強光暈。

## 📍 Phase 3: 物理層深化與應變分析 (⏳ 下一階段)
- [x] **Task 6.1:** 重構 UI 引入 Tabs 並實作物理公式白盒化可視化。
- [ ] **Task 6.2:** 引入原子散射因子 ($f_j$) 與結構因子 ($F_{hkl}$)，實現基於「斑點亮度(強度)」的聯合擬合。
- [ ] **Task 6.3:** 支持複雜晶系選擇 (FCC, BCC, 六方等) 的一鍵初始化。
- [ ] **Task 6.4:** 2D 局部應變圖 (Strain Mapping) 渲染輸出。

## 📍 Phase 4: Hybrid AI (端到端深度學習引入) (🔮 未來願景)
- [ ] 訓練輕量級 CNN / U-Net 替代傳統 CV，實現在極高噪點、缺陷情況下的魯棒性特徵提取。

---
*DeepDiffra v0.1.0-alpha - 為材料科學家打造的自動化分析工具*
