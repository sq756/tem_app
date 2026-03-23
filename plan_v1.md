# DeepDiffra 物理分析引擎 - 系統架構 MVP (v1.1)

## 一、 系統整體架構
1. **感知層 (Perception)**: 傳統 CV 提取斑點坐標 $\mathbf{g}_{exp}$ (OpenCV)。
2. **物理層 (Forward)**: 全鏈路可微 PyTorch 渲染理論斑點 $\mathbf{g}_{calc}$。
3. **優化層 (Optimizer)**: 加權 Chamfer Distance 擬合誤差，Adam 優化參數。
4. **交互層 (UI & Viz)**: 基於 **Streamlit** 的 Web 介面，Matplotlib 可視化結果。

## 二、 核心模組與任務追蹤
- [x] Task 1: 感知層開發 (FFT, Peak Detection)
- [x] Task 2: 物理層開發 (Reciprocal base, Rotation, Ewald truncation)
- [x] Task 3: 優化層開發 (Chamfer Loss, Physical penalty)
- [ ] Task 4: 交互層開發 (Streamlit App, Viz Engine)

## 三、 模組詳細設計 (詳見專案文件)
... [後續補充詳情]
