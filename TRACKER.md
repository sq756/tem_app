# DeepDiffra 任务追踪面板

## 模块二：物理层 (Forward) [已完成]
- [x] Task 2.1: 搭建 PyTorch 基础类，初始化 10 维可导参数 `[a, b, c, alpha, beta, gamma, theta, phi, psi, s_max]` (对应文件: `core/physics_engine.py`)
- [x] Task 2.2: 编写矩阵运算：正空间到倒易空间的转换算子 `get_reciprocal_base()`
- [x] Task 2.3: 编写 3D 倒易矢量生成器 `generate_3d_grid(h_max, k_max, l_max)`
- [x] Task 2.4: 编写 3D 空间欧拉角旋转算子 `rotate_to_zone_axis()`
- [x] Task 2.5: 编写 Ewald 球平滑截断算子 `apply_ewald_sphere_truncation()`
- [x] Task 2.6: 整合 `forward()` 流程，输出 2D 理论斑点坐标
- [x] Task 2.7: 编写物理层单元测试与梯度检验 (tests/test_physics.py)

## 模块一：感知层 (Perception) [进行中]
- [x] Task 1.1: 编写图像分块與 FFT 预处理函数 `preprocess_image()`
- [x] Task 1.2: 实现传统 CV 峰值提取 `extract_peaks()`
- [x] Task 1.3: 將像素坐标归一化並對齊倒易空間 (G-vector alignment)
- [ ] Task 5.2: 升級 Perception 算法，引入亞像素級斑點定位

## 模块三：优化层 (Optimizer) [已完成]
- [x] Task 3.1: 编写 Chamfer Distance 损失函数 `weighted_chamfer_loss()`
- [x] Task 3.2: 添加物理约束惩罚项 `physical_constraints_penalty()`
- [x] Task 3.3: 搭建 Adam 优化循环 `optimize_lattice()`

## 模块四：交互层 (UI & Viz) [已完成]
- [x] Task 4.1: 编写 Matplotlib 无头衍射图谱渲染器 (core/viz_engine.py)
- [x] Task 4.2: 构建 Streamlit 主程序 (app.py)
- [x] Task 4.3: 手动初始化与 Loss 优化预热 (app.py debug)
- [x] Task 4.4: 引入物理先验约束，防止晶格坍缩 (Optimizer & LR)
- [x] Task 4.6: 移除 (0,0,0) 斑點並優化初始張力 (強制由外向內收斂)
- [x] Task 4.7: 實作可學習比例尺並修復動態渲染 (物理與像素空間對齊)
- [x] Task 4.8: 增加 UI 初始旋轉滑桿與剛性晶格鎖定 (引導旋轉避開局部解)

## 模块五：自动化与深度可視化 [进行中]
- [x] Task 5.1: 實作自動標尺檢測與 OCR 識別 (core/scale_reader.py)
- [x] Task 6.1: 重構 UI 引入 Tabs 並實作物理公式白盒化可視化 (app.py)
- [ ] Task 6.2: 引入原子散射因子與結構因子擬合
- [ ] Task 4.5: 實現 2D 應變圖 (Strain Map) 生成
