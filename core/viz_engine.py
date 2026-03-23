import matplotlib
matplotlib.use('Agg') # 強制使用無頭後端 (Headless mode)
import matplotlib.pyplot as plt
import numpy as np

def plot_diffraction_overlay(bg_img, target_spots, pred_spots, save_path=None):
    """
    將實驗斑點 (Target) 與理論斑點 (Pred) 疊加在背景圖 (FFT) 上。
    
    Args:
        bg_img: 背景 2D numpy array (FFT 振幅譜)
        target_spots: 實驗提取斑點 [N, 2] (x, y)
        pred_spots: 理論預測斑點 [M, 2] (x, y)
        save_path: 圖片保存路徑 (可為 None)
        
    Returns:
        matplotlib.figure.Figure: 繪製好的 Figure 物件
    """
    fig = plt.figure(figsize=(8, 8))
    
    # 1. 繪製背景圖
    plt.imshow(bg_img, cmap='gray', vmin=np.percentile(bg_img, 5), vmax=np.percentile(bg_img, 95))
    
    # 2. 繪製實驗斑點 (綠色空心圓)
    if target_spots is not None and len(target_spots) > 0:
        plt.scatter(target_spots[:, 0], target_spots[:, 1], 
                    s=100, edgecolors='g', facecolors='none', label='Experiment (GT)')
    
    # 3. 繪製理論預測斑點 (紅色十字)
    if pred_spots is not None and len(pred_spots) > 0:
        plt.scatter(pred_spots[:, 0], pred_spots[:, 1], 
                    s=80, marker='+', color='r', label='AI Prediction')
    
    plt.title("Diffraction Pattern Overlay (DeepDiffra)")
    plt.legend()
    plt.axis('off')
    
    # 4. 如果提供了路徑則保存
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

if __name__ == "__main__":
    print("Testing viz_engine.py...")
    
    # 生成測試背景 (隨機噪點)
    test_bg = np.random.rand(512, 512)
    
    # 生成測試坐標 (中心區域)
    test_target = np.array([[256, 256], [200, 200], [300, 310]])
    test_pred = np.array([[258, 257], [202, 198], [295, 305]])
    
    plot_diffraction_overlay(test_bg, test_target, test_pred, save_path="test_render.png")
    
    print("渲染成功，請在 VS Code 中點擊查看 test_render.png")
