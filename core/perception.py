import cv2
import numpy as np

def preprocess_image(image_path, patch_size=128):
    """
    對 HRTEM 圖像進行預處理：
    1. 讀取圖像
    2. 如果圖像過大，切分為指定大小的 Patch (此處簡化為取中心 Patch)
    3. 執行 FFT 變換獲取振幅譜
    
    Returns:
        np.ndarray: FFT 振幅譜 (Log scaled)
    """
    # 1. 讀取圖像 (灰階)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    h, w = img.shape
    
    # 2. 取中心 Patch (簡化實作)
    start_h = max(0, h // 2 - patch_size // 2)
    start_w = max(0, w // 2 - patch_size // 2)
    patch = img[start_h:start_h + patch_size, start_w:start_w + patch_size]
    
    # 3. FFT 變換
    f = np.fft.fft2(patch)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    
    return magnitude_spectrum

def extract_peaks(magnitude_spectrum, num_peaks=10, min_dist=5):
    """
    從 FFT 振幅譜中提取局部極大值 (斑點坐標)。
    
    Args:
        magnitude_spectrum: FFT 振幅譜
        num_peaks: 提取的前 N 個最亮斑點
        min_dist: 斑點間最小距離 (防止重複提取同一個點)
        
    Returns:
        np.ndarray: [N, 2] 的像素坐標 (x, y)
    """
    # 屏蔽中心透射斑 (DC 分量)
    h, w = magnitude_spectrum.shape
    center_y, center_x = h // 2, w // 2
    
    # 創建遮罩，將中心 10x10 區域置零
    mask_radius = 5
    magnitude_spectrum[center_y-mask_radius:center_y+mask_radius, 
                       center_x-mask_radius:center_x+mask_radius] = 0
    
    # 使用 OpenCV 的局部最大值尋找或簡單的高斯模糊後尋找
    # 此處採用簡單的高斯模糊 + 局部極大值排序
    blurred = cv2.GaussianBlur(magnitude_spectrum, (5, 5), 0)
    
    # 找到所有候選點
    # 此處簡化：使用 dilatated mask 來尋找局部極大值
    kernel = np.ones((min_dist, min_dist), np.uint8)
    dilated = cv2.dilate(blurred, kernel)
    peaks_mask = (blurred == dilated) & (blurred > 0)
    
    peak_coords = np.column_stack(np.where(peaks_mask)) # [row, col] -> [y, x]
    peak_values = blurred[peaks_mask]
    
    # 排序並取前 N 個
    sorted_indices = np.argsort(peak_values)[::-1]
    top_indices = sorted_indices[:num_peaks]
    
    # 返回 (x, y) 格式
    return peak_coords[top_indices][:, ::-1] 

def align_to_reciprocal_space(peak_coords, patch_size, pixel_size_angstrom):
    """
    將像素坐標對齊到倒易空間 (1/Angstrom)。
    
    Args:
        peak_coords: [N, 2] 的像素坐標 (x, y)
        patch_size: FFT Patch 大小 (L)
        pixel_size_angstrom: 圖像單位像素對應的實空間長度 (Angstrom/pixel)
        
    Returns:
        np.ndarray: [N, 2] 的倒易空間向量 (g_x, g_y)
    """
    center = patch_size // 2
    
    # 計算倒易空間採樣間隔 delta_g = 1 / (L * pixel_size)
    delta_g = 1.0 / (patch_size * pixel_size_angstrom)
    
    # 平移中心到 (0,0) 並縮放
    g_exp = (peak_coords - center) * delta_g
    
    return g_exp
