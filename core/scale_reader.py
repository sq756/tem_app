import cv2
import numpy as np
import re

def detect_scale_bar(image_path):
    """
    自動檢測 TEM 圖片右下角的白色比例尺長度與 OCR 數值。
    
    Returns:
        dict: {
            "pixel_width": float (標尺像素長度),
            "physical_value": float (標尺物理數值, 如 10),
            "unit": str (標尺單位, 如 'nm'),
            "detected": bool
        }
    """
    # 1. 讀取並轉換為灰度圖
    img = cv2.imread(image_path)
    if img is None:
        return {"detected": False}
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 2. 尋找白色長條標尺 (通常為高亮度)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    
    # 3. 獲取輪廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    scale_bar_pixel_width = None
    scale_roi = None

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        # 標尺特性：
        # - 寬高比很大 (寬 >> 高)
        # - 通常在圖片底部 (y > h * 0.7)
        # - 寬度在 50 到 w/2 之間
        aspect_ratio = cw / float(ch)
        if aspect_ratio > 10 and y > h * 0.7 and cw > 50:
            scale_bar_pixel_width = float(cw)
            # 截取標尺上方的區域進行 OCR (通常文字在標尺上方)
            roi_y = max(0, y - 50)
            scale_roi = gray[roi_y:y, x:x+cw]
            break

    if scale_bar_pixel_width is None:
        return {"detected": False}

    # 4. 嘗試進行 OCR 識別 (封裝防崩潰)
    physical_value = None
    unit = "nm" # 默認
    
    try:
        # 嘗試使用 Pytesseract (如果系統有安裝)
        import pytesseract
        # OCR 配置: 只識別數字和單位字母
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.nm '
        text = pytesseract.image_to_string(scale_roi, config=custom_config)
        
        # 提取數字
        num_match = re.search(r"(\d+\.?\d*)", text)
        if num_match:
            physical_value = float(num_match.group(1))
        
        # 提取單位
        if "nm" in text.lower():
            unit = "nm"
        elif "A" in text or "ang" in text.lower():
            unit = "A"
            
    except Exception as e:
        print(f"OCR Detection failed or skipped: {e}")
        # 如果 OCR 失敗，我們至少回傳像素寬度
        pass

    return {
        "pixel_width": scale_bar_pixel_width,
        "physical_value": physical_value,
        "unit": unit,
        "detected": True
    }

def get_pixel_size(detection_result):
    """
    根據檢測結果計算 pixel_size (Angstrom / pixel)。
    """
    if not detection_result.get("detected") or detection_result.get("physical_value") is None:
        return 0.02 # 回傳默認值
    
    pixel_w = detection_result["pixel_width"]
    phys_v = detection_result["physical_value"]
    unit = detection_result["unit"]
    
    # 單位轉換為 Angstrom
    if unit == "nm":
        phys_v *= 10.0
    
    return phys_v / pixel_w
