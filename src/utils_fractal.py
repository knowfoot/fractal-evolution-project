import numpy as np
import scipy.optimize

def box_counting_dimension(image_matrix, threshold=0.5):
    """
    计算二维图像的分形维数 (Box-counting Method)
    
    Parameters:
        image_matrix: 2D numpy array (image or landscape map)
        threshold: Binarization threshold
        
    Returns:
        Df: Estimated Fractal Dimension
    """
    # 二值化
    pixels = image_matrix > threshold
    
    # 确定盒子的尺度序列 (2^k)
    scales = np.logspace(1, np.log10(min(image_matrix.shape)), num=10, base=2, dtype=int)
    scales = np.unique(scales) # 去重
    scales = scales[scales > 1] # 去除过小的尺度
    
    counts = []
    
    for scale in scales:
        # 覆盖网格计算
        H, W = pixels.shape
        ns = 0
        for y in range(0, H, scale):
            for x in range(0, W, scale):
                box = pixels[y:y+scale, x:x+scale]
                if np.any(box):
                    ns += 1
        counts.append(ns)
    
    # 拟合 log(N) vs log(1/s)
    coeffs = np.polyfit(np.log(1/scales), np.log(counts), 1)
    Df = coeffs[0]
    
    return Df

# 示例调用
if __name__ == "__main__":
    # 生成一个谢尔宾斯基地毯作为测试
    print("Testing Box-Counting Algorithm...")
    # (此处省略生成分形图案的代码，仅作框架示意)
    pass