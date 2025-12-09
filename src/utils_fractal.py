import numpy as np
import scipy.optimize

def box_counting_dimension(image_matrix, threshold=0.5):
    """
    Compute fractal dimension of 2D image (Box-counting method)
    
    Parameters:
        image_matrix: 2D numpy array (image or landscape map)
        threshold: Binarization threshold
        
    Returns:
        Df: Estimated Fractal Dimension
    """
    # Binarization
    pixels = image_matrix > threshold
    
    # Determine box size sequence (2^k)
    scales = np.logspace(1, np.log10(min(image_matrix.shape)), num=10, base=2, dtype=int)
    scales = np.unique(scales) # Deduplicate
    scales = scales[scales > 1] # Remove scales that are too small
    
    counts = []
    
    for scale in scales:
        # Coverage grid calculation
        H, W = pixels.shape
        ns = 0
        for y in range(0, H, scale):
            for x in range(0, W, scale):
                box = pixels[y:y+scale, x:x+scale]
                if np.any(box):
                    ns += 1
        counts.append(ns)
    
    # Fit log(N) vs log(1/s)
    coeffs = np.polyfit(np.log(1/scales), np.log(counts), 1)
    Df = coeffs[0]
    
    return Df

# Example usage
if __name__ == "__main__":
    # Generate a Sierpinski carpet for testing
    print("Testing Box-Counting Algorithm...")
    # (Fractal generation code omitted; framework example only)
    pass
