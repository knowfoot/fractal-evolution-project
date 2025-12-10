import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

def generate_mock_data_H1():
    """生成符合 H1 预测的模拟数据 (用于演示)"""
    np.random.seed(42)
    n = 200
    # 模拟变量
    log_npp = np.random.normal(3, 1, n)
    temp = np.random.normal(15, 10, n)
    terrain = np.random.gamma(2, 2, n)
    hetero = np.random.beta(2, 5, n) # 空间异质性
    
    # 模拟效应: D_L = 1.25 + 0.23*log_NPP + ...
    epsilon = np.random.normal(0, 0.1, n)
    D_L = 1.25 + 0.23 * log_npp + 0.08 * (temp/100) + 0.11 * terrain + 0.15 * hetero + epsilon
    
    df = pd.DataFrame({
        'Log_NPP': log_npp,
        'Temperature': temp,
        'Terrain_Heterogeneity': terrain,
        'Spatial_Heterogeneity': hetero,
        'Landscape_Df': D_L,
        'EcoRegion': np.random.randint(1, 10, n) # 用于随机效应
    })
    return df

def analyze_H1(df):
    """
    H1: 线性混合效应模型 (LME)
    模型: D_L ~ log(NPP) + Control_Vars + (1|EcoRegion)
    """
    print("\n--- Running H1 Analysis (LME Model) ---")
    # 定义模型公式
    formula = "Landscape_Df ~ Log_NPP + Temperature + Terrain_Heterogeneity + Spatial_Heterogeneity"
    
    # 使用 statsmodels 的 MixedLM
    model = smf.mixedlm(formula, df, groups=df["EcoRegion"])
    result = model.fit()
    
    print(result.summary())
    return result

def generate_mock_data_H2():
    """生成符合 H2 预测的时序数据 (含滞后)"""
    t = np.linspace(0, 100, 200) # 时间 (Myr)
    # 能量突升事件 (如氧气)
    oxygen = 1 / (1 + np.exp(-(t - 30)/5)) 
    
    # 形态分形维数 D_M (滞后响应)
    lag = 1.2 # Myr
    response = 1 / (1 + np.exp(-(t - 30 - lag)/5))
    D_M = 1.2 + 0.5 * response + np.random.normal(0, 0.05, 200)
    
    return pd.DataFrame({'Time': t, 'Energy_Proxy': oxygen, 'Morph_Df': D_M})

def analyze_H2(df):
    """
    H2: 时序滞后相关性分析
    """
    print("\n--- Running H2 Analysis (Time-Series Lag) ---")
    
    # 简单的互相关分析寻找最佳滞后
    cross_corr = np.correlate(df['Energy_Proxy'], df['Morph_Df'], mode='full')
    lags = np.arange(-len(df)+1, len(df))
    best_lag_idx = np.argmax(cross_corr)
    best_lag = lags[best_lag_idx]
    
    print(f"Estimated Lag: {best_lag * (df['Time'][1]-df['Time'][0]):.2f} Myr")
    
    # 绘图验证
    fig, ax1 = plt.subplots()
    ax1.plot(df['Time'], df['Energy_Proxy'], 'b-', label='Energy (Oxygen)')
    ax1.set_xlabel('Time (Myr)')
    ax1.set_ylabel('Energy Input', color='b')
    
    ax2 = ax1.twinx()
    ax2.plot(df['Time'], df['Morph_Df'], 'r--', label='Complexity (D_M)')
    ax2.set_ylabel('Fractal Dimension', color='r')
    
    plt.title("H2: Lagged Response of Complexity to Energy Surge")
    plt.savefig("h2_timeseries_result.png")

if __name__ == "__main__":
    # 1. H1 分析
    try:
        # 尝试加载真实数据 (用户需替换此处路径)
        data_h1 = pd.read_csv("data/global_ecoregions.csv")
    except FileNotFoundError:
        print("Warning: Real data not found. Generating SYNTHETIC data for demonstration.")
        data_h1 = generate_mock_data_H1()
    
    analyze_H1(data_h1)
    
    # 2. H2 分析
    data_h2 = generate_mock_data_H2()
    analyze_H2(data_h2)