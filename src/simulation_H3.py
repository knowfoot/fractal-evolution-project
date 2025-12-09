import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

class EvolutionSimulation:
    """
    H3: 能量受限的演化主体模型 (Agent-Based Model)
    对应论文 v1.9 中的模拟预测：E > Ec(lambda) 时 D_f 上升。
    """
    def __init__(self, energy_flux, disturbance_rate, n_agents=1000, n_steps=500):
        self.E = energy_flux          # 能量通量 (E)
        self.lam = disturbance_rate   # 扰动频率 (lambda)
        self.n_agents = n_agents
        self.n_steps = n_steps
        
        # 初始化群体：分形维数 D_f 随机分布在 [1.0, 2.0]
        self.population_Df = np.random.uniform(1.0, 1.5, n_agents)
        self.history_mean_Df = []

    def step(self):
        """单步演化逻辑"""
        # 1. 能量获取 (Energy Intake): 假设 P_in ~ E * D_f^gamma (gamma=0.75 WBE标度)
        gamma = 0.75
        energy_intake = self.E * (self.population_Df ** gamma)
        
        # 2. 维持成本 (Maintenance Cost): C ~ D_f (结构越复杂，维持成本越高)
        cost = 0.5 * self.population_Df
        
        # 3. 净能量与生存概率
        net_energy = energy_intake - cost
        survival_prob = 1 / (1 + np.exp(-10 * net_energy)) # Sigmoid 函数
        
        # 4. 环境扰动 (Disturbance): 高 D_f 更脆弱 (Fragility ~ D_f)
        # 扰动发生概率为 self.lam
        if np.random.random() < self.lam:
            # 扰动发生，D_f 越高，死亡率越高
            disturbance_impact = self.population_Df / 3.0 # 归一化影响
            survival_prob *= (1 - disturbance_impact)
        
        # 5. 选择 (Selection)
        survivors_mask = np.random.random(len(self.population_Df)) < survival_prob
        survivors = self.population_Df[survivors_mask]
        
        # 6. 繁殖与变异 (Reproduction & Mutation)
        if len(survivors) > 0:
            # 补齐种群数量
            n_offspring = self.n_agents - len(survivors)
            parents = np.random.choice(survivors, n_offspring)
            # 变异：随机漂移 + 定向分形优化尝试
            mutation = np.random.normal(0, 0.05, n_offspring) 
            offspring = parents + mutation
            offspring = np.clip(offspring, 1.0, 3.0) # 限制 D_f 范围
            
            self.population_Df = np.concatenate([survivors, offspring])
        else:
            # 灭绝重置 (极低概率)
            self.population_Df = np.random.uniform(1.0, 1.5, self.n_agents)
            
        self.history_mean_Df.append(np.mean(self.population_Df))

    def run(self):
        for _ in range(self.n_steps):
            self.step()
        return self.history_mean_Df

def run_parameter_sweep():
    """运行参数扫描，生成相图"""
    energies = np.linspace(0.01, 10, 20)
    disturbances = np.linspace(0, 2, 20)
    
    results = np.zeros((len(disturbances), len(energies)))
    
    print("Running Simulation Sweep (H3 Verification)...")
    for i, lam in enumerate(tqdm(disturbances)):
        for j, E in enumerate(energies):
            sim = EvolutionSimulation(energy_flux=E, disturbance_rate=lam, n_steps=200)
            history = sim.run()
            # 记录最后50步的平均 D_f 变化趋势 (斜率)，正值代表复杂化
            trend = np.mean(np.diff(history[-50:]))
            # 或者简单地记录最终 D_f 是否显著增加
            final_Df = np.mean(history[-50:])
            initial_Df = history[0]
            success = 1 if final_Df > (initial_Df + 0.2) else 0
            results[i, j] = success

    # 绘图
    plt.figure(figsize=(10, 8))
    plt.imshow(results, extent=[0.01, 10, 2, 0], aspect='auto', cmap='viridis')
    plt.colorbar(label='Probability of Complexity Increase')
    plt.xlabel('Energy Flux (E)')
    plt.ylabel('Disturbance Rate ($\lambda$)')
    
    # 绘制临界线 Ec = 0.1 * lambda^0.5
    lambda_seq = np.linspace(0, 2, 100)
    E_c = 4.0 * (lambda_seq ** 0.5) # 系数需根据模拟尺度调整，此处为演示
    plt.plot(E_c, lambda_seq, 'r--', linewidth=2, label=r'Critical Line $E_c \propto \lambda^{0.5}$')
    
    plt.title('H3: Phase Diagram of Evolutionary Direction')
    plt.legend()
    plt.savefig('h3_simulation_result.png')
    print("Simulation complete. Result saved to h3_simulation_result.png")

if __name__ == "__main__":
    run_parameter_sweep()