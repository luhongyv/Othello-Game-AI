import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np

# ==========================================
# 1. 全局学术绘图样式设置
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['figure.dpi'] = 300
sns.set_style("whitegrid")

# ==========================================
# 2. 读取数据与清洗
# ==========================================
csv_filename = "learning_curve_data.csv"
try:
    df = pd.read_csv(csv_filename)
except FileNotFoundError:
    print(f"找不到 {csv_filename} 文件，请确保路径正确！")
    exit()

# 如果写入的是小数 (如 0.945)，则转换为百分比形式 (94.5)
if df['Win_Rate_vs_Baseline'].max() <= 1.0:
    df['Win_Rate_vs_Baseline'] = df['Win_Rate_vs_Baseline'] * 100

def extract_iteration_num(label):
    """将文本标签转换为 X 轴的连续数值"""
    if 'Final' in str(label):
        return 20
    match = re.search(r'\d+', str(label))
    if match:
        return int(match.group())
    return 0

df['Iteration_Num'] = df['Iteration_Label'].apply(extract_iteration_num)

# 规范化图例名称
df['Algorithm'] = df['Algorithm'].replace({
    'Pure_RL': 'Tabula Rasa (Pure RL)',
    'SL_RL_Hybrid': 'Hybrid Pipeline (SL + RL)'
})

# 分离数据并按 X 轴排序
df_pure = df[df['Algorithm'] == 'Tabula Rasa (Pure RL)'].sort_values('Iteration_Num')
df_hybrid = df[df['Algorithm'] == 'Hybrid Pipeline (SL + RL)'].sort_values('Iteration_Num')

x_pure = df_pure['Iteration_Num'].to_numpy()
y_pure = df_pure['Win_Rate_vs_Baseline'].to_numpy()

x_hybrid = df_hybrid['Iteration_Num'].to_numpy()
y_hybrid = df_hybrid['Win_Rate_vs_Baseline'].to_numpy()

# ==========================================
# 3. 绘制折线图
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(x_pure, y_pure, marker='o', markersize=8, linewidth=2.5, 
        color='#2c7bb6', label='Tabula Rasa (Pure RL)', markeredgecolor='black')
ax.plot(x_hybrid, y_hybrid, marker='s', markersize=8, linewidth=2.5, 
        color='#d7191c', label='Hybrid Pipeline (SL + RL)', markeredgecolor='black')

# ==========================================
# 4. 图表美化与【动态防遮挡】标签标注
# ==========================================
ax.set_title('Learning Curve Comparison: Pure RL vs. Hybrid Pipeline', pad=15, weight='bold')
ax.set_xlabel('Self-Play Iteration', weight='bold')
ax.set_ylabel('Win Rate vs Greedy Baseline (%)', weight='bold')

# 设置 Y 轴范围并添加 50% 基准线 (适应早期可能的极低胜率)
ax.set_ylim(0, 105)
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.7)
ax.text(min(np.min(x_pure), np.min(x_hybrid)), 52, '50% Win Rate', color='gray', style='italic')

def get_dynamic_offset(x_current, y_current, x_other, y_other, is_pure):
    """
    动态计算标签偏移量：如果在相近的迭代轮数内两根线发生交汇，
    胜率高的数据将标签放在上方，胜率低的数据放在下方。
    """
    close_idx = np.where(np.abs(x_other - x_current) <= 1.5)[0]
    if len(close_idx) > 0:
        y_competitor = y_other[close_idx[0]]
        if is_pure:
            return (0, 8) if y_current >= y_competitor else (0, -15)
        else:
            return (0, 8) if y_current > y_competitor else (0, -15)
    return (0, 8) # 周围没有重合点时，默认在上方

# 标注 Pure RL
for x, y in zip(x_pure, y_pure):
    offset = get_dynamic_offset(x, y, x_hybrid, y_hybrid, is_pure=True)
    ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=offset, 
                ha='center', fontsize=10, weight='bold', color='#2c7bb6')

# 标注 Hybrid
for x, y in zip(x_hybrid, y_hybrid):
    offset = get_dynamic_offset(x, y, x_pure, y_pure, is_pure=False)
    ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=offset, 
                ha='center', fontsize=10, weight='bold', color='#d7191c')

# 图例设置
ax.legend(loc='lower right', frameon=True, shadow=True)

# 保存与输出
plt.tight_layout()
output_filename = "Learning_Curve_Comparison_Final.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"\n[Success] 完美折线图已保存至 {output_filename}")