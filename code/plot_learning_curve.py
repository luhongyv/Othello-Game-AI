import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np

# ==========================================
# 1. Global academic plotting style settings
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
# 2. Read data and data cleaning
# ==========================================
csv_filename = "learning_curve_data.csv"
try:
    df = pd.read_csv(csv_filename)
except FileNotFoundError:
    print(f"Cannot find {csv_filename} file, please ensure the path is correct!")
    exit()

# If the written value is a decimal (e.g., 0.945), convert it to percentage form (94.5)
if df['Win_Rate_vs_Baseline'].max() <= 1.0:
    df['Win_Rate_vs_Baseline'] = df['Win_Rate_vs_Baseline'] * 100

def extract_iteration_num(label):
    """Convert text labels to continuous numerical values for X-axis"""
    if 'Final' in str(label):
        return 20
    match = re.search(r'\d+', str(label))
    if match:
        return int(match.group())
    return 0

df['Iteration_Num'] = df['Iteration_Label'].apply(extract_iteration_num)

# Normalize legend names
df['Algorithm'] = df['Algorithm'].replace({
    'Pure_RL': 'Tabula Rasa (Pure RL)',
    'SL_RL_Hybrid': 'Hybrid Pipeline (SL + RL)'
})

# Separate data and sort by X-axis
df_pure = df[df['Algorithm'] == 'Tabula Rasa (Pure RL)'].sort_values('Iteration_Num')
df_hybrid = df[df['Algorithm'] == 'Hybrid Pipeline (SL + RL)'].sort_values('Iteration_Num')

x_pure = df_pure['Iteration_Num'].to_numpy()
y_pure = df_pure['Win_Rate_vs_Baseline'].to_numpy()

x_hybrid = df_hybrid['Iteration_Num'].to_numpy()
y_hybrid = df_hybrid['Win_Rate_vs_Baseline'].to_numpy()

# ==========================================
# 3. Plot line chart
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(x_pure, y_pure, marker='o', markersize=8, linewidth=2.5, 
        color='#2c7bb6', label='Tabula Rasa (Pure RL)', markeredgecolor='black')
ax.plot(x_hybrid, y_hybrid, marker='s', markersize=8, linewidth=2.5, 
        color='#d7191c', label='Hybrid Pipeline (SL + RL)', markeredgecolor='black')

# ==========================================
# 4. Chart beautification and dynamic label annotation to prevent overlapping
# ==========================================
ax.set_title('Learning Curve Comparison: Pure RL vs. Hybrid Pipeline', pad=15, weight='bold')
ax.set_xlabel('Self-Play Iteration', weight='bold')
ax.set_ylabel('Win Rate vs Greedy Baseline (%)', weight='bold')

# Set Y-axis range and add 50% baseline (accommodating possible extremely low win rates in early stages)
ax.set_ylim(0, 105)
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.7)
ax.text(min(np.min(x_pure), np.min(x_hybrid)), 52, '50% Win Rate', color='gray', style='italic')

def get_dynamic_offset(x_current, y_current, x_other, y_other, is_pure):
    """
    Dynamically calculate label offset: if two lines converge within similar iteration numbers,
    the data with higher win rate will have its label placed above, and lower win rate below.
    """
    close_idx = np.where(np.abs(x_other - x_current) <= 1.5)[0]
    if len(close_idx) > 0:
        y_competitor = y_other[close_idx[0]]
        if is_pure:
            return (0, 8) if y_current >= y_competitor else (0, -15)
        else:
            return (0, 8) if y_current > y_competitor else (0, -15)
    return (0, 8) # Default to above when there are no overlapping points nearby

# Annotate Pure RL
for x, y in zip(x_pure, y_pure):
    offset = get_dynamic_offset(x, y, x_hybrid, y_hybrid, is_pure=True)
    ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=offset, 
                ha='center', fontsize=10, weight='bold', color='#2c7bb6')

# Annotate Hybrid
for x, y in zip(x_hybrid, y_hybrid):
    offset = get_dynamic_offset(x, y, x_pure, y_pure, is_pure=False)
    ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=offset, 
                ha='center', fontsize=10, weight='bold', color='#d7191c')

# Legend settings
ax.legend(loc='lower right', frameon=True, shadow=True)

# Save and output
plt.tight_layout()
output_filename = "Learning_Curve_Comparison_Final.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"\n[Success] Perfect line chart has been saved to {output_filename}")
