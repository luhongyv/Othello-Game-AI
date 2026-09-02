import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['figure.dpi'] = 300

agents_y = ['Pure RL', 'SL+RL Hybrid', 'Minimax (Dynamic)', 'Greedy MCTS']
agents_x = ['Pure RL', 'SL+RL', 'Minimax', 'Greedy'] 

win_rates = np.array([
    [np.nan, 78.5, 78.6, 95.9],  
    [21.5, np.nan, 55.5, 90.8],  
    [21.4, 44.5, np.nan, 89.6],  
    [4.1,   9.2,  10.4, np.nan]   
])

df_matrix = pd.DataFrame(win_rates, index=agents_y, columns=agents_x)

plt.figure(figsize=(8, 5.5))  

ax = sns.heatmap(df_matrix, annot=True, fmt=".1f", cmap="Blues", 
            cbar_kws={'label': 'Win Rate (%)'}, vmin=0, vmax=100,
            linewidths=1, linecolor='white', 
            annot_kws={"size": 12, "weight": "bold"})


plt.xticks(rotation=0)
plt.yticks(rotation=0)

plt.title("Cross-play Win Rate Matrix", pad=20, weight='bold')
plt.ylabel("Agent (Playing as Black/White averaged)", weight='bold')
plt.xlabel("Opponent", weight='bold', labelpad=15)

plt.tight_layout()
plt.savefig("cross_play_heatmap.png", dpi=300, bbox_inches='tight')
print("Heatmap has been updated and saved to cross_play_heatmap.png")
plt.close()

plt.figure(figsize=(9, 5))

eval_opponents = ['vs Greedy MCTS (Expert)', 'vs Minimax (Tactical Ceiling)']
pure_rl_scores = [95.9, 78.6]
hybrid_scores = [90.8, 55.5]

x = np.arange(len(eval_opponents))
width = 0.35 

fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width/2, pure_rl_scores, width, label='Pure RL (Tabula Rasa)', color='#2c7bb6', edgecolor='black')
rects2 = ax.bar(x + width/2, hybrid_scores, width, label='SL+RL Hybrid (Expert Guided)', color='#d7191c', edgecolor='black')

ax.axhline(y=50, color='gray', linestyle='--', alpha=0.7)
ax.text(-0.4, 52, '50% Win Rate', color='gray', style='italic')

ax.set_ylabel('Win Rate (%)', weight='bold')
ax.set_title('Performance Comparison Against Baseline Opponents', pad=15, weight='bold')
ax.set_xticks(x)
ax.set_xticklabels(eval_opponents, weight='bold')
ax.legend(loc='upper right')
ax.set_ylim(0, 110)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', weight='bold')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig("baseline_comparison_bar.png", dpi=300, bbox_inches='tight')
print("The bar chart has been saved to baseline_comparison_bar.png")
plt.close()
