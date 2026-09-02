import os
import sys
import time
import random
import numpy as np
import gc

sys.path.append(os.getcwd())
from GAME.OthelloGame import OthelloGame

# Prevent duplicate C++ thread libraries from causing interference
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'

def get_random_board(game, target_empty_spots):
    """Simulate a realistic game by playing random moves until the board
    has the specified number of empty spots remaining."""
    while True:
        board = game.getInitBoard()
        cur_player = 1
        empty = 64 - 4 
        
        while empty > target_empty_spots:
            valids = game.getValidMoves(board, cur_player)
            valid_acts = np.where(valids == 1)[0]
            if len(valid_acts) == 0:
                action = game.getActionSize() - 1
            else:
                action = random.choice(valid_acts)
            
            board, cur_player = game.getNextState(board, cur_player, action)
            empty = 64 - np.count_nonzero(board)
            
            if game.getGameEnded(board, 1) != 0:
                break
                
        if empty <= target_empty_spots:
            return board, cur_player

# ==========================================
# Refactor: Stateless NumPy-based probe
# Eliminates memory corruption issues caused by native Python lists
# ==========================================
class SpeedTester:
    def __init__(self, game):
        self.game = game
        self.node_count = 0

    def count_nodes(self, board, player, depth):
        self.node_count = 0
        start_time = time.time()
        self._alphabeta(board, player, depth, -float('inf'), float('inf'))
        cost_time = time.time() - start_time
        return self.node_count, cost_time

    def _alphabeta(self, board, player, depth, alpha, beta):
        self.node_count += 1
        if depth == 0:
            return
            
        # Use NumPy to quickly get valid moves (automatically handles pass/skip)
        valids = self.game.getValidMoves(board, player)
        valid_acts = np.where(valids == 1)[0]
        
        if len(valid_acts) == 0:
            return

        for action in valid_acts:
            # getNextState returns a fresh NumPy array (safe to pass to recursion)
            next_board, next_player = self.game.getNextState(board, player, action)
            self._alphabeta(next_board, next_player, depth - 1, -beta, -alpha)


def generate_and_save_table(table_data):
    import matplotlib.pyplot as plt
    
    save_dir = "TRAINNING_CHARTS"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    columns = ["Game Phase", "Empty Spots", "Depth", "Nodes Expanded", "Time Cost (s)", "Speed (Nodes/s)", "b_eff"]
    
    csv_filename = os.path.join(save_dir, 'Minimax_Profiling_Data.csv')
    with open(csv_filename, 'w', encoding='utf-8') as f:
        f.write(",".join(columns) + "\n")
        for row in table_data:
            f.write(",".join(map(str, row)) + "\n")
    
    fig, ax = plt.subplots(figsize=(12, len(table_data) * 0.4 + 1.5))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=table_data, colLabels=columns, loc='center', cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#1f77b4') 
        elif row % 2 == 0:
            cell.set_facecolor('#f3f3f3') 

    plt.title("Minimax (Alpha-Beta) Profiling Results", weight='bold', size=16, pad=20)
    
    png_filename = os.path.join(save_dir, 'Minimax_Profiling_Table.png')
    plt.savefig(png_filename, dpi=300, bbox_inches='tight')
    
    print("\n" + "="*70)
    print(f" Table image successfully saved to: {png_filename}")
    print(f" CSV data successfully saved to:   {csv_filename}")
    print("="*70)

if __name__ == "__main__":
    game = OthelloGame(8)
    # Pass the game instance into the tester
    tester = SpeedTester(game) 
    
    test_cases = [
        {"empty": 60, "depths": [4, 5, 6], "phase": "Early Game"},
        {"empty": 50, "depths": [4, 5, 6], "phase": "Early-Mid Transition"}, 
        {"empty": 40, "depths": [3, 4, 5], "phase": "Mid Game"},
        {"empty": 30, "depths": [3, 4, 5], "phase": "Peak Complexity"}, 
        {"empty": 20, "depths": [4, 5, 6], "phase": "Late Game"},
        {"empty": 10, "depths": [8, 9, 10], "phase": "End Game"}
    ]
    
    print(" MINIMAX COMPLEXITY PROFILER IS RUNNING...\n")
    
    all_table_data = []
    
    for case in test_cases:
        target_empty = case["empty"]
        board, player = get_random_board(game, target_empty)
        
        # Removed py_board conversion; use the original NumPy array board
        
        prev_nodes = None
        prev_depth = None
        
        for idx, d in enumerate(case["depths"]):
            phase_display = case["phase"] if idx == 0 else ""
            empty_display = str(target_empty) if idx == 0 else ""
            
            print(f"Testing {case['phase']} (Empty: {target_empty}) at Depth {d}...")
            
            # Pass NumPy board directly
            nodes, cost = tester.count_nodes(board, player, d)
            cost = max(cost, 0.0001) 
            speed = int(nodes / cost)
            
            if prev_nodes is None:
                b_eff_display = "-" 
            else:
                depth_diff = d - prev_depth
                b_eff = (nodes / prev_nodes) ** (1.0 / depth_diff)
                b_eff_display = f"{b_eff:.2f}"
            
            prev_nodes = nodes
            prev_depth = d
            
            formatted_nodes = f"{nodes:,}"
            formatted_cost = f"{cost:.4f}"
            formatted_speed = f"{speed:,}"
            
            all_table_data.append([
                phase_display, 
                empty_display, 
                str(d), 
                formatted_nodes, 
                formatted_cost, 
                formatted_speed,
                b_eff_display 
            ])

            gc.collect()       
            time.sleep(0.5)
            
    generate_and_save_table(all_table_data)
