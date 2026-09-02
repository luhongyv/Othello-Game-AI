import os
import sys
import faulthandler
import json
crash_log = open("crash_report.txt", "w")
faulthandler.enable(file=crash_log)

# Ultimate C++ thread lock
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import random
import time
import gc

sys.path.append(os.getcwd()) 

from GAME.OthelloGame import OthelloGame 
from AGENTS.OthelloPlayers import MinimaxPlayer, GreedyMCTSPlayer

def run_comprehensive_benchmark(time_limit=1.0, num_games_per_stage=20):
    game = OthelloGame(8)
    
    # Test stages (if you're only missing the last three groups, change back to [50, 55, 60])
    test_stages = [5,10,15,20,25,30,35,40,45, 50, 55, 60]
    greedy_win_rates = []
    minimax_win_rates = []
    
    print("Initializing GreedyMCTS Player...")
    greedy_player = GreedyMCTSPlayer(game, time_limit=time_limit)
    
    print("Initializing Minimax Player...")
    minimax_player = MinimaxPlayer(game, time_limit=time_limit)
    
    print(f"\n Start Ablation Test ({len(test_stages)} stages, {num_games_per_stage} games per stage)")
    print("==================================================================================")
    
    for empty_spots in test_stages:
        print(f"\nTesting phase: {empty_spots} empty spots remaining (Game Complexity: {64-empty_spots}/64)")
        
        checkpoint_file = f"checkpoint_{empty_spots}_spots.json"
        
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                ckpt = json.load(f)
                start_game = ckpt['games_played']
                greedy_wins = ckpt['greedy_wins']
                minimax_wins = ckpt['minimax_wins']
                draws = ckpt['draws']
            print(f"  发现存档！正在从第 {start_game + 1} 局恢复进度...")
        else:
            start_game = 0
            greedy_wins = 0
            minimax_wins = 0
            draws = 0
            
        if start_game >= num_games_per_stage:
            print(f"  本阶段 {num_games_per_stage} 局已全部跑完！")
            greedy_win_rate = (greedy_wins / num_games_per_stage) * 100
            minimax_win_rate = (minimax_wins / num_games_per_stage) * 100
            greedy_win_rates.append(greedy_win_rate)
            minimax_win_rates.append(minimax_win_rate)
            continue # If the games for this stage are already completed, skip to the next empty-spots test
        
        for i in range(start_game, num_games_per_stage):
            board = game.getInitBoard()
            cur_player = 1
            
            steps_to_play = 60 - empty_spots
            for _ in range(steps_to_play):
                if game.getGameEnded(board, cur_player) != 0:
                    break 
                valids = game.getValidMoves(board, cur_player)
                valid_actions = np.where(valids == 1)[0]
                action = random.choice(valid_actions)
                board, cur_player = game.getNextState(board, cur_player, action)
                
            if i % 2 == 0:
                p1_play, p2_play = greedy_player.play, minimax_player.play
                p1_name, p2_name = "GreedyMCTS", "Minimax"
            else:
                p1_play, p2_play = minimax_player.play, greedy_player.play
                p1_name, p2_name = "Minimax", "GreedyMCTS"
                
            while game.getGameEnded(board, cur_player) == 0:
                canonical_board = game.getCanonicalForm(board, cur_player)
                
                if cur_player == 1:
                    action = p1_play(canonical_board)
                else:
                    action = p2_play(canonical_board)
                    
                board, cur_player = game.getNextState(board, cur_player, action)
                gc.collect()
                time.sleep(0.05)
                
            result = game.getGameEnded(board, 1)
            
            if result == 1:
                winner = p1_name
            elif result == -1:
                winner = p2_name
            else:
                winner = "Draw"
                
            if winner == "GreedyMCTS":
                greedy_wins += 1
            elif winner == "Minimax":
                minimax_wins += 1
            else:
                draws += 1
                
            gc.collect()

            with open(checkpoint_file, 'w') as f:
                json.dump({
                    'games_played': i + 1,
                    'greedy_wins': greedy_wins,
                    'minimax_wins': minimax_wins,
                    'draws': draws
                }, f)

            print(f" Game {i+1}/{num_games_per_stage} finished | Winner: {winner:<12} | Current MCTS Wins: {greedy_wins}")
            
        greedy_win_rate = (greedy_wins / num_games_per_stage) * 100
        minimax_win_rate = (minimax_wins / num_games_per_stage) * 100
        
        greedy_win_rates.append(greedy_win_rate)
        minimax_win_rates.append(minimax_win_rate)
        
        print(f"   [Result] GreedyMCTS: {greedy_win_rate:.1f}% | Minimax: {minimax_win_rate:.1f}% | Draws: {draws}")
        
    return test_stages, greedy_win_rates, minimax_win_rates


def plot_results(stages, greedy_rates, minimax_rates):
    
    import matplotlib.pyplot as plt
    import os
    
    save_dir = "TRAINNING_CHARTS"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"\n Created new directory for charts: {save_dir}/")
    
    # Force revert to the default classic white style
    plt.style.use('default')

    plt.figure(figsize=(10, 6))
    
    # Plot line charts
    plt.plot(stages, greedy_rates, marker='o', linewidth=3, markersize=10, label='GreedyMCTS', color='#ff7f0e')
    plt.plot(stages, minimax_rates, marker='s', linewidth=3, markersize=10, label='Minimax (Alpha-Beta)', color='#1f77b4')
    
    # Configure chart formatting
    plt.title('Performance Comparison: GreedyMCTS vs Minimax at Different Game Stages\n(Time Limit: 1.0s / move)', fontsize=16, pad=15)
    plt.xlabel('Remaining Empty Spots (Game Complexity)', fontsize=14)
    plt.ylabel('Win Rate (%)', fontsize=14)
    
    # Reverse X-axis
    plt.xlim(max(stages) + 5, min(stages) - 5)
    plt.ylim(-5, 105)
    
    # Gray dashed grid
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='best')
    
    # Add data labels
    for i, txt in enumerate(greedy_rates):
        plt.annotate(f"{txt:.0f}%", (stages[i], greedy_rates[i] + 3), fontsize=12, color='#ff7f0e', ha='center')
    for i, txt in enumerate(minimax_rates):
        plt.annotate(f"{txt:.0f}%", (stages[i], minimax_rates[i] - 6), fontsize=12, color='#1f77b4', ha='center')
        
    plt.tight_layout()
    
    output_filename = os.path.join(save_dir, 'GreedyMCTS_vs_Minimax_Final.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n Figure successfully saved to: {output_filename}")
    
    plt.show()

if __name__ == "__main__":
    stages, greedy_rates, minimax_rates = run_comprehensive_benchmark(time_limit=1.0, num_games_per_stage=40)
    plot_results(stages, greedy_rates, minimax_rates)
