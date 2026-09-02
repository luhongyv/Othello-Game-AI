import os
import sys
import faulthandler
import json

# Setup crash log
crash_log = open("crash_report_az.txt", "w")
faulthandler.enable(file=crash_log)

# Lock C++ threads to prevent CPU overload
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import torch
torch.set_num_threads(1)
torch.backends.cudnn.enabled = False

import numpy as np
import random
import time
import gc

sys.path.append(os.getcwd()) 

from GAME.OthelloGame import OthelloGame 
from AGENTS.OthelloPlayers import GreedyMCTSPlayer, AlphaZeroMCTS
from CNN.NNetTrainer import NNetWrapper

def run_ultimate_benchmark(time_limit=1.0, num_games_per_stage=40):
    game = OthelloGame(8)
    
    test_stages = [5,10,15,20,25,30,35,40,45, 50, 55, 60]
    az_win_rates = []
    greedy_win_rates = []
    
    print("==================================================================================")
    print(" AlphaZero VS GreedyMCTS ")
    print("==================================================================================")
    
    # 1. Initialize AlphaZero
    print("Loading AlphaZero Best Model... ")
    az_net = NNetWrapper(game)
    #az_net.to_cpu() 
    model_folder = os.path.join(os.getcwd(), "checkpoint")
    model_filename = "resnet_expert.pth"
    model_path = os.path.join(model_folder, model_filename)
    
    if os.path.exists(model_path):
        az_net.load_checkpoint(folder=model_folder, filename=model_filename)
    else:
        print(f"Error: {model_path} not found! Please check your folder. ")
        return None, None, None
        
    az_mcts = AlphaZeroMCTS(game, az_net, num_sims=1000, c_puct=1.0)
    
    # 2. Initialize GreedyMCTS
    print("Initializing GreedyMCTS Player... ")
    greedy_player = GreedyMCTSPlayer(game, time_limit=time_limit)
    
    print(f"\n Start Benchmark: {len(test_stages)} stages, {num_games_per_stage} games per stage")
    print("==================================================================================")
    
    for empty_spots in test_stages:
        print(f"\n Testing phase: {empty_spots} empty spots remaining")
        
        checkpoint_file = f"checkpoint_az_vs_greedy_{empty_spots}_spots.json"
        
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r') as f:
                    ckpt = json.load(f)
                    start_game = ckpt['games_played']
                    az_wins = ckpt.get('az_wins', 0)
                    greedy_wins = ckpt.get('greedy_wins', 0)
                    draws = ckpt['draws']
                print(f" Checkpoint found! Resuming from game {start_game + 1}... ")
            except json.JSONDecodeError:
                print(f" Broken checkpoint found and ignored. Restarting from game 1... ")
                start_game = 0
                az_wins = 0
                greedy_wins = 0
                draws = 0
        else:
            start_game = 0
            az_wins = 0
            greedy_wins = 0
            draws = 0
            
        if start_game >= num_games_per_stage:
            print(f" All {num_games_per_stage} games for this stage are completed! ")
            az_win_rate = (az_wins / num_games_per_stage) * 100
            greedy_win_rate = (greedy_wins / num_games_per_stage) * 100
            az_win_rates.append(az_win_rate)
            greedy_win_rates.append(greedy_win_rate)
            continue 
        
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
                p1_name, p2_name = "AlphaZero", "GreedyMCTS"
            else:
                p1_name, p2_name = "GreedyMCTS", "AlphaZero"
                
            while game.getGameEnded(board, cur_player) == 0:
                canonical_board = game.getCanonicalForm(board, cur_player)
                
                is_az_turn = (cur_player == 1 and p1_name == "AlphaZero") or (cur_player == -1 and p2_name == "AlphaZero")
                
                if is_az_turn:
                    if len(az_mcts.Ps) > 20000:
                        az_mcts.Qsa.clear()
                        az_mcts.Nsa.clear()
                        az_mcts.Ns.clear()
                        az_mcts.Ps.clear()
                        az_mcts.Es.clear()
                        az_mcts.Vs.clear()
                        gc.collect()
                        
                    probs = az_mcts.getAction(canonical_board, temp=0)
                    action = np.argmax(probs)
                else:
                    action = greedy_player.play(canonical_board)
                    
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
                
            if winner == "AlphaZero":
                az_wins += 1
            elif winner == "GreedyMCTS":
                greedy_wins += 1
            else:
                draws += 1
                
            gc.collect()

            with open(checkpoint_file, 'w') as f:
                json.dump({
                    'games_played': i + 1,
                    'az_wins': az_wins,
                    'greedy_wins': greedy_wins,
                    'draws': draws
                }, f)

            print(f" Game {i+1}/{num_games_per_stage} finished | Winner: {winner:<12} | Current AZ Wins: {az_wins} ")
            
        az_win_rate = (az_wins / num_games_per_stage) * 100
        greedy_win_rate = (greedy_wins / num_games_per_stage) * 100
        
        az_win_rates.append(az_win_rate)
        greedy_win_rates.append(greedy_win_rate)
        
        print(f" [Result] AlphaZero: {az_win_rate:.1f}% | GreedyMCTS: {greedy_win_rate:.1f}% | Draws: {draws} ")
        
    return test_stages, az_win_rates, greedy_win_rates

def plot_results(stages, greedy_rates, az_rates):
    
    import matplotlib.pyplot as plt
    import os
    
    save_dir = "TRAINNING_CHARTS"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"\n Created new directory for charts: {save_dir}/")
    
    
    plt.style.use('default')

    plt.figure(figsize=(10, 6))
    
    
    plt.plot(stages, greedy_rates, marker='o', linewidth=3, markersize=10, label='GreedyMCTS', color='#ff7f0e')
    plt.plot(stages, az_rates, marker='s', linewidth=3, markersize=10, label='AlphaZero', color='#1f77b4')
    
    
    plt.title('Performance Comparison: GreedyMCTS vs AlphaZero at Different Game Stages\n(Time Limit: 1.0s / move)', fontsize=16, pad=15)
    plt.xlabel('Remaining Empty Spots (Game Complexity)', fontsize=14)
    plt.ylabel('Win Rate (%)', fontsize=14)
    
   
    plt.xlim(max(stages) + 5, min(stages) - 5)
    plt.ylim(-5, 105)
    
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='best')
    
    
    for i, txt in enumerate(greedy_rates):
        plt.annotate(f"{txt:.0f}%", (stages[i], greedy_rates[i] + 3), fontsize=12, color='#ff7f0e', ha='center')
    for i, txt in enumerate(az_rates):
        plt.annotate(f"{txt:.0f}%", (stages[i], az_rates[i] - 6), fontsize=12, color='#1f77b4', ha='center')
        
    plt.tight_layout()
    
    output_filename = os.path.join(save_dir, 'GreedyMCTS_vs_AlphaZero_ResNet(1000).png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n Figure successfully saved to: {output_filename}")
    
    plt.show()

if __name__ == "__main__":
    stages, az_rates, greedy_rates = run_ultimate_benchmark(time_limit=1.0, num_games_per_stage=40)
    if stages is not None:
        plot_results(stages, greedy_rates, az_rates)
