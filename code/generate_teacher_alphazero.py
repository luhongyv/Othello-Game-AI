import os
import sys
import numpy as np
import pickle
import time
import gc
import random 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
set_seed(42)

sys.path.append(os.getcwd())
from GAME.OthelloGame import OthelloGame
from AGENTS.OthelloPlayers import GreedyMCTSPlayer

def get_symmetries(board, pi):
    """
    Data Augmentation: Utilizes the rotational and reflectional symmetries 
    of the Othello board to augment 1 sample into 8 samples.
    """     
    pi_board = np.reshape(pi[:-1], (8, 8))
    symmetries = []
    
    for i in range(1, 5):
        for j in [True, False]:
            new_b = np.rot90(board, i)
            new_pi = np.rot90(pi_board, i)
            if j:
                new_b = np.fliplr(new_b)
                new_pi = np.fliplr(new_pi)
            
            new_pi_flat = list(new_pi.ravel()) + [pi[-1]]
            symmetries.append((new_b, new_pi_flat))
            
    return symmetries

def collect_expert_data(num_games=100, time_limit=0.5, save_name="expert_data_test_100.pkl"):
    print("==================================================")
    print("Initializing Expert Data Generation Pipeline...")
    print(f"Target Games: {num_games}")
    print(f"Time Limit per Move: {time_limit}s")
    print("==================================================")
    
    game = OthelloGame(8)
    teacher = GreedyMCTSPlayer(game, time_limit=time_limit)
    
    dataset = [] 
    start_game = 0
    
    if os.path.exists(save_name):
        print(f"Found existing checkpoint file: {save_name}")
        try:
            with open(save_name, 'rb') as f:
                dataset = pickle.load(f)
            start_game = len(dataset) // 480
            print(f"Successfully loaded checkpoint. Current samples: {len(dataset)}")
            print(f"Resuming generation from game {start_game + 1}...")
        except Exception as e:
            print(f"Warning: Failed to load checkpoint. Starting from scratch. Reason: {e}")
            dataset = []
            start_game = 0

    if start_game >= num_games:
        print("Target number of games already reached. No further generation required.")
        return

    # 记录全局开始时间
    global_start_time = time.time()
    
    for g in range(start_game, num_games):
        game_start_time = time.time()  # 记录单局开始时间
        
        board = game.getInitBoard()
        cur_player = 1
        game_history = [] 
        
        while game.getGameEnded(board, cur_player) == 0:
            canonical_board = game.getCanonicalForm(board, cur_player)
            action = teacher.play(canonical_board)
            
            policy_target = np.zeros(game.getActionSize())
            policy_target[action] = 1.0
            
            syms = get_symmetries(canonical_board, policy_target)
            for b, p in syms:
                game_history.append({
                    'state': b,
                    'policy': p,
                    'player': cur_player
                })
            
            board, cur_player = game.getNextState(board, cur_player, action)
            
        final_result = game.getGameEnded(board, 1)
        
        for step in game_history:
            if final_result == 0:
                reward = 0.0
            else:
                reward = 1.0 if final_result == step['player'] else -1.0
                
            dataset.append((step['state'], step['policy'], reward))
            
        # 统计单局时间与总耗时
        game_elapsed = time.time() - game_start_time
        total_elapsed = time.time() - global_start_time
        avg_time_per_game = total_elapsed / ((g + 1) - start_game)
        
        print(f"Game {g+1}/{num_games} done | Game Time: {game_elapsed:.1f}s | "
              f"Avg Time/Game: {avg_time_per_game:.1f}s | Total Samples: {len(dataset)}")
        
        # Save checkpoint every 10 games
        if (g + 1) % 10 == 0:
            with open(save_name, 'wb') as f:
                pickle.dump(dataset, f)
            print(f">>> Checkpoint saved to {save_name} at Game {g+1}")
            
        gc.collect()    
        
    # Final save after loop completion
    with open(save_name, 'wb') as f:
        pickle.dump(dataset, f)
        
    # 计算最终总耗时
    final_total_seconds = time.time() - global_start_time
    final_hours = final_total_seconds // 3600
    final_minutes = (final_total_seconds % 3600) // 60
        
    print("==================================================")
    print("Data Generation Completed Successfully!")
    print(f"Total Samples Generated: {len(dataset)}")
    print(f"Final Dataset Saved to: {save_name}")
    print(f"Total Time Spent for {num_games - start_game} games: {int(final_hours)}h {int(final_minutes)}m ({final_total_seconds:.2f} seconds)")
    print("==================================================")

if __name__ == "__main__":
    collect_expert_data(
        num_games=1000,                
        time_limit=0.5,                
        save_name="data_training_GreedyMCTS_test.pkl" 
    )