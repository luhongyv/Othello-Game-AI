# -*- coding: utf-8 -*-
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'

import sys
import numpy as np
import torch
torch.set_num_threads(1) 

import time
import gc
import pickle

sys.path.append(os.getcwd())
from GAME.OthelloGame import OthelloGame
from CNN.NNetTrainer import NNetWrapper
from AGENTS.OthelloPlayers import AlphaZeroMCTS

def run_mcts_budget_sweep():
    print("=" * 60)
    print("Phase 1: MCTS Budget Sweep (N vs 2N Preliminary Test)")
    print("=" * 60)

    game = OthelloGame(8)
    
    nnet = NNetWrapper(game)
    checkpoint_path = os.path.join("checkpoint", "resnet_expert.pth")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Cannot find model {checkpoint_path}. Please ensure the base model exists.")
        return
        
    print("Loading base model: resnet_expert.pth ...")
    nnet.load_checkpoint(folder="checkpoint", filename="resnet_expert.pth")
    
    #budget_pairs = [(50, 100), (100, 200), (150, 300), (200, 400),(400, 800),(800, 1600)]
    budget_pairs = [(400, 800),(800, 1600)]
    total_games = 400 
    ckpt_dir = "checkpoint"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    ckpt_file = os.path.join(ckpt_dir, "mcts_sweep.pkl")
    
    start_pair_idx = 0
    start_game_idx = 0
    wins_N = 0
    wins_2N = 0
    draws = 0
    results = []
    accumulated_time = 0.0

    if os.path.exists(ckpt_file):
        with open(ckpt_file, "rb") as f:
            state = pickle.load(f)
        start_pair_idx = state['pair_idx']
        start_game_idx = state['game_idx']
        wins_N = state['wins_N']
        wins_2N = state['wins_2N']
        draws = state['draws']
        results = state['results']
        accumulated_time = state['accumulated_time']
        
        if start_pair_idx < len(budget_pairs):
            curr_n, curr_2n = budget_pairs[start_pair_idx]
            print(f"Recovered from crash! Resuming Matchup ({curr_n} vs {curr_2n}) from game {start_game_idx+1}/{total_games}...")
        else:
            print(f"Found checkpoint, but all matchups are already completed!")

    with torch.no_grad():
        for pair_idx in range(start_pair_idx, len(budget_pairs)):
            n, two_n = budget_pairs[pair_idx]
            print(f"\n>>> Starting Matchup: {n} Simulations vs {two_n} Simulations <<<")
            
            for g in range(start_game_idx, total_games):
                game_start_time = time.time()
                board = game.getInitBoard()
                cur_player = 1
                
                is_N_first = (g % 2 == 0)
                
                mcts_N = AlphaZeroMCTS(game, nnet, num_sims=n, c_puct=1.0)
                mcts_2N = AlphaZeroMCTS(game, nnet, num_sims=two_n, c_puct=1.0)
                move_count = 0
                while game.getGameEnded(board, cur_player) == 0:
                    canonical_board = game.getCanonicalForm(board, cur_player)
                    
                    current_temp = 1 if move_count < 4 else 0
                    
                    if (cur_player == 1 and is_N_first) or (cur_player == -1 and not is_N_first):
                        pi = mcts_N.getAction(canonical_board, temp=current_temp, add_noise=False) 
                    else:
                        pi = mcts_2N.getAction(canonical_board, temp=current_temp, add_noise=False)
                        
                    if current_temp == 0:
                        action = np.argmax(pi) 
                    else:
                        pi = np.array(pi)
                        pi /= np.sum(pi)
                        action = np.random.choice(len(pi), p=pi)    
                    
                    board, cur_player = game.getNextState(board, cur_player, action)
                    
                    move_count += 1  
                    time.sleep(0.01)  
                result = game.getGameEnded(board, 1)
                
                if result != 0:
                    if is_N_first:
                        if result == 1: wins_N += 1
                        else: wins_2N += 1
                    else:
                        if result == 1: wins_2N += 1
                        else: wins_N += 1
                else:
                    draws += 1
                    
                accumulated_time += (time.time() - game_start_time)
                
                print(f"Game {g+1}/{total_games} | {two_n}-Sims Wins: {wins_2N} | {n}-Sims Wins: {wins_N} | Draws: {draws}", end='\r', flush=True)
                sys.stdout.flush() 
                if (g + 1) % 10 == 0 or (g + 1) == total_games:
                    state = {
                        'pair_idx': pair_idx,
                        'game_idx': g + 1,
                        'wins_N': wins_N,
                        'wins_2N': wins_2N,
                        'draws': draws,
                        'results': results,
                        'accumulated_time': accumulated_time
                    }
                    safe_tmp_file = ckpt_file + ".tmp"
                    try:
                        with open(safe_tmp_file, "wb") as f:
                            pickle.dump(state, f)
                        os.replace(safe_tmp_file, ckpt_file)
                    except OSError as e:
                        pass
                
                del mcts_N, mcts_2N
                if torch.cuda.is_available():
                   torch.cuda.empty_cache()
                
            gc.collect() 
                
            win_rate_2N = (wins_2N / total_games) * 100
            print(f"\nMatchup {n} vs {two_n} Finished in {accumulated_time:.1f}s.")
            print(f"Result: {two_n}-Sims won {win_rate_2N:.1f}% of games.")
            
            results.append({
                'n': n,
                'two_n': two_n,
                'wins_2N': wins_2N,
                'wins_N': wins_N,
                'draws': draws,
                'win_rate_2N': win_rate_2N,
                'time': accumulated_time
            })
            start_game_idx = total_games
            start_game_idx = 0
            wins_N = 0
            wins_2N = 0
            draws = 0
            accumulated_time = 0.0
            
            gc.collect()

    if os.path.exists(ckpt_file):
        os.remove(ckpt_file)


    print("\n" + "="*80)
    print("MCTS Budget Sweep Test Results (400 Games per Matchup)")
    print("="*80)
    header = f"| {'Matchup (N vs 2N)':^17} | {'2N Wins':^9} | {'N Wins':^8} | {'Draws':^7} | {'2N Win Rate':^13} | {'Time (s)':^10} |"
    print(header)
    print(f"|{'-'*19}|{'-'*11}|{'-'*10}|{'-'*9}|{'-'*15}|{'-'*12}|")
    
    for r in results:
        match_str = f"{r['n']} vs {r['two_n']}"
        win_rate_str = f"{r['win_rate_2N']:.1f}%" 
        row = f"| {match_str:^17} | {r['wins_2N']:^9} | {r['wins_N']:^8} | {r['draws']:^7} | {win_rate_str:^13} | {r['time']:^10.1f} |"
        print(row)
    print("="*80)
    
    import matplotlib.pyplot as plt
    
    save_dir = "TRAINNING_CHARTS"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    columns = ["Matchup (N vs 2N)", "2N Wins", "N Wins", "Draws", "2N Win Rate", "Time (s)"]
    cell_text = []
    
    for r in results:
        match_str = f"{r['n']} vs {r['two_n']}"
        win_rate_str = f"{r['win_rate_2N']:.1f}%"
        cell_text.append([
            match_str, 
            str(r['wins_2N']), 
            str(r['wins_N']), 
            str(r['draws']), 
            win_rate_str, 
            f"{r['time']:.1f}"
        ])
        
    fig, ax = plt.subplots(figsize=(10, len(cell_text) * 0.6 + 1.5), dpi=300)
    ax.axis('off')
    
    table = ax.table(cellText=cell_text, colLabels=columns, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    
    for j, col in enumerate(columns):
        cell = table[0, j]
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#2980b9') 
        
    for idx in range(1, len(cell_text) + 1):
        for j in range(len(columns)):
            cell = table[idx, j]
            if idx % 2 == 0:
                cell.set_facecolor('#f4f6f7')
                
    plt.title(f"MCTS Budget Sweep Results (Total Games = {total_games})", fontweight="bold", fontsize=14, pad=15)
    plt.tight_layout()
    
    save_path_png = os.path.join(save_dir, "MCTS_Budget_Sweep_Table.png")
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nHigh-quality PNG table successfully saved to: {save_path_png}")

    print("\nAnalysis Tip for your thesis:")
    print("Look for the point of diminishing returns. If (200 vs 400) yields a win rate close to 50%,")
    print("it means doubling the computation from 200 to 400 doesn't provide significant statistical advantage.")
    print("In that case, choosing 200 simulations for your main experiment is highly defensible!")

if __name__ == "__main__":
    run_mcts_budget_sweep()
