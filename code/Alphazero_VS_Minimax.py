import os
import sys
import faulthandler
faulthandler.enable()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
torch.set_num_threads(1)
torch.backends.cudnn.enabled = False

import numpy as np
import random
import time
import gc

from GAME.OthelloGame import OthelloGame 
from AGENTS.OthelloPlayers import MinimaxPlayer, AlphaZeroMCTS
from CNN.NNetTrainer import NNetWrapper

def play_benchmark(num_games=30):
    """
    AlphaZero (MCTS + Neural Network) VS Traditional Minimax (Alpha-Beta)
    """
    print("==========================================================")
    print("Strict Control-Variable Benchmark: AlphaZero VS Minimax")
    print("==========================================================")
    game = OthelloGame(8)
    
    # Load the trained AlphaZero model
    print("Loading the 20th generation AlphaZero Best Model...")
    az_net = NNetWrapper(game)
    az_net.to_cpu() # Force CPU inference to avoid overheating or overload
    
    model_path = os.path.join(os.getcwd(), "MODELS", "best_model.pth")
    if os.path.exists(model_path):
        az_net.load_checkpoint(folder=os.path.join(os.getcwd(), "MODELS"), filename="best_model.pth")
    else:
        print(f"Error: {model_path} not found! Please check your MODELS folder.")
        return

    # 2. Instantiate the two players (strictly align physical thinking resources)
    # Give AlphaZero 200 simulations so its per-move time aligns closely with Minimax's 1.0 second
    az_mcts = AlphaZeroMCTS(game, az_net, num_sims=200, c_puct=1.0)
    minimax_player = MinimaxPlayer(game, time_limit=1.0)

    print(f"\nBenchmark Started: Running {num_games} Games...")
    print("----------------------------------------------------------")
    
    az_wins = 0
    minimax_wins = 0
    draws = 0

    for i in range(num_games):
        board = game.getInitBoard()
        cur_player = 1  # 1 is black, -1 is white
        
        # Odd-numbered games: AlphaZero plays black (first), Minimax plays white (second)
        # Even-numbered games: Minimax plays black (first), AlphaZero plays white (second)
        az_color = 1 if i % 2 == 0 else -1
        
        while game.getGameEnded(board, 1) == 0:
            # AlphaZero's turn to move
            if cur_player == az_color:
                # Periodically clean AlphaZero's memory tree to free physical memory
                if len(az_mcts.Ps) > 20000:
                    az_mcts.Qsa.clear()
                    az_mcts.Nsa.clear()
                    az_mcts.Ns.clear()
                    az_mcts.Ps.clear()
                    az_mcts.Es.clear()
                    az_mcts.Vs.clear()
                    gc.collect()
                    
                # AlphaZero must view the board in the current canonical orientation (relative to the player)
                canonical_board = game.getCanonicalForm(board, cur_player)
                probs = az_mcts.getAction(canonical_board, temp=0)
                action = np.argmax(probs)
            
            # Minimax's turn to move
            else:

                action = minimax_player.play(board)
            
            # Execute the move
            board, next_player = game.getNextState(board, cur_player, action)
            cur_player = next_player

        # Settle the game result
        r = game.getGameEnded(board, 1)
        if (r == 1 and az_color == 1) or (r == -1 and az_color == -1):
            az_wins += 1
            result_str = "Winner: AlphaZero"
        elif r == 0:
            draws += 1
            result_str = "Draw"
        else:
            minimax_wins += 1
            result_str = "Winner: Minimax"
            
        print(f"Game {i+1:02d}/{num_games} finished -> {result_str} (Score: AZ {game.getScore(board, az_color)} VS Minimax {game.getScore(board, -az_color)})")
        
        # Run deep garbage collection after each game
        gc.collect()

    print("\n================== Final Benchmark Results ==================")
    print(f"AlphaZero Wins: {az_wins} ({az_wins/num_games*100:.1f}%)")
    print(f"Minimax Wins:   {minimax_wins} ({minimax_wins/num_games*100:.1f}%)")
    print(f"Draws:          {draws}")
    print("=============================================================")

if __name__ == "__main__":
    play_benchmark(num_games=30)
if __name__ == "__main__":
    play_benchmark(num_games=30)
