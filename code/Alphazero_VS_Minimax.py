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
    
    # 1. 加载训练好的 AlphaZero 大脑
    print("Loading the 20th generation AlphaZero Best Model...")
    az_net = NNetWrapper(game)
    az_net.to_cpu() # 强制 CPU 推理，防发热过载
    
    model_path = os.path.join(os.getcwd(), "MODELS", "best_model.pth")
    if os.path.exists(model_path):
        az_net.load_checkpoint(folder=os.path.join(os.getcwd(), "MODELS"), filename="best_model.pth")
    else:
        print(f"Error: {model_path} not found! Please check your MODELS folder.")
        return

    # 2. 实例化两个选手（严格对齐物理思考资源）
    # 给 AlphaZero 200 次搜索，使其单步耗时与 Minimax 的 1.0 秒高度对齐
    az_mcts = AlphaZeroMCTS(game, az_net, num_sims=200, c_puct=1.0)
    minimax_player = MinimaxPlayer(game, time_limit=1.0)

    print(f"\nBenchmark Started: Running {num_games} Games...")
    print("----------------------------------------------------------")
    
    az_wins = 0
    minimax_wins = 0
    draws = 0

    for i in range(num_games):
        board = game.getInitBoard()
        cur_player = 1  # 1 为黑方，-1 为白方
        
        # 奇数局：AlphaZero 执黑先手，Minimax 执白后手
        # 偶数局：Minimax 执黑先手，AlphaZero 执白后手
        az_color = 1 if i % 2 == 0 else -1
        
        while game.getGameEnded(board, 1) == 0:
            # 轮到 AlphaZero 下棋
            if cur_player == az_color:
                # 动态打扫 AlphaZero 的记忆树，腾出物理内存
                if len(az_mcts.Ps) > 20000:
                    az_mcts.Qsa.clear()
                    az_mcts.Nsa.clear()
                    az_mcts.Ns.clear()
                    az_mcts.Ps.clear()
                    az_mcts.Es.clear()
                    az_mcts.Vs.clear()
                    gc.collect()
                    
                # AlphaZero 必须在当前的旋转视角（canonical_board）下看盘
                canonical_board = game.getCanonicalForm(board, cur_player)
                probs = az_mcts.getAction(canonical_board, temp=0)
                action = np.argmax(probs)
            
            # 轮到 Minimax 下棋
            else:

                action = minimax_player.play(board)
            
            # 执行落子
            board, next_player = game.getNextState(board, cur_player, action)
            cur_player = next_player

        # 结算本局
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
        
        # 每局结束深度释放内存
        gc.collect()

    print("\n================== Final Benchmark Results ==================")
    print(f"AlphaZero Wins: {az_wins} ({az_wins/num_games*100:.1f}%)")
    print(f"Minimax Wins:   {minimax_wins} ({minimax_wins/num_games*100:.1f}%)")
    print(f"Draws:          {draws}")
    print("=============================================================")

if __name__ == "__main__":
    play_benchmark(num_games=30)