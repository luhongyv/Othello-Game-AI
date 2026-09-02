import sys
import os
import csv
import time
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from GAME.OthelloGame import OthelloGame
from AGENTS.OthelloPlayers import GreedyOthelloPlayer, AlphaZeroMCTS
from CORE.Arena import Arena
from CNN.NNetTrainer import NNetWrapper

class AlphaZeroPlayer:
    def __init__(self, game, model_folder, model_file, mcts_sims=400):
        self.game = game
        self.nnet = NNetWrapper(game)
        self.nnet.load_checkpoint(folder=model_folder, filename=model_file)
        self.mcts_args = {'numMCTSSims': mcts_sims, 'cpuct': 1.0}
        self.mcts = None
        self.reset_mcts()
        
    def reset_mcts(self):
        """每局对局后重置 MCTS 树，防止内存泄露"""
        self.mcts = AlphaZeroMCTS(
            self.game, 
            self.nnet, 
            num_sims=self.mcts_args['numMCTSSims'], 
            c_puct=self.mcts_args['cpuct']
        )
        
    def __call__(self, board):
        pi = self.mcts.getAction(board, temp=0)
        return np.argmax(pi)

def run_evaluation(agent_name, model_file, baseline_func, game, num_games=100):
    """单模型对抗评估函数"""
    print(f"\n[{time.strftime('%H:%M:%S')}] Evaluating {model_file} (Total Games: {num_games}, MCTS Sims: 400) ...")
    MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MODELS')
    
    try:
        test_agent = AlphaZeroPlayer(game, model_folder=MODELS_DIR, model_file=model_file, mcts_sims=400)
    except FileNotFoundError:
        print(f"  [Warning] Model {model_file} not found. Skipping...")
        return None
        
    arena = Arena(
        player1=test_agent, 
        player2=baseline_func, 
        game=game, 
        p1_name=agent_name, 
        p2_name="Greedy_Baseline",
        log_file="learning_curve_log.txt"
    )
    
    # 临时重定向标准输出，避免控制台刷屏
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            w1, w2, _ = arena.play_games(num_games=num_games, num_opening_random_moves=4, save_freq=25)
        finally:
            sys.stdout = old_stdout
            
    print(f"  -> Result: {agent_name} Win Rate: {w1 * 100:.1f}%")
    return w1

if __name__ == "__main__":
    game = OthelloGame(8)
    greedy_baseline = GreedyOthelloPlayer(game).play
    
    checkpoints_to_test = {
        "Pure_RL": [
            ("Iter_2", "model_Pure_RL_iter_2.pth"),
            ("Iter_5", "model_Pure_RL_iter_5.pth"),
            ("Iter_8", "model_Pure_RL_iter_8.pth"),
            ("Iter_13", "model_Pure_RL_iter_13.pth"),
            ("Iter_18", "model_Pure_RL_iter_18.pth"),
            ("Final", "model_Pure_RL.pth")
        ],
        "SL_RL_Hybrid": [
            ("Iter_0", "model_Greedy_SL.pth"),
            ("Iter_1", "model_Greedy_SL_RL_iter_1.pth"),
            ("Iter_5", "model_Greedy_SL_RL_iter_5.pth"),
            ("Iter_8", "model_Greedy_SL_RL_iter_8.pth"),
            ("Iter_14", "model_Greedy_SL_RL_iter_14.pth"),
            ("Iter_19", "model_Greedy_SL_RL_iter_19.pth"),
            ("Final", "model_Greedy_SL_RL.pth")
        ]
    }
    
    csv_filename = "learning_curve_data.csv"
    
    # 自动加载已完成的记录，实现完美的断点续传
    completed_set = set()
    file_exists = os.path.exists(csv_filename)
    
    if file_exists:
        with open(csv_filename, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None) # 跳过表头
            for row in reader:
                if len(row) >= 2:
                    completed_set.add((row[0], row[1])) # 记录 (Algorithm, Iteration_Label)
                    
    # 根据文件是否存在选择追加模式或写入模式
    mode = 'a' if file_exists else 'w'
    with open(csv_filename, mode=mode, newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Algorithm", "Iteration_Label", "Model_File", "Win_Rate_vs_Baseline"])
        
        for algo_name, ckpt_list in checkpoints_to_test.items():
            print(f"\n================ Evaluating {algo_name} ================")
            for iter_label, filename in ckpt_list:
                # 检查该检查点是否已经跑过并写入 CSV
                if (algo_name, iter_label) in completed_set:
                    print(f"  [Resume Skip] {algo_name} - {iter_label} ({filename}) 已经完成，自动跳过！")
                    continue
                    
                # 正式运行评估（如需2000局可在此处修改参数）
                win_rate = run_evaluation(f"{algo_name}_{iter_label}", filename, greedy_baseline, game, num_games=2000)
                if win_rate is not None:
                    writer.writerow([algo_name, iter_label, filename, round(win_rate, 3)])
                    file.flush()
                    completed_set.add((algo_name, iter_label))
                    
    print(f"\n[DONE] All evaluations finished. Data saved to {csv_filename}")