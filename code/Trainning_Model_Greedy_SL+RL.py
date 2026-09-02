import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import numpy as np
import torch
import time
from collections import deque
import gc
import json
import pickle
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42) # 激活锁

sys.path.append(os.getcwd())
from GAME.OthelloGame import OthelloGame
from CNN.NNetTrainer import NNetWrapper
from AGENTS.OthelloPlayers import AlphaZeroMCTS

from Trainning_Model_Greedy_SL import AlphaZeroTrainer, OthelloDataset

class HybridSelfPlayPipeline:
    def __init__(self, game):
        self.game = game
        self.models_dir = os.path.join(os.getcwd(), "MODELS")
        self.ckpt_dir = os.path.join(os.getcwd(), "checkpoint")
        if not os.path.exists(self.models_dir): os.makedirs(self.models_dir)
        if not os.path.exists(self.ckpt_dir): os.makedirs(self.ckpt_dir)
        
        self.nnet = NNetWrapper(self.game)
        self.pnet = NNetWrapper(self.game)
        
        # --- 核心控制变量 (必须与 Pure RL 严格对齐) ---
        self.num_iters = 20          
        self.num_eps = 100           
        self.tempThreshold = 15      
        self.update_threshold = 0.55 
        self.maxlen_of_queue = 30000 
        self.arena_games = 400       
        self.epochs = 10             
        self.batch_size = 64         
        self.lr = 0.001             
        self.num_mcts_sims = 400    
        self.c_puct = 1.0           
        
        self.history_dataset = deque(maxlen=self.maxlen_of_queue)
        self.history = []
        self.start_iter = 1
        
        self.state_file = os.path.join(self.ckpt_dir, "pipeline_state_SL_RL.json") 
        self.buffer_file = os.path.join(self.ckpt_dir, "replay_buffer_SL_RL.pkl")
        
        # ==========================================
        # 1. 独家冷启动桥接逻辑 (SL -> RL 的基石)
        # ==========================================
        try:
            self.nnet.load_checkpoint(folder=self.models_dir, filename="model_Greedy_SL_RL.pth")
            print("Loaded existing model_Greedy_SL_RL.pth. Continuing hybrid evolution...")
        except Exception:
            print("No existing hybrid model. Starting from Supervised Pre-trained Expert Model (model_Greedy_SL.pth)...")
            try:
                self.nnet.load_checkpoint(folder=self.models_dir, filename="model_Greedy_SL.pth")
                # 立刻另存为 SL+RL 的专属文件，防止覆盖原专家数据
                self.nnet.save_checkpoint(folder=self.models_dir, filename="model_Greedy_SL_RL.pth")
            except Exception:
                print("Error: model_Greedy_SL.pth NOT FOUND! Please ensure Trainning_Model_Greedy_SL.py has successfully completed.")
                sys.exit(1)

        # 2. 读取 JSON 进度存档
        if os.path.exists(self.state_file):
            print(f"Found pipeline state at {self.state_file}!")
            with open(self.state_file, "r", encoding="utf-8") as f:
                self.history = json.load(f)
            if self.history:
                self.start_iter = self.history[-1]['iteration'] + 1
                if self.start_iter <= self.num_iters:
                    print(f"Fast-forwarding: Resuming directly from Iteration {self.start_iter}/{self.num_iters}")
                else:
                    print(f"All {self.num_iters} iterations completed.")
                    
        # 3. 读取 PKL 经验池存档
        if os.path.exists(self.buffer_file):
            print(f"Found existing replay buffer at {self.buffer_file}!")
            try:
                with open(self.buffer_file, "rb") as f:
                    saved_buffer = pickle.load(f)
                    self.history_dataset.extend(saved_buffer)
                print(f"Successfully loaded {len(self.history_dataset)} historical samples into replay buffer.")
            except Exception as e:
                print(f"Warning: Failed to load replay buffer. Starting with empty buffer. Error: {e}")

    def execute_episode(self):
        train_examples = []
        board = self.game.getInitBoard()
        cur_player = 1
        episode_step = 0
        
        mcts = AlphaZeroMCTS(self.game, self.nnet, num_sims=self.num_mcts_sims, c_puct=self.c_puct)
        
        while True:
            episode_step += 1
            canonical_board = self.game.getCanonicalForm(board, cur_player)
            temp = int(episode_step < self.tempThreshold)
            pi = mcts.getAction(canonical_board, temp=temp, add_noise=True)
            
            sym = self.game.getSymmetries(canonical_board, pi)
            for b, p in sym:
                train_examples.append([b, cur_player, p, None])
                
            action = np.random.choice(len(pi), p=pi)
            board, cur_player = self.game.getNextState(board, cur_player, action)
            
            r = self.game.getGameEnded(board, 1)
            if r != 0:
                res = [(x[0], x[2], r * (1 if x[1] == cur_player else -1)) for x in train_examples]
                del mcts
                del train_examples
                del sym
                return res

    def play_arena(self):
        print(f"Starting Arena: Challenger (New) vs Champion (Old) - {self.arena_games} games")
        pwins, nwins, draws = 0, 0, 0
        
        with torch.no_grad():
            for g in range(self.arena_games):
                
                # 【内存泄漏彻底修复】：每一局对战，分配干净独立的搜索树！
                pmcts = AlphaZeroMCTS(self.game, self.pnet, num_sims=self.num_mcts_sims, c_puct=self.c_puct)
                nmcts = AlphaZeroMCTS(self.game, self.nnet, num_sims=self.num_mcts_sims, c_puct=self.c_puct)
                
                board = self.game.getInitBoard()
                cur_player = 1
                p_color = 1 if g % 2 == 0 else -1
                
                while self.game.getGameEnded(board, cur_player) == 0:
                    canonical_board = self.game.getCanonicalForm(board, cur_player)
                    if cur_player == p_color:
                        pi = pmcts.getAction(canonical_board, temp=0, add_noise=False)
                    else:
                        pi = nmcts.getAction(canonical_board, temp=0, add_noise=False)
                        
                    action = np.argmax(pi)
                    board, cur_player = self.game.getNextState(board, cur_player, action)
                    
                result = self.game.getGameEnded(board, 1)
                if result == p_color:
                    pwins += 1
                elif result == -p_color:
                    nwins += 1
                else:
                    draws += 1
                    
                print(f"Arena Game {g+1}/{self.arena_games} completed | New Wins: {pwins}, Old: {nwins}, Draws: {draws}", end='\r')
                
                del pmcts, nmcts
                gc.collect()
        
        print()
        return pwins, nwins, draws

    def learn(self):
        for i in range(self.start_iter, self.num_iters + 1):
            print(f"\n==========================================")
            print(f"Starting SL+RL Hybrid Iteration: {i}/{self.num_iters}") 
            print(f"==========================================")
            
            print(f"Starting self-play (Target: {self.num_eps} episodes)...")
            start_time = time.time()
            
            iteration_train_examples = []
            start_eps = 0
            temp_ckpt_file = os.path.join(self.ckpt_dir, f"iter_{i}_selfplay_ckpt_SL_RL.pkl")
            
            # 读取局级微存档
            if os.path.exists(temp_ckpt_file):
                with open(temp_ckpt_file, "rb") as f:
                    ckpt_data = pickle.load(f)
                iteration_train_examples = ckpt_data['examples']
                start_eps = ckpt_data['eps_completed']
                print(f"Recovered! Resuming self-play from game {start_eps+1}/{self.num_eps}...")

            with torch.no_grad():
                for eps in range(start_eps, self.num_eps):
                    print(f"  -> Playing Game {eps+1}/{self.num_eps} ...", end=" ", flush=True)
                    new_examples = self.execute_episode()
                    iteration_train_examples += new_examples
                    print(f"Done! (Collected {len(new_examples)} steps)")
                    
                    safe_tmp_file = temp_ckpt_file + ".tmp"
                    with open(safe_tmp_file, "wb") as f:
                        pickle.dump({'eps_completed': eps + 1, 'examples': iteration_train_examples}, f)
                    os.replace(safe_tmp_file, temp_ckpt_file) 
                        
            print(f"\nSelf-play completed in {time.time()-start_time:.1f}s.")
            
            # 扩展全局滑动经验池并保存硬盘
            self.history_dataset.extend(iteration_train_examples)
            print(f"Total historical samples in replay buffer: {len(self.history_dataset)}")
            with open(self.buffer_file, "wb") as f:
                pickle.dump(list(self.history_dataset), f)
            print("Replay buffer saved to disk safely.")
            
            del iteration_train_examples
            gc.collect()   
            
            self.pnet.nnet.load_state_dict(self.nnet.nnet.state_dict())
            print("Starting to fine-tune the new model...")
            
            # 【完美对接 SL 的训练器】：把经验池存为 Dataset 支持的临时文件
            temp_train_data = os.path.join(self.ckpt_dir, "temp_train_data_SL_RL.pkl")
            with open(temp_train_data, "wb") as f:
                pickle.dump(list(self.history_dataset), f)
            dataset = OthelloDataset(temp_train_data)
            
            trainer = AlphaZeroTrainer(self.pnet.nnet, lr=self.lr, batch_size=self.batch_size, epochs=self.epochs)
            trainer.train(dataset)
            
            pwins, nwins, draws = self.play_arena()
            total_wins = pwins + nwins
            win_rate = float(pwins) / total_wins if total_wins > 0 else 0.0
            
            self.history.append({
                'iteration': i,
                'pwins': pwins,  
                'nwins': nwins, 
                'draws': draws,  
                'win_rate': win_rate
            })
            
            if total_wins > 0 and win_rate >= self.update_threshold:
                print(f"Evolution successful! New model win rate: {win_rate*100:.1f}%. Replacing old model.")
                self.nnet.nnet.load_state_dict(self.pnet.nnet.state_dict())
                self.nnet.save_checkpoint(folder=self.models_dir, filename=f"model_Greedy_SL_RL_iter_{i}.pth")
                self.nnet.save_checkpoint(folder=self.models_dir, filename="model_Greedy_SL_RL.pth")
            else:
                print(f"Evolution failed. New model win rate: {win_rate*100:.1f}%.")

            # 保存 JSON 进度
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=4)
            print(f"Iteration {i} state fully saved! Safe to interrupt.")
            
            if os.path.exists(temp_ckpt_file):
                os.remove(temp_ckpt_file)
                
        print("\nAll SL+RL Hybrid iterations completed.")

if __name__ == "__main__":
    game = OthelloGame(8)
    pipeline = HybridSelfPlayPipeline(game)
    pipeline.learn()