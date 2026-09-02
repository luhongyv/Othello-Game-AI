import os
import faulthandler
faulthandler.enable()
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
torch.set_num_threads(1)
torch.backends.cudnn.enabled = False 
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
import gc
import sys
import time
import numpy as np
import datetime
from collections import deque
from random import shuffle
import random
import json
import pickle

# Ensure correct path addressing
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from GAME.OthelloGame import OthelloGame
from CNN.NNetTrainer import NNetWrapper
from AGENTS.OthelloPlayers import AlphaZeroMCTS
from CORE.Arena import Arena

def set_seed(seed):                
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class DualLogger(object):
    def __init__(self, stream, filepath):
        self.terminal = stream
        self.log = open(filepath, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def update_and_save_charts(log_path, charts_dir):
    import re
    import matplotlib
    matplotlib.use('Agg')  
    import matplotlib.pyplot as plt
    
    if not os.path.exists(log_path):
        return
        
    with open(log_path, 'r', encoding='utf-8') as f:
        log_data = f.read()
        
    policy_losses = [float(x) for x in re.findall(r'Policy Loss: ([\d\.]+)', log_data)]
    value_losses = [float(x) for x in re.findall(r'Value Loss: ([\d\.]+)', log_data)]
    win_rates = [float(x) for x in re.findall(r'Win Rate = ([\d\.]+)', log_data)]
    
    if len(policy_losses) == 0 and len(win_rates) == 0:
        return
        
    plt.figure(figsize=(15, 5))
    
    if len(policy_losses) > 0:
        plt.subplot(1, 3, 1)
        plt.plot(policy_losses, color='blue', label='Policy Loss')
        plt.title('Policy Loss (Action Prediction)')
        plt.xlabel('Training Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        
    if len(value_losses) > 0:
        plt.subplot(1, 3, 2)
        plt.plot(value_losses, color='red', label='Value Loss')
        plt.title('Value Loss (Win/Loss Prediction)')
        plt.xlabel('Training Epochs')
        plt.grid(True)
        
    if len(win_rates) > 0:
        plt.subplot(1, 3, 3)
        plt.plot(win_rates, marker='o', color='green', label='Win Rate vs Best')
        plt.axhline(y=55.0, color='gray', linestyle='--', label='Upgrade Threshold (55%)')
        plt.title('Arena Win Rate (Evolution)')
        plt.xlabel('Iteration')
        plt.ylabel('Win Rate (%)')
        plt.xticks(range(len(win_rates)), range(1, len(win_rates) + 1))
        plt.legend()
        plt.grid(True)
        
    plt.tight_layout()
    base_name = os.path.basename(log_path).replace('.txt', '.png')
    chart_path = os.path.join(charts_dir, base_name)
    plt.savefig(chart_path, dpi=150)
    plt.close()

class PureRLPipeline:
    def __init__(self, game):
        self.game = game
        self.models_dir = os.path.join(os.getcwd(), "MODELS")
        self.charts_dir = os.path.join(os.getcwd(), "TRAINNING_CHARTS")
        self.log_dir = os.path.join(os.getcwd(), "LOG")
        self.ckpt_dir = os.path.join(os.getcwd(), "checkpoint")
        
        for d in [self.models_dir, self.charts_dir, self.log_dir, self.ckpt_dir]:
            os.makedirs(d, exist_ok=True)
            
        run_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(self.log_dir, f"log_{run_time}.txt")
        sys.stdout = DualLogger(sys.stdout, self.log_path)
        sys.stderr = DualLogger(sys.stderr, self.log_path)

        self.num_iters = 20
        self.num_eps = 100
        self.temp_threshold = 15
        self.update_threshold = 0.55
        self.maxlen_of_queue = 30000
        self.num_mcts_sims = 400
        self.arena_games = 400
        self.epochs = 10
        self.batch_size = 64
        self.lr = 0.001
        self.c_puct = 1.0

        self.nnet_current = NNetWrapper(self.game)
        self.nnet_best = NNetWrapper(self.game)
        
        try:
            self.nnet_best.load_checkpoint(folder=self.models_dir, filename="model_Pure_RL.pth")
            self.nnet_current.load_checkpoint(folder=self.models_dir, filename="model_Pure_RL.pth")
            print("Loaded existing model_Pure_RL.pth. Continuing evolution...")
        except Exception:
            print("No existing model_Pure_RL.pth found. Starting from scratch (Tabula Rasa)...")
            self.nnet_best.save_checkpoint(folder=self.models_dir, filename="model_Pure_RL.pth")

        self.history_dataset = deque(maxlen=self.maxlen_of_queue)
        self.start_iter = 1
        
        self.state_file = os.path.join(self.ckpt_dir, "pipeline_state_Pure_RL.json")
        self.buffer_file = os.path.join(self.ckpt_dir, "replay_buffer_Pure_RL.pkl")
        
        if os.path.exists(self.state_file):
            print(f"Found existing pipeline state at {self.state_file}!")
            with open(self.state_file, "r", encoding="utf-8") as f:
                state_data = json.load(f)
                self.start_iter = state_data.get('iteration', 0) + 1
                if self.start_iter <= self.num_iters:
                    print(f"Fast-forwarding: Resuming directly from Iteration {self.start_iter}/{self.num_iters}")
                else:
                    print(f"All {self.num_iters} iterations were already completed previously.")
                    
        if os.path.exists(self.buffer_file):
            print(f"Found existing replay buffer at {self.buffer_file}!")
            try:
                with open(self.buffer_file, "rb") as f:
                    saved_buffer = pickle.load(f)
                    self.history_dataset.extend(saved_buffer)
                print(f"Successfully loaded {len(self.history_dataset)} historical samples into replay buffer.")
            except Exception as e:
                print(f"Warning: Failed to load replay buffer. Error: {e}")

    def execute_episode(self):
        train_examples = []
        board = self.game.getInitBoard()
        cur_player = 1
        episode_step = 0
        
        mcts = AlphaZeroMCTS(self.game, self.nnet_best, num_sims=self.num_mcts_sims, c_puct=self.c_puct)
        
        while True:
            episode_step += 1
            canonical_board = self.game.getCanonicalForm(board, cur_player)
            time.sleep(0.01)
            temp = int(episode_step < self.temp_threshold)
            
            pi = mcts.getAction(canonical_board, temp=temp)
            train_examples.append([canonical_board, cur_player, pi, None])
            
            action = np.random.choice(len(pi), p=pi)
            board, cur_player = self.game.getNextState(board, cur_player, action)
            
            r = self.game.getGameEnded(board, cur_player)
            if r != 0:
                for step_idx in range(len(train_examples)):
                    is_same_player = (train_examples[step_idx][1] == cur_player)
                    train_examples[step_idx][3] = r if is_same_player else -r
                del mcts
                return [(x[0], x[2], x[3]) for x in train_examples]

    def play_arena(self, iteration):
        print(f"[3/3] Arena Evaluation: Current VS Best ({self.arena_games} Games)...")
        self.nnet_current.to_gpu()
        self.nnet_best.to_gpu()
        
        start_g = 0
        pwins, nwins, draws = 0, 0, 0
        
        arena_ckpt_file = os.path.join(self.ckpt_dir, f"iter_{iteration}_arena_ckpt_Pure.pkl")
        
        if os.path.exists(arena_ckpt_file):
            with open(arena_ckpt_file, 'rb') as f:
                ckpt = pickle.load(f)
                start_g = ckpt['g_completed']
                pwins = ckpt['pwins']
                nwins = ckpt['nwins']
                draws = ckpt['draws']
            print(f"  -> Resuming Arena from game {start_g+1}/{self.arena_games}...")
            
        with torch.no_grad():
            for g in range(start_g, self.arena_games):
                pmcts_current = AlphaZeroMCTS(self.game, self.nnet_current, num_sims=self.num_mcts_sims, c_puct=self.c_puct)
                pmcts_best = AlphaZeroMCTS(self.game, self.nnet_best, num_sims=self.num_mcts_sims, c_puct=self.c_puct)
                
                board = self.game.getInitBoard()
                cur_player = 1
                p_color = 1 if g % 2 == 0 else -1
                
                while self.game.getGameEnded(board, cur_player) == 0:
                    canonical_board = self.game.getCanonicalForm(board, cur_player)
                    if cur_player == p_color:
                        pi = pmcts_current.getAction(canonical_board, temp=0)
                    else:
                        pi = pmcts_best.getAction(canonical_board, temp=0)
                        
                    action = np.argmax(pi)
                    board, cur_player = self.game.getNextState(board, cur_player, action)
                    
                result = self.game.getGameEnded(board, 1)
                if result == p_color:
                    pwins += 1
                elif result == -p_color:
                    nwins += 1
                else:
                    draws += 1
                    
                safe_tmp_file = arena_ckpt_file + ".tmp"
                with open(safe_tmp_file, "wb") as f:
                    pickle.dump({'g_completed': g + 1, 'pwins': pwins, 'nwins': nwins, 'draws': draws}, f)
                os.replace(safe_tmp_file, arena_ckpt_file)
                
                print(f"  -> Arena Game {g+1}/{self.arena_games} completed | New Wins: {pwins}, Old: {nwins}, Draws: {draws}", end='\r')
                
                del pmcts_current, pmcts_best
                gc.collect()
                
        print()
        
        if os.path.exists(arena_ckpt_file):
            os.remove(arena_ckpt_file)
            
        return pwins, nwins, draws

    def learn(self):
        print(" AlphaZero Pure RL Evolution Loop Starting...")
        print("========================================================")
        
        for i in range(self.start_iter, self.num_iters + 1):
            print(f"\n" + "-"*50)
            print(f" ITERATION {i}/{self.num_iters}")
            print("-" * 50)
            
            temp_model_name = f"temp_current_iter_{i}.pth"
            temp_model_path = os.path.join(self.models_dir, temp_model_name)
            arena_ckpt_file = os.path.join(self.ckpt_dir, f"iter_{i}_arena_ckpt_Pure.pkl")
            eps_ckpt_file = os.path.join(self.ckpt_dir, f"iter_{i}_eps_ckpt_Pure.pkl")
            
            skip_phase_1 = os.path.exists(temp_model_path) or os.path.exists(arena_ckpt_file)
            
            if not skip_phase_1:
                print(f"[1/3] Self-Play: Generating {self.num_eps} episodes...")
                self.nnet_best.to_gpu()
                iteration_data = []
                
                start_eps = 0
                if os.path.exists(eps_ckpt_file):
                    with open(eps_ckpt_file, 'rb') as f:
                        ckpt = pickle.load(f)
                        start_eps = ckpt['eps_completed']
                        iteration_data = ckpt['examples']
                    print(f"Resuming self-play from game {start_eps+1}/{self.num_eps}...")

                for eps in range(start_eps, self.num_eps):
                    new_examples = self.execute_episode()
                    iteration_data += new_examples
                    
                    safe_tmp_file = eps_ckpt_file + ".tmp"
                    with open(safe_tmp_file, "wb") as f:
                        pickle.dump({'eps_completed': eps + 1, 'examples': iteration_data}, f)
                    os.replace(safe_tmp_file, eps_ckpt_file)
                    
                    sys.stdout.write(f"\r  -> Episode {eps+1}/{self.num_eps} completed.")
                    sys.stdout.flush()
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        
                print(f"\n  -> Generated {len(iteration_data)} new samples.")
                
                if os.path.exists(eps_ckpt_file):
                    os.remove(eps_ckpt_file)
                    
                self.history_dataset.extend(iteration_data)
                print(f"  -> Total replay buffer size: {len(self.history_dataset)}")
                
                with open(self.buffer_file, "wb") as f:
                    pickle.dump(list(self.history_dataset), f)
            else:
                print(f"[1/3] Self-Play Phase safely skipped (Data already in buffer).")
            
            if os.path.exists(temp_model_path):
                print(f"\n[2/3] Found trained temporary model. Loading weights to resume evaluation...")
                self.nnet_current.load_checkpoint(folder=self.models_dir, filename=temp_model_name)
            else:
                if os.path.exists(arena_ckpt_file):
                    print("Found old Arena scores but lost trained weights. Clearing Arena checkpoint to ensure fairness.")
                    os.remove(arena_ckpt_file)
                    
                print(f"\n[2/3] Training Current Model...")
                self.nnet_current.to_gpu()
                self.nnet_current.nnet.load_state_dict(self.nnet_best.nnet.state_dict())
                
                train_data = list(self.history_dataset)
                shuffle(train_data)
                self.nnet_current.train(train_data, epochs=self.epochs, batch_size=self.batch_size)
                
                self.nnet_current.save_checkpoint(folder=self.models_dir, filename=temp_model_name)
                print(f"  -> Model weights safely solidified for Arena evaluation.")
            
            p1_wins, p2_wins, draws = self.play_arena(i)
            
            if p1_wins + p2_wins == 0:
                win_rate = 0
            else:
                win_rate = p1_wins / (p1_wins + p2_wins)
                
            print(f"\nResult: Current Net Win Rate = {win_rate*100:.1f}% (Threshold: {self.update_threshold*100:.1f}%)")
            
            if win_rate >= self.update_threshold:
                print("ACCEPTED! Upgrading model_Pure_RL.pth...")
                self.nnet_current.save_checkpoint(folder=self.models_dir, filename=f"model_Pure_RL_iter_{i}.pth")
                self.nnet_current.save_checkpoint(folder=self.models_dir, filename="model_Pure_RL.pth")
                self.nnet_best.load_checkpoint(folder=self.models_dir, filename="model_Pure_RL.pth")
            else:
                print("REJECTED. Keeping the old best model.")
                
            try:
                update_and_save_charts(self.log_path, self.charts_dir)
                print(f"[Auto-Plot] Real-time charts updated in {self.charts_dir}")
            except Exception as plot_error:
                print(f"[Warning] Failed to auto-update charts: {plot_error}") 

            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump({'iteration': i, 'win_rate': win_rate}, f, indent=4)
                
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)
                
            print(f"Iteration {i} state fully saved! Safe to interrupt.")

        print(" Evolution Complete!")

if __name__ == "__main__":
    game = OthelloGame(8)
    pipeline = PureRLPipeline(game)
    pipeline.learn()