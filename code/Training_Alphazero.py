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

# Ensure correct path addressing
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from GAME.OthelloGame import OthelloGame
from CNN.NNetTrainer import NNetWrapper
from AGENTS.OthelloPlayers import AlphaZeroMCTS
from CORE.Arena import Arena

# ==========================================
# Hyperparameters Configuration
# ==========================================
args = {
    'num_iters': 20,              # Total number of evolution generations
    'num_eps': 30,                # Number of self-play games per generation
    'temp_threshold': 15,        # Steps before temperature drops to 0 (Explore -> Exploit)
    'update_threshold': 0.55,    # Minimum win rate to accept the new neural net
    'maxlen_of_queue': 30000,    # Max capacity of the experience replay buffer
    'num_mcts_sims': 30,         # Number of MCTS simulations per move
    'arena_games': 30,            # Number of games played in the Arena
    'epochs': 10,                 # Training epochs per generation
    'batch_size': 64           # Training batch size
}

class DualLogger(object):
    """同时向控制台流和日志文件流写入输出的双向记录器"""
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
    """自动解析日志文件并实时静默绘制/更新进化图表"""
    import re
    import matplotlib
    matplotlib.use('Agg')  # 激活无界面后端，防止弹出 GUI 窗口阻塞主训练循环
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
    
    # 1. 策略损失图
    if len(policy_losses) > 0:
        plt.subplot(1, 3, 1)
        plt.plot(policy_losses, color='blue', label='Policy Loss')
        plt.title('Policy Loss (Action Prediction)')
        plt.xlabel('Training Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        
    # 2. 价值损失图
    if len(value_losses) > 0:
        plt.subplot(1, 3, 2)
        plt.plot(value_losses, color='red', label='Value Loss')
        plt.title('Value Loss (Win/Loss Prediction)')
        plt.xlabel('Training Epochs')
        plt.grid(True)
        
    # 3. 胜率进化图
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

def execute_episode(game, nnet):
    """
    Executes a single episode of self-play using the current best neural network.
    Returns a list of training examples (state, policy, value).
    """
    train_examples = []
    board = game.getInitBoard()
    cur_player = 1
    episode_step = 0
    
    # CRITICAL: Create a new MCTS instance for every episode to clear the search tree memory
    mcts = AlphaZeroMCTS(game, nnet, num_sims=args['num_mcts_sims'], c_puct=1.0)
    
    while True:
        episode_step += 1
        canonical_board = game.getCanonicalForm(board, cur_player)
        time.sleep(0.01)
        # Temperature control: explore early, exploit later
        temp = int(episode_step < args['temp_threshold'])
        
        pi = mcts.getAction(canonical_board, temp=temp)
        train_examples.append([canonical_board, cur_player, pi, None])
        
        # Choose action based on the probability distribution
        action = np.random.choice(len(pi), p=pi)
        board, cur_player = game.getNextState(board, cur_player, action)
        
        r = game.getGameEnded(board, cur_player)
        if r != 0:
            # Assign labels (values) to the collected states based on the final outcome
            for step_idx in range(len(train_examples)):
                is_same_player = (train_examples[step_idx][1] == cur_player)
                train_examples[step_idx][3] = r if is_same_player else -r
            return [(x[0], x[2], x[3]) for x in train_examples]

if __name__ == "__main__":
    log_dir = os.path.join(os.getcwd(), "LOG")
    charts_dir = os.path.join(os.getcwd(), "TRAINNING_CHARTS")
    models_dir = os.path.join(os.getcwd(), "MODELS")
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    run_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_path = os.path.join(log_dir, f"log_{run_time}.txt")
    
    sys.stdout = DualLogger(sys.stdout, log_path)
    sys.stderr = DualLogger(sys.stderr, log_path)


    print(" AlphaZero Infinite Evolution Loop Starting...")
    print("========================================================")
    
    game = OthelloGame(8)
    
    # Initialize Current Model (to be trained) and Best Model (data generator)
    nnet_current = NNetWrapper(game)
    nnet_best = NNetWrapper(game)
        
    # Attempt to load an existing best model
    try:
        nnet_best.load_checkpoint(folder=models_dir, filename="best_model.pth")
        nnet_current.load_checkpoint(folder=models_dir, filename="best_model.pth")
        print("Loaded existing best_model.pth. Continuing evolution...")
    except Exception as e:
        print("No existing best_model.pth found. Starting from scratch...")
        nnet_best.save_checkpoint(folder=models_dir, filename="best_model.pth")
        
    history_dataset = deque(maxlen=args['maxlen_of_queue'])
    
    for i in range(1, args['num_iters'] + 1):
        print(f"\n" + "-"*50)
        print(f" ITERATION {i}/{args['num_iters']}")
        print("-" * 50)
        
        # --- Phase 1: Self-Play ---
        print(f"[1/3] Self-Play: Generating {args['num_eps']} episodes...")
        nnet_best.to_cpu()
        iteration_data = []
        for eps in range(args['num_eps']):
            iteration_data += execute_episode(game, nnet_best)
            sys.stdout.write(f"\r  -> Episode {eps+1}/{args['num_eps']} completed.")
            sys.stdout.flush()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        print(f"\n  -> Generated {len(iteration_data)} new samples.")
        
        history_dataset.extend(iteration_data)
        print(f"  -> Total replay buffer size: {len(history_dataset)}")
        
        # --- Phase 2: Training ---
        print(f"\n[2/3] Training Current Model...")
        nnet_current.to_gpu()  # Ensure training happens on GPU if available
        # Copy the best model weights into the current model before training
        nnet_current.nnet.load_state_dict(nnet_best.nnet.state_dict())
        
        train_data = list(history_dataset)
        shuffle(train_data)
        nnet_current.train(train_data, epochs=args['epochs'], batch_size=args['batch_size'])
        
        # --- Phase 3: Arena Evaluation ---
        print(f"\n[3/3] Arena Evaluation: Current VS Best ({args['arena_games']} Games)...")
        nnet_current.to_cpu()
        nnet_best.to_cpu()
        # Use a fresh MCTS for evaluation
        pmcts_current = AlphaZeroMCTS(game, nnet_current, num_sims=args['num_mcts_sims'])
        pmcts_best = AlphaZeroMCTS(game, nnet_best, num_sims=args['num_mcts_sims'])

        arena = Arena(player1=pmcts_current, player2=pmcts_best, game=game, p1_name="Current_Net", p2_name="Best_Net")
        
        # Suppress Arena step-by-step output by setting verbose=False
        p1_wins, p2_wins, draws = arena.play_games(num_games=args['arena_games'], verbose=False)
        
        # Calculate win rate 
        if p1_wins + p2_wins == 0:
            win_rate = 0
        else:
            win_rate = p1_wins / (p1_wins + p2_wins)
            
        print(f"\nResult: Current Net Win Rate = {win_rate*100:.1f}% (Threshold: {args['update_threshold']*100:.1f}%)")
        
        if win_rate >= args['update_threshold']:
            print(">>> ACCEPTED! Upgrading best_model.pth...")
            nnet_current.save_checkpoint(folder=models_dir, filename="best_model.pth")
            # Sync best network
            nnet_best.load_checkpoint(folder=models_dir, filename="best_model.pth")
        else:
            print(">>> REJECTED. Keeping the old best model.")

        try:
            update_and_save_charts(log_path, charts_dir)
            print(f">>> [Auto-Plot] Real-time charts updated in 'TRAINNING_CHARTS/evolution_history.png'")
        except Exception as plot_error:
            print(f">>> [Warning] Failed to auto-update charts: {plot_error}")    

    print("\n========================================================")
    print(" Evolution Complete!")
    print("========================================================")
