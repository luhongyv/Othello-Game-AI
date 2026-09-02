import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import numpy as np
import torch
import time
from collections import deque
import gc
import json  
sys.path.append(os.getcwd())
from GAME.OthelloGame import OthelloGame
from CNN.NNetTrainer import NNetWrapper
from AGENTS.OthelloPlayers import AlphaZeroMCTS
# Reuse the trainer code you wrote earlier
from train_alphazero_Gready import AlphaZeroTrainer, OthelloDataset

class SelfPlayPipeline:
    def __init__(self, game):
        self.game = game
        
        # 1. Initialize the current strongest model (old brain)
        self.nnet = NNetWrapper(self.game)
        checkpoint_path = os.path.join("checkpoint", "resnet_expert.pth")
        if not os.path.exists(checkpoint_path):
            print("Error: resnet_expert.pth not found! Please check the path.") # Expert model weights not found, force exit
            sys.exit()
        self.nnet.load_checkpoint(folder="checkpoint", filename="resnet_expert.pth")
        
        # 2. Initialize the challenger model (new brain)
        self.pnet = NNetWrapper(self.game)
        
        # --- Core training parameters (MVP minimal version) ---
        self.num_iters = 5        # Total number of evolution cycles
        self.num_eps = 100         # Number of self-play episodes
        self.tempThreshold = 15   # First 15 steps with high exploration (Temp=1)
        self.update_threshold = 0.55 # New model win rate >= 55% to replace old model
        self.arena_games = 40     # Number of arena games between new and old brain
        self.epochs = 2           # Only fine-tune for 2 epochs each time new data is obtained
        self.batch_size = 64      # Training batch size
        
        self.history = []
        self.start_iter = 1
        self.state_file = os.path.join("checkpoint", "pipeline_state.json")
        
        # Check if there is an interrupted history record
        if os.path.exists(self.state_file):
            print(f"Found existing pipeline state at {self.state_file}!")
            with open(self.state_file, "r", encoding="utf-8") as f:
                self.history = json.load(f)
            
            if self.history:
                # Find the highest completed iteration from last run, add 1 to get the iteration to start this time
                self.start_iter = self.history[-1]['iteration'] + 1
                if self.start_iter <= self.num_iters:
                    print(f"Fast-forwarding: Resuming directly from Iteration {self.start_iter}/{self.num_iters}")
                else:
                    print(f"All {self.num_iters} iterations were already completed previously.")

    def execute_episode(self):
        """Core action: play one game against itself, return training data"""
        train_examples = []
        board = self.game.getInitBoard()
        cur_player = 1
        episode_step = 0
        
        # Instantiate MCTS with noise (exploration maxed out)
        mcts = AlphaZeroMCTS(self.game, self.nnet, num_sims=100, c_puct=1.0)
        
        while True:
            episode_step += 1
            canonical_board = self.game.getCanonicalForm(board, cur_player)
            
            # Keep exploration for first 15 steps (temp=1), then become absolutely greedy (temp=0)
            temp = int(episode_step < self.tempThreshold)
            
            # Get action probabilities! (enable add_noise)
            pi = mcts.getAction(canonical_board, temp=temp, add_noise=True)
            
            # Record data (from current player's perspective, outcome unknown so fill None for now)
            sym = self.game.getSymmetries(canonical_board, pi)
            for b, p in sym:
                train_examples.append([b, cur_player, p, None])
                
            # Make a move
            action = np.random.choice(len(pi), p=pi)
            board, cur_player = self.game.getNextState(board, cur_player, action)
            
            # Check if the game is over
            r = self.game.getGameEnded(board, 1)
            if r != 0:
                # Game over, backtrack to assign real Value labels
                res = [(x[0], x[2], r * (1 if x[1] == cur_player else -1)) for x in train_examples]
                del mcts
                del train_examples
                del sym
                gc.collect()
                
                return res

    def play_arena(self):
        print(f"Starting Arena: Challenger (New) vs Champion (Old) - {self.arena_games} games") # Start the arena
        pwins, nwins, draws = 0, 0, 0
        
        # Both sides have equal computing power in arena, both 200 simulations
        pmcts = AlphaZeroMCTS(self.game, self.pnet, num_sims=200, c_puct=1.0)
        nmcts = AlphaZeroMCTS(self.game, self.nnet, num_sims=200, c_puct=1.0)
        
        with torch.no_grad():
            for g in range(self.arena_games):
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
        
        print()
        # Manually destroy MCTS trees after arena as well
        del pmcts, nmcts
        import gc
        gc.collect()
        
        return pwins, nwins, draws

    def learn(self):
        """Reinforcement learning main loop"""
        for i in range(self.start_iter, self.num_iters + 1):
            print(f"\n==========================================")
            print(f"Starting Reinforcement Learning Iteration: {i}/{self.num_iters}") # Start a new RL iteration
            print(f"==========================================")
            
            # ==========================================
            # 1. Self-play to collect data (with in-episode checkpoint resume mechanism)
            # ==========================================
            print(f"Starting self-play (Target: {self.num_eps} episodes)...")
            start_time = time.time()
            
            import pickle
            import os
            iteration_train_examples = []
            start_eps = 0
            
            # Define the name of the in-episode temporary checkpoint file (e.g. iter_1_selfplay_ckpt.pkl)
            temp_ckpt_file = os.path.join("checkpoint", f"iter_{i}_selfplay_ckpt.pkl")
            
            # Check: if it crashed midway, directly read the remaining data and episode count from disk
            if os.path.exists(temp_ckpt_file):
                with open(temp_ckpt_file, "rb") as f:
                    ckpt_data = pickle.load(f)
                iteration_train_examples = ckpt_data['examples']
                start_eps = ckpt_data['eps_completed']
                print(f"  -> Recovered from crash! Resuming self-play from game {start_eps+1}/{self.num_eps}...")

            with torch.no_grad(): # Keep gradients disabled, greatly saves memory
                for eps in range(start_eps, self.num_eps):
                    print(f"  -> Playing Game {eps+1}/{self.num_eps} ...", end=" ", flush=True)
                    
                    # Play a complete game
                    new_examples = self.execute_episode()
                    iteration_train_examples += new_examples
                    
                    print(f"Done! (Collected {len(new_examples)} steps)")
                    
                    safe_tmp_file = temp_ckpt_file + ".tmp"
                    with open(safe_tmp_file, "wb") as f:
                        pickle.dump({'eps_completed': eps + 1, 'examples': iteration_train_examples}, f)
                    os.replace(safe_tmp_file, temp_ckpt_file) # Instant replacement, prevents corrupted files from crashes
                        
            print(f"\nSelf-play completed in {time.time()-start_time:.1f}s. Total {len(iteration_train_examples)} new samples.")
            
            temp_data_file = "temp_selfplay_data.pkl"
            with open(temp_data_file, "wb") as f:
                pickle.dump(iteration_train_examples, f)
            
            del iteration_train_examples
            gc.collect()   
            dataset = OthelloDataset(temp_data_file)
            
            # ==========================================
            # Subsequent fine-tuning and arena process (keep as is)
            # ==========================================
            
            # 2. Copy current old brain parameters to new brain, prepare for fine-tuning
            self.pnet.nnet.load_state_dict(self.nnet.nnet.state_dict())
            
            # 3. Train the new brain (fine-tune)
            print("Starting to fine-tune the new model...")
            trainer = AlphaZeroTrainer(self.pnet.nnet, lr=0.0005, batch_size=self.batch_size, epochs=self.epochs)
            trainer.train(dataset)
            
            # 4. Play arena to determine the winner
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
                # Win rate exceeds threshold, accept the new model
                print(f"Evolution successful! New model win rate: {win_rate*100:.1f}%. Replacing old model.")
                self.nnet.nnet.load_state_dict(self.pnet.nnet.state_dict())
                # Save the historical model for this iteration
                self.nnet.save_checkpoint(folder="checkpoint", filename=f"resnet_rl_iter_{i}.pth")
                # Overwrite the current strongest model, as the baseline for the next round
                self.nnet.save_checkpoint(folder="checkpoint", filename="resnet_expert.pth")
            else:
                # Win rate insufficient, discard the new model
                print(f"Evolution failed. New model win rate: {win_rate*100:.1f}%. Discarding new data.")

            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=4)
            print(f"Iteration {i} state saved to checkpoint! It is now safe to interrupt the script if needed.")
            if os.path.exists(temp_ckpt_file):
                os.remove(temp_ckpt_file)

        print("\nAll reinforcement learning iterations completed.")
        self.generate_result_table()

    def generate_result_table(self):
        """
        Prints an academic table in the terminal, saves it as a Markdown file, 
        and renders a high-quality PNG table for publication.
        """
        if not self.history:
            return
            
        header = f"| {'Iteration':^11} | {'New Wins':^12} | {'Old Wins':^12} | {'Draws':^9} | {'Win Rate':^12} | {'Status':^55} |"
        separator = f"|{'-'*13}|{'-'*14}|{'-'*14}|{'-'*11}|{'-'*14}|{'-'*57}|"
        
        table_str = [header, separator]
        
        for data in self.history:
            i = data['iteration']
            pwins = data['pwins']
            nwins = data['nwins']
            draws = data.get('draws', 0)  
            win_rate = data['win_rate'] * 100
            
            if win_rate >= self.update_threshold * 100:
                status = "Evolution Successful"
            else:
                status = "Evolution Failed"
                
            row = f"| {i:^11} | {pwins:^12} | {nwins:^12} | {draws:^9} | {f'{win_rate:.1f}%':^12} | {status:<55} |"
            table_str.append(row)
            
        final_table = "\n".join(table_str)
        
        # 1. Print to terminal
        print("\n" + "="*80)
        print("AlphaZero Self-Play RL Proof of Concept (PoC) Test Results")
        print("="*80)
        print(final_table)
        print("="*80 + "\n")
        
        save_dir = "TRAINNING_CHARTS"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # 2. Save as Markdown
        save_path_md = os.path.join(save_dir, "SelfPlay_Evolution_Table.md")
        with open(save_path_md, "w", encoding="utf-8") as f:
            f.write("### Table X: AlphaZero Self-Play RL Proof of Concept Test Results\n")
            f.write("*(Test Conditions: 400 MCTS simulations per move, 40 total games, evolution threshold 55%)*\n\n")
            f.write(final_table + "\n")
            
        print(f" Markdown table successfully saved to: {save_path_md}")

        import matplotlib.pyplot as plt  
        columns = ["Iteration", "New Wins", "Old Wins", "Draws", "Win Rate", "Status"]
        cell_text = []
        for data in self.history:
            win_rate_val = data['win_rate'] * 100
            status_text = "Evolution Successful" if win_rate_val >= self.update_threshold * 100 else "Failed"
            cell_text.append([
                str(data['iteration']), 
                str(data['pwins']), 
                str(data['nwins']), 
                str(data.get('draws', 0)), 
                f"{win_rate_val:.1f}%", 
                status_text
            ])
            
        # Dynamically calculate table height
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
                # Status highlight
                if j == 5: 
                    if "Successful" in cell_text[idx-1][5]:
                        cell.set_text_props(color='#27ae60') # Green for success
                    else:
                        cell.set_text_props(color='#c0392b') # Red for failure
                # Zebra stripe alternating row colors
                if idx % 2 == 0:
                    cell.set_facecolor('#f4f6f7')
                    
        plt.title("AlphaZero Self-Play RL Proof of Concept Test Results", fontweight="bold", fontsize=14, pad=15)
        plt.tight_layout()
        
        save_path_png = os.path.join(save_dir, "SelfPlay_Evolution_Table.png")
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"PNG image successfully saved to: {save_path_png}")

if __name__ == "__main__":
    game = OthelloGame(8)
    pipeline = SelfPlayPipeline(game)
    pipeline.learn()
