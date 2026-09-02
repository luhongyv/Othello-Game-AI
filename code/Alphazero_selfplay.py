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

# 借用你之前写好的训练器代码
from train_alphazero_Gready import AlphaZeroTrainer, OthelloDataset

class SelfPlayPipeline:
    def __init__(self, game):
        self.game = game
        
        # 1. 初始化当前最强模型 (老大脑)
        self.nnet = NNetWrapper(self.game)
        checkpoint_path = os.path.join("checkpoint", "resnet_expert.pth")
        if not os.path.exists(checkpoint_path):
            print("Error: resnet_expert.pth not found! Please check the path.") # 找不到专家模型权重，强制退出
            sys.exit()
        self.nnet.load_checkpoint(folder="checkpoint", filename="resnet_expert.pth")
        
        # 2. 初始化挑战者模型 (新大脑)
        self.pnet = NNetWrapper(self.game)
        
        # --- 核心训练参数 (MVP 极简版) ---
        self.num_iters = 5        # 总共进行几次进化大循环
        self.num_eps = 100         # 自我对弈局数
        self.tempThreshold = 15   # 前 15 步带有极高探索性 (Temp=1)
        self.update_threshold = 0.55 # 新模型胜率 >= 55% 才能替代老模型
        self.arena_games = 40     # 新老大脑打擂台的局数
        self.epochs = 2           # 每次拿到新数据，只微调 2 个 Epoch
        self.batch_size = 64      # 训练批次大小
        
        self.history = []
        self.start_iter = 1
        self.state_file = os.path.join("checkpoint", "pipeline_state.json")
        
        # 检查是否存在中断的历史记录
        if os.path.exists(self.state_file):
            print(f"Found existing pipeline state at {self.state_file}!")
            with open(self.state_file, "r", encoding="utf-8") as f:
                self.history = json.load(f)
            
            if self.history:
                # 找到上一次完成的最高轮次，加 1 就是本次要启动的轮次
                self.start_iter = self.history[-1]['iteration'] + 1
                if self.start_iter <= self.num_iters:
                    print(f"Fast-forwarding: Resuming directly from Iteration {self.start_iter}/{self.num_iters}")
                else:
                    print(f"All {self.num_iters} iterations were already completed previously.")

    def execute_episode(self):
        """核心动作：打自己一局，返回训练数据"""
        train_examples = []
        board = self.game.getInitBoard()
        cur_player = 1
        episode_step = 0
        
        # 实例化带有噪声的 MCTS (探索度拉满)
        mcts = AlphaZeroMCTS(self.game, self.nnet, num_sims=100, c_puct=1.0)
        
        while True:
            episode_step += 1
            canonical_board = self.game.getCanonicalForm(board, cur_player)
            
            # 前 15 步保持探索 (temp=1)，之后变成绝对贪心 (temp=0)
            temp = int(episode_step < self.tempThreshold)
            
            # 获取动作概率！(开启 add_noise)
            pi = mcts.getAction(canonical_board, temp=temp, add_noise=True)
            
            # 记录数据 (当前 player 的视角，此时胜负未知先填 None)
            sym = self.game.getSymmetries(canonical_board, pi)
            for b, p in sym:
                train_examples.append([b, cur_player, p, None])
                
            # 落子
            action = np.random.choice(len(pi), p=pi)
            board, cur_player = self.game.getNextState(board, cur_player, action)
            
            # 检查游戏是否结束
            r = self.game.getGameEnded(board, 1)
            if r != 0:
                # 游戏结束，回溯赋予真实的 Value 标签
                res = [(x[0], x[2], r * (1 if x[1] == cur_player else -1)) for x in train_examples]
                del mcts
                del train_examples
                del sym
                gc.collect()
                
                return res
    def play_arena(self):
        print(f"Starting Arena: Challenger (New) vs Champion (Old) - {self.arena_games} games") # 开始打擂台
        pwins, nwins, draws = 0, 0, 0
        
        # 擂台赛中双方算力公平，都为 200 次模拟
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
        # 打完擂台同样要手动粉碎 MCTS 树
        del pmcts, nmcts
        import gc
        gc.collect()
        
        return pwins, nwins, draws

    def learn(self):
        """强化学习主循环"""
        for i in range(self.start_iter, self.num_iters + 1):
            print(f"\n==========================================")
            print(f"Starting Reinforcement Learning Iteration: {i}/{self.num_iters}") # 开始新的 RL 迭代
            print(f"==========================================")
            
            # ==========================================
            # 1. 自我对弈收集数据 (带局内断点重续机制)
            # ==========================================
            print(f"Starting self-play (Target: {self.num_eps} episodes)...")
            start_time = time.time()
            
            import pickle
            import os
            iteration_train_examples = []
            start_eps = 0
            
            # 定义局内临时断点文件的名字 (例如: iter_1_selfplay_ckpt.pkl)
            temp_ckpt_file = os.path.join("checkpoint", f"iter_{i}_selfplay_ckpt.pkl")
            
            # 👉 检查：如果中途闪退过，直接读取硬盘里残存的数据和局数
            if os.path.exists(temp_ckpt_file):
                with open(temp_ckpt_file, "rb") as f:
                    ckpt_data = pickle.load(f)
                iteration_train_examples = ckpt_data['examples']
                start_eps = ckpt_data['eps_completed']
                print(f"  -> 🔍 Recovered from crash! Resuming self-play from game {start_eps+1}/{self.num_eps}...")

            with torch.no_grad(): # 保持关闭梯度，极大节省内存
                for eps in range(start_eps, self.num_eps):
                    print(f"  -> Playing Game {eps+1}/{self.num_eps} ...", end=" ", flush=True)
                    
                    # 下完完整的一局
                    new_examples = self.execute_episode()
                    iteration_train_examples += new_examples
                    
                    print(f"Done! (Collected {len(new_examples)} steps)")
                    
                    safe_tmp_file = temp_ckpt_file + ".tmp"
                    with open(safe_tmp_file, "wb") as f:
                        pickle.dump({'eps_completed': eps + 1, 'examples': iteration_train_examples}, f)
                    os.replace(safe_tmp_file, temp_ckpt_file) # 瞬间替换，防闪退坏档
                        
            print(f"\nSelf-play completed in {time.time()-start_time:.1f}s. Total {len(iteration_train_examples)} new samples.")
            
            temp_data_file = "temp_selfplay_data.pkl"
            with open(temp_data_file, "wb") as f:
                pickle.dump(iteration_train_examples, f)
            
            del iteration_train_examples
            gc.collect()   
            dataset = OthelloDataset(temp_data_file)
            
            # ==========================================
            # 后续的微调与打擂台流程 (保持原样)
            # ==========================================
            
            # 2. 将当前老大脑参数复制给新大脑，准备微调
            self.pnet.nnet.load_state_dict(self.nnet.nnet.state_dict())
            
            # 3. 训练新大脑 (微调)
            print("Starting to fine-tune the new model...")
            trainer = AlphaZeroTrainer(self.pnet.nnet, lr=0.0005, batch_size=self.batch_size, epochs=self.epochs)
            trainer.train(dataset)
            
            # 4. 打擂台决出胜负
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
                # 胜率超过阈值，接受新模型
                print(f"Evolution successful! New model win rate: {win_rate*100:.1f}%. Replacing old model.")
                self.nnet.nnet.load_state_dict(self.pnet.nnet.state_dict())
                # 保存本次迭代的历史模型
                self.nnet.save_checkpoint(folder="checkpoint", filename=f"resnet_rl_iter_{i}.pth")
                # 覆盖当前最强模型，作为下一轮的基准
                self.nnet.save_checkpoint(folder="checkpoint", filename="resnet_expert.pth")
            else:
                # 胜率不足，丢弃新模型
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
        
        # 1. 打印到终端
        print("\n" + "="*80)
        print("AlphaZero Self-Play RL Proof of Concept (PoC) Test Results")
        print("="*80)
        print(final_table)
        print("="*80 + "\n")
        
        save_dir = "TRAINNING_CHARTS"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # 2. 保存为 Markdown
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
            
        # 动态计算表格高度
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
                # Status 状态高亮
                if j == 5: 
                    if "Successful" in cell_text[idx-1][5]:
                        cell.set_text_props(color='#27ae60') # 绿色成功
                    else:
                        cell.set_text_props(color='#c0392b') # 红色失败
                # 斑马纹交替行颜色
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