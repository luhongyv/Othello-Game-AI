import sys
import os
import time
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GAME.OthelloGame import OthelloGame
from AGENTS.OthelloPlayers import PureMCTSPlayer

def execute_episode(game, mcts_player):
    """
    执行一局完整的左右互搏，收集 (s, pi, z) 训练数据。
    """
    train_examples = []
    board = game.getInitBoard()
    cur_player = 1
    episode_step = 0
    
    print("thinking ...")
    
    while True:
        episode_step += 1
        # 1. 始终使用“当前玩家视角”的棋盘
        canonical_board = game.getCanonicalForm(board, cur_player)
        
        # 2. 让 MCTS 思考，获得动作概率分布 pi
        pi = mcts_player.getAction(canonical_board)
        
        # 【记录数据】：存下当前的 盘面、概率、以及是谁在下棋
        train_examples.append([canonical_board, cur_player, pi, None])
        
        # 3. 根据概率 pi 随机抽样选择一个动作去下
        # 这保证了 AI 不会每局都下出完全一模一样的棋
        action = np.random.choice(len(pi), p=pi)
        
        # 4. 执行动作，进入下一个状态
        board, cur_player = game.getNextState(board, cur_player, action)
        
        # 5. 判断游戏是否结束
        r = game.getGameEnded(board, cur_player)
        if r != 0:
            # 游戏结束了！开始给之前记录的数据打上最终胜负标签 z
            print(f"Episode finished. Total steps: {episode_step}")
            
            # 回溯打标签：如果最后的赢家和下这步棋的人是同一个，z就是赢(+1/胜负值)，否则就是输(-1/胜负值)
            # 因为 r 是相对于游戏结束时的 cur_player 而言的，所以要做一次判断
            for step_idx in range(len(train_examples)):
                is_same_player = (train_examples[step_idx][1] == cur_player)
                # 存入最终的 (s, pi, z)
                train_examples[step_idx][3] = r if is_same_player else -r
                
            # 返回干净的 (s, pi, z) 数据集
            return [(x[0], x[2], x[3]) for x in train_examples]

if __name__ == "__main__":
    game = OthelloGame(8)
    # Quick test with 0.5s limit
    mcts_generator = PureMCTSPlayer(game, time_limit=0.5)
    
    # Run one episode
    dataset = execute_episode(game, mcts_generator)
    
    print("\n" + "="*50)
    print("Data generation successful!")
    print(f"Generated {len(dataset)} training samples from this episode.")
    print("="*50)
    
    # Sample the very first move's data
    s, pi, z = dataset[0]
    
    print("\n[Sample Data Check: First Move]")
    print("1. Board State 's' (CNN Input Feature):")
    print(s)
    
    print(f"\n2. Policy Target 'pi' (Action Probabilities, Length: {len(pi)}):")
    for act, prob in enumerate(pi):
        if prob > 0:
            row, col = int(act / 8), act % 8
            print(f"  - Action [{row},{col}] (Index {act}): Probability {prob*100:.1f}%")
            
    print(f"\n3. Value Target 'z' (Game Outcome Prediction):")
    print(f"  - Final Outcome: {z} (1=Win, -1=Loss, 0=Draw)")
    print("="*50 + "\n")