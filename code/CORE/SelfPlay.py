import sys
import os
import time
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GAME.OthelloGame import OthelloGame
from AGENTS.OthelloPlayers import PureMCTSPlayer

def execute_episode(game, mcts_player):
    """
    Execute a complete self-play episode, collecting (s, pi, z) training data.
    """
    train_examples = []
    board = game.getInitBoard()
    cur_player = 1
    episode_step = 0
    
    print("thinking ...")
    
    while True:
        episode_step += 1
        # 1. Always use the canonical form of the board from the current player's perspective
        canonical_board = game.getCanonicalForm(board, cur_player)
        
        # 2. Let MCTS think and get the action probability distribution pi
        pi = mcts_player.getAction(canonical_board)
        
        # Record data: store the current board state, probability, and the player making the move
        train_examples.append([canonical_board, cur_player, pi, None])
        
        # 3. Sample an action based on the probability distribution pi
        # This ensures that the AI doesn't play the exact same moves in every game
        action = np.random.choice(len(pi), p=pi)
        
        # 4. Execute the action and move to the next state
        board, cur_player = game.getNextState(board, cur_player, action)
        
        # 5. Check if the game has ended
        r = game.getGameEnded(board, cur_player)
        if r != 0:
            # The game is finished! Now label the previously recorded data with the final outcome z
            print(f"Episode finished. Total steps: {episode_step}")
            
            # Backtrack and label: if the final winner is the same player who made this move, z is win (+1/outcome value), otherwise loss (-1/outcome value)
            # Since r is relative to the cur_player at the end of the game, we need to check if they are the same player
            for step_idx in range(len(train_examples)):
                is_same_player = (train_examples[step_idx][1] == cur_player)
                # Store the final (s, pi, z)
                train_examples[step_idx][3] = r if is_same_player else -r
                
            # Return the clean (s, pi, z) dataset
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
