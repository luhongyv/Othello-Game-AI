import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from GAME.OthelloGame import OthelloGame
from AGENTS.OthelloPlayers import GreedyOthelloPlayer, MinimaxPlayer, AlphaZeroMCTS
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
        self.mcts = AlphaZeroMCTS(
            self.game, 
            self.nnet, 
            num_sims=self.mcts_args['numMCTSSims'], 
            c_puct=self.mcts_args['cpuct']
        )
        
    def __call__(self, board):
        pi = self.mcts.getAction(board, temp=0)
        return np.argmax(pi)


def is_matchup_done(log_file, name1, name2):
    """Most secure macro check: as long as the log has the final settlement result for this match, skip it directly"""
    if not os.path.exists(log_file):
        return False
    with open(log_file, "r", encoding="utf-8") as f:
        content = f.read()
    start_pattern = f"TOURNAMENT START: {name1} vs {name2}"
    if start_pattern in content:
        pos = content.find(start_pattern)
        if "FINAL STATISTICAL RESULTS" in content[pos:]:
            return True
    return False


if __name__ == "__main__":
    game = OthelloGame(8)
    MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MODELS')
    
    pure_rl_agent = AlphaZeroPlayer(game, model_folder=MODELS_DIR, model_file='model_Pure_RL.pth')
    sl_rl_agent = AlphaZeroPlayer(game, model_folder=MODELS_DIR, model_file='model_Greedy_SL_RL.pth')
    
    competitors = [
        ("Pure_RL_Agent", pure_rl_agent),
        ("SL_RL_Hybrid_Agent", sl_rl_agent),
        ("Greedy_MCTS_Baseline", GreedyOthelloPlayer(game).play),
        ("Minimax_Dynamic", MinimaxPlayer(game, time_limit=2.0).play)
    ]
    
    TOTAL_GAMES_PER_MATCHUP = 1000
    RANDOM_OPENING_PLIES = 4
    LOG_FILE = "arena_full_tournament_log.txt"
    
    for i in range(len(competitors)):
        for j in range(i + 1, len(competitors)):
            name1, p1 = competitors[i]
            name2, p2 = competitors[j]
            
            # Macro defense: if this set of 1000 games has been completely finished in the log, never repeat it on restart
            if is_matchup_done(LOG_FILE, name1, name2):
                print(f"\n[Safe Skip] {name1} vs {name2} fully completed in log, skipping!")
                continue
            
            print(f"\n>>>>>>>> Preparing Matchup: {name1} vs {name2} <<<<<<<<")
            
            arena = Arena(
                player1=p1, 
                player2=p2, 
                game=game, 
                p1_name=name1, 
                p2_name=name2,
                log_file=LOG_FILE
            )
            
            arena.play_games(
                num_games=TOTAL_GAMES_PER_MATCHUP, 
                num_opening_random_moves=RANDOM_OPENING_PLIES, 
                save_freq=50
            )
            
            print(f">>>>>>>> Matchup {name1} vs {name2} Finished! Check log file. <<<<<<<<\n")
