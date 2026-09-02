import sys
import os
import numpy as np
import math
import time

# Ensure cross-folder module loading works correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GAME.OthelloGame import OthelloGame
from AGENTS.OthelloPlayers import RandomPlayer, GreedyOthelloPlayer, HumanPlayer, MinimaxPlayer


class Arena:
    """
    Bulletproof Arena class for rigorous large-scale Othello Tournament evaluation.
    Features: 
    - Opening randomization to avoid standard-game traps.
    - Real-time logging to prevent data loss on crashes.
    - Automatic MCTS Tree clearing (reset) between games to strictly prevent OOM.
    """
    def __init__(self, player1, player2, game, p1_name="Player_1", p2_name="Player_2", log_file="arena_results_log.txt"):
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.p1_name = p1_name
        self.p2_name = p2_name
        self.log_file = log_file

    def _log_to_file(self, message):
        """Safely writes intermediate results to disk immediately to prevent data loss."""
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")
        print(message)

    def play_match(self, p1, p2, display=False, num_opening_random_moves=4):
        """Executes a single match with randomized openings."""
        board = self.game.getInitBoard()
        cur_player = 1
        step = 0
        
        while self.game.getGameEnded(board, cur_player) == 0:
            step += 1
            if display:
                print(f"Step {step} | Current turn: {'Black(1)' if cur_player==1 else 'White(-1)'}")
            
            # --- Phase 1: Opening Randomization (Soft Plies) ---
            if step <= num_opening_random_moves:
                valid_moves = self.game.getValidMoves(board, cur_player)
                valid_indices = np.where(valid_moves == 1)[0]
                if len(valid_indices) > 0:
                    action = np.random.choice(valid_indices)
                else:
                    action = self.game.n * self.game.n  # Pass move
            else:
                # --- Phase 2: True AI Battle ---
                player = p1 if cur_player == 1 else p2
                
                # Compatible with AlphaZero __call__ and Legacy functions
                if cur_player == 1:
                    action = player(board)
                else:
                    # Player 2 always needs canonical perspective
                    canonical_board = self.game.getCanonicalForm(board, cur_player)
                    action = player(canonical_board)
                        
            board, cur_player = self.game.getNextState(board, cur_player, action)
            
        return cur_player * self.game.getGameEnded(board, cur_player)

    def play_games(self, num_games=1000, num_opening_random_moves=4, save_freq=50):
        """
        Plays multiple games synchronously and writes checkpoints to disk.
        """
        if num_games % 2 != 0:
            num_games += 1
            
        p1_wins, p2_wins, draws = 0, 0, 0
        half_games = num_games // 2
        
        self._log_to_file(f"========== TOURNAMENT START: {self.p1_name} vs {self.p2_name} ==========")
        self._log_to_file(f"Total Games: {num_games} | Random Opening Plies: {num_opening_random_moves}")
        self._log_to_file(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        start_time = time.time()

        # --- First Half: P1 is Black (1) ---
        self._log_to_file(f"[Phase 1] {self.p1_name} (Black) vs {self.p2_name} (White)")
        for i in range(half_games):
            
            if hasattr(self.player1, 'reset_mcts'): self.player1.reset_mcts()
            if hasattr(self.player2, 'reset_mcts'): self.player2.reset_mcts()
                
            res = self.play_match(self.player1, self.player2, display=False, num_opening_random_moves=num_opening_random_moves)
            if res == 1: p1_wins += 1
            elif res == -1: p2_wins += 1
            else: draws += 1
            
            if (i + 1) % save_freq == 0:
                self._log_to_file(f"  [Progress] Phase 1: {i+1}/{half_games} games done. Current P1 Wins: {p1_wins}, P2 Wins: {p2_wins}, Draws: {draws}")

        # --- Second Half: P2 is Black (1) ---
        self._log_to_file(f"\n[Phase 2] {self.p2_name} (Black) vs {self.p1_name} (White)")
        for i in range(half_games):
            
            if hasattr(self.player1, 'reset_mcts'): self.player1.reset_mcts()
            if hasattr(self.player2, 'reset_mcts'): self.player2.reset_mcts()
                
            res = self.play_match(self.player2, self.player1, display=False, num_opening_random_moves=num_opening_random_moves)
            if res == 1: p2_wins += 1    
            elif res == -1: p1_wins += 1 
            else: draws += 1
            
            if (i + 1) % save_freq == 0:
                self._log_to_file(f"  [Progress] Phase 2: {i+1}/{half_games} games done. Current P1 Wins: {p1_wins}, P2 Wins: {p2_wins}, Draws: {draws}")

        # --- Statistical Analysis ---
        total_valid_games = p1_wins + p2_wins + draws
        win_rate_p1 = (p1_wins + 0.5 * draws) / total_valid_games
        win_rate_p2 = (p2_wins + 0.5 * draws) / total_valid_games
        moe = 1.96 * math.sqrt((win_rate_p1 * (1 - win_rate_p1)) / total_valid_games) * 100
        
        elapsed = time.time() - start_time

        self._log_to_file("\n=================== FINAL STATISTICAL RESULTS ===================")
        self._log_to_file(f" Matches Evaluated : {total_valid_games}")
        self._log_to_file(f" Total Time Taken  : {elapsed/3600:.2f} Hours")
        self._log_to_file(f" {self.p1_name} Wins : {p1_wins} ({p1_wins/total_valid_games*100:.1f}%)")
        self._log_to_file(f" {self.p2_name} Wins : {p2_wins} ({p2_wins/total_valid_games*100:.1f}%)")
        self._log_to_file(f" Draws             : {draws} ({draws/total_valid_games*100:.1f}%)")
        self._log_to_file(f"-----------------------------------------------------------------")
        self._log_to_file(f" Effective Win Rate ({self.p1_name}): {win_rate_p1*100:.2f}% ± {moe:.2f}% (95% CI)")
        self._log_to_file(f" Effective Win Rate ({self.p2_name}): {win_rate_p2*100:.2f}% ± {moe:.2f}% (95% CI)")
        self._log_to_file("=================================================================\n")
        
        return win_rate_p1, win_rate_p2, moe