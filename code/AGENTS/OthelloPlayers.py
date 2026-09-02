import gc
import numpy as np
import random
import math
import time
#import torch
from GAME.OthelloLogic import Board 

class TimeoutException(Exception):
    pass

class RandomPlayer():
    
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1) 
        valid_actions = np.where(valids == 1)[0]
        if len(valid_actions) == 0:
            return self.game.getActionSize() - 1 
        a = np.random.choice(valid_actions)
        return a


class HumanPlayer():
        
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valid_moves = self.game.getValidMoves(board, 1)
        
        print("Legal move points:", end="")
        for i in range(len(valid_moves)):
            if valid_moves[i]:
                if i == self.game.getActionSize() - 1:
                    print("[Pass]", end=" ")
                else:
                    print(f"[{int(i / self.game.n)} {int(i % self.game.n)}]", end=" ")
        print()

        while True:
            input_move = input("Please enter the coordinates of the move you want to make (format: row, column, e.g., '3 4'). Enter 'pass' to skip this move.: ")
            
            if input_move.lower() == 'pass':
                a = self.game.getActionSize() - 1
            else:
                try:
                    x, y = [int(i) for i in input_move.split()]
                    a = self.game.n * x + y
                except ValueError:
                    print("Format error. Please enter the number again.")
                    continue
            
            if a < len(valid_moves) and valid_moves[a]:
                break
            else:
                print("Please choose a new location for the illegal move.")
                
        return a


class GreedyOthelloPlayer():

    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)
        best_score = -float('inf')
        best_actions = []
        
        for a in range(self.game.getActionSize()):
            if valids[a] == 0:
                continue
            
            nextBoard, _ = self.game.getNextState(board, 1, a)
            score = self.game.getScore(nextBoard, 1)
            
            if score > best_score:
                best_score = score
                best_actions = [a] 
            elif score == best_score:
                best_actions.append(a)
            
        return random.choice(best_actions)# return best score(random)
    
    
class MinimaxPlayer():
    def __init__(self, game, time_limit=1.0):
        self.game = game
        self.time_limit = time_limit 
        self.start_time = 0
        self.n = game.n
        self.node_count = 0
        self.board_pool = [[[0]*self.n for _ in range(self.n)] for _ in range(65)]

    def play(self, board):
        self.start_time = time.time()
        self.node_count = 0
        
        py_board = np.asarray(board, dtype=int).tolist()
        b = Board(self.n)
        b.pieces = py_board
        
        empty_spots = sum(row.count(0) for row in py_board)
        
        def calculate_dynamic_depth(empty, t_limit):
            n_safe = 100000 * t_limit 
            
            if empty == 60: b_eff = 5.79
            elif empty >= 50: b_eff = 12.02
            elif empty >= 40: b_eff = 13.27
            elif empty >= 30: b_eff = 13.85  
            elif empty >= 20: b_eff = 8.80
            elif empty >= 10: b_eff = 1.98
            else: 
                return 15 
            
            # d = log(N_safe) / log(b_eff)
            d_safe = math.floor(math.log(n_safe, b_eff))
            return d_safe

        max_depth = calculate_dynamic_depth(empty_spots, self.time_limit)
            
        valid_moves = b.get_legal_moves(1)
        
        if not valid_moves:
            return self.n * self.n
            
        best_safe_move = valid_moves[0]
        depth = 1
        
        try:
            while True:
                current_move = self._search_at_depth(b, depth)
                if current_move is not None:
                    best_safe_move = current_move
                depth += 1
                
                if depth > max_depth:
                    break
                    
        except TimeoutException:
            pass
       
        return best_safe_move[0] * self.n + best_safe_move[1]

    def _search_at_depth(self, b, target_depth):
        best_actions = []
        best_value = -float('inf')
        
        valid_moves = b.get_legal_moves(1)
        
        for move in valid_moves:
            self.node_count += 1
            
            if self.node_count & 1023 == 0:
                if time.time() - self.start_time > self.time_limit:
                    raise TimeoutException()

            for r in range(self.n):
                self.board_pool[target_depth][r][:] = b.pieces[r]
            
            b.execute_move(move, 1)
            
            value = -self._negamax(b, -1, target_depth - 1, -float('inf'), float('inf'))
            
            for r in range(self.n):
                b.pieces[r][:] = self.board_pool[target_depth][r]
            
            if value > best_value:
                best_value = value
                best_actions = [move]
            elif value == best_value:
                best_actions.append(move)
                
        return random.choice(best_actions) if best_actions else None

    def _negamax(self, b, player, depth, alpha, beta):
        self.node_count += 1
        if self.node_count & 1023 == 0:
            if time.time() - self.start_time > self.time_limit:
                raise TimeoutException()

        my_moves = b.get_legal_moves(player)
        opp_moves = b.get_legal_moves(-player)
        
        if not my_moves and not opp_moves:
            diff = b.countDiff(player)
            if diff > 0: return 10000
            elif diff < 0: return -10000
            else: return 0

        if depth == 0:
            return b.countDiff(player)

        if not my_moves:
            return -self._negamax(b, -player, depth - 1, -beta, -alpha)

        best_value = -float('inf')
        
        for move in my_moves:
            for r in range(self.n):
                self.board_pool[depth][r][:] = b.pieces[r]
            
            b.execute_move(move, player)
            
            value = -self._negamax(b, -player, depth - 1, -beta, -alpha)
            
            for r in range(self.n):
                b.pieces[r][:] = self.board_pool[depth][r]
            
            best_value = max(best_value, value)
            alpha = max(alpha, value)
            
            if alpha >= beta:
                break 
                
        return best_value

    def _evaluate(self, board, game_ended):
        if game_ended == 1:
            return 10000
        elif game_ended == -1:
            return -10000
            
        return self.game.getScore(board, 1)

class MCTSNode:
    def __init__(self, game, board, parent=None, action=None):
        self.game = game
        self.board = board      
        self.parent = parent     
        self.action = action     
        self.children = {}       
        self.visits = 0          
        self.value_sum = 0   
        self.value_sq_sum = 0    
        self.is_expanded = False 

    def expand(self):
        valids = self.game.getValidMoves(self.board, 1)
        valid_actions = np.where(valids == 1)[0]
        
        if len(valid_actions) == 0:
            valid_actions = [self.game.getActionSize() - 1]

        for a in valid_actions:
            if a not in self.children:
                next_board, next_player = self.game.getNextState(self.board, 1, a)
                canonical_board = self.game.getCanonicalForm(next_board, next_player)
                self.children[a] = MCTSNode(self.game, canonical_board, parent=self, action=a)
                
        self.is_expanded = True

    def select_child(self, c_puct=1.414):
        best_score = -float('inf')
        best_action = -1
        best_child = None

        corners = {0, 7, 56, 63}
        danger_zones = {1, 8, 9, 6, 14, 15, 48, 49, 57, 54, 55, 62}

        for action, child in self.children.items():
           
            h_bias = 0.0
            if action in corners:
                h_bias = 1.5   
            elif action in danger_zones:
                h_bias = -1.5  

            if child.visits == 0:
                ucb = float('inf') + h_bias
            else:
                mean = child.value_sum / child.visits
                
                
                variance = (child.value_sq_sum / child.visits) - (mean ** 2)
                variance = max(0.0, variance) 
                
                v_term = variance + math.sqrt(2 * math.log(self.visits) / child.visits)
                explore_term = c_puct * math.sqrt(math.log(self.visits) / child.visits) * min(0.25, v_term)
                
                
                bias_term = h_bias / (child.visits + 1)
                
                ucb = mean + explore_term + bias_term
            
            if ucb > best_score:
                best_score = ucb
                best_action = action
                best_child = child
        
        return best_action, best_child
       

class PureMCTSPlayer():
   
    def __init__(self, game, time_limit=4.5):
        self.game = game
        self.time_limit = time_limit 
        self.max_rollout_depth = 12 

    def play(self, board):
        root = MCTSNode(self.game, board)
        start_time = time.time()

        while time.time() - start_time < self.time_limit:
            node = root
            while node.is_expanded and len(node.children) > 0:
                _, node = node.select_child()

            game_ended = self.game.getGameEnded(node.board, 1)
            if game_ended == 0:
                node.expand()
                action = list(node.children.keys())[0]
                node = node.children[action]

            value = self._rollout(node.board)

            
            while node is not None:
                node.visits += 1
                node.value_sum += value
                node.value_sq_sum += value ** 2 
                value = -value 
                node = node.parent
                
        best_action = -1
        most_visits = -1
        for action, child in root.children.items():
            if child.visits > most_visits:
                most_visits = child.visits
                best_action = action
                
        return best_action
    def getAction(self, board):
        root = MCTSNode(self.game, board)
        start_time = time.time()

        while time.time() - start_time < self.time_limit:
            node = root
            while node.is_expanded and len(node.children) > 0:
                _, node = node.select_child()

            game_ended = self.game.getGameEnded(node.board, 1)
            if game_ended == 0:
                node.expand()
                action = list(node.children.keys())[0]
                node = node.children[action]

            value = self._rollout(node.board)

            while node is not None:
                node.visits += 1
                node.value_sum += value
                node.value_sq_sum += value ** 2 
                value = -value 
                node = node.parent

        pi = [0] * self.game.getActionSize()
        
        for action, child in root.children.items():
            pi[action] = child.visits
            
        sum_visits = sum(pi)
        if sum_visits > 0:
            pi = [x / sum_visits for x in pi]
        else:
            valids = self.game.getValidMoves(board, 1)
            pi = valids / np.sum(valids)
            
        return pi

    def _rollout(self, board):
        cur_board = board
        perspective = 1 
        step_count = 0
        
        corners = {0, 7, 56, 63}
        safe_edges = {2, 3, 4, 5, 16, 24, 32, 40, 23, 31, 39, 47, 58, 59, 60, 61}
        danger_zones = {1, 8, 9, 6, 14, 15, 48, 49, 57, 54, 55, 62}
        
        while self.game.getGameEnded(cur_board, 1) == 0:
            
            valids = self.game.getValidMoves(cur_board, 1)
            valid_actions = np.where(valids == 1)[0]
            
            if step_count >= self.max_rollout_depth:

                my_moves = len([x for x in valid_actions if x != self.game.getActionSize() - 1])
                
                opp_board = self.game.getCanonicalForm(cur_board, -1)
                opp_valids = np.where(self.game.getValidMoves(opp_board, 1) == 1)[0]
                opp_moves = len([x for x in opp_valids if x != self.game.getActionSize() - 1])
                
                if my_moves > opp_moves: return 0.6 * perspective
                elif my_moves < opp_moves: return -0.6 * perspective
                else: return 0
                
            if len(valid_actions) == 0:
                a = self.game.getActionSize() - 1
            else:
                
                if random.random() < 0.10: 
                    
                    safe_random = [act for act in valid_actions if act not in danger_zones]
                    if safe_random:
                        a = random.choice(safe_random)
                    else:
                        a = random.choice(valid_actions)
                else:
                    
                    tier1 = [act for act in valid_actions if act in corners]
                    if tier1:
                        a = random.choice(tier1)
                    else:
                        tier2 = [act for act in valid_actions if act in safe_edges]
                        if tier2:
                            a = random.choice(tier2)
                        else:
                            safe_actions = [act for act in valid_actions if act not in danger_zones]
                            if safe_actions:
                                a = random.choice(safe_actions)
                            else:
                                a = random.choice(valid_actions)
                
            next_board, next_player = self.game.getNextState(cur_board, 1, a)
            cur_board = self.game.getCanonicalForm(next_board, next_player)
            perspective *= -1 
            step_count += 1
            
        return self.game.getGameEnded(cur_board, 1) * perspective

class GreedyMCTSNode:
    def __init__(self, state, player, move=None, n=8):
        self.state = state  
        self.player = player  
        self.move = move  
        self.children = []
        self.n = n
        
        self.untried_moves = self._get_legal_moves()
        self.visits = 0
        self.wins = 0.0

    def _get_legal_moves(self):
        b = Board(self.n)
        b.pieces = [row[:] for row in self.state]
        moves = b.get_legal_moves(self.player)
        
        if not moves:
            if not b.get_legal_moves(-self.player):
                return [] 
            return [None]  
        return moves

    def uct_select_child(self):
        return max(self.children, key=lambda c: c.wins / c.visits + 1.414 * math.sqrt(math.log(self.visits) / c.visits))

    def expand(self):
        move = self.untried_moves.pop()
        next_state = [row[:] for row in self.state]
        
        if move is not None:
            b = Board(self.n)
            b.pieces = next_state
            b.execute_move(move, self.player)
            next_state = b.pieces
            
        child = GreedyMCTSNode(next_state, -self.player, move=move, n=self.n)
        self.children.append(child)
        return child


class GreedyMCTSPlayer():
    def __init__(self, game, time_limit=1.0):
        self.game = game
        self.time_limit = time_limit
        self.n = game.n

    def play(self, board):
        start_time = time.time()
        
        py_board = np.asarray(board, dtype=int).tolist()
        root = GreedyMCTSNode(py_board, 1, n=self.n)
        
        if not root.untried_moves and not root.children:
            return self.n * self.n
            
        while time.time() - start_time < self.time_limit:
            node = root
            search_path = [node]
            
            while not node.untried_moves and node.children:
                node = node.uct_select_child()
                search_path.append(node)
                
            if node.untried_moves:
                node = node.expand()
                search_path.append(node)
                
            result = self._greedy_rollout(node.state, node.player)
            
            for n in reversed(search_path):
                n.visits += 1
                n.wins += (1.0 - result)
                result = 1.0 - result
                
        best_child = max(root.children, key=lambda c: c.visits)
        best_move = best_child.move
        
        root = None
        search_path = None
        gc.collect()
        
        if best_move is None:
            return self.n * self.n
        return best_move[0] * self.n + best_move[1]

    def _greedy_rollout(self, state, player):
        b = Board(self.n)
        b.pieces = [row[:] for row in state]
        current_player = player

        temp_board = Board(self.n)
        temp_board.pieces = [[0]*self.n for _ in range(self.n)]
        
        while True:
            my_moves = b.get_legal_moves(current_player)
            opp_moves = b.get_legal_moves(-current_player)
            
            if not my_moves and not opp_moves:
                diff = b.countDiff(player)
                if diff > 0: return 1.0  
                elif diff < 0: return 0.0 
                else: return 0.5 
                
            if not my_moves:
                current_player = -current_player
                continue
                
            best_move = None
            best_diff = -float('inf')
            
            for move in my_moves:
                for r in range(self.n):
                    temp_board.pieces[r][:] = b.pieces[r]
                    
                temp_board.execute_move(move, current_player)

                diff = temp_board.countDiff(current_player)
                if diff > best_diff:
                    best_diff = diff
                    best_move = move
                    
            b.execute_move(best_move, current_player)
            current_player = -current_player
                
class AlphaZeroMCTS:

    def __init__(self, game, nnet_wrapper, num_sims=50, c_puct=1.0):
        self.game = game
        self.nnet = nnet_wrapper 
        self.num_sims = num_sims 
        self.c_puct = c_puct
        
        self.Qsa = {}  
        self.Nsa = {}  
        self.Ns = {}   
        self.Ps = {}   
        self.Es = {}  
        self.Vs = {}   

    def getAction(self, canonicalBoard, temp=1, add_noise=False):
        
        if len(self.Ps) > 30000:
            self.Qsa.clear()
            self.Nsa.clear()
            self.Ns.clear()
            self.Ps.clear()
            self.Es.clear()
            self.Vs.clear()
            import gc
            gc.collect()
            
        s = canonicalBoard.tobytes()

        if s not in self.Ps:
            self.search(canonicalBoard, depth=0)
            sims_to_run = self.num_sims - 1
        else:
            sims_to_run = self.num_sims

        if add_noise:
            valids = self.Vs[s]
           
            valid_moves = np.where(np.array(valids))[0] 
            
            if len(valid_moves) > 0:
                
                noise = np.random.dirichlet([0.25] * len(valid_moves))
                epsilon = 0.25 
                
                for i, move in enumerate(valid_moves):
                    
                    self.Ps[s][move] = (1 - epsilon) * self.Ps[s][move] + epsilon * noise[i]

        
        for i in range(sims_to_run):
            self.search(canonicalBoard, depth=0)
        counts = [self.Nsa.get((s, a), 0) for a in range(self.game.getActionSize())]

        
        if temp == 0:
            
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, canonicalBoard, depth=0):
        
        if depth > 60:
            return 0
        
        s = canonicalBoard.tobytes()

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] != 0:
            
            return -self.Es[s] 

       
        if s not in self.Ps:
            
            pi, v = self.nnet.predict(canonicalBoard)
            v = float(v)

            # Mask invalid moves 
            valids = self.game.getValidMoves(canonicalBoard, 1)
            pi = pi * valids  
            sum_Ps_s = np.sum(pi)
            
            if sum_Ps_s > 0:
                pi /= sum_Ps_s  
            else:
               
                pi = pi + valids
                pi /= np.sum(pi)

            
            self.Ps[s] = pi
            self.Vs[s] = valids
            self.Ns[s] = 0
            
            return -v

        
        valids = self.Vs[s]
        cur_best_u = -float('inf')
        best_act = -1

        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    
                    u = self.Qsa[(s, a)] + self.c_puct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:
                   
                    u = self.c_puct * self.Ps[s][a] * math.sqrt(self.Ns[s] + 1e-8)  

                if u > cur_best_u:
                    cur_best_u = u
                    best_act = a

        if best_act == -1:
            valid_moves = [i for i, v in enumerate(valids) if v == 1]
            best_act = valid_moves[0] if valid_moves else (self.game.getActionSize() - 1)
            
        a = best_act

        
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        
        v = self.search(next_s, depth=depth + 1)

        #Backup
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1

        return -v    
