import sys
import os

sys.path.append(os.getcwd()) 

from GAME.OthelloLogic import Board 
import numpy as np

class OthelloGame():
    def __init__(self, n=8):
        self.n = n

    def getInitBoard(self):
        """Returns the initialized NumPy board matrix"""
        b = Board(self.n)
        return np.array(b.pieces, dtype=int)

    def getBoardSize(self):
        """Returns the board size (n, n)"""
        return (self.n, self.n)

    def getActionSize(self):
        """
        Returns the total number of actions.
        n*n is the total number of cells, plus one action representing 'pass' (skip when no moves are available).
        """
        return self.n * self.n + 1

    def getNextState(self, board, player, action):
        """
        Execute an action and return the next state.
        """
        # Strong purification and hard copy: allocate a completely contiguous, view-pollution-free independent memory block
        np_board = np.array(board, dtype=int, copy=True)
        act = int(action)
        
        # Use the completely purified act for boundary checking
        if act == self.n * self.n:
            return (np_board, -int(player))
        
        b = Board(self.n)
        b.pieces = np_board.tolist()

        move = (int(act / self.n), int(act % self.n))
        b.execute_move(move, int(player))
        
        # Return as a pure, independent NumPy contiguous integer array
        return (np.array(b.pieces, dtype=int), -int(player))

    def getValidMoves(self, board, player):
        """Returns a binary vector of length n*n+1, where 1 indicates a legal move and 0 indicates an illegal move."""
        valids = [0] * self.getActionSize()
        b = Board(self.n)
        
        # Ensure contiguity to completely prevent pointer corruption due to stride misalignment
        safe_arr = np.array(board, dtype=int, copy=True)
        b.pieces = safe_arr.tolist()

        legalMoves = b.get_legal_moves(int(player))
        
        if len(legalMoves) == 0:
            valids[-1] = 1 
            return np.array(valids, dtype=int)
        
        for x, y in legalMoves: 
            valids[self.n * x + y] = 1
        return np.array(valids, dtype=int)

    def getGameEnded(self, board, player):
        """
        Returns the game state.
        0 means the game is not finished; 1 means the current player won; -1 means lost; minimum value indicates a draw.
        """
        b = Board(self.n)
        
        # Core battle: completely fix the crash vulnerability through copy=True
        safe_arr = np.array(board, dtype=int, copy=True)
        b.pieces = safe_arr.tolist()
        
        if b.has_legal_moves(int(player)):
            return 0
        if b.has_legal_moves(-int(player)):
            return 0

        diff = b.countDiff(int(player))
        if diff > 0:
            return 1
        return -1
    
    def getScore(self, board, player):
        """Returns the count of the specified player's pieces minus the opponent's pieces"""
        b = Board(self.n)
        
        # Ensure safety during evaluation phase
        safe_arr = np.array(board, dtype=int, copy=True)
        b.pieces = safe_arr.tolist()
        return b.countDiff(int(player))
    
    def getCanonicalForm(self, board, player):
        """
        Returns the board from the current player's perspective.
        """
        # First, use copy=True to break any view chains coming from above
        np_board = np.array(board, dtype=int, copy=True)
        canonical = int(player) * np_board
        
        # Important: multiplication will again produce stride views inside NumPy! We must use copy=True again to hard-clone it into contiguity
        return np.array(canonical, dtype=int, copy=True)

    def getSymmetries(self, board, pi):
        """
        Data augmentation: Othello has rotational and mirror symmetry.
        """
        assert(len(pi) == self.n**2 + 1)
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        l = []

        # Ensure contiguity
        int_board = np.array(board, dtype=int, copy=True)

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(int_board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.flipud(newB)
                    newPi = np.flipud(newPi)
                
                # Symmetric board transformations produce many fragmented memory views; convert all to hard copies with independent physical memory
                l += [(np.array(newB, dtype=int, copy=True), list(newPi.ravel()) + [pi[-1]])]
        return l

    def stringRepresentation(self, board):
        """Convert the board to a string as a key for the MCTS dictionary"""
        # Ensure contiguity and unity of memory before converting to binary string
        return np.array(board, dtype=int, copy=True).tobytes()
