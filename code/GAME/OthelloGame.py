import sys
import os

sys.path.append(os.getcwd()) 

from GAME.OthelloLogic import Board 
import numpy as np

class OthelloGame():
    def __init__(self, n=8):
        self.n = n

    def getInitBoard(self):
        """返回初始化的 NumPy 棋盘矩阵"""
        b = Board(self.n)
        return np.array(b.pieces, dtype=int)

    def getBoardSize(self):
        """返回棋盘尺寸 (n, n)"""
        return (self.n, self.n)

    def getActionSize(self):
        """
        返回所有动作的数量。
        n*n 是所有格子的总数，外加一个动作代表 'pass'（无子可下只能跳过）。
        """
        return self.n * self.n + 1

    def getNextState(self, board, player, action):
        """
        执行一个动作并返回下一个状态。
        """
        # 🌟 强清洗并硬拷贝：物理开辟一片完全连续、无任何视图污染的独立内存块！
        np_board = np.array(board, dtype=int, copy=True)
        act = int(action)
        
        # 🌟 使用完全净化的 act 进行边界判定
        if act == self.n * self.n:
            return (np_board, -int(player))
        
        b = Board(self.n)
        b.pieces = np_board.tolist()

        move = (int(act / self.n), int(act % self.n))
        b.execute_move(move, int(player))
        
        # 返回时强保也是纯正的、独立的 NumPy 连续整型数组
        return (np.array(b.pieces, dtype=int), -int(player))

    def getValidMoves(self, board, player):
        """返回一个长度为 n*n+1 的二进制向量，1表示合法走子，0表示非法。"""
        valids = [0] * self.getActionSize()
        b = Board(self.n)
        
        # 🌟 强保连续，彻底杜绝 .tolist() 指针因 Stride 跨度错位暴毙
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
        返回游戏状态。
        0 表示未结束；1 表示当前玩家赢了；-1 表示输了；极小值表示平局。
        """
        b = Board(self.n)
        
        # 🌟🌟🌟 核心攻坚战场：通过 copy=True 彻底修复刚才的闪退死穴！
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
        """返回指定玩家的棋子数减去对手的棋子数"""
        b = Board(self.n)
        
        # 🌟 强保评估阶段安全
        safe_arr = np.array(board, dtype=int, copy=True)
        b.pieces = safe_arr.tolist()
        return b.countDiff(int(player))
    
    def getCanonicalForm(self, board, player):
        """
        返回当前玩家视角下的棋盘。
        """
        # 🌟 先通过 copy=True 截断上层传进来的任何视图链接
        np_board = np.array(board, dtype=int, copy=True)
        canonical = int(player) * np_board
        
        # 🌟 重点：乘法会再次在 NumPy 内部产生 Stride 视图！我们必须再次通过 copy=True 将其连续化硬克隆！
        return np.array(canonical, dtype=int, copy=True)

    def getSymmetries(self, board, pi):
        """
        数据增强：黑白棋具有旋转和镜像对称性。
        """
        assert(len(pi) == self.n**2 + 1)
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        l = []

        # 🌟 强保连续
        int_board = np.array(board, dtype=int, copy=True)

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(int_board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.flipud(newB)
                    newPi = np.flipud(newPi)
                
                # 🌟 对称局面转换会制造大量碎片内存视图，全部强转成拥有物理独立内存的连续硬副本
                l += [(np.array(newB, dtype=int, copy=True), list(newPi.ravel()) + [pi[-1]])]
        return l

    def stringRepresentation(self, board):
        """将棋盘转为字符串，作为 MCTS 字典的 Key"""
        # 🌟 强保转二进制串之前，内存是连续且单一的
        return np.array(board, dtype=int, copy=True).tobytes()