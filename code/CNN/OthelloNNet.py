import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):

    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection! 
        out = F.relu(out)
        return out

class OthelloNNet(nn.Module):

    def __init__(self, game, num_channels=128, num_res_blocks=4):
        super(OthelloNNet, self).__init__()
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        # 1. Initial Convolutional Block 
        self.conv_initial = nn.Conv2d(1, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_initial = nn.BatchNorm2d(num_channels)

        # 2. Residual Tower 
        self.res_blocks = nn.ModuleList([ResBlock(num_channels) for _ in range(num_res_blocks)])

        # 3. Policy Head 
        self.conv_policy = nn.Conv2d(num_channels, 2, kernel_size=1, bias=False)
        self.bn_policy = nn.BatchNorm2d(2)
        self.fc_policy = nn.Linear(2 * self.board_x * self.board_y, self.action_size)

        # 4. Value Head 
        self.conv_value = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False)
        self.bn_value = nn.BatchNorm2d(1)
        self.fc_value1 = nn.Linear(1 * self.board_x * self.board_y, 256)
        self.fc_value2 = nn.Linear(256, 1)

    def forward(self, s):
        s = s.view(-1, 1, self.board_x, self.board_y)

        # Initial Block
        s = F.relu(self.bn_initial(self.conv_initial(s)))

        # Pass through all Residual Blocks
        for res_block in self.res_blocks:
            s = res_block(s)

        # Policy Head Output
        pi = F.relu(self.bn_policy(self.conv_policy(s)))
        pi = pi.view(-1, 2 * self.board_x * self.board_y)
        pi = self.fc_policy(pi)
        pi = F.log_softmax(pi, dim=1)

        # Value Head Output
        v = F.relu(self.bn_value(self.conv_value(s)))
        v = v.view(-1, 1 * self.board_x * self.board_y)
        v = F.relu(self.fc_value1(v))
        v = torch.tanh(self.fc_value2(v))

        return pi, v

if __name__ == "__main__":
    from GAME.OthelloGame import OthelloGame
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print("==================================================")
    print("Initializing AlphaZero ResNet...") 
    game = OthelloGame(8)
    nnet = OthelloNNet(game)
    print("ResNet successfully constructed!") 
    
    dummy_boards = torch.randn(2, 8, 8) 
    print(f"Input tensor shape: {dummy_boards.shape}") 
    
    out_pi, out_v = nnet(dummy_boards)
    
    print("--------------------------------------------------")
    print(f"Policy head output shape: {out_pi.shape}") 
    print(f"Value head output shape: {out_v.shape}") 
    print("==================================================")
