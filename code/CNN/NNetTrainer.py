import os
import torch
import torch.optim as optim
import numpy as np
import sys
import time

# Ensure the root directory is in the search path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Global import - must not be inside any function

from CNN.OthelloNNet import OthelloNNet

class NNetWrapper:
    """
    Neural network wrapper for training, saving, loading, and predicting.
    """
    def __init__(self, game):
        # Now OthelloNNet can be globally recognized
        self.nnet = OthelloNNet(game)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nnet.to(self.device)

    def to_cpu(self):
        self.device = torch.device("cpu")
        self.nnet.to(self.device)

    def to_gpu(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.nnet.to(self.device)        

    def predict(self, board):

        board_tensor = torch.FloatTensor(board.astype(np.float64)).unsqueeze(0).to(self.device)
        
        self.nnet.eval() 
        
        with torch.no_grad(): 
            pi, v = self.nnet(board_tensor)
            
        
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def train(self, dataset, epochs=10, batch_size=64):
        print(f"Starting training on device: {self.device}")
        optimizer = optim.Adam(self.nnet.parameters(), lr=0.001, weight_decay=1e-4)

        for epoch in range(epochs):
            self.nnet.train()
            np.random.shuffle(dataset)
            
            total_pi_loss = 0.0
            total_v_loss = 0.0
            batch_count = 0

            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i+batch_size]
                
                boards, target_pis, target_vs = list(zip(*batch))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64)).to(self.device)
                target_pis = torch.FloatTensor(np.array(target_pis)).to(self.device)
                target_vs = torch.FloatTensor(np.array(target_vs).astype(np.float64)).to(self.device)

                # Forward pass
                out_pi, out_v = self.nnet(boards)

                # Calculate losses
                l_pi = -torch.sum(target_pis * out_pi) / target_pis.size()[0]
                l_v = torch.sum((target_vs - out_v.view(-1)) ** 2) / target_vs.size()[0]
                total_loss = l_pi + l_v

                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                total_pi_loss += l_pi.item()
                total_v_loss += l_v.item()
                batch_count += 1
                time.sleep(0.05) 

            print(f"Epoch {epoch+1:02d}/{epochs} | Policy Loss: {total_pi_loss/batch_count:.4f} | Value Loss: {total_v_loss/batch_count:.4f}")

    def save_checkpoint(self, folder="models", filename="best_model.pth"):
        if not os.path.exists(folder):
            os.makedirs(folder)
        filepath = os.path.join(folder, filename)
        torch.save(self.nnet.state_dict(), filepath)
        print(f"Model saved to: {filepath}")

    def load_checkpoint(self, folder="models", filename="best_model.pth"):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise Exception(f"Model not found: {filepath}")
        self.nnet.load_state_dict(torch.load(filepath, map_location=self.device, weights_only=True))
        print(f"Model loaded: {filepath}")
