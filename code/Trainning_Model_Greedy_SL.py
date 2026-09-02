import os
import sys
import time
import pickle
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42) 

sys.path.append(os.getcwd())
from CNN.NNetTrainer import OthelloNNet 
from GAME.OthelloGame import OthelloGame

class OthelloDataset(Dataset):
    """
    Standard PyTorch Dataset for loading the expert data.
    """
    def __init__(self, data_path):
        print(f"Loading dataset from {data_path}...")
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)
        print(f"Loaded {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        board, pi, v = self.data[idx]
        
        # Convert numpy arrays to PyTorch tensors
        # Board shape needs to be (1, 8, 8) for the ResNet
        board_tensor = torch.FloatTensor(board.astype(np.float32))
        pi_tensor = torch.FloatTensor(np.array(pi, dtype=np.float32))
        v_tensor = torch.FloatTensor(np.array([v], dtype=np.float32))
        
        return board_tensor, pi_tensor, v_tensor

class AlphaZeroTrainer:
    """
    Supervised Learning Trainer for the ResNet.
    """
    def __init__(self, model, lr=0.001, batch_size=64, epochs=10):
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Use GPU if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Training on device: {self.device}")
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)

    def loss_pi(self, targets, outputs):
        """
        Policy Loss: Cross Entropy.
        Since our network outputs log_softmax, we use manual dot product for Cross Entropy.
        """
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        """
        Value Loss: Mean Squared Error.
        """
        return torch.sum((targets.view(-1) - outputs.view(-1)) ** 2) / targets.size()[0]

    def train(self, dataset):
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        print("Starting ResNet Training...")
        
        for epoch in range(self.epochs):
            self.model.train()
            pi_losses = []
            v_losses = []
            start_time = time.time()
            
            for batch_idx, (boards, pis, vs) in enumerate(dataloader):
                boards, pis, vs = boards.to(self.device), pis.to(self.device), vs.to(self.device)
                
                # 1. Forward pass
                out_pi, out_v = self.model(boards)
                
                # 2. Calculate Loss (Combined Policy and Value)
                l_pi = self.loss_pi(pis, out_pi)
                l_v = self.loss_v(vs, out_v)
                total_loss = l_pi + l_v
                
                # 3. Backward pass and optimize
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                pi_losses.append(l_pi.item())
                v_losses.append(l_v.item())
                
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1}/{self.epochs} | "
                  f"Policy Loss: {np.mean(pi_losses):.4f} | "
                  f"Value Loss: {np.mean(v_losses):.4f} | "
                  f"Time: {elapsed:.2f}s")
            
        print("Training complete.")

    def save_model(self, folder="checkpoint", filename="resnet_expert.pth"):
        if not os.path.exists(folder):
            os.makedirs(folder)
        filepath = os.path.join(folder, filename)
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")

if __name__ == "__main__":
    # 1. Initialize Game and Model
    game = OthelloGame(8)
    resnet = OthelloNNet(game)
    
    # 2. Load the full dataset 
    dataset_file = "expert_data_full.pkl" 
    if not os.path.exists(dataset_file):
        print(f"Error: {dataset_file} not found. Please run data generation first.")
        sys.exit(1)
        
    dataset = OthelloDataset(dataset_file)
    
    # 3. Train the network 
    # 离线监督学习阶段：由于数据集极其庞大，恢复较大的 batch_size 和 epochs，让模型充分吸收专家知识
    trainer = AlphaZeroTrainer(resnet, lr=0.001, batch_size=512, epochs=20)
    trainer.train(dataset)
    
    # 4. Save the trained weights
    models_dir = os.path.join(os.getcwd(), "MODELS")
    trainer.save_model(folder=models_dir, filename="model_Greedy_SL.pth")