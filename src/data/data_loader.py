from pathlib import Path
import torch
from torch_geometric.data import Data
from typing import Tuple, Optional
import numpy as np
from .base import BaseDataLoader
from .downloader import DataDownloader

class CoraDataLoader(BaseDataLoader):
    """Data loader for the Cora dataset."""
    
    def __init__(self, root: Path, val_ratio: float = 0.1, test_ratio: float = 0.1):
        super().__init__(root)
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.downloader = DataDownloader(root)
        
    def load(self) -> Data:
        """Load the Cora dataset."""
        # Download if needed and get PyG dataset
        dataset = self.downloader.download_dataset('Cora')
        if dataset is None:
            raise ValueError("Failed to load Cora dataset")
            
        self.dataset = dataset
        self.processed_data = dataset[0]  # Cora has only one graph
        self.process()
        
        return self.processed_data
    
    def process(self) -> None:
        """Process the raw data into required format."""
        if self.processed_data is None:
            raise ValueError("Data not loaded")
            
        # Add split masks if they don't exist
        if not hasattr(self.processed_data, 'train_mask'):
            train_mask, val_mask, test_mask = self.get_split_masks()
            self.processed_data.train_mask = train_mask
            self.processed_data.val_mask = val_mask
            self.processed_data.test_mask = test_mask
    
    def get_split_masks(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate train/val/test split masks."""
        if self.processed_data is None:
            raise ValueError("Data not loaded")
            
        num_nodes = self.processed_data.num_nodes
        indices = np.random.permutation(num_nodes)
        
        test_size = int(num_nodes * self.test_ratio)
        val_size = int(num_nodes * self.val_ratio)
        train_size = num_nodes - test_size - val_size
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True
        
        return train_mask, val_mask, test_mask