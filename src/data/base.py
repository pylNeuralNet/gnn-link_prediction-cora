from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Optional
import torch
from torch_geometric.data import Dataset, Data
import logging

class BaseDataLoader(ABC):
    """Abstract base class for data loading and processing."""
    
    def __init__(self, root: Path):
        self.root = Path(root)
        self.logger = logging.getLogger(__name__)
        self.dataset: Optional[Dataset] = None
        self.processed_data: Optional[Data] = None
        
    @abstractmethod
    def load(self) -> Data:
        """Load and return the processed dataset."""
        pass
    
    @abstractmethod
    def process(self) -> None:
        """Process the raw data into the required format."""
        pass
    
    @abstractmethod
    def get_split_masks(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get train/val/test split masks."""
        pass
    
    def get_graph_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get edge_index, node features, and labels."""
        if self.processed_data is None:
            raise ValueError("Data not loaded. Call load() first.")
        return (
            self.processed_data.edge_index,
            self.processed_data.x,
            self.processed_data.y
        )