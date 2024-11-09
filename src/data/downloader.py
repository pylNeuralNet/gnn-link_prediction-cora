from pathlib import Path
import logging
import torch_geometric.datasets as pyg_datasets
from typing import Optional

class DataDownloader:
    """Handles dataset downloading using PyTorch Geometric."""
    
    def __init__(self, root: Path):
        self.root = root
        self.logger = logging.getLogger(__name__)
        
    def download_dataset(self, name: str) -> Optional[pyg_datasets.Planetoid]:
        """
        Download a dataset using PyTorch Geometric.
        
        Args:
            name: Dataset name ('Cora', 'Citeseer', etc.)
            
        Returns:
            PyG dataset or None if download fails
        """
        self.logger.info(f"Downloading/loading {name} dataset to {self.root}")
        
        try:
            dataset = pyg_datasets.Planetoid(
                root=str(self.root),
                name=name
            )
            self.logger.info(f"Successfully loaded {name} dataset")
            return dataset
            
        except Exception as e:
            self.logger.error(f"Error downloading {name} dataset: {str(e)}")
            raise