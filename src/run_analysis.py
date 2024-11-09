import logging
import yaml
from pathlib import Path
import torch
from torch_geometric.data import Data
import networkx as nx
from itertools import combinations
import sys

from data.data_loader import CoraDataLoader
from metrics.common_neighbors import CommonNeighbors

def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent

def setup_logging(config: dict) -> None:
    """Setup logging configuration."""
    log_path = get_project_root() / config['logging']['file']
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=config['logging']['level'],
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(str(log_path)),
            logging.StreamHandler(sys.stdout)
        ]
    )

def load_config() -> dict:
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent / 'config' / 'config.yaml'
    with open(config_path) as f:
        return yaml.safe_load(f)

def to_networkx(data: Data) -> nx.Graph:
    """Convert PyG data to NetworkX graph."""
    return nx.from_edgelist(data.edge_index.t().tolist())

def get_candidate_pairs(G: nx.Graph) -> list:
    """Get all possible node pairs that are not connected."""
    all_nodes = list(G.nodes())
    return [(i, j) for i, j in combinations(all_nodes, 2) 
            if not G.has_edge(i, j)]

def main():
    # Load configuration
    config = load_config()
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    try:
        # Setup data loader
        data_root = get_project_root() / config['paths']['data_root']
        data_loader = CoraDataLoader(
            root=data_root,
            val_ratio=config['data']['datasets']['cora']['splits']['val'],
            test_ratio=config['data']['datasets']['cora']['splits']['test']
        )
        
        # Load data
        data = data_loader.load()
        logger.info(f"Dataset loaded: {data.num_nodes} nodes, {data.num_edges} edges")
        
        # Convert to NetworkX for link prediction
        G = to_networkx(data)
        
        # Get candidate pairs
        candidate_pairs = get_candidate_pairs(G)
        logger.info(f"Generated {len(candidate_pairs)} candidate pairs")
        
        # Calculate Common Neighbors scores
        cn_metric = CommonNeighbors(G)
        predictions = [
            (i, j, cn_metric.compute_score(i, j))
            for i, j in candidate_pairs
        ]
        
        # Sort predictions
        predictions.sort(key=lambda x: x[2], reverse=True)
        
        # Get top-k predictions
        top_k = config['metrics']['common_neighbors']['top_k']
        top_predictions = predictions[:top_k]
        
        # Log results
        logger.info(f"\nTop {top_k} predicted links:")
        for node1, node2, score in top_predictions:
            logger.info(f"Node {node1} - Node {node2}: Score = {score}")
            
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()