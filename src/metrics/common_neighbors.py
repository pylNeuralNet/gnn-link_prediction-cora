from typing import Tuple, List, Set
import networkx as nx
from metrics.base_metric import BaseLinkPredictionMetric

class CommonNeighbors(BaseLinkPredictionMetric):
    """Implementation of Common Neighbors metric with temporal awareness."""
    
    def __init__(self, graph: nx.Graph, use_temporal: bool = True):
        super().__init__(graph)
        self.use_temporal = use_temporal
    
    def get_temporal_neighbors(self, node: int, timestamp: float) -> Set[int]:
        """Get neighbors of a node up to a specific timestamp."""
        if not self.use_temporal:
            return self.get_neighbors(node)
            
        temporal_neighbors = set()
        for neighbor in self.graph.neighbors(node):
            edge_timestamp = self.graph[node][neighbor]['timestamp']
            if edge_timestamp <= timestamp:
                temporal_neighbors.add(neighbor)
        return temporal_neighbors
    
    def compute_score(self, node_i: int, node_j: int, timestamp: float = None) -> float:
        """
        Compute Common Neighbors score for a node pair.
        
        Args:
            node_i: First node
            node_j: Second node
            timestamp: Optional timestamp for temporal analysis
            
        Returns:
            float: Number of common neighbors
        """
        if self.use_temporal and timestamp is not None:
            neighbors_i = self.get_temporal_neighbors(node_i, timestamp)
            neighbors_j = self.get_temporal_neighbors(node_j, timestamp)
        else:
            neighbors_i = self.get_neighbors(node_i)
            neighbors_j = self.get_neighbors(node_j)
            
        return len(neighbors_i.intersection(neighbors_j))
    
    def compute_all_scores(self, node_pairs: List[Tuple[int, int]], 
                          timestamps: List[float] = None) -> List[float]:
        """Compute Common Neighbors scores for multiple node pairs."""
        scores = []
        
        if timestamps and len(timestamps) != len(node_pairs):
            raise ValueError("Number of timestamps must match number of node pairs")
            
        for idx, (node_i, node_j) in enumerate(node_pairs):
            timestamp = timestamps[idx] if timestamps else None
            scores.append(self.compute_score(node_i, node_j, timestamp))
            
        return scores