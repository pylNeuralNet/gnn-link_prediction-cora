from abc import ABC, abstractmethod
import networkx as nx
from typing import Tuple, List, Set

class BaseLinkPredictionMetric(ABC):
    """Abstract base class for link prediction metrics."""
    
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        
    @abstractmethod
    def compute_score(self, node_i: int, node_j: int) -> float:
        """Compute the metric score for a node pair."""
        pass
    
    @abstractmethod
    def compute_all_scores(self, node_pairs: List[Tuple[int, int]]) -> List[float]:
        """Compute scores for multiple node pairs."""
        pass
    
    def get_neighbors(self, node: int) -> Set[int]:
        """Get neighbors of a node."""
        return set(self.graph.neighbors(node))