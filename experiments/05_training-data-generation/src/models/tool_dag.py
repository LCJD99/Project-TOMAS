"""Tool DAG (Directed Acyclic Graph) data structure."""

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict

from .tool_node import SampledNode, ToolNode, Link


@dataclass
class ToolDAG:
    """Represents a complete Tool DAG with all its components."""
    
    id: str
    seed: int
    n_tools: int
    type: str
    sampled_nodes: List[SampledNode]
    sampled_links: List[Link]
    instruction: str
    tool_steps: List[str]
    tool_nodes: List[ToolNode]
    tool_links: List[Link]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolDAG':
        """Create a ToolDAG from a dictionary."""
        return cls(
            id=data['id'],
            seed=data['seed'],
            n_tools=data['n_tools'],
            type=data['type'],
            sampled_nodes=[SampledNode.from_dict(node) for node in data['sampled_nodes']],
            sampled_links=[Link.from_dict(link) for link in data['sampled_links']],
            instruction=data['instruction'],
            tool_steps=data['tool_steps'],
            tool_nodes=[ToolNode.from_dict(node) for node in data['tool_nodes']],
            tool_links=[Link.from_dict(link) for link in data['tool_links']]
        )
    
    def get_node_degrees(self) -> Dict[str, Tuple[int, int]]:
        """
        Calculate in-degree and out-degree for each node.
        
        Returns:
            Dict mapping task name to (in_degree, out_degree) tuple
        """
        in_degree = defaultdict(int)
        out_degree = defaultdict(int)
        
        # Initialize all nodes with 0 degree
        for node in self.tool_nodes:
            if node.task not in in_degree:
                in_degree[node.task] = 0
            if node.task not in out_degree:
                out_degree[node.task] = 0
        
        # Count degrees from links
        for link in self.tool_links:
            out_degree[link.source] += 1
            in_degree[link.target] += 1
        
        # Combine into single dict
        result = {}
        all_tasks = set(in_degree.keys()) | set(out_degree.keys())
        for task in all_tasks:
            result[task] = (in_degree[task], out_degree[task])
        
        return result
    
    def get_all_node_degree_pairs(self) -> List[Tuple[int, int]]:
        """
        Get list of all (in_degree, out_degree) pairs for this DAG.
        
        Returns:
            List of (in_degree, out_degree) tuples
        """
        degrees = self.get_node_degrees()
        return list(degrees.values())
