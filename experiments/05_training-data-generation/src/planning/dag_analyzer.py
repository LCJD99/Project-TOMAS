"""
DAG analyzer for task dependency analysis and topological sorting.

Analyzes DAG structure from task JSON to support scheduling decisions.
"""

from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict, deque


class DAGNode:
    """Represents a node in the DAG."""
    
    def __init__(self, name: str, task_type: str, arguments: List[str]):
        self.name = name
        self.task_type = task_type  # Original task type (with prefix removed)
        self.arguments = arguments
        self.predecessors: Set[str] = set()
        self.successors: Set[str] = set()
    
    def __repr__(self):
        return f"DAGNode({self.name}, type={self.task_type})"


class DAGAnalyzer:
    """Analyzes DAG structure for scheduling."""
    
    def __init__(self, task: Dict[str, Any]):
        """
        Initialize analyzer with task data.
        
        Args:
            task: Task dictionary from tasks_augmented.json
        """
        self.task = task
        self.nodes: Dict[str, DAGNode] = {}
        self.edges: List[Tuple[str, str]] = []
        
        # Parse task structure
        self._parse_task()
    
    def _parse_task(self):
        """Parse task structure into nodes and edges."""
        # Create nodes
        for node_data in self.task['tool_nodes']:
            name = node_data['task']
            arguments = node_data.get('arguments', [])
            
            # Extract base task type (remove prefix like T1_, T2_)
            task_type = name
            if '_' in name:
                parts = name.split('_', 1)
                if len(parts[0]) >= 2 and parts[0][0] == 'T' and parts[0][1:].isdigit():
                    task_type = parts[1]
            
            self.nodes[name] = DAGNode(name, task_type, arguments)
        
        # Create edges
        for link in self.task['tool_links']:
            source = link['source']
            target = link['target']
            self.edges.append((source, target))
            
            # Update node connections
            if source in self.nodes and target in self.nodes:
                self.nodes[source].successors.add(target)
                self.nodes[target].predecessors.add(source)
    
    def topological_sort(self) -> List[List[str]]:
        """
        Perform topological sort and return nodes grouped by levels.
        
        Nodes at the same level can potentially execute in parallel.
        
        Returns:
            List of levels, where each level is a list of node names
        """
        # Calculate in-degrees
        in_degree = {name: len(node.predecessors) for name, node in self.nodes.items()}
        
        # Queue for nodes with in-degree 0
        queue = deque([name for name, deg in in_degree.items() if deg == 0])
        
        levels = []
        
        while queue:
            # All nodes in current queue are at the same level
            current_level = list(queue)
            levels.append(current_level)
            
            # Process all nodes in current level
            next_queue = []
            for node_name in current_level:
                node = self.nodes[node_name]
                
                # Reduce in-degree of successors
                for successor in node.successors:
                    in_degree[successor] -= 1
                    if in_degree[successor] == 0:
                        next_queue.append(successor)
            
            queue = deque(next_queue)
        
        # Check if all nodes were processed (detect cycles)
        if sum(len(level) for level in levels) != len(self.nodes):
            raise ValueError("DAG contains a cycle!")
        
        return levels
    
    def get_dependencies(self, node_name: str) -> Set[str]:
        """
        Get all direct dependencies (predecessors) of a node.
        
        Args:
            node_name: Name of the node
            
        Returns:
            Set of predecessor node names
        """
        if node_name not in self.nodes:
            raise ValueError(f"Node not found: {node_name}")
        
        return self.nodes[node_name].predecessors.copy()
    
    def get_all_ancestors(self, node_name: str) -> Set[str]:
        """
        Get all ancestor nodes (transitive closure of predecessors).
        
        Args:
            node_name: Name of the node
            
        Returns:
            Set of all ancestor node names
        """
        if node_name not in self.nodes:
            raise ValueError(f"Node not found: {node_name}")
        
        ancestors = set()
        queue = deque([node_name])
        visited = {node_name}
        
        while queue:
            current = queue.popleft()
            for pred in self.nodes[current].predecessors:
                if pred not in visited:
                    visited.add(pred)
                    ancestors.add(pred)
                    queue.append(pred)
        
        return ancestors
    
    def get_output_nodes(self) -> List[str]:
        """
        Get nodes with no successors (output nodes).
        
        Returns:
            List of output node names
        """
        return [name for name, node in self.nodes.items() 
                if len(node.successors) == 0]
    
    def get_parallel_groups(self) -> List[List[str]]:
        """
        Get groups of nodes that can execute in parallel.
        
        This is essentially the topological sort levels, excluding START.
        
        Returns:
            List of parallel groups (each group is a list of node names)
        """
        levels = self.topological_sort()
        
        # Filter out START nodes and empty levels
        parallel_groups = []
        for level in levels:
            # Remove START nodes
            filtered = [node for node in level if self.nodes[node].task_type != 'START']
            if filtered:
                parallel_groups.append(filtered)
        
        return parallel_groups
    
    def can_execute_parallel(self, node1: str, node2: str) -> bool:
        """
        Check if two nodes can execute in parallel (no dependency between them).
        
        Args:
            node1: First node name
            node2: Second node name
            
        Returns:
            True if nodes can execute in parallel
        """
        # Check if either is an ancestor of the other
        return (node2 not in self.get_all_ancestors(node1) and 
                node1 not in self.get_all_ancestors(node2))
    
    def get_node(self, node_name: str) -> DAGNode:
        """Get node by name."""
        if node_name not in self.nodes:
            raise ValueError(f"Node not found: {node_name}")
        return self.nodes[node_name]


if __name__ == "__main__":
    # Test with sample task data
    import json
    from pathlib import Path
    
    print("Testing DAGAnalyzer")
    print("=" * 60)
    
    # Load a sample task
    data_dir = Path(__file__).parent.parent.parent / "data"
    with open(data_dir / "tasks_augmented.json", 'r') as f:
        data = json.load(f)
    
    # Test with a chain task
    chain_tasks = [t for t in data['data'] if t['type'] == 'chain']
    if chain_tasks:
        task = chain_tasks[0]
        print(f"\nAnalyzing chain task: {task['id']}")
        print(f"Nodes: {[n['task'] for n in task['tool_nodes']]}")
        print(f"Links: {[(l['source'], l['target']) for l in task['tool_links']]}")
        
        analyzer = DAGAnalyzer(task)
        
        print("\nTopological sort (levels):")
        levels = analyzer.topological_sort()
        for i, level in enumerate(levels):
            print(f"  Level {i}: {level}")
        
        print("\nParallel groups (excluding START):")
        groups = analyzer.get_parallel_groups()
        for i, group in enumerate(groups):
            print(f"  Group {i}: {group}")
        
        print("\nOutput nodes:")
        outputs = analyzer.get_output_nodes()
        print(f"  {outputs}")
    
    # Test with a merged task
    print("\n" + "=" * 60)
    merged_tasks = [t for t in data['data'] if t['type'] == 'merged']
    if merged_tasks:
        task = merged_tasks[0]
        print(f"\nAnalyzing merged task: {task['id']}")
        print(f"Nodes: {[n['task'] for n in task['tool_nodes']]}")
        print(f"Links: {[(l['source'], l['target']) for l in task['tool_links']]}")
        
        analyzer = DAGAnalyzer(task)
        
        print("\nTopological sort (levels):")
        levels = analyzer.topological_sort()
        for i, level in enumerate(levels):
            print(f"  Level {i}: {level}")
        
        print("\nParallel groups:")
        groups = analyzer.get_parallel_groups()
        for i, group in enumerate(groups):
            print(f"  Group {i}: {group}")
        
        # Test dependencies
        if len(groups) > 0 and len(groups[0]) > 0:
            node = groups[0][0]
            deps = analyzer.get_dependencies(node)
            print(f"\nDependencies of {node}: {deps}")
        
        # Test parallel execution check
        if len(groups) > 0 and len(groups[0]) >= 2:
            node1, node2 = groups[0][0], groups[0][1]
            can_parallel = analyzer.can_execute_parallel(node1, node2)
            print(f"\nCan {node1} and {node2} execute in parallel? {can_parallel}")
