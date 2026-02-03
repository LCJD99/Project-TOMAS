"""
Core scheduling algorithm for tool execution planning.

Solves multi-dimensional resource allocation problem:
- Assigns optimal resource configurations to each tool node
- Respects DAG dependencies 
- Minimizes total execution latency
- Handles parallel vs serial execution trade-offs
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from itertools import product

# Handle both module import and standalone execution
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.planning.resource_profiler import ResourceProfiler, ResourceConfig
    from src.planning.dag_analyzer import DAGAnalyzer
    from src.planning import tool_mapper
else:
    from .resource_profiler import ResourceProfiler, ResourceConfig
    from .dag_analyzer import DAGAnalyzer
    from . import tool_mapper


class ScheduleAssignment:
    """Represents a resource assignment for a single node."""
    
    def __init__(self, node_name: str, config: ResourceConfig, 
                 start_time: float, end_time: float):
        self.node_name = node_name
        self.config = config
        self.start_time = start_time
        self.end_time = end_time
    
    def __repr__(self):
        duration = self.end_time - self.start_time
        return (f"Assignment({self.node_name}: {self.config}, "
                f"t={self.start_time:.2f}-{self.end_time:.2f}, "
                f"duration={duration:.2f}s)")


class ExecutionPlan:
    """Complete execution plan for a task."""
    
    def __init__(self, assignments: Dict[str, ScheduleAssignment], 
                 total_latency: float):
        self.assignments = assignments
        self.total_latency = total_latency
    
    def get_assignment(self, node_name: str) -> Optional[ScheduleAssignment]:
        """Get assignment for a specific node."""
        return self.assignments.get(node_name)
    
    def __repr__(self):
        return f"ExecutionPlan(total_latency={self.total_latency:.2f}s, nodes={len(self.assignments)})"


class BruteForceScheduler:
    """
    Brute-force scheduler that searches all possible configurations.
    
    Algorithm:
    1. Process DAG level by level (topological order)
    2. For each level, try all feasible config combinations
    3. For each combination, decide parallel vs serial execution
    4. Track resource availability and time
    5. Select minimum latency plan
    
    Key constraints:
    - Parallel nodes: resources accumulate (must fit in available)
    - Serial nodes: resources don't accumulate (run one at a time)
    - Dependencies: node can't start until predecessors finish
    """
    
    def __init__(self, profiler: ResourceProfiler):
        self.profiler = profiler
    
    def schedule(self, dag: DAGAnalyzer, available_resources: ResourceConfig) -> ExecutionPlan:
        """
        Generate optimal execution plan for a DAG.
        
        Args:
            dag: Task DAG structure
            available_resources: Maximum available system resources
            
        Returns:
            ExecutionPlan with assignments and total latency
        """
        # Get topological levels (returns List[List[str]])
        levels = dag.topological_sort()
        
        # Initialize best plan
        best_plan = None
        best_latency = float('inf')
        
        # Generate all possible config assignments
        for config_assignment in self._generate_config_assignments(dag, levels, available_resources):
            # Try to execute this assignment
            plan = self._simulate_execution(dag, levels, config_assignment, available_resources)
            
            if plan and plan.total_latency < best_latency:
                best_plan = plan
                best_latency = plan.total_latency
        
        if best_plan is None:
            raise RuntimeError("No feasible execution plan found")
        
        return best_plan
    
    def _generate_config_assignments(self, dag: DAGAnalyzer, 
                                     levels: List[List[str]],
                                     available_resources: ResourceConfig):
        """
        Generate all possible resource configuration assignments.
        
        For each node, get all feasible configs and try all combinations.
        
        Yields:
            Dict mapping node_name -> (ResourceConfig, latency) or None for virtual nodes
        """
        # Get feasible configs for each node
        node_configs = {}
        for level in levels:
            for node_name in level:
                # Skip virtual nodes like START
                if tool_mapper.is_virtual_node(node_name):
                    node_configs[node_name] = [None]  # No config needed
                    continue
                
                # Get profiling tool name (handles prefix removal internally)
                profiling_tool = tool_mapper.task_to_profiling_name(node_name)
                if profiling_tool is None:
                    # Virtual node
                    node_configs[node_name] = [None]
                    continue
                
                # Get all feasible configs for this tool
                # Returns List[Tuple[ResourceConfig, latency]]
                feasible = self.profiler.get_feasible_configs(
                    profiling_tool, 
                    available_resources
                )
                
                if not feasible:
                    # No feasible config - can't schedule this task
                    return
                
                node_configs[node_name] = feasible
        
        # Generate all combinations
        node_names = []
        config_lists = []
        
        for node_name in sorted(node_configs.keys()):
            node_names.append(node_name)
            config_lists.append(node_configs[node_name])
        
        # Yield each combination
        for config_combo in product(*config_lists):
            assignment = dict(zip(node_names, config_combo))
            yield assignment
    
    def _simulate_execution(self, dag: DAGAnalyzer, 
                           levels: List[List[str]],
                           config_assignment: Dict[str, Optional[Tuple[ResourceConfig, float]]],
                           available_resources: ResourceConfig) -> Optional[ExecutionPlan]:
        """
        Simulate execution with given config assignment.
        
        Tries both parallel and serial execution at each level,
        selects the one with minimum latency that fits resources.
        
        Args:
            dag: Task DAG
            levels: Topological levels (node names)
            config_assignment: (ResourceConfig, latency) tuple for each node or None for virtual
            available_resources: System resource limits
            
        Returns:
            ExecutionPlan if feasible, None otherwise
        """
        assignments = {}
        current_time = 0.0
        
        for level_idx, level in enumerate(levels):
            # Skip empty levels
            if not level:
                continue
            
            # Filter out virtual nodes
            real_nodes = [n for n in level if not tool_mapper.is_virtual_node(n)]
            
            if not real_nodes:
                continue
            
            # For single-node levels, no choice needed
            if len(real_nodes) == 1:
                node_name = real_nodes[0]
                config_data = config_assignment[node_name]
                
                if config_data is None:
                    continue
                    
                config, latency = config_data
                
                # Find latest predecessor end time
                pred_end_time = self._get_predecessor_end_time(dag, node_name, assignments)
                start_time = max(current_time, pred_end_time)
                end_time = start_time + latency
                
                assignments[node_name] = ScheduleAssignment(
                    node_name, config, start_time, end_time
                )
                
                current_time = end_time
                continue
            
            # For multi-node levels, try parallel vs serial
            # Try parallel execution
            parallel_plan = self._try_parallel_execution(
                dag, real_nodes, config_assignment, assignments, 
                available_resources, current_time
            )
            
            # Try serial execution
            serial_plan = self._try_serial_execution(
                dag, real_nodes, config_assignment, assignments,
                current_time
            )
            
            # Select best feasible plan
            if parallel_plan and serial_plan:
                # Both feasible - pick minimum latency
                if parallel_plan['latency'] <= serial_plan['latency']:
                    selected = parallel_plan
                else:
                    selected = serial_plan
            elif parallel_plan:
                selected = parallel_plan
            elif serial_plan:
                selected = serial_plan
            else:
                # Neither feasible - can't schedule this assignment
                return None
            
            # Update assignments and time
            for node_name, assignment in selected['assignments'].items():
                assignments[node_name] = assignment
            
            current_time = selected['end_time']
        
        # Calculate total latency (max end time)
        if assignments:
            total_latency = max(a.end_time for a in assignments.values())
        else:
            total_latency = 0.0
        
        return ExecutionPlan(assignments, total_latency)
    
    def _try_parallel_execution(self, dag: DAGAnalyzer,
                               nodes: List[str],
                               config_assignment: Dict[str, Optional[Tuple[ResourceConfig, float]]],
                               current_assignments: Dict[str, ScheduleAssignment],
                               available_resources: ResourceConfig,
                               current_time: float) -> Optional[Dict]:
        """
        Try parallel execution of nodes in a level.
        
        Returns:
            Dict with 'assignments', 'latency', 'end_time' if feasible, None otherwise
        """
        # Calculate cumulative resource usage
        total_cpu_core = 0
        total_cpu_mem = 0.0
        total_gpu_sm = 0
        total_gpu_mem = 0.0
        
        node_data = []
        
        for node_name in nodes:
            config_data = config_assignment[node_name]
            if config_data is None:
                continue
            
            config, latency = config_data
            
            total_cpu_core += config.cpu_core
            total_cpu_mem += config.cpu_mem_gb
            total_gpu_sm += config.gpu_sm
            total_gpu_mem += config.gpu_mem_gb
            
            # Find when this node can start (after all predecessors)
            pred_end_time = self._get_predecessor_end_time(dag, node_name, current_assignments)
            start_time = max(current_time, pred_end_time)
            
            node_data.append({
                'node_name': node_name,
                'config': config,
                'latency': latency,
                'start_time': start_time
            })
        
        # Check if resources fit
        if (total_cpu_core > available_resources.cpu_core or
            total_cpu_mem > available_resources.cpu_mem_gb or
            total_gpu_sm > available_resources.gpu_sm or
            total_gpu_mem > available_resources.gpu_mem_gb):
            return None  # Resources don't fit
        
        # All nodes run in parallel - latency is max of all latencies
        # (accounting for different start times due to dependencies)
        assignments = {}
        max_end_time = current_time
        
        for data in node_data:
            start_time = data['start_time']
            end_time = start_time + data['latency']
            
            assignments[data['node_name']] = ScheduleAssignment(
                data['node_name'],
                data['config'],
                start_time,
                end_time
            )
            
            max_end_time = max(max_end_time, end_time)
        
        return {
            'assignments': assignments,
            'latency': max_end_time - current_time,
            'end_time': max_end_time
        }
    
    def _try_serial_execution(self, dag: DAGAnalyzer,
                             nodes: List[str],
                             config_assignment: Dict[str, Optional[Tuple[ResourceConfig, float]]],
                             current_assignments: Dict[str, ScheduleAssignment],
                             current_time: float) -> Optional[Dict]:
        """
        Try serial execution of nodes in a level.
        
        Returns:
            Dict with 'assignments', 'latency', 'end_time' if feasible, None otherwise
        """
        assignments = {}
        time = current_time
        
        for node_name in nodes:
            config_data = config_assignment[node_name]
            if config_data is None:
                continue
            
            config, latency = config_data
            
            # Serial execution: wait for predecessors, then run
            pred_end_time = self._get_predecessor_end_time(dag, node_name, current_assignments)
            start_time = max(time, pred_end_time)
            end_time = start_time + latency
            
            assignments[node_name] = ScheduleAssignment(
                node_name,
                config,
                start_time,
                end_time
            )
            
            # Next node starts after this one
            time = end_time
        
        return {
            'assignments': assignments,
            'latency': time - current_time,
            'end_time': time
        }
    
    def _get_predecessor_end_time(self, dag: DAGAnalyzer, 
                                  node_name: str,
                                  assignments: Dict[str, ScheduleAssignment]) -> float:
        """
        Get the latest end time of all predecessors.
        
        Node can't start until all predecessors finish.
        """
        node = dag.get_node(node_name)
        
        if not node.predecessors:
            return 0.0
        
        max_end = 0.0
        for pred_name in node.predecessors:
            if pred_name in assignments:
                max_end = max(max_end, assignments[pred_name].end_time)
        
        return max_end


if __name__ == "__main__":
    import json
    
    print("Testing BruteForceScheduler")
    print("=" * 60)
    
    # Load data
    base_path = Path(__file__).parent.parent.parent
    
    with open(base_path / "data/tasks_augmented.json") as f:
        tasks_data = json.load(f)
        tasks = tasks_data['data']  # Extract data list
    
    with open(base_path / "data/system.json") as f:
        system = json.load(f)
    
    # Initialize components
    profiler = ResourceProfiler(str(base_path / "data/profiling.csv"))
    scheduler = BruteForceScheduler(profiler)
    
    # Find a chain task
    print("\nTest 1: Chain task")
    print("-" * 60)
    chain_task = next((t for t in tasks if t['type'] == 'chain'), None)
    
    if chain_task:
        print(f"Task ID: {chain_task['id']}")
        print(f"Task structure keys: {list(chain_task.keys())}")
        
        # Parse DAG
        dag = DAGAnalyzer(chain_task)
        levels = dag.topological_sort()
        
        print(f"\nTopological levels: {len(levels)}")
        for i, level in enumerate(levels):
            print(f"  Level {i}: {level}")
        
        # Schedule with high resources
        available = ResourceConfig(
            cpu_core=16,
            cpu_mem_gb=32.0,
            gpu_sm=100,
            gpu_mem_gb=16.0
        )
        
        print(f"\nScheduling with resources: {available}")
        plan = scheduler.schedule(dag, available)
        
        print(f"\nExecution plan:")
        print(f"  Total latency: {plan.total_latency:.2f}s")
        print(f"  Assignments:")
        for node_name in sorted(plan.assignments.keys()):
            assignment = plan.assignments[node_name]
            print(f"    {assignment}")
    else:
        print("No chain task found. Trying single_with_start task instead...")
        single_task = next((t for t in tasks if t['type'] == 'single_with_start'), None)
        if single_task:
            print(f"Task ID: {single_task['id']}")
            dag = DAGAnalyzer(single_task)
            levels = dag.topological_sort()
            
            print(f"\nTopological levels: {len(levels)}")
            for i, level in enumerate(levels):
                print(f"  Level {i}: {level}")
            
            available = ResourceConfig(
                cpu_core=16,
                cpu_mem_gb=32.0,
                gpu_sm=100,
                gpu_mem_gb=16.0
            )
            
            print(f"\nScheduling with resources: {available}")
            plan = scheduler.schedule(dag, available)
            
            print(f"\nExecution plan:")
            print(f"  Total latency: {plan.total_latency:.2f}s")
            print(f"  Assignments:")
            for node_name in sorted(plan.assignments.keys()):
                assignment = plan.assignments[node_name]
                print(f"    {assignment}")
    
    # Test with a merged task
    print("\n" + "=" * 60)
    print("\nTest 2: Merged task (parallel branches)")
    print("-" * 60)
    merged_task = next((t for t in tasks if t['type'] == 'merged'), None)
    
    if merged_task:
        print(f"Task ID: {merged_task['id']}")
        
        # Parse DAG
        dag = DAGAnalyzer(merged_task)
        levels = dag.topological_sort()
        
        print(f"\nTopological levels: {len(levels)}")
        for i, level in enumerate(levels):
            print(f"  Level {i}: {level}")
        
        # Schedule with medium resources (should force serial execution)
        available = ResourceConfig(
            cpu_core=8,
            cpu_mem_gb=16.0,
            gpu_sm=50,
            gpu_mem_gb=8.0
        )
        
        print(f"\nScheduling with resources: {available}")
        plan = scheduler.schedule(dag, available)
        
        print(f"\nExecution plan:")
        print(f"  Total latency: {plan.total_latency:.2f}s")
        print(f"  Assignments:")
        for node_name in sorted(plan.assignments.keys()):
            assignment = plan.assignments[node_name]
            print(f"    {assignment}")
        
        # Compare with high resources (might allow parallel)
        print("\n" + "-" * 60)
        available_high = ResourceConfig(
            cpu_core=16,
            cpu_mem_gb=32.0,
            gpu_sm=100,
            gpu_mem_gb=16.0
        )
        
        print(f"\nScheduling with high resources: {available_high}")
        plan_high = scheduler.schedule(dag, available_high)
        
        print(f"\nExecution plan:")
        print(f"  Total latency: {plan_high.total_latency:.2f}s")
        print(f"  Speedup: {plan.total_latency / plan_high.total_latency:.2f}x")
    else:
        print("No merged task found in dataset.")
