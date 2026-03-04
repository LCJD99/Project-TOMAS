"""
Execution language generator.

Generates execution sequences following EBNF grammar with:
- <EXEC>, <WAIT>, <FINISH> operators
- Tool tokens with resource configurations
- Reference variables (<REF_N>)
- Data dependencies (both static and dynamic)
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Optional

# Handle both module import and standalone execution
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.planning.scheduler import ExecutionPlan, ScheduleAssignment
    from src.planning.dag_analyzer import DAGAnalyzer
    from src.planning.resource_profiler import ResourceConfig
    from src.planning import tool_mapper
    from src.schema.token_schema import generate_token_name, format_token_for_model, RESOURCE_BINS
else:
    from .scheduler import ExecutionPlan, ScheduleAssignment
    from .dag_analyzer import DAGAnalyzer
    from .resource_profiler import ResourceConfig
    from . import tool_mapper
    from ..schema.token_schema import generate_token_name, format_token_for_model, RESOURCE_BINS


class ExecutionLanguageGenerator:
    """Generates execution language from execution plan and DAG."""
    
    def __init__(self, dag: DAGAnalyzer, plan: ExecutionPlan):
        """
        Initialize generator.
        
        Args:
            dag: Task DAG structure
            plan: Execution plan with resource assignments
        """
        self.dag = dag
        self.plan = plan
        
        # Map node names to reference IDs
        self.node_to_ref: Dict[str, int] = {}
        
        # Track which nodes have been assigned refs
        self._assign_references()
    
    def _assign_references(self):
        """Assign <REF_N> IDs to non-virtual nodes in execution order."""
        ref_id = 0
        
        # Sort nodes by start time to get execution order
        sorted_nodes = sorted(
            self.plan.assignments.items(),
            key=lambda x: x[1].start_time
        )
        
        for node_name, assignment in sorted_nodes:
            # Skip virtual nodes
            if tool_mapper.is_virtual_node(node_name):
                continue
            
            self.node_to_ref[node_name] = ref_id
            ref_id += 1
    
    def generate(self) -> str:
        """
        Generate complete execution sequence.
        
        Returns:
            Execution sequence string following EBNF grammar
        """
        statements = []
        
        # Sort nodes by start time
        sorted_assignments = sorted(
            self.plan.assignments.items(),
            key=lambda x: x[1].start_time
        )
        
        for node_name, assignment in sorted_assignments:
            # Skip virtual nodes like START
            if tool_mapper.is_virtual_node(node_name):
                continue
            
            # Generate statement for this node
            stmt = self._generate_statement(node_name, assignment)
            statements.append(stmt)
        
        # Generate FINISH statement with output nodes
        finish_stmt = self._generate_finish_statement()
        
        # Combine all statements
        program = '\n'.join(statements)
        if finish_stmt:
            program += '\n' + finish_stmt
        
        return program
    
    def _generate_statement(self, node_name: str, assignment: ScheduleAssignment) -> str:
        """
        Generate execution statement for a node.
        
        Format: <REF_N> = [<WAIT> <REF_X> ...] <EXEC> <TOOL_TOKEN>(...args...)
        
        Args:
            node_name: Node name
            assignment: Resource assignment for node
            
        Returns:
            Execution statement string
        """
        # Get reference ID for this node
        ref_id = self.node_to_ref[node_name]
        ref_var = f"<REF_{ref_id}>"
        
        # Get node data
        node = self.dag.get_node(node_name)
        
        # Generate tool token with resource configuration
        tool_token = self._generate_tool_token(node_name, assignment.config)
        
        # Generate arguments (mix of static values and references)
        args = self._generate_arguments(node)
        
        # Check if we need <WAIT> for dependencies
        wait_clause = self._generate_wait_clause(node_name, assignment)
        
        # Build statement
        if wait_clause:
            stmt = f"{ref_var} = {wait_clause} <EXEC> {tool_token}({args})"
        else:
            stmt = f"{ref_var} = <EXEC> {tool_token}({args})"
        
        return stmt
    
    def _generate_tool_token(self, node_name: str, config: ResourceConfig) -> str:
        """
        Generate tool token with resource configuration.
        
        Format: <TOOL_ABBREV_SMALL_CPU_CORE_CPU_MEM_GPU_SM_GPU_MEM>
        
        Args:
            node_name: Node name
            config: Resource configuration
            
        Returns:
            Formatted tool token
        """
        # Get profiling tool name
        profiling_tool = tool_mapper.task_to_profiling_name(node_name)
        
        # Map resource values to levels (low/medium/high)
        cpu_core_level = self._value_to_level('cpu_core', config.cpu_core)
        cpu_mem_level = self._value_to_level('cpu_mem_gb', config.cpu_mem_gb)
        gpu_sm_level = self._value_to_level('gpu_sm', config.gpu_sm)
        gpu_mem_level = self._value_to_level('gpu_mem_gb', config.gpu_mem_gb)
        
        # Generate token (always use 'small' input size per requirements)
        token = generate_token_name(
            profiling_tool,
            'small',  # All tasks use small input size
            cpu_core_level,
            cpu_mem_level,
            gpu_sm_level,
            gpu_mem_level
        )
        
        return format_token_for_model(token)
    
    def _value_to_level(self, resource_type: str, value: float) -> str:
        """
        Map resource value to level (low/medium/high).
        
        Args:
            resource_type: Resource type key
            value: Resource value
            
        Returns:
            Level string ('low', 'medium', or 'high')
        """
        bins = RESOURCE_BINS[resource_type]
        
        # Find closest bin
        if value <= bins['low']:
            return 'low'
        elif value <= bins['medium']:
            return 'medium'
        else:
            return 'high'
    
    def _generate_arguments(self, node) -> str:
        """
        Generate argument list for tool.
        
        Includes both:
        - Static arguments (e.g., "example.jpg", "What is in the image?")
        - Dynamic references (e.g., <REF_0>, <REF_1>)
        
        Args:
            node: DAG node
            
        Returns:
            Comma-separated argument string
        """
        args = []
        
        # Add static arguments from node data, skipping <node-N> placeholders
        # (predecessor outputs are appended below via <REF_x>)
        for arg in node.arguments:
            if re.match(r'^<node-\d+>$', str(arg)):
                continue
            args.append(f'"{arg}"')
        
        # Add dynamic references from predecessors (excluding START)
        for pred_name in sorted(node.predecessors):
            # Skip virtual nodes
            if tool_mapper.is_virtual_node(pred_name):
                continue
            
            # Add reference to predecessor's output
            if pred_name in self.node_to_ref:
                ref_id = self.node_to_ref[pred_name]
                args.append(f"<REF_{ref_id}>")
        
        return ', '.join(args)
    
    def _generate_wait_clause(self, node_name: str, assignment: ScheduleAssignment) -> str:
        """
        Generate <WAIT> clause if needed for resource dependencies.
        
        A node needs to wait if:
        1. It has data dependencies (handled by argument references)
        2. It needs resources to be freed from nodes that finished before it
        
        The <WAIT> clause lists nodes that must complete before this one can start,
        either for data or resource availability.
        
        Args:
            node_name: Node name
            assignment: Resource assignment
            
        Returns:
            Wait clause string or empty string
        """
        node = self.dag.get_node(node_name)
        wait_refs = set()
        
        # Add all real predecessors (data dependencies)
        for pred_name in node.predecessors:
            if not tool_mapper.is_virtual_node(pred_name):
                if pred_name in self.node_to_ref:
                    wait_refs.add(self.node_to_ref[pred_name])
        
        # Find nodes that must finish to free resources
        # (nodes that overlap in time but finished before this one starts)
        for other_name, other_assignment in self.plan.assignments.items():
            if other_name == node_name:
                continue
            
            if tool_mapper.is_virtual_node(other_name):
                continue
            
            # Check if other node's execution overlaps with when we need resources
            # We need to wait for nodes that are still running when we want to start
            if (other_assignment.start_time < assignment.start_time and 
                other_assignment.end_time >= assignment.start_time):
                # This node is still running when we want to start
                if other_name in self.node_to_ref:
                    wait_refs.add(self.node_to_ref[other_name])
        
        if not wait_refs:
            return ""
        
        # Sort refs for consistency
        sorted_refs = sorted(wait_refs)
        ref_list = ' '.join(f"<REF_{r}>" for r in sorted_refs)
        
        return f"<WAIT> {ref_list}"
    
    def _generate_finish_statement(self) -> str:
        """
        Generate <FINISH> statement with output nodes.

        Output nodes are nodes with out-degree 0 (no successors).

        Returns:
            Finish statement string
        """
        # Get output nodes from DAG
        output_node_names = self.dag.get_output_nodes()

        # Convert to reference IDs
        output_refs = []
        for node_name in output_node_names:
            if node_name in self.node_to_ref:
                output_refs.append(self.node_to_ref[node_name])

        # Sort for consistency
        output_refs.sort()

        if not output_refs:
            return "<FINISH>"

        ref_list = ' '.join(f"<REF_{r}>" for r in output_refs)
        return f"<FINISH> {ref_list}"

    def generate_json(self) -> Dict:
        """
        Generate a structured JSON representation of the execution plan.

        Semantically equivalent to the PLAN_START string but in a machine-
        readable dict format, useful for baseline comparisons.

        Schema::

            {
                "nodes": [
                    {
                        "index": 0,
                        "tool": "object_detection",   # profiling snake_case name
                        "arguments": ["example.jpg"], # static args + <node_N> for data deps
                        "cpu_core": 8,
                        "cpu_memory": 16.0,
                        "gpu_sm": 60,
                        "gpu_memory": 8.0
                    },
                    ...
                ],
                "links": [          # ALL execution-order edges (data dep + resource-mutex)
                    {"from": 0, "to": 1},
                    ...
                ]
            }

        ``links`` mirrors the <WAIT> semantics in PLAN_START: a link (A→B)
        exists whenever B would emit ``<WAIT> <REF_A>`` in the text form.
        Parallel nodes (no WAIT between them) have no link.

        Returns:
            Dict ready for JSON serialisation.
        """
        # Sort nodes by start time (same order as generate())
        sorted_assignments = sorted(
            self.plan.assignments.items(),
            key=lambda x: x[1].start_time
        )

        nodes = []
        for node_name, assignment in sorted_assignments:
            if tool_mapper.is_virtual_node(node_name):
                continue

            node = self.dag.get_node(node_name)
            idx = self.node_to_ref[node_name]
            profiling_tool = tool_mapper.task_to_profiling_name(node_name) or node_name

            # Build arguments: static values (skip <node-N> placeholders), then
            # append <node_N> references for each real predecessor (data deps).
            args: List[str] = []
            for arg in node.arguments:
                if re.match(r'^<node-\d+>$', str(arg)):
                    continue
                args.append(str(arg))

            for pred_name in sorted(node.predecessors):
                if tool_mapper.is_virtual_node(pred_name):
                    continue
                if pred_name in self.node_to_ref:
                    args.append(f"<node_{self.node_to_ref[pred_name]}>")

            nodes.append({
                'index': idx,
                'tool': profiling_tool,
                'arguments': args,
                'cpu_core': assignment.config.cpu_core,
                'cpu_memory': assignment.config.cpu_mem_gb,
                'gpu_sm': assignment.config.gpu_sm,
                'gpu_memory': assignment.config.gpu_mem_gb,
            })

        # Build links: same logic as _generate_wait_clause — one link per
        # (predecessor, node) pair where the predecessor's end_time causes
        # the node to wait (data dep or resource-mutex serialisation).
        link_set: set = set()

        for node_name, assignment in sorted_assignments:
            if tool_mapper.is_virtual_node(node_name):
                continue

            node = self.dag.get_node(node_name)
            to_idx = self.node_to_ref[node_name]
            wait_from: set = set()

            # Data dependencies
            for pred_name in node.predecessors:
                if not tool_mapper.is_virtual_node(pred_name):
                    if pred_name in self.node_to_ref:
                        wait_from.add(self.node_to_ref[pred_name])

            # Resource-mutex serialisation (overlapping time windows)
            for other_name, other_assignment in self.plan.assignments.items():
                if other_name == node_name or tool_mapper.is_virtual_node(other_name):
                    continue
                if (other_assignment.start_time < assignment.start_time and
                        other_assignment.end_time >= assignment.start_time):
                    if other_name in self.node_to_ref:
                        wait_from.add(self.node_to_ref[other_name])

            for from_idx in wait_from:
                link_set.add((from_idx, to_idx))

        links = [{'from': f, 'to': t} for f, t in sorted(link_set)]

        return {'nodes': nodes, 'links': links}


if __name__ == "__main__":
    import json
    from src.planning.resource_profiler import ResourceProfiler
    from src.planning.scheduler import BruteForceScheduler
    
    print("Testing ExecutionLanguageGenerator")
    print("=" * 60)
    
    # Load data
    base_path = Path(__file__).parent.parent.parent
    
    with open(base_path / "data/tasks_augmented.json") as f:
        tasks_data = json.load(f)
        tasks = tasks_data['data']
    
    # Initialize components
    profiler = ResourceProfiler(str(base_path / "data/profiling.csv"))
    scheduler = BruteForceScheduler(profiler)
    
    # Test with a chain task
    print("\nTest 1: Chain task")
    print("-" * 60)
    chain_task = next((t for t in tasks if t['type'] == 'chain'), None)
    
    if chain_task:
        print(f"Task ID: {chain_task['id']}")
        
        # Parse DAG and schedule
        dag = DAGAnalyzer(chain_task)
        
        available = ResourceConfig(
            cpu_core=16,
            cpu_mem_gb=32.0,
            gpu_sm=100,
            gpu_mem_gb=16.0
        )
        
        plan = scheduler.schedule(dag, available)
        
        print(f"\nExecution plan (latency: {plan.total_latency:.2f}s):")
        for node_name in sorted(plan.assignments.keys()):
            assignment = plan.assignments[node_name]
            print(f"  {assignment}")
        
        # Generate execution language
        generator = ExecutionLanguageGenerator(dag, plan)
        program = generator.generate()
        
        print(f"\nGenerated execution sequence:")
        print(program)
    
    # Test with a merged task
    print("\n" + "=" * 60)
    print("\nTest 2: Merged task (parallel branches)")
    print("-" * 60)
    merged_task = next((t for t in tasks if t['type'] == 'merged'), None)
    
    if merged_task:
        print(f"Task ID: {merged_task['id']}")
        
        # Parse DAG
        dag = DAGAnalyzer(merged_task)
        
        # Schedule with medium resources (serial execution)
        available = ResourceConfig(
            cpu_core=8,
            cpu_mem_gb=16.0,
            gpu_sm=50,
            gpu_mem_gb=8.0
        )
        
        plan = scheduler.schedule(dag, available)
        
        print(f"\nExecution plan (latency: {plan.total_latency:.2f}s):")
        for node_name in sorted(plan.assignments.keys()):
            assignment = plan.assignments[node_name]
            print(f"  {assignment}")
        
        # Generate execution language
        generator = ExecutionLanguageGenerator(dag, plan)
        program = generator.generate()
        
        print(f"\nGenerated execution sequence:")
        print(program)
        
        # Compare with high resources (parallel execution)
        print("\n" + "-" * 60)
        print("With high resources (parallel execution):")
        
        available_high = ResourceConfig(
            cpu_core=16,
            cpu_mem_gb=32.0,
            gpu_sm=100,
            gpu_mem_gb=16.0
        )
        
        plan_high = scheduler.schedule(dag, available_high)
        
        print(f"\nExecution plan (latency: {plan_high.total_latency:.2f}s):")
        for node_name in sorted(plan_high.assignments.keys()):
            assignment = plan_high.assignments[node_name]
            print(f"  {assignment}")
        
        # Generate execution language
        generator_high = ExecutionLanguageGenerator(dag, plan_high)
        program_high = generator_high.generate()
        
        print(f"\nGenerated execution sequence:")
        print(program_high)
