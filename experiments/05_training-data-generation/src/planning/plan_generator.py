"""
Plan generator - integrates all components to generate execution plans.

Orchestrates:
1. Resource scenario generation
2. DAG analysis
3. Scheduling
4. Execution language generation

Outputs execution plans in the required JSON format.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Handle both module import and standalone execution
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.planning.resource_profiler import ResourceProfiler, ResourceConfig
    from src.planning.dag_analyzer import DAGAnalyzer
    from src.planning.scheduler import BruteForceScheduler
    from src.planning.execution_language import ExecutionLanguageGenerator
    from src.planning.resource_scenarios import ScenarioGenerator
else:
    from .resource_profiler import ResourceProfiler, ResourceConfig
    from .dag_analyzer import DAGAnalyzer
    from .scheduler import BruteForceScheduler
    from .execution_language import ExecutionLanguageGenerator
    from .resource_scenarios import ScenarioGenerator


class PlanGenerator:
    """
    Orchestrates execution plan generation for tasks.
    
    For each task:
    1. Generates N resource scenarios
    2. For each scenario, schedules the task
    3. Generates execution language
    4. Formats output JSON
    """
    
    def __init__(self, profiler: ResourceProfiler, system_state: Dict[str, float],
                 num_scenarios: int = 3, seed: int = 42):
        """
        Initialize plan generator.
        
        Args:
            profiler: Resource profiler with performance data
            system_state: Maximum system resources
            num_scenarios: Number of scenarios per task
            seed: Random seed for scenario generation
        """
        self.profiler = profiler
        self.system_state = system_state
        self.num_scenarios = num_scenarios
        self.seed = seed
        
        # Initialize scheduler and scenario generator
        self.scheduler = BruteForceScheduler(profiler)
        self.scenario_generator = ScenarioGenerator(system_state, seed=seed)
    
    @staticmethod
    def _generate_task_plans_worker(args):
        """
        Worker function for parallel task processing.
        
        This static method is used by ProcessPoolExecutor to process tasks in parallel.
        It recreates the necessary objects in each worker process.
        
        Args:
            args: Tuple of (task, profiler_path, system_state, num_scenarios, seed)
            
        Returns:
            Dictionary with task_id and list of plans, or None if error
        """
        task, profiler_path, system_state, num_scenarios, worker_seed = args
        
        try:
            # Recreate profiler and generator in worker process (suppress verbose output)
            profiler = ResourceProfiler(profiler_path, verbose=False)
            scheduler = BruteForceScheduler(profiler)
            scenario_generator = ScenarioGenerator(system_state, seed=worker_seed)
            
            task_id = task['id']
            
            # Parse DAG
            dag = DAGAnalyzer(task)
            
            # Generate resource scenarios
            scenarios = scenario_generator.generate_scenarios(num_scenarios)
            
            # Generate plan for each scenario
            plans = []
            for scenario_idx, scenario in enumerate(scenarios):
                try:
                    # Schedule task with this scenario's resources
                    plan = scheduler.schedule(dag, scenario.available)
                    
                    # Generate execution language
                    lang_generator = ExecutionLanguageGenerator(dag, plan)
                    execution_sequence = lang_generator.generate()
                    
                    # Format plan
                    plan_dict = {
                        'scenario_id': scenario_idx + 1,
                        'SYSTEM_STATE': scenario.to_dict(),
                        'USER_QUESTION': PlanGenerator._format_user_question_static(task),
                        'PLAN_START': execution_sequence,
                        'total_latency_ms': round(plan.total_latency, 2)
                    }
                    
                    plans.append(plan_dict)
                    
                except Exception as e:
                    # If scheduling fails for this scenario, skip it
                    from tqdm import tqdm
                    tqdm.write(f"Warning: Failed to schedule task {task_id} scenario {scenario_idx + 1}: {e}")
                    continue
            
            return {
                'task_id': task_id,
                'plans': plans
            }
            
        except Exception as e:
            from tqdm import tqdm
            tqdm.write(f"Error processing task {task.get('id', 'unknown')}: {e}")
            return None
    
    @staticmethod
    def _format_user_question_static(task: Dict[str, Any]) -> str:
        """
        Static version of _format_user_question for use in worker processes.
        
        Args:
            task: Task dictionary
            
        Returns:
            Formatted user question string
        """
        # Always use the instruction field when present — it contains the full
        # natural-language question for all task types (single, chain, merged, dag).
        instruction = task.get('instruction', '')
        if instruction:
            return instruction
        
        # Fallback: reconstruct from tool nodes when no instruction is available.
        tool_nodes = task.get('tool_nodes', [])
        real_nodes = [n for n in tool_nodes if n['task'] != 'START']
        
        questions = []
        for i, node in enumerate(real_nodes, 1):
            task_name = node['task']
            # Strip Tx_ prefix if present
            if '_' in task_name:
                parts = task_name.split('_', 1)
                if parts[0].startswith('T') and parts[0][1:].isdigit():
                    task_name = parts[1]
            args_str = ', '.join(f'"{arg}"' for arg in node.get('arguments', []))
            questions.append(f"{i}. {task_name}: {args_str}")
        
        return '\n'.join(questions)
    
    def generate_task_plans(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate execution plans for a single task.
        
        Args:
            task: Task dictionary from tasks_augmented.json
            
        Returns:
            Dictionary with task_id and list of plans (one per scenario)
        """
        task_id = task['id']
        
        # Parse DAG
        dag = DAGAnalyzer(task)
        
        # Generate resource scenarios
        scenarios = self.scenario_generator.generate_scenarios(self.num_scenarios)
        
        # Generate plan for each scenario with progress bar
        plans = []
        for scenario_idx, scenario in enumerate(scenarios):
            try:
                # Schedule task with this scenario's resources
                plan = self.scheduler.schedule(dag, scenario.available)
                
                # Generate execution language
                lang_generator = ExecutionLanguageGenerator(dag, plan)
                execution_sequence = lang_generator.generate()
                
                # Format plan
                plan_dict = {
                    'scenario_id': scenario_idx + 1,
                    'SYSTEM_STATE': scenario.to_dict(),
                    'USER_QUESTION': self._format_user_question(task),
                    'PLAN_START': execution_sequence,
                    'total_latency_ms': round(plan.total_latency, 2)
                }
                
                plans.append(plan_dict)
                
            except Exception as e:
                # If scheduling fails for this scenario, skip it
                # Use tqdm.write to avoid interfering with progress bar
                from tqdm import tqdm
                tqdm.write(f"Warning: Failed to schedule task {task_id} with scenario {scenario_idx + 1}: {e}")
                continue
        
        return {
            'task_id': task_id,
            'plans': plans
        }
    
    def _format_user_question(self, task: Dict[str, Any]) -> str:
        """
        Format user question from task instruction.
        
        For merged tasks, we need to create a numbered list.
        For single tasks, use the instruction as-is.
        
        Args:
            task: Task dictionary
            
        Returns:
            Formatted user question string
        """
        return self._format_user_question_static(task)
    
    def generate_all_plans(self, tasks: List[Dict[str, Any]], 
                          max_tasks: int = None,
                          num_workers: int = 1,
                          profiler_path: str = None) -> List[Dict[str, Any]]:
        """
        Generate execution plans for all tasks.
        
        Args:
            tasks: List of task dictionaries
            max_tasks: Maximum number of tasks to process (None = all)
            num_workers: Number of parallel workers (1 = sequential, >1 = parallel)
            profiler_path: Path to profiling CSV (required for parallel processing)
            
        Returns:
            List of plan dictionaries
        """
        task_limit = max_tasks if max_tasks is not None else len(tasks)
        tasks_to_process = tasks[:task_limit]
        
        if num_workers > 1:
            # Parallel processing
            if profiler_path is None:
                raise ValueError("profiler_path is required for parallel processing")
            
            return self._generate_all_plans_parallel(
                tasks_to_process, num_workers, profiler_path
            )
        else:
            # Sequential processing
            return self._generate_all_plans_sequential(tasks_to_process)
    
    def _generate_all_plans_sequential(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate plans sequentially (single-threaded).
        
        Args:
            tasks: List of task dictionaries
            
        Returns:
            List of plan dictionaries
        """
        all_plans = []
        
        # Use tqdm for progress bar
        with tqdm(total=len(tasks), desc="Generating plans", unit="task") as pbar:
            for task in tasks:
                try:
                    plans = self.generate_task_plans(task)
                    all_plans.append(plans)
                    
                    # Update progress bar with additional info
                    pbar.set_postfix({
                        'task_id': task.get('id', 'unknown')[:8],
                        'plans': len(plans['plans'])
                    })
                    pbar.update(1)
                    
                except Exception as e:
                    pbar.write(f"Error processing task {task.get('id', 'unknown')}: {e}")
                    pbar.update(1)
                    continue
        
        return all_plans
    
    def _generate_all_plans_parallel(self, tasks: List[Dict[str, Any]], 
                                     num_workers: int,
                                     profiler_path: str) -> List[Dict[str, Any]]:
        """
        Generate plans in parallel using multiple processes.
        
        Args:
            tasks: List of task dictionaries
            num_workers: Number of parallel workers
            profiler_path: Path to profiling CSV
            
        Returns:
            List of plan dictionaries (in original order)
        """
        # Prepare arguments for each task
        # Use different seeds for each task to ensure diverse scenarios
        task_args = [
            (task, profiler_path, self.system_state, self.num_scenarios, self.seed + i)
            for i, task in enumerate(tasks)
        ]
        
        all_plans = []
        
        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(self._generate_task_plans_worker, args): idx
                for idx, args in enumerate(task_args)
            }
            
            # Process results with progress bar
            with tqdm(total=len(tasks), desc="Generating plans", unit="task") as pbar:
                # Collect results as they complete
                results = [None] * len(tasks)
                
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result = future.result()
                        if result is not None:
                            results[idx] = result
                            pbar.set_postfix({
                                'task_id': result['task_id'][:8],
                                'plans': len(result['plans'])
                            })
                    except Exception as e:
                        pbar.write(f"Error in worker for task {tasks[idx].get('id', 'unknown')}: {e}")
                    finally:
                        pbar.update(1)
                
                # Filter out None results and maintain order
                all_plans = [r for r in results if r is not None]
        
        return all_plans


if __name__ == "__main__":
    print("Testing PlanGenerator")
    print("=" * 60)
    
    # Load data
    base_path = Path(__file__).parent.parent.parent
    
    with open(base_path / "data/tasks_augmented.json") as f:
        tasks_data = json.load(f)
        tasks = tasks_data['data']
    
    with open(base_path / "data/system.json") as f:
        system_state = json.load(f)
    
    # Initialize profiler
    profiler = ResourceProfiler(str(base_path / "data/profiling.csv"))
    
    # Create plan generator
    generator = PlanGenerator(
        profiler=profiler,
        system_state=system_state,
        num_scenarios=3,
        seed=42
    )
    
    # Test with first task
    print("\nTest 1: Single task")
    print("-" * 60)
    
    single_task = tasks[0]
    print(f"Task ID: {single_task['id']}")
    print(f"Type: {single_task['type']}")
    
    result = generator.generate_task_plans(single_task)
    
    print(f"\nGenerated {len(result['plans'])} plans")
    
    for plan in result['plans']:
        print(f"\nScenario {plan['scenario_id']}:")
        print(f"  System: {plan['SYSTEM_STATE']}")
        print(f"  Question: {plan['USER_QUESTION'][:100]}...")
        print(f"  Plan:\n    {plan['PLAN_START']}")
    
    # Test with merged task
    print("\n" + "=" * 60)
    print("\nTest 2: Merged task")
    print("-" * 60)
    
    merged_task = next((t for t in tasks if t['type'] == 'merged'), None)
    
    if merged_task:
        print(f"Task ID: {merged_task['id']}")
        print(f"Type: {merged_task['type']}")
        
        result = generator.generate_task_plans(merged_task)
        
        print(f"\nGenerated {len(result['plans'])} plans")
        
        for plan in result['plans']:
            print(f"\nScenario {plan['scenario_id']}:")
            print(f"  System: {plan['SYSTEM_STATE']}")
            print(f"  Question:")
            for line in plan['USER_QUESTION'].split('\n'):
                print(f"    {line}")
            print(f"  Plan:")
            for line in plan['PLAN_START'].split('\n'):
                print(f"    {line}")
    
    # Test batch processing
    print("\n" + "=" * 60)
    print("\nTest 3: Batch processing (first 5 tasks)")
    print("-" * 60)
    
    all_plans = generator.generate_all_plans(tasks, max_tasks=5)
    
    print(f"\nProcessed {len(all_plans)} tasks")
    print(f"Total plans generated: {sum(len(p['plans']) for p in all_plans)}")
    
    # Show summary
    for task_plans in all_plans:
        task_id = task_plans['task_id']
        num_plans = len(task_plans['plans'])
        print(f"  Task {task_id}: {num_plans} plans")
    
    # Save sample output
    output_file = base_path / "data/sample_execution_plans.json"
    with open(output_file, 'w') as f:
        json.dump(all_plans, f, indent=2)
    
    print(f"\nSaved sample output to: {output_file}")
