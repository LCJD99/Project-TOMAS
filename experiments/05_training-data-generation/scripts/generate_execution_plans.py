#!/usr/bin/env python3
"""
Generate execution plans for all tasks.

Usage:
    python scripts/generate_execution_plans.py \\
        --tasks data/tasks_augmented.json \\
        --system data/system.json \\
        --profiling data/profiling.csv \\
        --output data/execution_plans.json \\
        --batch-dir data/gt \\
        --batch-size 100 \\
        --scenarios 3 \\
        --seed 42
"""

import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.planning.resource_profiler import ResourceProfiler
from src.planning.plan_generator import PlanGenerator


def main():
    parser = argparse.ArgumentParser(
        description='Generate execution plans for task scheduling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--tasks',
        type=str,
        required=True,
        help='Path to tasks_augmented.json'
    )
    
    parser.add_argument(
        '--system',
        type=str,
        required=True,
        help='Path to system.json with resource limits'
    )
    
    parser.add_argument(
        '--profiling',
        type=str,
        required=True,
        help='Path to profiling.csv with performance data'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output path for combined execution_plans.json'
    )
    
    parser.add_argument(
        '--batch-dir',
        type=str,
        default=None,
        help='Directory for batched output files (optional)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Number of tasks per batch file (default: 100)'
    )
    
    parser.add_argument(
        '--scenarios',
        type=int,
        default=3,
        help='Number of resource scenarios per task (default: 3)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for scenario generation (default: 42)'
    )
    
    parser.add_argument(
        '--max-tasks',
        type=int,
        default=None,
        help='Maximum number of tasks to process (for testing)'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of parallel workers (default: 1). Use 8 for multi-core processing.'
    )
    
    parser.add_argument(
        '--task-filter',
        type=str,
        choices=['all', 'single', 'multi', 'merged'],
        default='all',
        help='Filter tasks by type: "single" (single_with_start), "multi" (chain/merged/dag), "merged" (merged only), "all" (default: all)'
    )
    
    args = parser.parse_args()
    
    # Load input data
    print("Loading input data...")
    
    with open(args.tasks) as f:
        tasks_data = json.load(f)
        all_tasks = tasks_data['data']
    
    with open(args.system) as f:
        system_state = json.load(f)
    
    # Filter tasks by type
    if args.task_filter == 'single':
        tasks = [t for t in all_tasks if t.get('type') == 'single_with_start']
        print(f"Loaded {len(all_tasks)} tasks, filtered to {len(tasks)} single tasks")
    elif args.task_filter == 'multi':
        tasks = [t for t in all_tasks if t.get('type') != 'single_with_start']
        print(f"Loaded {len(all_tasks)} tasks, filtered to {len(tasks)} multi-step tasks")
    elif args.task_filter == 'merged':
        tasks = [t for t in all_tasks if t.get('type') == 'merged']
        print(f"Loaded {len(all_tasks)} tasks, filtered to {len(tasks)} merged tasks")
    else:
        tasks = all_tasks
        print(f"Loaded {len(tasks)} tasks")
    
    print(f"System resources: {system_state}")
    
    # Initialize profiler
    print(f"\nLoading profiling data from {args.profiling}...")
    profiler = ResourceProfiler(args.profiling)
    
    # Create plan generator
    print(f"\nInitializing plan generator...")
    print(f"  Scenarios per task: {args.scenarios}")
    print(f"  Random seed: {args.seed}")
    print(f"  Parallel workers: {args.workers}")
    
    generator = PlanGenerator(
        profiler=profiler,
        system_state=system_state,
        num_scenarios=args.scenarios,
        seed=args.seed
    )
    
    # Generate plans
    print(f"\nGenerating execution plans...")
    
    max_tasks = args.max_tasks if args.max_tasks is not None else len(tasks)
    print(f"Processing {max_tasks} tasks...")
    
    all_plans = generator.generate_all_plans(
        tasks, 
        max_tasks=max_tasks,
        num_workers=args.workers,
        profiler_path=args.profiling if args.workers > 1 else None
    )
    
    print(f"\nGeneration complete!")
    print(f"  Tasks processed: {len(all_plans)}")
    print(f"  Total plans: {sum(len(p['plans']) for p in all_plans)}")
    
    # Save combined output
    print(f"\nSaving combined output to {args.output}...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_plans, f, indent=2)
    
    print(f"Saved {len(all_plans)} tasks to {args.output}")
    
    # Save batched output if requested
    if args.batch_dir:
        print(f"\nCreating batched output in {args.batch_dir}...")
        batch_dir = Path(args.batch_dir)
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        batch_num = 0
        num_batches = (len(all_plans) + args.batch_size - 1) // args.batch_size
        
        with tqdm(total=num_batches, desc="Writing batches", unit="batch") as pbar:
            for i in range(0, len(all_plans), args.batch_size):
                batch = all_plans[i:i + args.batch_size]
                batch_file = batch_dir / f"batch_{batch_num:03d}.json"
                
                with open(batch_file, 'w') as f:
                    json.dump(batch, f, indent=2)
                
                pbar.set_postfix({'file': batch_file.name, 'tasks': len(batch)})
                pbar.update(1)
                batch_num += 1
        
        print(f"\nCreated {batch_num} batch files")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    
    total_plans = sum(len(p['plans']) for p in all_plans)
    avg_plans = total_plans / len(all_plans) if all_plans else 0
    
    print(f"Tasks processed: {len(all_plans)}")
    print(f"Total plans generated: {total_plans}")
    print(f"Average plans per task: {avg_plans:.2f}")
    
    # Count tasks by type
    task_types = {}
    for task_plan in all_plans:
        task_id = task_plan['task_id']
        # Find original task
        task = next((t for t in tasks if t['id'] == task_id), None)
        if task:
            task_type = task.get('type', 'unknown')
            task_types[task_type] = task_types.get(task_type, 0) + 1
    
    print("\nTasks by type:")
    for task_type, count in sorted(task_types.items()):
        print(f"  {task_type}: {count}")
    
    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
