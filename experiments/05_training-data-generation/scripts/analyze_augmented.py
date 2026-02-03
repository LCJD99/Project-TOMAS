#!/usr/bin/env python3
"""
Analyze augmented tasks dataset.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import ToolDAG


def main():
    """Analyze the augmented dataset."""
    data_file = Path(__file__).parent.parent / "data" / "tasks_augmented_test.json"
    
    if not data_file.exists():
        print(f"Error: {data_file} not found!")
        print("Please run augment_tasks.py first.")
        sys.exit(1)
    
    print(f"Loading augmented tasks from {data_file}...")
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("\n" + "=" * 80)
    print("AUGMENTED DATASET METADATA")
    print("=" * 80)
    if 'metadata' in data:
        for key, value in data['metadata'].items():
            print(f"{key}: {value}")
    
    tasks = [ToolDAG.from_dict(task_data) for task_data in data['data']]
    
    print("\n" + "=" * 80)
    print("TASK STATISTICS")
    print("=" * 80)
    print(f"Total tasks: {len(tasks)}")
    
    # Categorize by type
    type_counts = defaultdict(int)
    for task in tasks:
        task_type = task.type
        type_counts[task_type] += 1
    
    print("\nTask types:")
    for task_type, count in sorted(type_counts.items()):
        print(f"  {task_type}: {count}")
    
    # Analyze merged tasks specifically
    merged_tasks = [t for t in tasks if t.type == 'merged']
    print(f"\nMerged tasks analysis:")
    print(f"  Total merged tasks: {len(merged_tasks)}")
    
    if merged_tasks:
        avg_nodes = sum(len(t.tool_nodes) for t in merged_tasks) / len(merged_tasks)
        avg_links = sum(len(t.tool_links) for t in merged_tasks) / len(merged_tasks)
        min_nodes = min(len(t.tool_nodes) for t in merged_tasks)
        max_nodes = max(len(t.tool_nodes) for t in merged_tasks)
        
        print(f"  Average nodes per merged task: {avg_nodes:.2f}")
        print(f"  Average links per merged task: {avg_links:.2f}")
        print(f"  Min nodes: {min_nodes}, Max nodes: {max_nodes}")
    
    # Overall degree statistics
    print("\n" + "=" * 80)
    print("DEGREE DISTRIBUTION (ALL TASKS)")
    print("=" * 80)
    
    degree_stats = defaultdict(int)
    for task in tasks:
        degree_pairs = task.get_all_node_degree_pairs()
        for in_deg, out_deg in degree_pairs:
            in_cat = str(in_deg) if in_deg < 4 else "4+"
            out_cat = str(out_deg) if out_deg < 4 else "4+"
            degree_stats[(in_cat, out_cat)] += 1
    
    print(f"{'In-Degree':<12} {'Out-Degree':<12} {'Count':<10}")
    print("-" * 80)
    
    degree_order = ["0", "1", "2", "3", "4+"]
    sorted_stats = sorted(
        degree_stats.items(),
        key=lambda x: (degree_order.index(x[0][0]), degree_order.index(x[0][1]))
    )
    
    for (in_deg, out_deg), count in sorted_stats:
        print(f"{in_deg:<12} {out_deg:<12} {count:<10}")
    
    print("=" * 80)
    
    # Show sample merged task
    if merged_tasks:
        print("\n" + "=" * 80)
        print("SAMPLE MERGED TASK")
        print("=" * 80)
        sample = merged_tasks[0]
        print(f"ID: {sample.id}")
        print(f"N_tools: {sample.n_tools}")
        print(f"\nTool nodes:")
        for node in sample.tool_nodes:
            print(f"  - {node.task}")
        print(f"\nTool links:")
        for link in sample.tool_links:
            print(f"  - {link.source} -> {link.target}")
        
        degrees = sample.get_node_degrees()
        print(f"\nNode degrees:")
        for task_name, (in_deg, out_deg) in degrees.items():
            print(f"  - {task_name}: in={in_deg}, out={out_deg}")
        print("=" * 80)


if __name__ == "__main__":
    main()
