#!/usr/bin/env python3
"""
Script to analyze tasks.json and compute statistics on in-degree/out-degree combinations.

This script loads the tasks.json file, creates ToolDAG objects, and computes statistics
about the distribution of nodes based on their in-degree and out-degree.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Tuple, List
from collections import defaultdict

# Add parent directory to path to import models
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import ToolDAG


def load_tasks(file_path: str) -> List[ToolDAG]:
    """
    Load tasks from JSON file and convert to ToolDAG objects.
    
    Args:
        file_path: Path to tasks.json file
        
    Returns:
        List of ToolDAG objects
    """
    print(f"Loading tasks from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    tasks = [ToolDAG.from_dict(task_data) for task_data in data['data']]
    print(f"Loaded {len(tasks)} tasks successfully.\n")
    return tasks


def categorize_degree(degree: int) -> str:
    """
    Categorize a degree value into bins: 0, 1, 2, 3, 4+
    
    Args:
        degree: The degree value
        
    Returns:
        String representation of the category
    """
    if degree >= 4:
        return "4+"
    return str(degree)


def compute_degree_statistics(tasks: List[ToolDAG]) -> Dict[Tuple[str, str], int]:
    """
    Compute statistics on (in_degree, out_degree) combinations.
    
    Groups degrees into categories: 0, 1, 2, 3, 4+
    
    Args:
        tasks: List of ToolDAG objects
        
    Returns:
        Dictionary mapping (in_degree_category, out_degree_category) to count of nodes
    """
    stats = defaultdict(int)
    
    for task in tasks:
        degree_pairs = task.get_all_node_degree_pairs()
        for in_deg, out_deg in degree_pairs:
            in_cat = categorize_degree(in_deg)
            out_cat = categorize_degree(out_deg)
            stats[(in_cat, out_cat)] += 1
    
    return dict(stats)


def print_statistics(tasks: List[ToolDAG], stats: Dict[Tuple[str, str], int]):
    """
    Print formatted statistics.
    
    Args:
        tasks: List of ToolDAG objects
        stats: Statistics dictionary
    """
    print("=" * 80)
    print("TASK STATISTICS")
    print("=" * 80)
    print(f"Total number of tasks: {len(tasks)}")
    
    # Count total nodes across all tasks
    total_nodes = sum(len(task.tool_nodes) for task in tasks)
    print(f"Total number of nodes: {total_nodes}")
    
    # Count total links across all tasks
    total_links = sum(len(task.tool_links) for task in tasks)
    print(f"Total number of links: {total_links}\n")
    
    print("=" * 80)
    print("NODE DEGREE DISTRIBUTION")
    print("=" * 80)
    print(f"{'In-Degree':<12} {'Out-Degree':<12} {'Count':<10}")
    print("-" * 80)
    
    # Sort by in-degree category, then out-degree category
    degree_order = ["0", "1", "2", "3", "4+"]
    sorted_stats = sorted(
        stats.items(),
        key=lambda x: (degree_order.index(x[0][0]), degree_order.index(x[0][1]))
    )
    
    for (in_deg, out_deg), count in sorted_stats:
        print(f"{in_deg:<12} {out_deg:<12} {count:<10}")
    
    print("=" * 80)
    
    # Summary by in-degree only
    print("\nSummary by In-Degree:")
    in_deg_summary = defaultdict(int)
    for (in_deg, _), count in stats.items():
        in_deg_summary[in_deg] += count
    
    for deg in degree_order:
        if deg in in_deg_summary:
            print(f"  In-Degree {deg}: {in_deg_summary[deg]} nodes")
    
    # Summary by out-degree only
    print("\nSummary by Out-Degree:")
    out_deg_summary = defaultdict(int)
    for (_, out_deg), count in stats.items():
        out_deg_summary[out_deg] += count
    
    for deg in degree_order:
        if deg in out_deg_summary:
            print(f"  Out-Degree {deg}: {out_deg_summary[deg]} nodes")
    
    print("\n" + "=" * 80)


def main():
    """Main function to run the analysis."""
    # Determine paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    tasks_file = project_root / "data" / "tasks.json"
    
    if not tasks_file.exists():
        print(f"Error: {tasks_file} not found!")
        sys.exit(1)
    
    # Load tasks
    tasks = load_tasks(str(tasks_file))
    
    # Compute statistics
    print("Computing degree statistics...")
    stats = compute_degree_statistics(tasks)
    print("Done.\n")
    
    # Print results
    print_statistics(tasks, stats)


if __name__ == "__main__":
    main()
