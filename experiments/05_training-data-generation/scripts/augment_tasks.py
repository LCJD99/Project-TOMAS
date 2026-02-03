#!/usr/bin/env python3
"""
Data augmentation script for Tool DAG tasks.

This script augments the task dataset by:
1. Adding a unified START node to each DAG structure
2. Randomly selecting N tasks and merging their DAGs as subgraphs
3. Connecting all subgraphs to the START node
4. Generating augmented tasks and saving to tasks_augmented.json
"""

import json
import random
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Set
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import ToolDAG


class TaskAugmentor:
    """Handles task augmentation by merging multiple DAGs."""
    
    START_NODE_TASK = "START"
    
    def __init__(self, seed: int = 42):
        """
        Initialize the augmentor.
        
        Args:
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        self.next_id = 1
    
    def add_start_node_to_dag(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add a unified START node to a single DAG.
        
        Args:
            task_data: Original task data dictionary
            
        Returns:
            Modified task data with START node added
        """
        # Create a deep copy to avoid modifying original
        augmented = json.loads(json.dumps(task_data))
        
        # Add START node to sampled_nodes
        start_sampled_node = {
            "task": self.START_NODE_TASK,
            "input-type": [],
            "output-type": ["signal"]
        }
        augmented['sampled_nodes'].insert(0, start_sampled_node)
        
        # Add START node to tool_nodes
        start_tool_node = {
            "task": self.START_NODE_TASK,
            "arguments": []
        }
        augmented['tool_nodes'].insert(0, start_tool_node)
        
        # Find all root nodes (nodes with in-degree 0)
        in_degrees = defaultdict(int)
        for link in augmented['tool_links']:
            in_degrees[link['target']] += 1
        
        # All existing nodes
        all_nodes = {node['task'] for node in augmented['tool_nodes'][1:]}  # Exclude START
        root_nodes = [node for node in all_nodes if in_degrees[node] == 0]
        
        # If no root nodes found (shouldn't happen), connect to all nodes
        if not root_nodes:
            root_nodes = list(all_nodes)
        
        # Add links from START to all root nodes
        new_links = []
        for root in root_nodes:
            new_link = {
                "source": self.START_NODE_TASK,
                "target": root
            }
            new_links.append(new_link)
            
            # Also add to sampled_links if it exists
            augmented['sampled_links'].insert(0, {
                "source": self.START_NODE_TASK,
                "target": root
            })
        
        # Insert new links at the beginning
        augmented['tool_links'] = new_links + augmented['tool_links']
        
        # Update n_tools
        augmented['n_tools'] = len(augmented['tool_nodes'])
        
        # Update type to indicate it's augmented
        if augmented['type'] == 'single':
            augmented['type'] = 'single_with_start'
        
        return augmented
    
    def merge_tasks(self, tasks_data: List[Dict[str, Any]], 
                   original_ids: List[str]) -> Dict[str, Any]:
        """
        Merge multiple tasks into a single augmented task.
        
        Args:
            tasks_data: List of task data dictionaries to merge
            original_ids: List of original task IDs for reference
            
        Returns:
            New augmented task data dictionary
        """
        # Generate new ID
        new_id = f"AUG_{self.next_id:08d}"
        self.next_id += 1
        
        # Create merged task structure
        merged = {
            "id": new_id,
            "seed": random.randint(0, 1000000),
            "n_tools": 1,  # Will be updated
            "type": "merged",
            "original_task_ids": original_ids,  # Track source tasks
            "sampled_nodes": [],
            "sampled_links": [],
            "instruction": "",
            "tool_steps": [],
            "tool_nodes": [],
            "tool_links": []
        }
        
        # Add START node
        start_sampled_node = {
            "task": self.START_NODE_TASK,
            "input-type": [],
            "output-type": ["signal"]
        }
        merged['sampled_nodes'].append(start_sampled_node)
        
        start_tool_node = {
            "task": self.START_NODE_TASK,
            "arguments": []
        }
        merged['tool_nodes'].append(start_tool_node)
        
        # Merge all subtasks
        for i, task_data in enumerate(tasks_data):
            task_prefix = f"T{i+1}"
            
            # Create local mapping for this subtask
            local_mapping = {}  # original_name -> unique_name for this subtask
            
            # Merge sampled_nodes
            for node in task_data['sampled_nodes']:
                original_name = node['task']
                
                # Create unique name with task prefix
                if len(tasks_data) > 1:  # Only add prefix if merging multiple tasks
                    unique_name = f"{task_prefix}_{original_name}"
                else:
                    unique_name = original_name
                
                local_mapping[original_name] = unique_name
                
                new_node = node.copy()
                new_node['task'] = unique_name
                merged['sampled_nodes'].append(new_node)
            
            # Merge tool_nodes
            for node in task_data['tool_nodes']:
                original_name = node['task']
                unique_name = local_mapping.get(original_name, original_name)
                
                new_node = node.copy()
                new_node['task'] = unique_name
                merged['tool_nodes'].append(new_node)
            
            # Merge sampled_links with renamed nodes
            for link in task_data['sampled_links']:
                source = local_mapping.get(link['source'], link['source'])
                target = local_mapping.get(link['target'], link['target'])
                merged['sampled_links'].append({
                    "source": source,
                    "target": target
                })
            
            # Merge tool_links with renamed nodes
            for link in task_data['tool_links']:
                source = local_mapping.get(link['source'], link['source'])
                target = local_mapping.get(link['target'], link['target'])
                merged['tool_links'].append({
                    "source": source,
                    "target": target
                })
            
            # Collect instructions (will format later)
            # Store temporarily as list
            if 'instruction_list' not in merged:
                merged['instruction_list'] = []
            merged['instruction_list'].append(task_data['instruction'])
            
            # Merge tool_steps
            for step in task_data['tool_steps']:
                merged['tool_steps'].append(f"[Subtask {i+1}] {step}")
        
        # Find root nodes in merged graph (excluding START)
        in_degrees = defaultdict(int)
        for link in merged['tool_links']:
            in_degrees[link['target']] += 1
        
        all_nodes = {node['task'] for node in merged['tool_nodes'][1:]}  # Exclude START
        root_nodes = [node for node in all_nodes if in_degrees[node] == 0]
        
        # If no root nodes, use all nodes
        if not root_nodes:
            root_nodes = list(all_nodes)
        
        # Connect START to all root nodes
        for root in root_nodes:
            merged['sampled_links'].insert(0, {
                "source": self.START_NODE_TASK,
                "target": root
            })
            merged['tool_links'].insert(0, {
                "source": self.START_NODE_TASK,
                "target": root
            })
        
        # Add initial step about START
        merged['tool_steps'].insert(0, "Step 0: Initialize the workflow with the START node")
        
        # Format instruction with numbered list
        if 'instruction_list' in merged and merged['instruction_list']:
            if len(merged['instruction_list']) == 1:
                # Single instruction, no numbering needed
                merged['instruction'] = merged['instruction_list'][0]
            else:
                # Multiple instructions, use numbered format
                instruction_parts = []
                for i, instr in enumerate(merged['instruction_list'], 1):
                    instruction_parts.append(f"{i}. {instr}")
                merged['instruction'] = "\n\n".join(instruction_parts)
            # Clean up temporary field
            del merged['instruction_list']
        
        # Update n_tools
        merged['n_tools'] = len(merged['tool_nodes'])
        
        return merged
    
    def augment_dataset(self, 
                       input_file: str,
                       output_file: str,
                       num_augmented: int = 1000,
                       merge_count: int = 3) -> None:
        """
        Generate augmented dataset.
        
        Args:
            input_file: Path to input tasks.json
            output_file: Path to output tasks_augmented.json
            num_augmented: Number of augmented tasks to generate
            merge_count: Number of tasks to merge per augmented task
        """
        print(f"Loading tasks from {input_file}...")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        original_tasks = data['data']
        print(f"Loaded {len(original_tasks)} original tasks")
        
        # Step 1: Add START node to all original tasks
        print(f"\nStep 1: Adding START node to all original tasks...")
        tasks_with_start = []
        for task in original_tasks:
            augmented = self.add_start_node_to_dag(task)
            tasks_with_start.append(augmented)
        print(f"Added START node to {len(tasks_with_start)} tasks")
        
        # Step 2: Generate merged tasks
        print(f"\nStep 2: Generating {num_augmented} merged tasks...")
        print(f"Each merged task combines {merge_count} random subtasks")
        
        augmented_tasks = []
        for i in range(num_augmented):
            # Randomly select merge_count tasks
            selected_tasks = random.sample(original_tasks, merge_count)
            selected_ids = [t['id'] for t in selected_tasks]
            
            # Merge them
            merged_task = self.merge_tasks(selected_tasks, selected_ids)
            augmented_tasks.append(merged_task)
            
            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{num_augmented} merged tasks")
        
        print(f"Generated {len(augmented_tasks)} merged tasks")
        
        # Step 3: Combine all tasks
        print(f"\nStep 3: Combining all tasks...")
        all_tasks = tasks_with_start + augmented_tasks
        
        output_data = {
            "data": all_tasks,
            "metadata": {
                "original_count": len(tasks_with_start),
                "augmented_count": len(augmented_tasks),
                "total_count": len(all_tasks),
                "merge_count": merge_count
            }
        }
        
        # Step 4: Save to file
        print(f"\nStep 4: Saving to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nAugmentation complete!")
        print(f"  Original tasks (with START): {len(tasks_with_start)}")
        print(f"  Augmented merged tasks: {len(augmented_tasks)}")
        print(f"  Total tasks: {len(all_tasks)}")
        print(f"  Output file: {output_file}")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Augment tool DAG task dataset by merging multiple tasks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='data/tasks.json',
        help='Path to input tasks JSON file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/tasks_augmented.json',
        help='Path to output augmented tasks JSON file'
    )
    
    parser.add_argument(
        '--num-augmented',
        type=int,
        default=1000,
        help='Number of augmented tasks to generate'
    )
    
    parser.add_argument(
        '--merge-count',
        type=int,
        default=3,
        help='Number of tasks to merge per augmented task'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Resolve paths relative to script location
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    input_path = project_root / args.input
    output_path = project_root / args.output
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Run augmentation
    augmentor = TaskAugmentor(seed=args.seed)
    augmentor.augment_dataset(
        input_file=str(input_path),
        output_file=str(output_path),
        num_augmented=args.num_augmented,
        merge_count=args.merge_count
    )


if __name__ == "__main__":
    main()
