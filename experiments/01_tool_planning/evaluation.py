import json
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as prfs
import click

@dataclass
class TaskParameter:
    """Represents task parameters"""
    task_id: str
    params: Dict[str, Any]

@dataclass
class TaskNode:
    """Represents a task node"""
    node_id: str
    tool_name: str
    parameters: List[TaskParameter]
    dependencies: List[str]
    original_task_ids: List[str]

@dataclass
class TaskPlan:
    """Represents a task plan"""
    task_nodes: List[TaskNode]

@dataclass
class BatchTask:
    """Represents a batch task"""
    task_ids: List[str]
    plan: TaskPlan

class TaskGraphAnalyzer:
    """Task graph analyzer"""
    
    def __init__(self):
        self.batches: List[BatchTask] = []
        
    def load_from_json(self, json_file_path: str):
        """Load task planning data from JSON file"""
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.batches = []
        for batch_data in data:
            # Parse task parameters
            task_nodes = []
            for node_data in batch_data['plan']['task_nodes']:
                parameters = [
                    TaskParameter(
                        task_id=param['task_id'],
                        params=param['params']
                    ) for param in node_data['parameters']
                ]
                
                task_node = TaskNode(
                    node_id=node_data['node_id'],
                    tool_name=node_data['tool_name'],
                    parameters=parameters,
                    dependencies=node_data['dependencies'],
                    original_task_ids=node_data['original_task_ids']
                )
                task_nodes.append(task_node)
            
            task_plan = TaskPlan(task_nodes=task_nodes)
            batch_task = BatchTask(
                task_ids=batch_data['task_ids'],
                plan=task_plan
            )
            self.batches.append(batch_task)
    
    def create_graph_for_batch(self, batch_index: int) -> nx.DiGraph:
        """Create NetworkX graph for specified batch"""
        if batch_index >= len(self.batches):
            raise IndexError(f"Batch index {batch_index} out of range")
        
        batch = self.batches[batch_index]
        G = nx.DiGraph()
        
        # Add nodes
        for node in batch.plan.task_nodes:
            G.add_node(node.node_id, 
                      tool_name=node.tool_name,
                      task_ids=node.original_task_ids,
                      parameter_count=len(node.parameters))
        
        # Add edges (dependencies)
        for node in batch.plan.task_nodes:
            for dep in node.dependencies:
                G.add_edge(dep, node.node_id)
        
        return G
    
    def analyze_tool_reuse(self, batch_index: int) -> Dict[str, Any]:
        """Analyze tool reuse patterns"""
        if batch_index >= len(self.batches):
            raise IndexError(f"Batch index {batch_index} out of range")
        
        batch = self.batches[batch_index]
        tool_usage = {}
        tool_reuse_info = {}
        
        # Count usage frequency and nodes for each tool
        for node in batch.plan.task_nodes:
            tool_name = node.tool_name
            if tool_name not in tool_usage:
                tool_usage[tool_name] = []
            tool_usage[tool_name].append({
                'node_id': node.node_id,
                'task_ids': node.original_task_ids,
                'param_count': len(node.parameters)
            })
        
        # Analyze reuse patterns
        for tool_name, usage_list in tool_usage.items():
            tool_reuse_info[tool_name] = {
                'usage_count': len(usage_list),
                'is_reused': len(usage_list) > 1,
                'nodes': usage_list,
                'total_tasks': sum(len(usage['task_ids']) for usage in usage_list)
            }
        
        return {
            'batch_task_ids': batch.task_ids,
            'tool_reuse_info': tool_reuse_info,
            'total_tools_used': len(tool_usage),
            'reused_tools': [tool for tool, info in tool_reuse_info.items() if info['is_reused']]
        }
    
    def visualize_batch_graph(self, batch_index: int, figsize: tuple = (12, 8), 
                            save_path: Optional[str] = None, show_task_ids: bool = True):
        """Visualize batch task graph"""
        if batch_index >= len(self.batches):
            raise IndexError(f"Batch index {batch_index} out of range")
        
        G = self.create_graph_for_batch(batch_index)
        batch = self.batches[batch_index]
        
        plt.figure(figsize=figsize)
        
        # Use hierarchical layout
        try:
            pos = nx.spring_layout(G, k=2, iterations=50)
        except:
            pos = nx.random_layout(G)
        
        # Assign colors based on tool types
        tool_names = list(set(node.tool_name for node in batch.plan.task_nodes))
        colors = plt.cm.Set3(np.linspace(0, 1, len(tool_names)))
        tool_color_map = dict(zip(tool_names, colors))
        
        # Draw nodes
        for node in batch.plan.task_nodes:
            tool_name = node.tool_name
            color = tool_color_map[tool_name]
            
            # Node size based on number of tasks contained
            node_size = 1000 + len(node.original_task_ids) * 500
            
            nx.draw_networkx_nodes(G, pos, nodelist=[node.node_id], 
                                 node_color=[color], node_size=node_size, 
                                 alpha=0.8, edgecolors='black', linewidths=1)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                             arrowsize=20, arrowstyle='->', width=2, alpha=0.6)
        
        # Draw labels
        labels = {}
        for node in batch.plan.task_nodes:
            if show_task_ids:
                task_ids_str = ', '.join(node.original_task_ids)
                labels[node.node_id] = f"{node.node_id}\n{node.tool_name}\nTasks: {task_ids_str}"
            else:
                labels[node.node_id] = f"{node.node_id}\n{node.tool_name}"
        
        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')
        
        # Create legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=tool_color_map[tool], markersize=10, 
                                    label=tool) for tool in tool_names]
        plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        plt.title(f"Task Planning Graph - Batch {batch_index}\nTask IDs: {', '.join(batch.task_ids)}", 
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_reuse_analysis_report(self, batch_index: int) -> str:
        """Generate tool reuse analysis report"""
        analysis = self.analyze_tool_reuse(batch_index)
        
        report = f"""
=== Batch {batch_index} Tool Reuse Analysis Report ===

Batch Task IDs: {', '.join(analysis['batch_task_ids'])}
Total Tools Used: {analysis['total_tools_used']}
Reused Tools: {len(analysis['reused_tools'])}

"""
        
        for tool_name, info in analysis['tool_reuse_info'].items():
            report += f"\nTool: {tool_name}\n"
            report += f"  - Usage Count: {info['usage_count']}\n"
            report += f"  - Is Reused: {'Yes' if info['is_reused'] else 'No'}\n"
            report += f"  - Total Tasks Involved: {info['total_tasks']}\n"
            report += f"  - Node Details:\n"
            
            for node in info['nodes']:
                report += f"    * {node['node_id']}: Tasks {', '.join(node['task_ids'])}\n"
        
        if analysis['reused_tools']:
            report += f"\nReused Tools: {', '.join(analysis['reused_tools'])}\n"
        else:
            report += f"\nNo tool reuse in this batch\n"
        
        return report
    
    def visualize_all_batches(self, save_dir: Optional[str] = None):
        """Visualize all batches"""
        for i in range(len(self.batches)):
            save_path = None
            if save_dir:
                save_path = f"{save_dir}/batch_{i}_graph.png"
            
            print(f"\n{'='*50}")
            print(f"Processing batch {i}...")
            print(self.generate_reuse_analysis_report(i))
            
            self.visualize_batch_graph(i, save_path=save_path)
    
    def load_ground_truth_data(self, ground_truth_file: str):
        """Load ground truth data from filtered_data.json format"""
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create mapping from task_id to ground truth tool_nodes
        self.ground_truth_map = {}
        for tool_group in data:
            for task_data in tool_group.get('data', []):
                task_id = task_data['id']
                tool_nodes = task_data.get('tool_nodes', [])
                self.ground_truth_map[task_id] = tool_nodes
    
    def flatten_tool_lists(self, gt_lists: List[List[str]], pred_lists: List[List[str]]) -> tuple:
        """Flatten tool lists for F1 calculation, similar to the original evaluate.py"""
        assert len(gt_lists) == len(pred_lists)
        
        gt_flat = []
        pred_flat = []
        
        for (sample_gt, sample_pred) in zip(gt_lists, pred_lists):
            union = set()
            union.update(sample_gt)
            union.update(sample_pred)
            
            for tool in union:
                if tool in sample_gt:
                    gt_flat.append(1)
                else:
                    gt_flat.append(0)
                    
                if tool in sample_pred:
                    pred_flat.append(1)
                else:
                    pred_flat.append(0)
        
        return gt_flat, pred_flat
    
    def analyze_f1_performance(self, ground_truth_file: str) -> Dict[str, Any]:
        """Analyze F1 performance against ground truth data"""
        # Load ground truth data
        self.load_ground_truth_data(ground_truth_file)
        
        results = {}
        
        for batch_idx, batch in enumerate(self.batches):
            batch_results = {
                'batch_id': batch_idx,
                'task_ids': batch.task_ids,
                'matched_tasks': [],
                'unmatched_tasks': [],
                'gt_tools': [],
                'pred_tools': []
            }
            
            # For each task in the batch, find ground truth
            for task_id in batch.task_ids:
                if task_id in self.ground_truth_map:
                    gt_tool_nodes = self.ground_truth_map[task_id]
                    gt_tools = [node['task'] for node in gt_tool_nodes]
                    
                    # Find predicted tools for this task
                    pred_tools = []
                    for node in batch.plan.task_nodes:
                        if task_id in node.original_task_ids:
                            pred_tools.append(node.tool_name)
                    
                    batch_results['matched_tasks'].append(task_id)
                    batch_results['gt_tools'].append(gt_tools)
                    batch_results['pred_tools'].append(pred_tools)
                else:
                    batch_results['unmatched_tasks'].append(task_id)
            
            # Calculate F1 scores if we have matched tasks
            if batch_results['matched_tasks']:
                gt_flat, pred_flat = self.flatten_tool_lists(
                    batch_results['gt_tools'], 
                    batch_results['pred_tools']
                )
                
                if gt_flat and pred_flat:
                    # Calculate metrics
                    precision, recall, f1, support = prfs(gt_flat, pred_flat, average='binary')
                    
                    batch_results['precision'] = precision
                    batch_results['recall'] = recall
                    batch_results['f1'] = f1
                    batch_results['support'] = support
                    
                    # Also calculate per-tool metrics
                    all_tools = set()
                    for gt_tools in batch_results['gt_tools']:
                        all_tools.update(gt_tools)
                    for pred_tools in batch_results['pred_tools']:
                        all_tools.update(pred_tools)
                    
                    tool_metrics = {}
                    for tool in all_tools:
                        tool_gt = []
                        tool_pred = []
                        
                        for gt_tools, pred_tools in zip(batch_results['gt_tools'], batch_results['pred_tools']):
                            tool_gt.append(1 if tool in gt_tools else 0)
                            tool_pred.append(1 if tool in pred_tools else 0)
                        
                        if any(tool_gt) or any(tool_pred):  # Only calculate if tool appears
                            tool_p, tool_r, tool_f1, tool_s = prfs(tool_gt, tool_pred, average='binary')
                            tool_metrics[tool] = {
                                'precision': tool_p,
                                'recall': tool_r,
                                'f1': tool_f1,
                                'support': tool_s
                            }
                    
                    batch_results['tool_metrics'] = tool_metrics
                else:
                    batch_results['precision'] = 0.0
                    batch_results['recall'] = 0.0
                    batch_results['f1'] = 0.0
                    batch_results['support'] = 0
                    batch_results['tool_metrics'] = {}
            else:
                batch_results['precision'] = 0.0
                batch_results['recall'] = 0.0
                batch_results['f1'] = 0.0
                batch_results['support'] = 0
                batch_results['tool_metrics'] = {}
            
            results[f'batch_{batch_idx}'] = batch_results
        
        # Calculate overall metrics
        all_gt_flat = []
        all_pred_flat = []
        
        for batch_key, batch_result in results.items():
            if batch_result['matched_tasks']:
                gt_flat, pred_flat = self.flatten_tool_lists(
                    batch_result['gt_tools'], 
                    batch_result['pred_tools']
                )
                all_gt_flat.extend(gt_flat)
                all_pred_flat.extend(pred_flat)
        
        if all_gt_flat and all_pred_flat:
            overall_precision, overall_recall, overall_f1, overall_support = prfs(
                all_gt_flat, all_pred_flat, average='binary'
            )
            
            results['overall'] = {
                'precision': overall_precision,
                'recall': overall_recall,
                'f1': overall_f1,
                'support': overall_support,
                'total_batches': len(self.batches),
                'matched_batches': sum(1 for r in results.values() if isinstance(r, dict) and r.get('matched_tasks'))
            }
        else:
            results['overall'] = {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'support': 0,
                'total_batches': len(self.batches),
                'matched_batches': 0
            }
        
        return results
    
    def extract_tool_reuse_triplets(self, batch_index: int) -> List[tuple]:
        """Extract tool reuse triplets <task1, task2, Tool> from prediction data"""
        if batch_index >= len(self.batches):
            raise IndexError(f"Batch index {batch_index} out of range")
        
        batch = self.batches[batch_index]
        triplets = []
        
        # For each tool node that handles multiple tasks
        for node in batch.plan.task_nodes:
            if len(node.original_task_ids) > 1:
                # Generate all pairs of tasks that reuse the same tool
                task_ids = node.original_task_ids
                for i in range(len(task_ids)):
                    for j in range(i + 1, len(task_ids)):
                        # Create triplet (task1, task2, tool) in sorted order
                        task1, task2 = sorted([task_ids[i], task_ids[j]])
                        triplet = (task1, task2, node.tool_name)
                        triplets.append(triplet)
        
        return triplets
    
    def extract_ground_truth_tool_reuse_triplets(self, task_ids: List[str]) -> List[tuple]:
        """Extract tool reuse triplets from ground truth data for given task IDs"""
        if not hasattr(self, 'ground_truth_map'):
            raise ValueError("Ground truth data not loaded. Call load_ground_truth_data first.")
        
        triplets = []
        
        # Find which tools are used by which tasks in ground truth
        tool_to_tasks = {}
        
        for task_id in task_ids:
            if task_id in self.ground_truth_map:
                tool_nodes = self.ground_truth_map[task_id]
                for tool_node in tool_nodes:
                    tool_name = tool_node['task']
                    if tool_name not in tool_to_tasks:
                        tool_to_tasks[tool_name] = []
                    tool_to_tasks[tool_name].append(task_id)
        
        # Generate triplets for tools that are reused (used by multiple tasks)
        for tool_name, using_tasks in tool_to_tasks.items():
            if len(using_tasks) > 1:
                # Generate all pairs of tasks that reuse the same tool
                for i in range(len(using_tasks)):
                    for j in range(i + 1, len(using_tasks)):
                        # Create triplet (task1, task2, tool) in sorted order
                        task1, task2 = sorted([using_tasks[i], using_tasks[j]])
                        triplet = (task1, task2, tool_name)
                        triplets.append(triplet)
        
        return triplets
    
    def analyze_tool_reuse_f1(self, ground_truth_file: str) -> Dict[str, Any]:
        """Analyze tool reuse F1 performance using triplet matching"""
        # Load ground truth data
        self.load_ground_truth_data(ground_truth_file)
        
        results = {}
        all_gt_triplets = []
        all_pred_triplets = []
        
        for batch_idx, batch in enumerate(self.batches):
            batch_results = {
                'batch_id': batch_idx,
                'task_ids': batch.task_ids,
                'gt_triplets': [],
                'pred_triplets': [],
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }
            
            # Extract prediction triplets
            pred_triplets = self.extract_tool_reuse_triplets(batch_idx)
            batch_results['pred_triplets'] = pred_triplets
            
            # Extract ground truth triplets for tasks in this batch
            gt_triplets = self.extract_ground_truth_tool_reuse_triplets(batch.task_ids)
            batch_results['gt_triplets'] = gt_triplets
            
            # Calculate F1 for this batch
            if gt_triplets or pred_triplets:
                gt_set = set(gt_triplets)
                pred_set = set(pred_triplets)
                
                # Calculate precision, recall, F1
                if pred_set:
                    precision = len(gt_set.intersection(pred_set)) / len(pred_set)
                else:
                    precision = 0.0
                
                if gt_set:
                    recall = len(gt_set.intersection(pred_set)) / len(gt_set)
                else:
                    recall = 0.0
                
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                else:
                    f1 = 0.0
                
                batch_results['precision'] = precision
                batch_results['recall'] = recall
                batch_results['f1'] = f1
                batch_results['true_positives'] = len(gt_set.intersection(pred_set))
                batch_results['false_positives'] = len(pred_set - gt_set)
                batch_results['false_negatives'] = len(gt_set - pred_set)
            
            results[f'batch_{batch_idx}'] = batch_results
            
            # Accumulate for overall metrics
            all_gt_triplets.extend(gt_triplets)
            all_pred_triplets.extend(pred_triplets)
        
        # Calculate overall metrics
        if all_gt_triplets or all_pred_triplets:
            gt_set = set(all_gt_triplets)
            pred_set = set(all_pred_triplets)
            
            if pred_set:
                overall_precision = len(gt_set.intersection(pred_set)) / len(pred_set)
            else:
                overall_precision = 0.0
            
            if gt_set:
                overall_recall = len(gt_set.intersection(pred_set)) / len(gt_set)
            else:
                overall_recall = 0.0
            
            if overall_precision + overall_recall > 0:
                overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall)
            else:
                overall_f1 = 0.0
            
            results['overall'] = {
                'precision': overall_precision,
                'recall': overall_recall,
                'f1': overall_f1,
                'total_gt_triplets': len(gt_set),
                'total_pred_triplets': len(pred_set),
                'true_positives': len(gt_set.intersection(pred_set)),
                'false_positives': len(pred_set - gt_set),
                'false_negatives': len(gt_set - pred_set),
                'total_batches': len(self.batches)
            }
        else:
            results['overall'] = {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'total_gt_triplets': 0,
                'total_pred_triplets': 0,
                'true_positives': 0,
                'false_positives': 0,
                'false_negatives': 0,
                'total_batches': len(self.batches)
            }
        
        return results
    
    def generate_tool_reuse_f1_report(self, ground_truth_file: str) -> str:
        """Generate tool reuse F1 analysis report"""
        results = self.analyze_tool_reuse_f1(ground_truth_file)
        
        report = "="*70 + "\n"
        report += "Tool Reuse F1 Performance Analysis Report\n"
        report += "="*70 + "\n\n"
        
        # Overall results
        overall = results['overall']
        report += f"Overall Tool Reuse Performance:\n"
        report += f"  - Precision: {overall['precision']:.4f}\n"
        report += f"  - Recall: {overall['recall']:.4f}\n"
        report += f"  - F1 Score: {overall['f1']:.4f}\n"
        report += f"  - Total GT Triplets: {overall['total_gt_triplets']}\n"
        report += f"  - Total Pred Triplets: {overall['total_pred_triplets']}\n"
        report += f"  - True Positives: {overall['true_positives']}\n"
        report += f"  - False Positives: {overall['false_positives']}\n"
        report += f"  - False Negatives: {overall['false_negatives']}\n"
        report += f"  - Total Batches: {overall['total_batches']}\n\n"
        
        # Per-batch results
        for batch_key, batch_result in results.items():
            if batch_key == 'overall':
                continue
                
            report += f"{batch_key.upper()} Tool Reuse Analysis:\n"
            report += f"  - Task IDs: {', '.join(batch_result['task_ids'])}\n"
            report += f"  - GT Triplets: {len(batch_result['gt_triplets'])}\n"
            report += f"  - Pred Triplets: {len(batch_result['pred_triplets'])}\n"
            report += f"  - Precision: {batch_result['precision']:.4f}\n"
            report += f"  - Recall: {batch_result['recall']:.4f}\n"
            report += f"  - F1 Score: {batch_result['f1']:.4f}\n"
            
            if batch_result['gt_triplets']:
                report += f"  - GT Triplets: {batch_result['gt_triplets']}\n"
            if batch_result['pred_triplets']:
                report += f"  - Pred Triplets: {batch_result['pred_triplets']}\n"
            
            if 'true_positives' in batch_result:
                report += f"  - True Positives: {batch_result['true_positives']}\n"
                report += f"  - False Positives: {batch_result['false_positives']}\n"
                report += f"  - False Negatives: {batch_result['false_negatives']}\n"
            
            report += "\n"
        
        return report

    def generate_f1_analysis_report(self, ground_truth_file: str) -> str:
        """Generate F1 analysis report"""
        results = self.analyze_f1_performance(ground_truth_file)
        
        report = "="*60 + "\n"
        report += "F1 Performance Analysis Report\n"
        report += "="*60 + "\n\n"
        
        # Overall results
        overall = results['overall']
        report += f"Overall Performance:\n"
        report += f"  - Precision: {overall['precision']:.4f}\n"
        report += f"  - Recall: {overall['recall']:.4f}\n"
        report += f"  - F1 Score: {overall['f1']:.4f}\n"
        report += f"  - Total Batches: {overall['total_batches']}\n"
        report += f"  - Matched Batches: {overall['matched_batches']}\n\n"
        
        # Per-batch results
        for batch_key, batch_result in results.items():
            if batch_key == 'overall':
                continue
                
            report += f"{batch_key.upper()}:\n"
            report += f"  - Task IDs: {', '.join(batch_result['task_ids'])}\n"
            report += f"  - Matched Tasks: {len(batch_result['matched_tasks'])}\n"
            report += f"  - Unmatched Tasks: {len(batch_result['unmatched_tasks'])}\n"
            
            if batch_result['matched_tasks']:
                report += f"  - Precision: {batch_result['precision']:.4f}\n"
                report += f"  - Recall: {batch_result['recall']:.4f}\n"
                report += f"  - F1 Score: {batch_result['f1']:.4f}\n"
                
            else:
                report += f"  - No matched tasks for evaluation\n"
                
            if batch_result['unmatched_tasks']:
                report += f"  - Unmatched Tasks: {', '.join(batch_result['unmatched_tasks'])}\n"
            
            report += "\n"
        
        return report


@click.command()
@click.option("--data_path", type=str, default="generated/zero_shot_response.json", help="Path to the data file.")
@click.option("--ground_truth", type=str, default="generated/filtered_data.json", help="Path to the ground truth file.")
@click.option("--analyze_reuse", is_flag=True, help="Enable tool reuse F1 analysis.")
def main(data_path, ground_truth, analyze_reuse):
    analyzer = TaskGraphAnalyzer()

    analyzer.load_from_json(data_path)

    # print(analyzer.generate_reuse_analysis_report(0))
    
    # analyzer.visualize_batch_graph(0)
    
    # F1 analysis against ground truth
    print("\n" + "="*60)
    print("F1 Performance Analysis")
    print("="*60)
    print(analyzer.generate_f1_analysis_report(ground_truth))
    
    # Tool reuse F1 analysis
    if analyze_reuse:
        print("\n" + "="*70)
        print("Tool Reuse F1 Performance Analysis")
        print("="*70)
        print(analyzer.generate_tool_reuse_f1_report(ground_truth))
    
    # analyzer.visualize_all_batches(save_dir='./output')

if __name__ == "__main__":
    main()