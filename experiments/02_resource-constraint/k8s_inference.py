#!/usr/bin/env python3
"""
Kubernetes-based inference runner for resource constraint experiments.

This script runs model inference experiments in Kubernetes pods with various
resource constraints and collects results for analysis.
"""

import argparse
import json
import os
import sys
import time
import csv
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any


class K8sExperimentRunner:
    """Manages Kubernetes-based resource constraint experiments."""
    
    def __init__(self, config_file: str = 'resource_configs.json'):
        """Initialize the experiment runner."""
        self.config_file = config_file
        self.config = self._load_config()
        self.results = []
        self.namespace = self.config['experiment_config']['namespace']
        
    def _load_config(self) -> dict:
        """Load experiment configuration from JSON file."""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: Configuration file '{self.config_file}' not found.")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in configuration file: {e}")
            sys.exit(1)
    
    def _run_command(self, command: List[str], capture_output: bool = True) -> subprocess.CompletedProcess:
        """Run a shell command and return the result."""
        try:
            result = subprocess.run(
                command,
                capture_output=capture_output,
                text=True,
                check=False
            )
            return result
        except Exception as e:
            print(f"Error running command {' '.join(command)}: {e}")
            raise
    
    def _wait_for_pod_completion(self, pod_name: str, timeout: int = 600) -> Dict[str, Any]:
        """
        Wait for a pod to complete and return its status and logs.
        
        Args:
            pod_name: Name of the pod to wait for
            timeout: Maximum time to wait in seconds
            
        Returns:
            Dictionary containing pod status, logs, and timing information
        """
        start_time = time.time()
        print(f"Waiting for pod {pod_name} to complete...")
        
        while time.time() - start_time < timeout:
            # Check pod status
            result = self._run_command([
                'k3s',
                'kubectl',
                'get',
                'pod',
                pod_name,
                '-n',
                self.namespace,
                '-o',
                'jsonpath={.status.phase}'
            ])
            
            if result.returncode != 0:
                return {
                    'status': 'Error',
                    'error': f"Failed to get pod status: {result.stderr}",
                    'execution_time': None,
                    'logs': None
                }
            
            phase = result.stdout.strip()
            
            if phase in ['Succeeded', 'Failed']:
                # Pod completed, get logs
                log_result = self._run_command([
                    'k3s',
                    'kubectl',
                    'logs',
                    pod_name,
                    '-n',
                    self.namespace
                ])
                
                # Get additional pod information
                info_result = self._run_command([
                    'k3s',
                    'kubectl',
                    'get',
                    'pod',
                    pod_name,
                    '-n',
                    self.namespace,
                    '-o',
                    'json'
                ])
                
                pod_info = {}
                if info_result.returncode == 0:
                    try:
                        pod_data = json.loads(info_result.stdout)
                        # Extract timing information
                        start_ts = pod_data.get('status', {}).get('startTime')
                        if 'finishTime' in pod_data.get('status', {}):
                            finish_ts = pod_data['status']['finishTime']
                        else:
                            # Use current time if finish time not available
                            finish_ts = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
                        
                        if start_ts and finish_ts:
                            start_dt = datetime.fromisoformat(start_ts.replace('Z', '+00:00'))
                            finish_dt = datetime.fromisoformat(finish_ts.replace('Z', '+00:00'))
                            pod_info['execution_time'] = (finish_dt - start_dt).total_seconds()
                        
                        # Get resource usage information if available
                        pod_info['node_name'] = pod_data.get('spec', {}).get('nodeName')
                        
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        print(f"Warning: Could not parse pod information: {e}")
                
                return {
                    'status': phase,
                    'logs': log_result.stdout if log_result.returncode == 0 else None,
                    'log_error': log_result.stderr if log_result.returncode != 0 else None,
                    'execution_time': pod_info.get('execution_time'),
                    'node_name': pod_info.get('node_name'),
                    'total_wait_time': time.time() - start_time
                }
            
            # Wait before checking again
            time.sleep(5)
        
        # Timeout reached
        return {
            'status': 'Timeout',
            'error': f"Pod did not complete within {timeout} seconds",
            'execution_time': None,
            'logs': None,
            'total_wait_time': timeout
        }
    
    def _parse_inference_results(self, logs: str) -> Dict[str, Any]:
        """Parse inference results from pod logs."""
        results = {
            'inference_time': None,
            'end_to_end_time': None,
            'gpu_memory_peak': None,
            'cpu_memory_peak': None,
            'error_occurred': False,
            'error_message': None
        }
        
        if not logs:
            results['error_occurred'] = True
            results['error_message'] = "No logs available"
            return results
        
        lines = logs.split('\n')
        for line in lines:
            line = line.strip()
            
            # Parse timing information
            if 'end to end time =' in line:
                try:
                    time_part = line.split('=')[1].strip()
                    results['end_to_end_time'] = float(time_part)
                except (IndexError, ValueError):
                    pass
            
            # Parse memory usage information
            if 'GPU Peak Memory Allocated:' in line:
                try:
                    memory_part = line.split(':')[1].strip().replace('GB', '').strip()
                    results['gpu_memory_peak'] = float(memory_part)
                except (IndexError, ValueError):
                    pass
            
            if 'CPU Process Memory (RSS):' in line:
                try:
                    memory_part = line.split(':')[1].strip().replace('GB', '').strip()
                    results['cpu_memory_peak'] = float(memory_part)
                except (IndexError, ValueError):
                    pass
            
            # Check for errors
            if any(error_keyword in line.lower() for error_keyword in ['error', 'exception', 'failed', 'traceback']):
                results['error_occurred'] = True
                if not results['error_message']:
                    results['error_message'] = line
        
        return results
    
    def run_single_experiment(self, pod_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single experiment with the given pod configuration.
        
        Args:
            pod_config: Configuration for the pod to run
            
        Returns:
            Dictionary containing experiment results
        """
        pod_name = pod_config['pod_name']
        yaml_file = pod_config['yaml_file']
        
        print(f"\n{'='*60}")
        print(f"Running experiment: {pod_name}")
        print(f"Config: {pod_config['config_name']}")
        print(f"Task: {pod_config['task']}")
        print(f"Resources: CPU={pod_config['resources']['cpu_cores']}, "
              f"Memory={pod_config['resources']['cpu_memory']}, "
              f"GPU={pod_config['resources']['gpu_fraction']}, "
              f"GPU Memory={pod_config['resources']['gpu_memory']}Mi")
        print(f"{'='*60}")
        
        # Apply the pod configuration
        apply_result = self._run_command(['k3s', 'kubectl', 'apply', '-f', yaml_file])
        
        if apply_result.returncode != 0:
            return {
                'pod_name': pod_name,
                'config_name': pod_config['config_name'],
                'task': pod_config['task'],
                'status': 'Apply Failed',
                'error': apply_result.stderr,
                'timestamp': datetime.now().isoformat(),
                **pod_config['resources']
            }
        
        print(f"Pod {pod_name} created successfully")
        
        # Wait for pod completion
        pod_result = self._wait_for_pod_completion(pod_name)
        
        # Parse inference results from logs
        inference_results = self._parse_inference_results(pod_result.get('logs', ''))
        
        # Clean up the pod
        cleanup_result = self._run_command([
            'k3s',
            'kubectl',
            'delete',
            'pod',
            pod_name,
            '-n',
            self.namespace,
            '--ignore-not-found=true'
        ])
        
        if cleanup_result.returncode != 0:
            print(f"Warning: Failed to delete pod {pod_name}: {cleanup_result.stderr}")
        else:
            print(f"Pod {pod_name} deleted successfully")
        
        # Compile results
        result = {
            'pod_name': pod_name,
            'config_name': pod_config['config_name'],
            'task': pod_config['task'],
            'status': pod_result['status'],
            'timestamp': datetime.now().isoformat(),
            'execution_time': pod_result.get('execution_time'),
            'total_wait_time': pod_result.get('total_wait_time'),
            'node_name': pod_result.get('node_name'),
            **pod_config['resources'],
            **inference_results
        }
        
        if pod_result.get('error'):
            result['error'] = pod_result['error']
        
        return result
    
    def run_all_experiments(self, manifest_file: str = 'experiment_manifest.json') -> List[Dict[str, Any]]:
        """
        Run all experiments defined in the manifest file.
        
        Args:
            manifest_file: Path to the experiment manifest JSON file
            
        Returns:
            List of experiment results
        """
        try:
            with open(manifest_file, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
        except FileNotFoundError:
            print(f"Error: Manifest file '{manifest_file}' not found.")
            print("Please run generate_k8s_configs.py first to generate the manifest.")
            return []
        
        configs = manifest['configurations']
        total_experiments = len(configs)
        
        print(f"Starting {total_experiments} experiments...")
        
        results = []
        for i, config in enumerate(configs, 1):
            print(f"\nProgress: {i}/{total_experiments}")
            
            try:
                result = self.run_single_experiment(config)
                results.append(result)
                
                # Print summary
                status = result['status']
                if status == 'Succeeded':
                    print(f"✓ Experiment completed successfully")
                    if result.get('end_to_end_time'):
                        print(f"  Execution time: {result['end_to_end_time']:.2f}s")
                else:
                    print(f"✗ Experiment failed with status: {status}")
                    if result.get('error'):
                        print(f"  Error: {result['error']}")
                
            except KeyboardInterrupt:
                print("\nExperiment interrupted by user")
                break
            except Exception as e:
                print(f"✗ Unexpected error in experiment: {e}")
                result = {
                    'pod_name': config['pod_name'],
                    'config_name': config['config_name'],
                    'task': config['task'],
                    'status': 'Exception',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat(),
                    **config['resources']
                }
                results.append(result)
        
        return results
    
    def save_results_to_csv(self, results: List[Dict[str, Any]], output_file: str = 'k8s_experiment_results.csv'):
        """Save experiment results to CSV file."""
        if not results:
            print("No results to save.")
            return
        
        # Define CSV columns
        fieldnames = [
            'timestamp',
            'pod_name',
            'config_name',
            'task',
            'status',
            'cpu_cores',
            'cpu_memory',
            'gpu_fraction',
            'gpu_memory',
            'execution_time',
            'end_to_end_time',
            'total_wait_time',
            'gpu_memory_peak',
            'cpu_memory_peak',
            'node_name',
            'error_occurred',
            'error_message',
            'error'
        ]
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                # Ensure all fields are present
                row = {field: result.get(field, '') for field in fieldnames}
                writer.writerow(row)
        
        print(f"\nResults saved to: {output_file}")
        print(f"Total experiments: {len(results)}")
        successful = sum(1 for r in results if r.get('status') == 'Succeeded')
        print(f"Successful experiments: {successful}")
        print(f"Failed experiments: {len(results) - successful}")


def main():
    parser = argparse.ArgumentParser(
        description='Run Kubernetes-based resource constraint experiments'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='resource_configs.json',
        help='Path to resource configuration file (default: resource_configs.json)'
    )
    parser.add_argument(
        '--manifest',
        type=str,
        default='experiment_manifest.json',
        help='Path to experiment manifest file (default: experiment_manifest.json)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='k8s_experiment_results.csv',
        help='Output CSV file for results (default: k8s_experiment_results.csv)'
    )
    parser.add_argument(
        '--single',
        type=str,
        help='Run single experiment by pod name'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what experiments would be run without executing them'
    )
    
    args = parser.parse_args()
    
    try:
        runner = K8sExperimentRunner(args.config)
        
        if args.single:
            # Run single experiment
            try:
                with open(args.manifest, 'r', encoding='utf-8') as f:
                    manifest = json.load(f)
            except FileNotFoundError:
                print(f"Error: Manifest file '{args.manifest}' not found.")
                return 1
            
            # Find the specific configuration
            config = None
            for cfg in manifest['configurations']:
                if cfg['pod_name'] == args.single:
                    config = cfg
                    break
            
            if not config:
                print(f"Error: Pod '{args.single}' not found in manifest.")
                return 1
            
            if args.dry_run:
                print(f"Would run experiment: {config['pod_name']}")
                print(f"  Config: {config['config_name']}")
                print(f"  Task: {config['task']}")
                return 0
            
            result = runner.run_single_experiment(config)
            runner.save_results_to_csv([result], args.output)
            
        else:
            # Run all experiments
            if args.dry_run:
                try:
                    with open(args.manifest, 'r', encoding='utf-8') as f:
                        manifest = json.load(f)
                    print(f"Would run {len(manifest['configurations'])} experiments")
                    for cfg in manifest['configurations']:
                        print(f"  - {cfg['pod_name']}: {cfg['config_name']} / {cfg['task']}")
                    return 0
                except FileNotFoundError:
                    print(f"Error: Manifest file '{args.manifest}' not found.")
                    return 1
            
            results = runner.run_all_experiments(args.manifest)
            if results:
                runner.save_results_to_csv(results, args.output)
        
        return 0
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
