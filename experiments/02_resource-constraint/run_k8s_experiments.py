#!/usr/bin/env python3
"""
Main experiment runner for Kubernetes-based resource constraint experiments.

This script orchestrates the complete experimental workflow:
1. Generates Kubernetes pod configurations from templates
2. Creates necessary ConfigMaps for inference scripts and test images
3. Runs experiments with various resource constraints
4. Collects and analyzes results
"""

import argparse
import json
import os
import sys
import subprocess
import base64
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any


class MainExperimentRunner:
    """Main controller for K8s resource constraint experiments."""
    
    def __init__(self, work_dir: str = '.'):
        """Initialize the experiment runner."""
        self.work_dir = Path(work_dir).resolve()
        self.namespace = 'default'
        
    def _run_command(self, command: List[str], cwd: str = None) -> subprocess.CompletedProcess:
        """Run a shell command and return the result."""
        if cwd is None:
            cwd = str(self.work_dir)
            
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                check=False
            )
            return result
        except Exception as e:
            print(f"Error running command {' '.join(command)}: {e}")
            raise
    
    def check_prerequisites(self) -> bool:
        """Check if all required files and tools are available."""
        print("Checking prerequisites...")
        
        # Check kubectl
        result = self._run_command(['k3s', 'kubectl', 'version', '--client'])
        if result.returncode != 0:
            print("Error: kubectl is not installed or not accessible")
            return False
        print("✓ kubectl is available")
        
        # Check cluster connectivity
        result = self._run_command(['k3s', 'kubectl', 'cluster-info'])
        if result.returncode != 0:
            print("Error: Cannot connect to Kubernetes cluster")
            return False
        print("✓ Kubernetes cluster is accessible")
        
        # Check required files
        required_files = [
            'resource_configs.json',
            'k8s-pod-template.yaml.j2',
            'generate_k8s_configs.py',
            'k8s_inference.py',
            'inference.py',
            'lena.png'
        ]
        
        missing_files = []
        for file in required_files:
            if not (self.work_dir / file).exists():
                missing_files.append(file)
        
        if missing_files:
            print(f"Error: Missing required files: {', '.join(missing_files)}")
            return False
        
        for file in required_files:
            print(f"✓ {file} found")
        
        # Check Python dependencies
        try:
            import jinja2
            print("✓ jinja2 is available")
        except ImportError:
            print("Error: jinja2 is not installed. Install with: pip install jinja2")
            return False
        
        return True
    
    def create_configmaps(self) -> bool:
        """Create ConfigMaps for inference script and test image."""
        print("\nCreating ConfigMaps...")
        
        # Create ConfigMap for inference script
        inference_script = self.work_dir / 'inference.py'
        result = self._run_command([
            'k3s',
            'kubectl',
            'create',
            'configmap',
            'inference-script',
            f'--from-file=inference.py={inference_script}',
            '-n',
            self.namespace,
            '--dry-run=client',
            '-o',
            'yaml'
        ])
        
        if result.returncode != 0:
            print(f"Error creating inference script ConfigMap: {result.stderr}")
            return False
        
        # Apply the ConfigMap
        print("Applying inference-script ConfigMap...")
        apply_process = subprocess.Popen(
            ['k3s', 'kubectl', 'apply', '-f', '-'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(self.work_dir)
        )
        stdout, stderr = apply_process.communicate(input=result.stdout)
        
        if apply_process.returncode != 0:
            print(f"Error applying inference script ConfigMap: {stderr}")
            return False
        print("✓ inference-script ConfigMap created")
        print("✓ Using hostPath for test image (lena.png) - no ConfigMap needed")
        
        return True
    
    def cleanup_configmaps(self) -> bool:
        """Clean up ConfigMaps created for the experiments."""
        print("\nCleaning up ConfigMaps...")
        
        configmaps = ['inference-script']
        for configmap in configmaps:
            result = self._run_command([
                'k3s',
                'kubectl',
                'delete',
                'configmap',
                configmap,
                '-n',
                self.namespace,
                '--ignore-not-found=true'
            ])
            
            if result.returncode != 0:
                print(f"Warning: Failed to delete ConfigMap {configmap}: {result.stderr}")
            else:
                print(f"✓ ConfigMap {configmap} deleted")
        
        return True
    
    def generate_configurations(self, config_file: str = 'resource_configs.json') -> bool:
        """Generate Kubernetes pod configurations."""
        print("\nGenerating Kubernetes configurations...")
        
        result = self._run_command([
            'python3', 'generate_k8s_configs.py',
            '--config', config_file,
            '--template', 'k8s-pod-template.yaml.j2',
            '--output-dir', 'k8s-configs',
            '--manifest', 'experiment_manifest.json'
        ])
        
        if result.returncode != 0:
            print(f"Error generating configurations: {result.stderr}")
            return False
        
        print(result.stdout)
        return True
    
    def run_experiments(self, manifest_file: str = 'experiment_manifest.json', 
                       output_file: str = None) -> bool:
        """Run all experiments and collect results."""
        print("\nRunning experiments...")
        
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f'k8s_experiment_results_{timestamp}.csv'
        
        result = self._run_command([
            'python3', 'k8s_inference.py',
            '--manifest', manifest_file,
            '--output', output_file
        ])
        
        if result.returncode != 0:
            print(f"Error running experiments: {result.stderr}")
            return False
        
        print(result.stdout)
        return True
    
    def analyze_results(self, results_file: str) -> Dict[str, Any]:
        """Perform basic analysis of experiment results."""
        print(f"\nAnalyzing results from {results_file}...")
        
        try:
            import pandas as pd
            df = pd.read_csv(results_file)
            
            analysis = {
                'total_experiments': len(df),
                'successful_experiments': len(df[df['status'] == 'Succeeded']),
                'failed_experiments': len(df[df['status'] != 'Succeeded']),
                'success_rate': len(df[df['status'] == 'Succeeded']) / len(df) * 100,
                'average_execution_time': df[df['status'] == 'Succeeded']['end_to_end_time'].mean(),
                'tasks_tested': df['task'].unique().tolist(),
                'configs_tested': df['config_name'].unique().tolist()
            }
            
            print(f"Analysis Summary:")
            print(f"  Total experiments: {analysis['total_experiments']}")
            print(f"  Successful: {analysis['successful_experiments']}")
            print(f"  Failed: {analysis['failed_experiments']}")
            print(f"  Success rate: {analysis['success_rate']:.1f}%")
            if not pd.isna(analysis['average_execution_time']):
                print(f"  Average execution time: {analysis['average_execution_time']:.2f}s")
            print(f"  Tasks tested: {', '.join(analysis['tasks_tested'])}")
            print(f"  Resource configs tested: {len(analysis['configs_tested'])}")
            
            return analysis
            
        except ImportError:
            print("pandas not available for detailed analysis")
            return {}
        except Exception as e:
            print(f"Error analyzing results: {e}")
            return {}


def main():
    parser = argparse.ArgumentParser(
        description='Run complete K8s resource constraint experiment workflow'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='resource_configs.json',
        help='Path to resource configuration file (default: resource_configs.json)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output CSV file for results (default: auto-generated with timestamp)'
    )
    parser.add_argument(
        '--skip-config-generation',
        action='store_true',
        help='Skip configuration generation step'
    )
    parser.add_argument(
        '--skip-experiments',
        action='store_true',
        help='Skip experiment execution step'
    )
    parser.add_argument(
        '--cleanup-only',
        action='store_true',
        help='Only perform cleanup operations'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Perform dry run without executing experiments'
    )
    parser.add_argument(
        '--work-dir',
        type=str,
        default='.',
        help='Working directory for experiments (default: current directory)'
    )
    
    args = parser.parse_args()
    
    try:
        runner = MainExperimentRunner(args.work_dir)
        
        if args.cleanup_only:
            runner.cleanup_configmaps()
            return 0
        
        # Check prerequisites
        if not runner.check_prerequisites():
            print("\nPrerequisite check failed. Please fix the issues above.")
            return 1
        
        # Create ConfigMaps
        if not runner.create_configmaps():
            print("\nFailed to create ConfigMaps.")
            return 1
        
        try:
            # Generate configurations
            if not args.skip_config_generation:
                if not runner.generate_configurations(args.config):
                    print("\nFailed to generate configurations.")
                    return 1
            
            # Run experiments
            if not args.skip_experiments and not args.dry_run:
                output_file = args.output
                if not runner.run_experiments(output_file=output_file):
                    print("\nFailed to run experiments.")
                    return 1
                
                # Analyze results if output file was generated
                if output_file or Path('k8s_experiment_results_*.csv').exists():
                    # Find the most recent results file if output_file not specified
                    if not output_file:
                        import glob
                        pattern = str(runner.work_dir / 'k8s_experiment_results_*.csv')
                        files = glob.glob(pattern)
                        if files:
                            output_file = max(files, key=os.path.getctime)
                    
                    if output_file and Path(output_file).exists():
                        runner.analyze_results(output_file)
            
            elif args.dry_run:
                print("\nDry run completed. Use --skip-experiments=false to run actual experiments.")
        
        finally:
            # Cleanup ConfigMaps
            runner.cleanup_configmaps()
        
        print("\n" + "="*60)
        print("Experiment workflow completed successfully!")
        print("="*60)
        
        return 0
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    exit(main())