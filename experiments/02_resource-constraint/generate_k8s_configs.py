#!/usr/bin/env python3
"""
Generate Kubernetes pod configurations from resource constraint specifications.

This script reads resource constraint configurations from a JSON file and generates 
Kubernetes pod YAML files using Jinja2 templates. Each configuration defines CPU cores,
CPU memory, GPU fraction, and GPU memory constraints for running model inference experiments.
"""

import json
import os
import argparse
from pathlib import Path
from jinja2 import Environment, FileSystemLoader


def load_config(config_file: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_pod_name(config_name: str, task: str) -> str:
    """Generate a unique pod name based on configuration and task."""
    return f"inference-{config_name}-{task.lower()}".replace('_', '-')


def generate_k8s_configs(
    config_file: str,
    template_file: str,
    output_dir: str,
    dry_run: bool = False
) -> list:
    """
    Generate Kubernetes pod configurations from templates.
    
    Args:
        config_file: Path to JSON configuration file
        template_file: Path to Jinja2 template file
        output_dir: Directory to save generated YAML files
        dry_run: If True, only print what would be generated
        
    Returns:
        List of generated configuration information
    """
    # Load configuration
    config = load_config(config_file)
    
    # Setup Jinja2 environment
    template_dir = os.path.dirname(os.path.abspath(template_file))
    template_name = os.path.basename(template_file)
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template(template_name)
    
    # Create output directory
    output_path = Path(output_dir)
    if not dry_run:
        output_path.mkdir(parents=True, exist_ok=True)
    
    generated_configs = []
    
    # Generate configs for each resource constraint and task combination
    for constraint in config['resource_constraints']:
        for task in config['tasks']:
            # Generate pod name
            pod_name = generate_pod_name(constraint['name'], task)
            
            # Prepare template variables
            template_vars = {
                'pod_name': pod_name,
                'config_name': constraint['name'],
                'task': task,
                'cpu_cores': constraint['cpu_cores'],
                'cpu_memory': constraint['cpu_memory'],
                'gpu_fraction': constraint['gpu_fraction'],
                'gpu_memory': constraint['gpu_memory'],
                'namespace': config['experiment_config']['namespace'],
                'image': config['experiment_config']['image'],
                'scheduler_name': config['experiment_config']['scheduler_name'],
                'image_path': config['experiment_config']['image_path'],
                'device': config['experiment_config']['device']
            }
            
            # Render template
            rendered_yaml = template.render(**template_vars)
            
            # Generate output filename
            output_filename = f"{pod_name}.yaml"
            output_file_path = output_path / output_filename
            
            if dry_run:
                print(f"Would generate: {output_file_path}")
                print(f"  - Resource config: {constraint['name']}")
                print(f"  - Task: {task}")
                print(f"  - CPU cores: {constraint['cpu_cores']}")
                print(f"  - CPU memory: {constraint['cpu_memory']}")
                print(f"  - GPU fraction: {constraint['gpu_fraction']}")
                print(f"  - GPU memory: {constraint['gpu_memory']}Mi")
                print()
            else:
                # Write YAML file
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    f.write(rendered_yaml)
                print(f"Generated: {output_file_path}")
            
            # Store configuration info
            config_info = {
                'pod_name': pod_name,
                'config_name': constraint['name'],
                'task': task,
                'yaml_file': str(output_file_path),
                'resources': {
                    'cpu_cores': constraint['cpu_cores'],
                    'cpu_memory': constraint['cpu_memory'],
                    'gpu_fraction': constraint['gpu_fraction'],
                    'gpu_memory': constraint['gpu_memory']
                }
            }
            generated_configs.append(config_info)
    
    return generated_configs


def save_experiment_manifest(configs: list, output_file: str):
    """Save experiment manifest with all generated configurations."""
    manifest = {
        'total_configs': len(configs),
        'configurations': configs
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    print(f"Experiment manifest saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate K8s pod configurations for resource constraint experiments'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='resource_configs.json',
        help='Path to resource configuration JSON file (default: resource_configs.json)'
    )
    parser.add_argument(
        '--template',
        type=str,
        default='k8s-pod-template.yaml.j2',
        help='Path to Jinja2 template file (default: k8s-pod-template.yaml.j2)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='k8s-configs',
        help='Output directory for generated YAML files (default: k8s-configs)'
    )
    parser.add_argument(
        '--manifest',
        type=str,
        default='experiment_manifest.json',
        help='Output file for experiment manifest (default: experiment_manifest.json)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be generated without creating files'
    )
    
    args = parser.parse_args()
    
    try:
        # Validate input files exist
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Configuration file not found: {args.config}")
        if not os.path.exists(args.template):
            raise FileNotFoundError(f"Template file not found: {args.template}")
        
        # Generate configurations
        print(f"Generating K8s configurations...")
        print(f"Config file: {args.config}")
        print(f"Template file: {args.template}")
        print(f"Output directory: {args.output_dir}")
        print()
        
        generated_configs = generate_k8s_configs(
            args.config,
            args.template,
            args.output_dir,
            args.dry_run
        )
        
        if not args.dry_run:
            # Save experiment manifest
            save_experiment_manifest(generated_configs, args.manifest)
        
        print(f"\nSummary:")
        print(f"  Total configurations generated: {len(generated_configs)}")
        print(f"  Resource constraints: {len(set(c['config_name'] for c in generated_configs))}")
        print(f"  Tasks: {len(set(c['task'] for c in generated_configs))}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())