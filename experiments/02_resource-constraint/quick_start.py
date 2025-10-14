#!/usr/bin/env python3
"""
Quick start script for K8s resource constraint experiments.
"""

import argparse
import subprocess
import sys
from datetime import datetime


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*50}")
    print(f"Step: {description}")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*50}")
    
    result = subprocess.run(command, text=True)
    
    if result.returncode != 0:
        print(f"❌ Failed: {description}")
        return False
    else:
        print(f"✅ Success: {description}")
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Quick start script for K8s resource constraint experiments'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only run validation, do not execute experiments'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='resource_configs.json',
        help='Configuration file to use'
    )
    
    args = parser.parse_args()
    
    print("🚀 Kubernetes Resource Constraint Experiments - Quick Start")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    steps = []
    
    # Step 1: Validation
    steps.append((['python3', 'validate_k8s_setup.py'], 'System validation'))
    
    if not args.validate_only:
        # Step 2: Install dependencies (optional)
        steps.append((['python3', 'setup_dependencies.py'], 'Setup dependencies'))
        
        # Step 3: Generate configurations
        steps.append(([
            'python3', 'generate_k8s_configs.py',
            '--config', args.config
        ], 'Generate K8s configurations'))
        
        # Step 4: Run experiments
        steps.append(([
            'python3', 'run_k8s_experiments.py',
            '--config', args.config
        ], 'Run experiments'))
    
    # Execute steps
    failed_steps = []
    for i, (command, description) in enumerate(steps, 1):
        print(f"\n🔄 Step {i}/{len(steps)}: {description}")
        
        if not run_command(command, description):
            failed_steps.append(description)
            if not args.validate_only:
                print(f"\n❌ Stopping execution due to failure in step: {description}")
                break
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 EXECUTION SUMMARY")
    print(f"{'='*60}")
    
    if failed_steps:
        print(f"❌ Failed steps: {len(failed_steps)}")
        for step in failed_steps:
            print(f"  - {step}")
        print("\n🔧 Please fix the issues above and try again.")
        return 1
    else:
        if args.validate_only:
            print("✅ Validation completed successfully!")
            print("\n🚀 Next steps:")
            print("  1. Install dependencies: pip install jinja2 pandas")
            print("  2. Ensure K8s cluster is accessible: kubectl cluster-info")
            print("  3. Run full experiments: python3 quick_start.py")
        else:
            print("🎉 All steps completed successfully!")
            print("\n📁 Check the following for results:")
            print("  - k8s-configs/: Generated Pod configurations")
            print("  - k8s_experiment_results_*.csv: Experiment results")
            print("  - experiment_manifest.json: Experiment metadata")
        
        return 0


if __name__ == '__main__':
    exit(main())