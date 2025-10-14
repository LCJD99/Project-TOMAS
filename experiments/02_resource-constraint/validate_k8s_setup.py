#!/usr/bin/env python3
"""
Validation script for K8s resource constraint experiment system.
"""

import os
import sys
import subprocess
import json
from pathlib import Path


def check_file_exists(filepath: str, description: str) -> bool:
    """Check if a file exists."""
    if os.path.exists(filepath):
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description}: {filepath} (NOT FOUND)")
        return False


def check_command_available(command: str) -> bool:
    """Check if a command is available."""
    try:
        result = subprocess.run([command, '--version'], 
                              capture_output=True, text=True, check=False)
        if result.returncode == 0:
            print(f"✓ {command} is available")
            return True
        else:
            print(f"✗ {command} is not available")
            return False
    except FileNotFoundError:
        print(f"✗ {command} is not available")
        return False


def check_python_package(package: str) -> bool:
    """Check if a Python package is importable."""
    try:
        __import__(package)
        print(f"✓ Python package '{package}' is available")
        return True
    except ImportError:
        print(f"✗ Python package '{package}' is not available")
        return False


def validate_json_config(config_file: str) -> bool:
    """Validate JSON configuration file."""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        required_keys = ['resource_constraints', 'tasks', 'experiment_config']
        for key in required_keys:
            if key not in config:
                print(f"✗ Missing key '{key}' in {config_file}")
                return False
        
        print(f"✓ JSON configuration is valid: {len(config['resource_constraints'])} configs, {len(config['tasks'])} tasks")
        return True
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"✗ JSON configuration error: {e}")
        return False


def test_config_generation() -> bool:
    """Test configuration generation."""
    try:
        result = subprocess.run([
            'python3', 'generate_k8s_configs.py',
            '--dry-run'
        ], capture_output=True, text=True, check=False)
        
        if result.returncode == 0:
            print("✓ Configuration generation test passed")
            return True
        else:
            print(f"✗ Configuration generation test failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Configuration generation test error: {e}")
        return False


def main():
    print("Kubernetes Resource Constraint Experiment System Validation")
    print("=" * 60)
    
    validation_results = []
    
    # Check required files
    print("\n1. Checking required files...")
    files_to_check = [
        ('resource_configs.json', 'Resource configuration file'),
        ('k8s-pod-template.yaml.j2', 'Jinja2 template file'),
        ('generate_k8s_configs.py', 'Configuration generation script'),
        ('k8s_inference.py', 'K8s inference script'),
        ('run_k8s_experiments.py', 'Main experiment script'),
        ('inference.py', 'Inference script'),
        ('lena.png', 'Test image'),
        ('README_K8S.md', 'Documentation')
    ]
    
    for filepath, description in files_to_check:
        validation_results.append(check_file_exists(filepath, description))
    
    # Check system commands
    print("\n2. Checking system commands...")
    commands_to_check = ['python3', 'kubectl']
    for command in commands_to_check:
        validation_results.append(check_command_available(command))
    
    # Check Python packages
    print("\n3. Checking Python packages...")
    packages_to_check = ['jinja2', 'json', 'argparse', 'pathlib', 'subprocess']
    for package in packages_to_check:
        validation_results.append(check_python_package(package))
    
    # Validate JSON configuration
    print("\n4. Validating JSON configuration...")
    validation_results.append(validate_json_config('resource_configs.json'))
    
    # Test configuration generation
    print("\n5. Testing configuration generation...")
    validation_results.append(test_config_generation())
    
    # Check Kubernetes connectivity (optional)
    print("\n6. Checking Kubernetes connectivity...")
    try:
        result = subprocess.run(['kubectl', 'cluster-info'], 
                              capture_output=True, text=True, check=False)
        if result.returncode == 0:
            print("✓ Kubernetes cluster is accessible")
            validation_results.append(True)
        else:
            print("⚠ Kubernetes cluster is not accessible (optional for setup validation)")
            validation_results.append(True)  # Don't fail validation for this
    except:
        print("⚠ kubectl not available or cluster not accessible (optional for setup validation)")
        validation_results.append(True)
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(validation_results)
    total = len(validation_results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All validation tests passed! System is ready for experiments.")
        return 0
    else:
        print("✗ Some validation tests failed. Please fix the issues above.")
        return 1


if __name__ == '__main__':
    exit(main())