#!/usr/bin/env python3
"""
Create required dependencies file for K8s experiments.
"""

def create_requirements_txt():
    """Create or update requirements.txt with necessary dependencies."""
    requirements = [
        "jinja2>=3.0.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "torch>=1.9.0",
        "transformers>=4.20.0",
        "Pillow>=8.0.0",
        "click>=8.0.0",
        "psutil>=5.8.0"
    ]
    
    with open('requirements.txt', 'w') as f:
        for req in requirements:
            f.write(f"{req}\n")
    
    print("Created requirements.txt with necessary dependencies")

if __name__ == '__main__':
    create_requirements_txt()