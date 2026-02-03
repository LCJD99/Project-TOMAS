"""
Resource profiler for loading and querying tool performance data.

Loads profiling.csv and provides efficient lookup of latency based on
tool name and resource configuration.
"""

import pandas as pd
from typing import Dict, List, Tuple, Optional
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from schema.token_schema import RESOURCE_BINS


class ResourceConfig:
    """Represents a resource configuration."""
    
    def __init__(self, cpu_core: int, cpu_mem_gb: float, 
                 gpu_sm: int, gpu_mem_gb: float):
        self.cpu_core = cpu_core
        self.cpu_mem_gb = cpu_mem_gb
        self.gpu_sm = gpu_sm
        self.gpu_mem_gb = gpu_mem_gb
    
    def __repr__(self):
        return f"Config(cpu={self.cpu_core}c/{self.cpu_mem_gb}GB, gpu={self.gpu_sm}sm/{self.gpu_mem_gb}GB)"
    
    def __eq__(self, other):
        if not isinstance(other, ResourceConfig):
            return False
        return (self.cpu_core == other.cpu_core and
                self.cpu_mem_gb == other.cpu_mem_gb and
                self.gpu_sm == other.gpu_sm and
                self.gpu_mem_gb == other.gpu_mem_gb)
    
    def __hash__(self):
        return hash((self.cpu_core, self.cpu_mem_gb, self.gpu_sm, self.gpu_mem_gb))
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'cpu_core': self.cpu_core,
            'cpu_mem_gb': self.cpu_mem_gb,
            'gpu_sm': self.gpu_sm,
            'gpu_mem_gb': self.gpu_mem_gb
        }
    
    def fits_in(self, available: 'ResourceConfig') -> bool:
        """Check if this config fits within available resources."""
        return (self.cpu_core <= available.cpu_core and
                self.cpu_mem_gb <= available.cpu_mem_gb and
                self.gpu_sm <= available.gpu_sm and
                self.gpu_mem_gb <= available.gpu_mem_gb)
    
    def subtract(self, other: 'ResourceConfig') -> 'ResourceConfig':
        """Subtract another config from this one."""
        return ResourceConfig(
            self.cpu_core - other.cpu_core,
            self.cpu_mem_gb - other.cpu_mem_gb,
            self.gpu_sm - other.gpu_sm,
            self.gpu_mem_gb - other.gpu_mem_gb
        )
    
    def add(self, other: 'ResourceConfig') -> 'ResourceConfig':
        """Add another config to this one."""
        return ResourceConfig(
            self.cpu_core + other.cpu_core,
            self.cpu_mem_gb + other.cpu_mem_gb,
            self.gpu_sm + other.gpu_sm,
            self.gpu_mem_gb + other.gpu_mem_gb
        )


class ResourceProfiler:
    """Loads and queries tool performance profiling data."""
    
    def __init__(self, profiling_csv_path: str):
        """
        Initialize profiler with CSV data.
        
        Args:
            profiling_csv_path: Path to profiling.csv file
        """
        self.df = pd.read_csv(profiling_csv_path)
        
        # Build index for fast lookup: (tool, input_size, config) -> latency
        self.latency_map = {}
        for _, row in self.df.iterrows():
            key = (
                row['tool'],
                row['input_size'],
                ResourceConfig(
                    int(row['cpu_core']),
                    float(row['cpu_mem_gb']),
                    int(row['gpu_sm']),
                    float(row['gpu_mem_gb'])
                )
            )
            self.latency_map[key] = float(row['latency_ms'])
        
        print(f"Loaded {len(self.latency_map)} profiling entries")
    
    def get_all_configs(self, tool_name: str, input_size: str = 'small') -> List[Tuple[ResourceConfig, float]]:
        """
        Get all available configurations for a tool.
        
        Args:
            tool_name: Tool name (profiling format, e.g., 'image_classification')
            input_size: Input size category
            
        Returns:
            List of (config, latency) tuples
        """
        configs = []
        for (tool, size, config), latency in self.latency_map.items():
            if tool == tool_name and size == input_size:
                configs.append((config, latency))
        
        return configs
    
    def get_feasible_configs(self, tool_name: str, available_resources: ResourceConfig,
                            input_size: str = 'small') -> List[Tuple[ResourceConfig, float]]:
        """
        Get configurations that fit within available resources.
        
        Args:
            tool_name: Tool name (profiling format)
            available_resources: Available resource limits
            input_size: Input size category
            
        Returns:
            List of (config, latency) tuples that fit in available resources
        """
        all_configs = self.get_all_configs(tool_name, input_size)
        feasible = [(cfg, lat) for cfg, lat in all_configs if cfg.fits_in(available_resources)]
        return feasible
    
    def get_latency(self, tool_name: str, config: ResourceConfig, 
                   input_size: str = 'small') -> Optional[float]:
        """
        Get latency for a specific configuration.
        
        Args:
            tool_name: Tool name (profiling format)
            config: Resource configuration
            input_size: Input size category
            
        Returns:
            Latency in milliseconds, or None if not found
        """
        key = (tool_name, input_size, config)
        return self.latency_map.get(key)
    
    def map_config_to_levels(self, config: ResourceConfig) -> Dict[str, str]:
        """
        Map concrete resource values to level names (low/medium/high).
        
        Args:
            config: Resource configuration
            
        Returns:
            Dictionary with level names for each resource type
        """
        levels = {}
        
        # CPU cores
        for level, value in RESOURCE_BINS['cpu_core'].items():
            if config.cpu_core == value:
                levels['cpu_core_level'] = level
                break
        
        # CPU memory
        for level, value in RESOURCE_BINS['cpu_mem_gb'].items():
            if abs(config.cpu_mem_gb - value) < 0.01:  # Float comparison
                levels['cpu_mem_level'] = level
                break
        
        # GPU SM
        for level, value in RESOURCE_BINS['gpu_sm'].items():
            if config.gpu_sm == value:
                levels['gpu_sm_level'] = level
                break
        
        # GPU memory
        for level, value in RESOURCE_BINS['gpu_mem_gb'].items():
            if abs(config.gpu_mem_gb - value) < 0.01:  # Float comparison
                levels['gpu_mem_level'] = level
                break
        
        if len(levels) != 4:
            raise ValueError(f"Could not map config to levels: {config}")
        
        return levels


if __name__ == "__main__":
    # Test the profiler
    print("Testing ResourceProfiler")
    print("=" * 60)
    
    # Load profiler
    data_dir = Path(__file__).parent.parent.parent / "data"
    profiler = ResourceProfiler(str(data_dir / "profiling.csv"))
    
    # Test get_all_configs
    print("\nAll configs for image_classification (first 5):")
    configs = profiler.get_all_configs('image_classification', 'small')
    for cfg, lat in configs[:5]:
        print(f"  {cfg} -> {lat:.1f}ms")
    
    # Test get_feasible_configs
    print("\nFeasible configs with limited resources:")
    available = ResourceConfig(4, 8.0, 40, 4.0)
    feasible = profiler.get_feasible_configs('image_classification', available, 'small')
    print(f"  Available: {available}")
    print(f"  Found {len(feasible)} feasible configs (showing first 5):")
    for cfg, lat in feasible[:5]:
        print(f"    {cfg} -> {lat:.1f}ms")
    
    # Test get_latency
    print("\nQuery specific config:")
    test_config = ResourceConfig(2, 4.0, 20, 2.0)
    latency = profiler.get_latency('image_classification', test_config, 'small')
    print(f"  {test_config} -> {latency}ms")
    
    # Test level mapping
    print("\nMap config to levels:")
    levels = profiler.map_config_to_levels(test_config)
    print(f"  {test_config}")
    print(f"  Levels: {levels}")
