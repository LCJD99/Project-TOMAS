"""
Resource scenario generator.

Generates multiple resource constraint scenarios to ensure different
tool configurations are selected under different resource availability.
"""

import random
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Union

# Handle both module import and standalone execution
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.planning.resource_profiler import ResourceConfig
else:
    from .resource_profiler import ResourceConfig


class ResourceScenario:
    """Represents a resource availability scenario."""
    
    def __init__(self, name: str, available: ResourceConfig):
        self.name = name
        self.available = available
    
    def __repr__(self):
        return f"Scenario({self.name}: {self.available})"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON output."""
        return {
            'cpu_core': self.available.cpu_core,
            'cpu_memory': self.available.cpu_mem_gb,
            'gpu_sm': self.available.gpu_sm,
            'gpu_memory': self.available.gpu_mem_gb
        }


class ScenarioGenerator:
    """Generates resource availability scenarios for a task."""
    
    def __init__(self, system_state: Mapping[str, Union[int, float]], seed: int = 42):
        """
        Initialize scenario generator.
        
        Args:
            system_state: Maximum system resources
            seed: Random seed for reproducibility
        """
        self.max_resources = ResourceConfig(
            cpu_core=int(system_state['cpu core']),
            cpu_mem_gb=float(system_state['cpu memory']),
            gpu_sm=int(system_state['gpu sm']),
            gpu_mem_gb=float(system_state['gpu memory'])
        )
        self.seed = seed
        random.seed(seed)

    @staticmethod
    def _quantize_resource(value: float, minimum: int) -> int:
        """Round resource value to nearest int and enforce minimum."""
        quantized = int(round(value))
        return max(minimum, quantized)

    def generate_scenarios(self, num_scenarios: int = 3) -> List[ResourceScenario]:
        """
        Generate different resource scenarios.
        
        The goal is to create scenarios with different resource levels
        that will likely lead to different tool configuration selections.
        
        Args:
            num_scenarios: Number of scenarios to generate
            
        Returns:
            List of ResourceScenario objects
        """
        scenarios = []
        
        if num_scenarios >= 1:
            # Scenario 1: High resources (90-100% of max)
            scenarios.append(self._generate_high_resource_scenario())
        
        if num_scenarios >= 2:
            # Scenario 2: Medium resources (50-70% of max)
            scenarios.append(self._generate_medium_resource_scenario())
        
        if num_scenarios >= 3:
            # Scenario 3: Low resources (30-50% of max)
            scenarios.append(self._generate_low_resource_scenario())
        
        # Generate additional scenarios with varying levels
        for i in range(3, num_scenarios):
            scenarios.append(self._generate_random_scenario(i))
        
        return scenarios
    
    def _generate_high_resource_scenario(self) -> ResourceScenario:
        """Generate high resource scenario (90-100% of max)."""
        factor = random.uniform(0.90, 1.0)

        available = ResourceConfig(
            cpu_core=self._quantize_resource(self.max_resources.cpu_core * factor, 2),
            cpu_mem_gb=self._quantize_resource(self.max_resources.cpu_mem_gb * factor, 4),
            gpu_sm=self._quantize_resource(self.max_resources.gpu_sm * factor, 20),
            gpu_mem_gb=self._quantize_resource(self.max_resources.gpu_mem_gb * factor, 2)
        )

        return ResourceScenario("high", available)

    def _generate_medium_resource_scenario(self) -> ResourceScenario:
        """Generate medium resource scenario (50-70% of max)."""
        factor = random.uniform(0.50, 0.70)

        available = ResourceConfig(
            cpu_core=self._quantize_resource(self.max_resources.cpu_core * factor, 2),
            cpu_mem_gb=self._quantize_resource(self.max_resources.cpu_mem_gb * factor, 4),
            gpu_sm=self._quantize_resource(self.max_resources.gpu_sm * factor, 20),
            gpu_mem_gb=self._quantize_resource(self.max_resources.gpu_mem_gb * factor, 2)
        )

        return ResourceScenario("medium", available)

    def _generate_low_resource_scenario(self) -> ResourceScenario:
        """Generate low resource scenario (30-50% of max)."""
        factor = random.uniform(0.30, 0.50)

        available = ResourceConfig(
            cpu_core=self._quantize_resource(self.max_resources.cpu_core * factor, 2),
            cpu_mem_gb=self._quantize_resource(self.max_resources.cpu_mem_gb * factor, 4),
            gpu_sm=self._quantize_resource(self.max_resources.gpu_sm * factor, 20),
            gpu_mem_gb=self._quantize_resource(self.max_resources.gpu_mem_gb * factor, 2)
        )

        return ResourceScenario("low", available)

    def _generate_random_scenario(self, index: int) -> ResourceScenario:
        """Generate a random resource scenario."""
        factor = random.uniform(0.25, 0.95)

        available = ResourceConfig(
            cpu_core=self._quantize_resource(self.max_resources.cpu_core * factor, 2),
            cpu_mem_gb=self._quantize_resource(self.max_resources.cpu_mem_gb * factor, 4),
            gpu_sm=self._quantize_resource(self.max_resources.gpu_sm * factor, 20),
            gpu_mem_gb=self._quantize_resource(self.max_resources.gpu_mem_gb * factor, 2)
        )

        return ResourceScenario(f"scenario_{index}", available)
    
    def generate_diverse_scenarios(self, num_scenarios: int = 3) -> List[ResourceScenario]:
        """
        Generate scenarios with maximum diversity.
        
        Uses different strategies for each resource type to ensure
        different configurations will be selected.
        
        Args:
            num_scenarios: Number of scenarios to generate
            
        Returns:
            List of ResourceScenario objects
        """
        scenarios = []
        
        # Strategy 1: Favor CPU resources, limit GPU
        if len(scenarios) < num_scenarios:
            available = ResourceConfig(
                cpu_core=self._quantize_resource(self.max_resources.cpu_core, 2),
                cpu_mem_gb=self._quantize_resource(self.max_resources.cpu_mem_gb, 4),
                gpu_sm=self._quantize_resource(self.max_resources.gpu_sm * 0.4, 20),
                gpu_mem_gb=self._quantize_resource(self.max_resources.gpu_mem_gb * 0.4, 2)
            )
            scenarios.append(ResourceScenario("cpu_favored", available))

        # Strategy 2: Favor GPU resources, limit CPU
        if len(scenarios) < num_scenarios:
            available = ResourceConfig(
                cpu_core=self._quantize_resource(self.max_resources.cpu_core * 0.4, 2),
                cpu_mem_gb=self._quantize_resource(self.max_resources.cpu_mem_gb * 0.4, 4),
                gpu_sm=self._quantize_resource(self.max_resources.gpu_sm, 20),
                gpu_mem_gb=self._quantize_resource(self.max_resources.gpu_mem_gb, 2)
            )
            scenarios.append(ResourceScenario("gpu_favored", available))

        # Strategy 3: Balanced but limited
        if len(scenarios) < num_scenarios:
            factor = 0.5
            available = ResourceConfig(
                cpu_core=self._quantize_resource(self.max_resources.cpu_core * factor, 2),
                cpu_mem_gb=self._quantize_resource(self.max_resources.cpu_mem_gb * factor, 4),
                gpu_sm=self._quantize_resource(self.max_resources.gpu_sm * factor, 20),
                gpu_mem_gb=self._quantize_resource(self.max_resources.gpu_mem_gb * factor, 2)
            )
            scenarios.append(ResourceScenario("balanced", available))
        
        # Fill remaining with standard scenarios
        while len(scenarios) < num_scenarios:
            scenarios.extend(self.generate_scenarios(1))
        
        return scenarios[:num_scenarios]


if __name__ == "__main__":
    # Test scenario generator
    print("Testing ScenarioGenerator")
    print("=" * 60)
    
    # System state
    system_state = {
        'cpu core': 16,
        'cpu memory': 32,
        'gpu sm': 100,
        'gpu memory': 16
    }
    
    generator = ScenarioGenerator(system_state, seed=42)
    
    print("\nStandard scenarios:")
    scenarios = generator.generate_scenarios(3)
    for i, scenario in enumerate(scenarios):
        print(f"\nScenario {i+1}: {scenario.name}")
        print(f"  {scenario.available}")
        print(f"  JSON: {scenario.to_dict()}")
    
    print("\n" + "=" * 60)
    print("\nDiverse scenarios:")
    diverse = generator.generate_diverse_scenarios(3)
    for i, scenario in enumerate(diverse):
        print(f"\nScenario {i+1}: {scenario.name}")
        print(f"  {scenario.available}")
