"""Tool Node representation for DAG structure."""

from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class SampledNode:
    """Represents a sampled node with task information."""
    task: str
    input_type: List[str]
    output_type: List[str]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SampledNode':
        return cls(
            task=data['task'],
            input_type=data['input-type'],
            output_type=data['output-type']
        )


@dataclass
class ToolNode:
    """Represents a tool node with execution arguments."""
    task: str
    arguments: List[str]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolNode':
        # Handle different field names for arguments
        # Some tasks use 'arguments', others use 'parameters', 'params', or 'input'
        arguments = data.get('arguments') or data.get('parameters') or data.get('params') or data.get('input') or []
        
        # Ensure arguments is always a list
        if not isinstance(arguments, list):
            arguments = [arguments] if arguments else []
        
        return cls(
            task=data['task'],
            arguments=arguments
        )


@dataclass
class Link:
    """Represents a link between two nodes."""
    source: str
    target: str
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Link':
        return cls(
            source=data['source'],
            target=data['target']
        )
