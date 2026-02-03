"""
Tool name mapping between tasks and profiling data.

Maps tool names from tasks_augmented.json to profiling.csv format.
"""

# Mapping from task JSON format to profiling CSV format
TASK_TO_PROFILING = {
    'Image Classification': 'image_classification',
    'Image-to-Text': 'image_captioning',
    'Translation': 'machine_translation',
    'Object Detection': 'object_detection',
    'Summarization': 'text_summarization',
    'Visual Question Answering': 'visual_question_answering',
    'Super Resolution': 'super_resolution',
    'START': None,  # Virtual node, no profiling data
}

# Reverse mapping
PROFILING_TO_TASK = {v: k for k, v in TASK_TO_PROFILING.items() if v is not None}


def task_to_profiling_name(task_name: str) -> str:
    """
    Convert task name to profiling name.
    
    Handles prefixed names like "T1_Translation" -> "machine_translation"
    
    Args:
        task_name: Task name from JSON (may include prefix like "T1_")
        
    Returns:
        Profiling tool name, or None if it's a virtual node
        
    Raises:
        ValueError: If tool name is not recognized
    """
    # Remove prefix like T1_, T2_, etc.
    clean_name = task_name
    if '_' in task_name:
        parts = task_name.split('_', 1)
        # Check if first part is like T1, T2, T3, etc.
        if len(parts[0]) >= 2 and parts[0][0] == 'T' and parts[0][1:].isdigit():
            clean_name = parts[1]
    
    # Look up in mapping
    if clean_name not in TASK_TO_PROFILING:
        raise ValueError(f"Unknown task name: {task_name} (cleaned: {clean_name})")
    
    return TASK_TO_PROFILING[clean_name]


def profiling_to_task_name(profiling_name: str) -> str:
    """
    Convert profiling name to task name.
    
    Args:
        profiling_name: Tool name from profiling CSV
        
    Returns:
        Task name
        
    Raises:
        ValueError: If profiling name is not recognized
    """
    if profiling_name not in PROFILING_TO_TASK:
        raise ValueError(f"Unknown profiling name: {profiling_name}")
    
    return PROFILING_TO_TASK[profiling_name]


def is_virtual_node(task_name: str) -> bool:
    """
    Check if a task name represents a virtual node (like START).
    
    Args:
        task_name: Task name from JSON
        
    Returns:
        True if it's a virtual node, False otherwise
    """
    # Remove prefix
    clean_name = task_name
    if '_' in task_name:
        parts = task_name.split('_', 1)
        if len(parts[0]) >= 2 and parts[0][0] == 'T' and parts[0][1:].isdigit():
            clean_name = parts[1]
    
    return clean_name == 'START' or TASK_TO_PROFILING.get(clean_name) is None


if __name__ == "__main__":
    # Test the mappings
    print("Testing tool_mapper.py")
    print("=" * 60)
    
    test_cases = [
        "Image Classification",
        "T1_Translation",
        "T2_Object Detection",
        "START",
        "Visual Question Answering",
    ]
    
    for test in test_cases:
        try:
            profiling = task_to_profiling_name(test)
            is_virtual = is_virtual_node(test)
            print(f"{test:30s} -> {profiling or 'None':30s} (virtual: {is_virtual})")
        except ValueError as e:
            print(f"{test:30s} -> ERROR: {e}")
    
    print("\n" + "=" * 60)
    print("Reverse mapping test:")
    for prof_name in ['image_classification', 'machine_translation']:
        task_name = profiling_to_task_name(prof_name)
        print(f"{prof_name:30s} -> {task_name}")
