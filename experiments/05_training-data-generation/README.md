# Tool DAG Analysis

This project provides data structures and analysis tools for representing and analyzing Tool DAG (Directed Acyclic Graph) structures from task data.

## Project Structure

```
.
├── src/
│   └── models/
│       ├── __init__.py           # Model exports
│       ├── tool_node.py          # Node and link data structures
│       └── tool_dag.py           # Main DAG data structure
├── scripts/
│   ├── analyze_tasks.py          # Task analysis script
│   ├── augment_tasks.py          # Data augmentation script
│   ├── analyze_augmented.py      # Augmented data analysis
│   └── example_usage.py          # Usage examples
└── data/
    ├── tasks.json                # Original task data
    └── tasks_augmented.json      # Augmented task data (generated)
```

## Data Structures

### ToolDAG

The main data structure representing a complete tool workflow DAG. Each DAG contains:

- `id`: Unique identifier
- `seed`: Random seed
- `n_tools`: Number of tools in the workflow
- `type`: Workflow type (e.g., "single", "chain", etc.)
- `sampled_nodes`: List of sampled nodes with task information
- `sampled_links`: Links between sampled nodes
- `instruction`: User instruction/query
- `tool_steps`: Step-by-step execution plan
- `tool_nodes`: Actual tool nodes with execution arguments
- `tool_links`: Links between tool nodes (defines the DAG structure)

### Node Types

**SampledNode**: Template node with task metadata
- `task`: Task name
- `input_type`: List of input types
- `output_type`: List of output types

**ToolNode**: Executable node with runtime arguments
- `task`: Task name
- `arguments`: List of arguments for execution

**Link**: Directed edge in the DAG
- `source`: Source task name
- `target`: Target task name

## Usage

### 1. Analyze Original Tasks

Run the analysis script to compute statistics on the original task data:

```bash
python scripts/analyze_tasks.py
```

This will:
1. Load all tasks from `data/tasks.json`
2. Parse them into ToolDAG objects
3. Compute node degree statistics (in-degree and out-degree)
4. Display results grouped by degree combinations

### 2. Data Augmentation

Generate augmented tasks by merging multiple DAGs:

```bash
# Basic usage (generates 1000 augmented tasks, each merging 3 subtasks)
python scripts/augment_tasks.py

# Custom parameters
python scripts/augment_tasks.py \
  --input data/tasks.json \
  --output data/tasks_augmented.json \
  --num-augmented 1000 \
  --merge-count 3 \
  --seed 42
```

**Parameters:**
- `--input`: Path to input tasks JSON file (default: `data/tasks.json`)
- `--output`: Path to output augmented tasks JSON file (default: `data/tasks_augmented.json`)
- `--num-augmented`: Number of augmented tasks to generate (default: 1000)
- `--merge-count`: Number of tasks to merge per augmented task (default: 3)
- `--seed`: Random seed for reproducibility (default: 42)

**What the augmentation does:**
1. Adds a unified START node to all original tasks
2. Randomly selects N tasks and merges their DAGs as subgraphs
3. Connects all subgraph root nodes to the START node
4. Saves both original tasks (with START) and merged tasks to output file

### 3. Analyze Augmented Dataset

After generating augmented data, analyze it:

```bash
python scripts/analyze_augmented.py
```

This provides detailed statistics on the augmented dataset including:
- Task type distribution
- Node and link statistics for merged tasks
- Degree distribution across all tasks
- Sample merged task examples

### Using the Data Structures

```python
from src.models import ToolDAG

# Load a task
import json
with open('data/tasks.json', 'r') as f:
    data = json.load(f)

# Create ToolDAG object
task = ToolDAG.from_dict(data['data'][0])

# Analyze node degrees
degrees = task.get_node_degrees()
# Returns: {'Task Name': (in_degree, out_degree), ...}

# Get all degree pairs
degree_pairs = task.get_all_node_degree_pairs()
# Returns: [(in_degree, out_degree), ...]
```

## Statistics Output

The analysis script provides:

1. **Total Statistics**: Number of tasks, nodes, and links
2. **Degree Distribution**: Count of nodes by (in-degree, out-degree) combinations
3. **In-Degree Summary**: Total nodes grouped by in-degree
4. **Out-Degree Summary**: Total nodes grouped by out-degree

Degrees are categorized as: 0, 1, 2, 3, 4+

## Example Output

```
================================================================================
TASK STATISTICS
================================================================================
Total number of tasks: 866
Total number of nodes: 931
Total number of links: 64

================================================================================
NODE DEGREE DISTRIBUTION
================================================================================
In-Degree    Out-Degree   Count     
--------------------------------------------------------------------------------
0            0            820       
0            1            46        
0            2            1         
1            0            48        
1            1            16        
================================================================================
```

This shows that most tasks (820) are single isolated nodes with no connections, while 64 nodes are involved in workflows with dependencies.

## Data Augmentation Results

After running the augmentation script with default parameters:

```
Original tasks (with START): 866
Augmented merged tasks: 1000
Total tasks: 1866

Merged task statistics:
  Average nodes per task: 4.23
  Average links per task: 3.23
  Min nodes: 4, Max nodes: 8
  Min links: 3, Max links: 7
```

The augmented dataset significantly increases the complexity of the DAG structures:
- **Original dataset**: Mostly isolated nodes (in-degree 0, out-degree 0)
- **Augmented dataset**: All tasks have a START node, merged tasks have multiple interconnected subtasks
- **Benefit**: Provides more complex workflow patterns for training and evaluation
