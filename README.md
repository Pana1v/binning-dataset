# Pick-and-Place Binning Dataset

A large-scale, standardized dataset for pick-and-place binning tasks with multiple object and bin configurations. This dataset is designed for benchmarking and training algorithms for robotic binning operations.

## Dataset Overview

This dataset contains 30,000 total scenarios across 6 variants, each with different object counts:
- **3 objects**: 5,000 scenarios
- **5 objects**: 5,000 scenarios
- **10 objects**: 5,000 scenarios
- **40 objects**: 5,000 scenarios
- **100 objects**: 5,000 scenarios
- **200 objects**: 5,000 scenarios

Each scenario includes:
- Object positions and types
- Bin positions (fixed at workspace corners)
- Robot start position
- Optimal greedy solution sequence and cost

## Bin Configuration

The dataset uses a fixed 4-bin configuration at the corners of a 100×100 workspace:

| Bin Index | Position | Description |
|-----------|----------|-------------|
| 0 | `[10, 10]` | Top-left corner |
| 1 | `[90, 10]` | Top-right corner |
| 2 | `[10, 90]` | Bottom-left corner |
| 3 | `[90, 90]` | Bottom-right corner |

```
Workspace Layout:
(0,0)                    (100,0)
  ┌─────────────────────────┐
  │                         │
  │  Bin 0      Bin 1       │
  │  [10,10]    [90,10]     │
  │                         │
  │                         │
  │                         │
  │  Bin 2      Bin 3       │
  │  [10,90]    [90,90]     │
  │                         │
  └─────────────────────────┘
(0,100)                  (100,100)
```

## Data Format

Each scenario is stored as a JSON object with the following schema:

```json
{
  "objects": [[x1, y1], [x2, y2], ...],
  "types": [0, 1, 2, 3, ...],
  "bins": [[10, 10], [90, 10], [10, 90], [90, 90]],
  "start": [x, y],
  "greedy_sequence": [2, 0, 1, ...],
  "greedy_cost": 123.45,
  "metadata": {
    "object_count": 10,
    "bin_count": 4,
    "bin_configuration": "corners",
    "random_seed": 12345
  }
}
```

### Field Descriptions

- **`objects`**: List of object positions `[x, y]` in the workspace (10-90 range)
- **`types`**: List of object types (0-3), where each type maps to a bin index
- **`bins`**: Fixed list of 4 bin positions at workspace corners
- **`start`**: Initial robot position `[x, y]` (randomly generated, 10-90 range)
- **`greedy_sequence`**: Optimal greedy pick sequence (list of object indices)
- **`greedy_cost`**: Total distance traveled by the greedy solution
- **`metadata`**: Additional information about the scenario

## Greedy Algorithm

The greedy solution minimizes the total distance traveled using the following strategy:

At each step, select the object that minimizes:
```
distance(robot_position, object) + distance(object, target_bin)
```

The algorithm:
1. Starts at the robot's initial position
2. For each remaining object, calculates: `d_pick + d_place`
3. Selects the object with minimum cost
4. Moves robot to that object's target bin
5. Repeats until all objects are placed

## Usage

### Loading the Dataset

```python
import json

# Load a dataset file
with open('data/dataset_10_objects.json', 'r') as f:
    scenarios = json.load(f)

# Access a scenario
scenario = scenarios[0]
print(f"Objects: {scenario['objects']}")
print(f"Types: {scenario['types']}")
print(f"Greedy sequence: {scenario['greedy_sequence']}")
print(f"Greedy cost: {scenario['greedy_cost']}")
```

### Working with Scenarios

```python
import numpy as np

# Convert to numpy arrays for easier manipulation
objects = np.array(scenario['objects'])
types = np.array(scenario['types'])
bins = np.array(scenario['bins'])
start = np.array(scenario['start'])

# Calculate distance between two points
def distance(p1, p2):
    return np.linalg.norm(p1 - p2)

# Example: Calculate cost of a custom sequence
def calculate_sequence_cost(sequence, objects, types, bins, start):
    cost = 0.0
    current_pos = start.copy()
    
    for obj_idx in sequence:
        obj_pos = objects[obj_idx]
        bin_pos = bins[types[obj_idx]]
        
        cost += distance(current_pos, obj_pos)
        cost += distance(obj_pos, bin_pos)
        current_pos = bin_pos
    
    return cost

# Compare custom sequence with greedy
custom_seq = [0, 1, 2]
custom_cost = calculate_sequence_cost(custom_seq, objects, types, bins, start)
greedy_cost = scenario['greedy_cost']

print(f"Custom cost: {custom_cost:.2f}")
print(f"Greedy cost: {greedy_cost:.2f}")
print(f"Improvement: {(custom_cost - greedy_cost) / greedy_cost * 100:.2f}%")
```

## Generating the Dataset

To regenerate the dataset files:

```bash
cd scripts
python generate_dataset.py
```

This will generate all 6 dataset files in the `data/` directory.

## Analyzing the Dataset

To analyze and validate the dataset:

```bash
cd scripts
python analyze_dataset.py
```

This will:
- Validate all scenarios
- Compute statistics (cost distributions, object spacing, etc.)
- Check for quality issues
- Generate summary reports

## Dataset Specifications

- **Workspace Size**: 100 × 100 units
- **Object Position Range**: 10-90 (ensuring objects stay within workspace)
- **Minimum Object Distance**: 0.1 units between any two objects (minimal constraint to prevent exact overlaps)
- **Robot Start Position**: Random within workspace (10-90 range)
- **Object Types**: Randomly assigned (0-3), uniformly distributed
- **Reproducibility**: Each scenario uses a fixed random seed (scenario index)


## File Structure

```
pick-place-binning-dataset/
├── data/
│   ├── dataset_3_objects.json
│   ├── dataset_5_objects.json
│   ├── dataset_10_objects.json
│   ├── dataset_40_objects.json
│   ├── dataset_100_objects.json
│   └── dataset_200_objects.json
├── scripts/
│   ├── generate_dataset.py
│   └── analyze_dataset.py
├── README.md
└── LICENSE
```

## Requirements

- Python 3.7+
- numpy
- tqdm (for progress bars during generation)

Install dependencies:
```bash
pip install numpy tqdm
```

## License

This dataset is released under the MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{pick_place_binning_2025,
  title={Pick-and-Place Binning Dataset},
  author={Pana1v},
  year={2025},
  url={https://github.com/Pana1v/binning-dataset}
}
```

## Versioning

Dataset versions are tracked using Git tags:
- `v1.0` - Initial release with 30,000 scenarios

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

### Reporting Issues

If you find any issues with the dataset or have suggestions for improvements, please open an issue on GitHub with:
- A clear description of the problem
- Steps to reproduce (if applicable)
- Expected vs actual behavior

## Acknowledgments

This dataset was generated for benchmarking pick-and-place binning algorithms and can be used for:
- Training machine learning models (e.g., GNNs, reinforcement learning)
- Benchmarking optimization algorithms
- Testing heuristic approaches
- Academic research

