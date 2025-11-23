import json
import os
import numpy as np
from typing import Dict, List
from collections import Counter

OBJECT_COUNTS = [3, 5, 10, 40, 100, 200]
NUM_BINS = 4
MIN_DISTANCE = 10.0


def load_dataset(filename: str) -> List[Dict]:
    """Load dataset from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)


def validate_dataset(dataset: List[Dict], expected_count: int) -> Dict:
    """Validate dataset and return statistics."""
    stats = {
        "total_scenarios": len(dataset),
        "valid_scenarios": 0,
        "invalid_scenarios": [],
        "object_count_errors": 0,
        "distance_violations": 0,
        "sequence_errors": 0,
        "cost_stats": {
            "min": float('inf'),
            "max": float('-inf'),
            "mean": 0.0,
            "std": 0.0
        },
        "object_spacing_stats": {
            "min": float('inf'),
            "max": float('-inf'),
            "mean": 0.0
        }
    }
    
    costs = []
    min_distances = []
    
    for i, scenario in enumerate(dataset):
        is_valid = True
        
        # Check object count
        if len(scenario["objects"]) != expected_count:
            stats["object_count_errors"] += 1
            is_valid = False
        
        if len(scenario["types"]) != expected_count:
            stats["object_count_errors"] += 1
            is_valid = False
        
        if len(scenario["greedy_sequence"]) != expected_count:
            stats["sequence_errors"] += 1
            is_valid = False
        
        # Check minimum distance between objects
        obj_locs = np.array(scenario["objects"])
        min_dist = float('inf')
        for j in range(len(obj_locs)):
            for k in range(j + 1, len(obj_locs)):
                dist = np.linalg.norm(obj_locs[j] - obj_locs[k])
                min_dist = min(min_dist, dist)
                if dist < MIN_DISTANCE - 0.1:
                    stats["distance_violations"] += 1
                    is_valid = False
        
        if min_dist < float('inf'):
            min_distances.append(min_dist)
        
        # Collect cost
        costs.append(scenario["greedy_cost"])
        
        if is_valid:
            stats["valid_scenarios"] += 1
        else:
            stats["invalid_scenarios"].append(i)
    
    # Compute cost statistics
    if costs:
        stats["cost_stats"]["min"] = float(np.min(costs))
        stats["cost_stats"]["max"] = float(np.max(costs))
        stats["cost_stats"]["mean"] = float(np.mean(costs))
        stats["cost_stats"]["std"] = float(np.std(costs))
    
    # Compute spacing statistics
    if min_distances:
        stats["object_spacing_stats"]["min"] = float(np.min(min_distances))
        stats["object_spacing_stats"]["max"] = float(np.max(min_distances))
        stats["object_spacing_stats"]["mean"] = float(np.mean(min_distances))
    
    return stats


def analyze_type_distribution(dataset: List[Dict]) -> Dict:
    """Analyze distribution of object types."""
    type_counts = Counter()
    total_objects = 0
    
    for scenario in dataset:
        for obj_type in scenario["types"]:
            type_counts[obj_type] += 1
            total_objects += 1
    
    distribution = {}
    for bin_idx in range(NUM_BINS):
        count = type_counts.get(bin_idx, 0)
        distribution[f"bin_{bin_idx}"] = {
            "count": count,
            "percentage": (count / total_objects * 100) if total_objects > 0 else 0.0
        }
    
    return distribution


def analyze_start_positions(dataset: List[Dict]) -> Dict:
    """Analyze robot start position distribution."""
    starts = np.array([scenario["start"] for scenario in dataset])
    
    return {
        "x": {
            "min": float(np.min(starts[:, 0])),
            "max": float(np.max(starts[:, 0])),
            "mean": float(np.mean(starts[:, 0])),
            "std": float(np.std(starts[:, 0]))
        },
        "y": {
            "min": float(np.min(starts[:, 1])),
            "max": float(np.max(starts[:, 1])),
            "mean": float(np.mean(starts[:, 1])),
            "std": float(np.std(starts[:, 1]))
        }
    }


def print_report(dataset_name: str, stats: Dict, type_dist: Dict, start_pos: Dict):
    """Print analysis report."""
    print("\n" + "=" * 70)
    print(f"Dataset Analysis: {dataset_name}")
    print("=" * 70)
    
    print(f"\n📊 Dataset Overview:")
    print(f"  Total scenarios: {stats['total_scenarios']}")
    print(f"  Valid scenarios: {stats['valid_scenarios']}")
    print(f"  Invalid scenarios: {len(stats['invalid_scenarios'])}")
    
    if stats['invalid_scenarios']:
        print(f"  ⚠️  Invalid scenario indices: {stats['invalid_scenarios'][:10]}...")
    
    print(f"\n🔍 Validation Results:")
    print(f"  Object count errors: {stats['object_count_errors']}")
    print(f"  Distance violations: {stats['distance_violations']}")
    print(f"  Sequence errors: {stats['sequence_errors']}")
    
    print(f"\n💰 Cost Statistics:")
    print(f"  Min cost: {stats['cost_stats']['min']:.2f}")
    print(f"  Max cost: {stats['cost_stats']['max']:.2f}")
    print(f"  Mean cost: {stats['cost_stats']['mean']:.2f}")
    print(f"  Std dev: {stats['cost_stats']['std']:.2f}")
    
    print(f"\n📏 Object Spacing Statistics:")
    print(f"  Min distance: {stats['object_spacing_stats']['min']:.2f}")
    print(f"  Max distance: {stats['object_spacing_stats']['max']:.2f}")
    print(f"  Mean distance: {stats['object_spacing_stats']['mean']:.2f}")
    
    print(f"\n🎯 Object Type Distribution:")
    for bin_idx in range(NUM_BINS):
        bin_info = type_dist[f"bin_{bin_idx}"]
        print(f"  Bin {bin_idx}: {bin_info['count']} objects ({bin_info['percentage']:.1f}%)")
    
    print(f"\n🤖 Robot Start Position Statistics:")
    print(f"  X: min={start_pos['x']['min']:.2f}, max={start_pos['x']['max']:.2f}, "
          f"mean={start_pos['x']['mean']:.2f}, std={start_pos['x']['std']:.2f}")
    print(f"  Y: min={start_pos['y']['min']:.2f}, max={start_pos['y']['max']:.2f}, "
          f"mean={start_pos['y']['mean']:.2f}, std={start_pos['y']['std']:.2f}")
    
    print("=" * 70)


def main():
    """Analyze all dataset files."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), "data")
    
    print("Dataset Analysis Tool")
    print("=" * 70)
    
    all_stats = {}
    
    for count in OBJECT_COUNTS:
        filename = os.path.join(data_dir, f"dataset_{count}_objects.json")
        
        if not os.path.exists(filename):
            print(f"\n⚠️  File not found: {filename}")
            continue
        
        print(f"\n📁 Loading {filename}...")
        dataset = load_dataset(filename)
        
        stats = validate_dataset(dataset, count)
        type_dist = analyze_type_distribution(dataset)
        start_pos = analyze_start_positions(dataset)
        
        print_report(f"dataset_{count}_objects.json", stats, type_dist, start_pos)
        
        all_stats[count] = {
            "stats": stats,
            "type_dist": type_dist,
            "start_pos": start_pos
        }
    
    # Summary across all datasets
    print("\n\n" + "=" * 70)
    print("Summary Across All Datasets")
    print("=" * 70)
    
    total_scenarios = sum(s["stats"]["total_scenarios"] for s in all_stats.values())
    total_valid = sum(s["stats"]["valid_scenarios"] for s in all_stats.values())
    
    print(f"\nTotal scenarios across all datasets: {total_scenarios}")
    print(f"Total valid scenarios: {total_valid}")
    print(f"Overall validity rate: {total_valid/total_scenarios*100:.2f}%")
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()

