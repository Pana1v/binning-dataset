import numpy as np
import json
import os
from typing import List, Dict
from tqdm import tqdm
import torch

# Constants
WORKSPACE_SIZE = 100.0
MIN_DISTANCE = 0.1
BIN_POSITIONS = [[10, 10], [90, 10], [10, 90], [90, 90]]
NUM_BINS = 4
NUM_SCENARIOS = 5000
OBJECT_COUNTS = [3, 5, 10, 40, 100, 200]

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_locations(n: int, min_dist: float = MIN_DISTANCE, seed: int = None) -> np.ndarray:
    """Generate random valid locations with minimum distance constraint (GPU-accelerated)."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    locs = []
    max_attempts = 10000
    
    while len(locs) < n:
        attempts = 0
        found = False
        
        while attempts < max_attempts and not found:
            cand = np.random.uniform(10, WORKSPACE_SIZE - 10, size=2)
            
            if not locs:
                locs.append(cand)
                found = True
            else:
                # GPU-accelerated distance calculation
                cand_tensor = torch.tensor(cand, device=device, dtype=torch.float32)
                locs_tensor = torch.tensor(locs, device=device, dtype=torch.float32)
                distances = torch.norm(locs_tensor - cand_tensor, dim=1)
                min_dist_gpu = torch.min(distances).item()
                
                if min_dist_gpu >= min_dist:
                    locs.append(cand)
                    found = True
            
            attempts += 1
        
        if not found:
            break
    
    return np.array(locs)


def greedy_solve(obj_locs: np.ndarray, obj_types: np.ndarray, 
                 bin_locs: List[List[float]], start_pos: np.ndarray) -> tuple:
    """
    Greedy solver that minimizes: distance(robot, object) + distance(object, bin)
    At each step, selects the object that minimizes this sum (GPU-accelerated).
    """
    # Convert to GPU tensors
    obj_locs_t = torch.tensor(obj_locs, device=device, dtype=torch.float32)
    bin_locs_t = torch.tensor(bin_locs, device=device, dtype=torch.float32)
    obj_types_t = torch.tensor(obj_types, device=device, dtype=torch.long)
    curr = torch.tensor(start_pos, device=device, dtype=torch.float32)
    
    remaining = set(range(len(obj_locs)))
    seq = []
    cost = 0.0
    
    while remaining:
        remaining_list = list(remaining)
        remaining_t = torch.tensor(remaining_list, device=device, dtype=torch.long)
        
        # Vectorized distance calculations on GPU
        obj_locs_remaining = obj_locs_t[remaining_t]
        d_pick = torch.norm(obj_locs_remaining - curr, dim=1)
        
        bin_indices = obj_types_t[remaining_t]
        bin_positions = bin_locs_t[bin_indices]
        d_place = torch.norm(obj_locs_remaining - bin_positions, dim=1)
        
        total_costs = d_pick + d_place
        
        # Find minimum
        min_idx_local = torch.argmin(total_costs).item()
        best_idx = remaining_list[min_idx_local]
        min_cost = total_costs[min_idx_local].item()
        
        seq.append(best_idx)
        remaining.discard(best_idx)
        curr = bin_locs_t[obj_types_t[best_idx]]
        cost += min_cost
    
    return seq, cost


def generate_scenario(object_count: int, bin_locs: List[List[float]], 
                     min_dist: float = MIN_DISTANCE, seed: int = None) -> Dict:
    """Generate a single scenario with objects, types, bins, and greedy solution."""
    if seed is not None:
        np.random.seed(seed)
    
    # Generate object positions
    obj_locs = get_locations(object_count, min_dist, seed)
    
    if len(obj_locs) < object_count:
        raise ValueError(f"Failed to generate {object_count} objects with min_dist={min_dist}")
    
    # Assign random types (0-3) to objects
    obj_types = np.random.randint(0, NUM_BINS, size=object_count)
    
    # Generate random robot start position
    start_pos = np.random.uniform(10, WORKSPACE_SIZE - 10, size=2)
    
    # Compute greedy solution
    greedy_seq, greedy_cost = greedy_solve(obj_locs, obj_types, bin_locs, start_pos)
    
    return {
        "objects": obj_locs.tolist(),
        "types": obj_types.tolist(),
        "bins": bin_locs,
        "start": start_pos.tolist(),
        "greedy_sequence": greedy_seq,
        "greedy_cost": float(greedy_cost),
        "metadata": {
            "object_count": object_count,
            "bin_count": NUM_BINS,
            "bin_configuration": "corners",
            "random_seed": seed if seed is not None else None
        }
    }


def save_dataset(scenarios: List[Dict], filename: str):
    """Save scenarios to JSON file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump(scenarios, f, indent=2)
    
    file_size = os.path.getsize(filename) / (1024 * 1024)  # Size in MB
    print(f"Dataset saved to {filename} ({len(scenarios)} scenarios, {file_size:.2f} MB)")


def validate_scenario(scenario: Dict, expected_count: int) -> bool:
    """Validate a scenario meets quality requirements (GPU-accelerated)."""
    if len(scenario["objects"]) != expected_count:
        return False
    if len(scenario["types"]) != expected_count:
        return False
    if len(scenario["bins"]) != NUM_BINS:
        return False
    if len(scenario["greedy_sequence"]) != expected_count:
        return False
    
    # GPU-accelerated distance check
    obj_locs = np.array(scenario["objects"])
    obj_locs_t = torch.tensor(obj_locs, device=device, dtype=torch.float32)
    
    # Compute pairwise distances using broadcasting
    diff = obj_locs_t.unsqueeze(0) - obj_locs_t.unsqueeze(1)
    distances = torch.norm(diff, dim=2)
    
    # Check lower triangle (excluding diagonal)
    mask = torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)
    min_dist = torch.min(distances[mask]).item()
    
    if min_dist < MIN_DISTANCE - 0.1:
        return False
    
    return True


def main():
    """Generate all dataset files."""
    print("Starting dataset generation...")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Object counts: {OBJECT_COUNTS}")
    print(f"Scenarios per count: {NUM_SCENARIOS}")
    print(f"Bin positions: {BIN_POSITIONS}")
    print(f"Minimum object distance: {MIN_DISTANCE}")
    print("-" * 60)
    
    for count in OBJECT_COUNTS:
        print(f"\nGenerating dataset for {count} objects...")
        scenarios = []
        failed = 0
        
        for i in tqdm(range(NUM_SCENARIOS), desc=f"Generating {count} objects"):
            try:
                scenario = generate_scenario(count, BIN_POSITIONS, seed=i)
                
                if validate_scenario(scenario, count):
                    scenarios.append(scenario)
                else:
                    failed += 1
                    if failed < 10:
                        print(f"Warning: Scenario {i} failed validation, retrying...")
            except Exception as e:
                failed += 1
                if failed < 10:
                    print(f"Error generating scenario {i}: {e}")
        
        if len(scenarios) < NUM_SCENARIOS:
            print(f"Warning: Only generated {len(scenarios)}/{NUM_SCENARIOS} valid scenarios")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(os.path.dirname(script_dir), "data")
        filename = os.path.join(data_dir, f"dataset_{count}_objects.json")
        save_dataset(scenarios, filename)
    
    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

