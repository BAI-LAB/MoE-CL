import argparse
import re
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np


# Predefined task orders for MTL5
TASK_ORDERS = {
    "order1": ["dbpedia", "amazon", "yahoo", "agnews"],
    "order2": ["dbpedia", "amazon", "agnews", "yahoo"],
    "order3": ["yahoo", "amazon", "agnews", "dbpedia"],
}

# All tasks in MTL5
ALL_TASKS = ["agnews", "amazon", "dbpedia", "yahoo"]


def parse_log_file(log_file: str, task_order: List[str]) -> np.ndarray:
    """
    Parse log file and extract accuracy matrix R.
    
    R[i, j] = accuracy on task j after training on task i
    
    Args:
        log_file: Path to log.txt
        task_order: List of task names in training order
        
    Returns:
        R: T x T accuracy matrix
    """
    T = len(task_order)
    R = np.zeros((T, T))
    
    # Create task name to index mapping (based on training order)
    task_to_idx = {task: i for i, task in enumerate(task_order)}
    
    # Read log file
    with open(log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Parse log entries
    # Format: "[timestamp]: Training {task} finished!" or "[timestamp]: {task} acc: {value}"
    current_trained_task = None
    current_trained_idx = -1
    
    acc_pattern = re.compile(r"\[.*?\]:\s*(\w+)\s+acc:\s*([\d.]+)")
    finish_pattern = re.compile(r"\[.*?\]:\s*Training\s+(\w+)\s+finished!")
    
    # Store accuracies for each training stage
    stage_accs = defaultdict(dict)  # stage_accs[trained_task_idx][test_task] = acc
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if this is a "Training finished" line
        finish_match = finish_pattern.match(line)
        if finish_match:
            trained_task = finish_match.group(1).lower()
            if trained_task in task_to_idx:
                current_trained_task = trained_task
                current_trained_idx = task_to_idx[trained_task]
            continue
        
        # Check if this is an accuracy line
        acc_match = acc_pattern.match(line)
        if acc_match and current_trained_idx >= 0:
            test_task = acc_match.group(1).lower()
            acc_value = float(acc_match.group(2))
            
            if test_task in task_to_idx:
                test_idx = task_to_idx[test_task]
                stage_accs[current_trained_idx][test_idx] = acc_value
    
    # Fill the R matrix
    for trained_idx, test_accs in stage_accs.items():
        for test_idx, acc in test_accs.items():
            R[trained_idx, test_idx] = acc
    
    return R


def calculate_metrics(R: np.ndarray, task_order: List[str], 
                      random_init_acc: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """
    Calculate ACC, BWT, and FWT metrics.
    
    Args:
        R: T x T accuracy matrix where R[i,j] = acc on task j after training task i
        task_order: List of task names in training order
        random_init_acc: Dict of {task_name: acc} for random initialization baseline (for FWT)
        
    Returns:
        Dictionary containing ACC, BWT, FWT metrics
    """
    T = len(task_order)
    
    # ACC: Average accuracy after learning all tasks
    # ACC = (1/T) * sum(R[T-1, j]) for j in 0..T-1
    ACC = np.mean(R[T-1, :])
    
    # BWT: Backward Transfer (measures forgetting)
    # BWT = (1/(T-1)) * sum(R[T-1, i] - R[i, i]) for i in 0..T-2
    if T > 1:
        bwt_sum = 0.0
        for i in range(T - 1):
            bwt_sum += R[T-1, i] - R[i, i]
        BWT = bwt_sum / (T - 1)
    else:
        BWT = 0.0
    
    # FWT: Forward Transfer (measures positive transfer to new tasks)
    # FWT = (1/(T-1)) * sum(R[i-1, i] - b_i) for i in 1..T-1
    # where b_i is random init baseline for task i
    if T > 1 and random_init_acc is not None:
        fwt_sum = 0.0
        task_to_idx = {task: i for i, task in enumerate(task_order)}
        for i in range(1, T):
            task_name = task_order[i]
            b_i = random_init_acc.get(task_name, 0.0)
            fwt_sum += R[i-1, i] - b_i
        FWT = fwt_sum / (T - 1)
    else:
        FWT = None  # Cannot compute without baseline
    
    return {
        "ACC": ACC,
        "BWT": BWT,
        "FWT": FWT,
    }


def parse_random_init_log(log_file: str) -> Dict[str, float]:
    """
    Parse random initialization log to get baseline accuracies.
    
    Args:
        log_file: Path to rand_init log.txt
        
    Returns:
        Dictionary of {task_name: baseline_acc}
    """
    baseline_acc = {}
    
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        acc_pattern = re.compile(r"\[.*?\]:\s*(\w+)\s+acc:\s*([\d.]+)")
        
        for line in lines:
            line = line.strip()
            acc_match = acc_pattern.match(line)
            if acc_match:
                task_name = acc_match.group(1).lower()
                acc_value = float(acc_match.group(2))
                baseline_acc[task_name] = acc_value
                
    except FileNotFoundError:
        print(f"Warning: Random init log file not found: {log_file}")
        
    return baseline_acc


def print_results(R: np.ndarray, task_order: List[str], metrics: Dict[str, float]):
    """Pretty print the results."""
    T = len(task_order)
    
    print("\n" + "=" * 70)
    print("MoE-CL Continual Learning Metrics (MTL5)")
    print("=" * 70)
    
    # Print task order
    print(f"\nTask Order: {' → '.join(task_order)}")
    
    # Print accuracy matrix
    print("\n" + "-" * 70)
    print("Accuracy Matrix R[i,j] (row i = after training task i, col j = test on task j)")
    print("-" * 70)
    
    # Header
    header = "Trained \\ Test |"
    for task in task_order:
        header += f" {task:>10} |"
    print(header)
    print("-" * len(header))
    
    # Rows
    for i, trained_task in enumerate(task_order):
        row = f"{trained_task:>14} |"
        for j in range(T):
            acc = R[i, j]
            if acc > 0:
                row += f" {acc*100:>9.2f}% |"
            else:
                row += f" {'N/A':>10} |"
        print(row)
    
    # Print metrics
    print("\n" + "-" * 70)
    print("Metrics")
    print("-" * 70)
    
    print(f"  ACC (Average Accuracy):     {metrics['ACC']*100:.2f}%")
    print(f"  BWT (Backward Transfer):    {metrics['BWT']*100:+.2f}%")
    if metrics['FWT'] is not None:
        print(f"  FWT (Forward Transfer):     {metrics['FWT']*100:+.2f}%")
    else:
        print(f"  FWT (Forward Transfer):     N/A (no random init baseline provided)")
    
    print("\n" + "=" * 70)
    
    # Interpretation
    print("\nInterpretation:")
    print(f"  - ACC: Final average performance across all tasks")
    print(f"  - BWT < 0: Forgetting occurred (negative = worse)")
    print(f"  - BWT > 0: Backward transfer (learning new tasks improved old ones)")
    if metrics['FWT'] is not None:
        print(f"  - FWT > 0: Positive forward transfer (prior knowledge helps new tasks)")
        print(f"  - FWT < 0: Negative transfer (prior knowledge hurts new tasks)")
    print()


def main():
    parser = argparse.ArgumentParser(description="Calculate BWT and FWT metrics for MoE-CL on MTL5")
    parser.add_argument(
        "--log_file", 
        type=str, 
        required=True,
        help="Path to the log.txt file (e.g., results/moe-cl/mtl5/order1/log.txt)"
    )
    parser.add_argument(
        "--order",
        type=str,
        default=None,
        choices=["order1", "order2", "order3"],
        help="Predefined task order (order1, order2, or order3)"
    )
    parser.add_argument(
        "--task_order",
        type=str,
        default=None,
        help="Custom task order as comma-separated list (e.g., dbpedia,amazon,yahoo,agnews)"
    )
    parser.add_argument(
        "--random_init_log",
        type=str,
        default=None,
        help="Path to random initialization log.txt for FWT baseline"
    )
    parser.add_argument(
        "--adapter_name",
        type=str,
        default="moe-cl",
        help="Adapter name (default: moe-cl)"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="mtl5",
        help="Benchmark name (default: mtl5)"
    )
    
    args = parser.parse_args()
    
    # Determine task order
    if args.task_order:
        task_order = [t.strip().lower() for t in args.task_order.split(",")]
    elif args.order:
        task_order = TASK_ORDERS[args.order]
    else:
        # Default to order1
        task_order = TASK_ORDERS["order1"]
        print(f"No order specified, using default: {task_order}")
    
    print(f"Task order: {task_order}")
    print(f"Log file: {args.log_file}")
    
    # Parse the log file
    R = parse_log_file(args.log_file, task_order)
    
    # Parse random init baseline if provided
    random_init_acc = None
    if args.random_init_log:
        random_init_acc = parse_random_init_log(args.random_init_log)
        print(f"Random init baseline: {random_init_acc}")
    else:
        # Try to find rand_init log automatically
        import os
        log_dir = os.path.dirname(args.log_file)
        parent_dir = os.path.dirname(log_dir)
        rand_init_log = os.path.join(parent_dir, "rand_init", "log.txt")
        if os.path.exists(rand_init_log):
            random_init_acc = parse_random_init_log(rand_init_log)
            print(f"Auto-detected random init log: {rand_init_log}")
            print(f"Random init baseline: {random_init_acc}")
    
    # Calculate metrics
    metrics = calculate_metrics(R, task_order, random_init_acc)
    
    # Print results
    print_results(R, task_order, metrics)
    
    # Return metrics for programmatic use
    return metrics


if __name__ == "__main__":
    main()
