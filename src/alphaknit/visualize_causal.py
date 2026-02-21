import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_causal_reality_map(history_path, output_path="artifacts/causal_reality_map.png"):
    """
    v6.6-F: Visualizes Hypothesis Survival vs. Training Phase.
    """
    if not os.path.exists(history_path):
        print(f"❌ Error: {history_path} not found.")
        return

    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = [row["epoch"] for row in history]
    hypo_names = list(history[0]["causal_confidence"].keys()) if history and "causal_confidence" in history[0] else []
    
    if not hypo_names:
        print("⚠️ No causal hypotheses found in history.")
        return

    # Map status to numeric values for plotting
    status_map = {
        "PROPOSED": 1,
        "VERIFIED": 2,
        "FALSIFIED": 0,
        "REJECTED_BY_CONTROL": -1
    }

    plt.figure(figsize=(12, 6))
    
    for name in hypo_names:
        status_history = []
        for row in history:
            status = row["causal_confidence"].get(name, "PROPOSED")
            status_history.append(status_map.get(status, 1))
        
        plt.step(epochs, status_history, where='post', label=name, alpha=0.8, linewidth=2)

    plt.yticks([-1, 0, 1, 2], ["REJECTED", "FALSIFIED", "PROPOSED", "VERIFIED"])
    plt.xlabel("Epoch")
    plt.ylabel("Scientific Status")
    plt.title("AlphaKnit v6.6-F: Causal Reality Mapping (Hypothesis Survival)")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"✅ Causal Reality Map saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=str, required=True, help="Path to training_history.json")
    parser.add_argument("--output", type=str, default="artifacts/causal_reality_map.png")
    args = parser.parse_args()
    
    plot_causal_reality_map(args.history, args.output)
